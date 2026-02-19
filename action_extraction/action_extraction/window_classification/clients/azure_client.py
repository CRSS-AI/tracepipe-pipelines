from __future__ import annotations

import json
import random
import time
from typing import Any, Optional

from openai import AzureOpenAI
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError, BadRequestError  # type: ignore

from action_extraction.extraction import logger
from action_extraction.window_classification.clients.client_protocol import (
    AgentResult,
    ClientProtocol,
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
)


class AzureClient(ClientProtocol):
    """Client for Azure OpenAI Chat Completions with tool calling support."""

    def __init__(self, *, api_key: str, max_retries: int = 3, timeout_s: int = 600):
        """
        Args:
            api_key: Azure OpenAI API key.
            max_retries: Max attempts for retryable failures.
            timeout_s: Request timeout in seconds (passed to the SDK call).
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout_s = timeout_s

        self.endpoint = (json.loads(json.dumps(str(__import__("os").environ.get("AZURE_OPENAI_ENDPOINT", ""))))).rstrip("/")  # keep endpoint robust
        self.api_version = __import__("os").environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")

        if not self.endpoint:
            logger.warning("AZURE_OPENAI_ENDPOINT not set. AzureClient may fail.")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            max_retries=0,  # we implement retries ourselves for consistent behavior
        )

    @staticmethod
    def _build_tools_payload(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tool definitions to OpenAI tool payload."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    @staticmethod
    def _safe_json_loads(value: Any) -> dict[str, Any]:
        """Tool call arguments can be dict, JSON string, or garbage. Always return a dict."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return {}
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _extract_tool_calls(message: Any) -> tuple[list[ToolCall], Optional[str]]:
        """Extract tool calls and assistant content from an Azure message object."""
        tool_calls: list[ToolCall] = []
        raw_calls = getattr(message, "tool_calls", None) or []
        content = getattr(message, "content", None)

        for tc in raw_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            name = getattr(func, "name", "") or ""
            args = getattr(func, "arguments", {})  # may be JSON str
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", "") or "",
                    name=name,
                    arguments=AzureClient._safe_json_loads(args),
                )
            )

        return tool_calls, content

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        """Retry for rate limits, transient network issues, and 5xx."""
        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True
        if isinstance(exc, APIError):
            status = getattr(exc, "status_code", None)
            if status in {408, 409, 425, 429, 500, 502, 503, 504}:
                return True
        msg = str(exc).lower()
        return any(k in msg for k in ["timeout", "timed out", "temporarily", "unavailable", "rate limit"])

    def generate(
        self,
        *,
        model: str,
        prompt: str | list[dict[str, Any]],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        providers: list[str] | None = None,
    ) -> str:
        """Generate a simple text response from a prompt.

        Args:
            model: Azure deployment name.
            prompt: Either a string prompt or a list of message dicts.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate.
            providers: Ignored for Azure.

        Returns:
            Generated text response.
        """
        messages: list[dict[str, Any]] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout_s,
                )
                
                if not response.choices:
                    raise ValueError("No choices returned from Azure API")
                
                return response.choices[0].message.content or ""
                
            except BadRequestError as e:
                logger.warning("Azure 400 Bad Request - context may have been exceeded: %s", str(e))
                raise
            except Exception as e:
                if self._is_retryable_exception(e) and attempt < self.max_retries:
                    sleep_s = (2**attempt) + random.random()
                    logger.warning(
                        "Azure generate retryable error (attempt %s/%s). Sleeping %.2fs. %s",
                        attempt,
                        self.max_retries,
                        sleep_s,
                        str(e),
                    )
                    time.sleep(sleep_s)
                    continue
                raise
        
        raise RuntimeError("Failed to generate response after retries")

    def run_agent(
        self,
        *,
        model: str,
        system_prompt: str,
        tools: list[ToolDefinition],
        tool_executor: ToolExecutor,
        max_iterations: int = 25,
        temperature: float = 0.0,
        providers: list[str] | None = None,
    ) -> AgentResult:
        """
        Run a tool calling loop until a terminal ToolResult is returned.

        Notes:
          - `model` is the Azure deployment name.
          - `providers` is ignored for Azure.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        tools_payload = self._build_tools_payload(tools)

        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_calls_count = 0

        max_tokens_per_turn = 1024
        max_no_tool_turns = 2
        no_tool_turns = 0

        for iteration in range(max_iterations):
            response = None
            last_error = ""

            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools_payload,
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=max_tokens_per_turn,
                        timeout=self.timeout_s,
                    )
                    break
                except Exception as e:
                    last_error = str(e)
                    retryable = self._is_retryable_exception(e)
                    if retryable and attempt < self.max_retries:
                        sleep_s = (2**attempt) + random.random()
                        logger.warning(
                            "Azure retryable error (attempt %s/%s). Sleeping %.2fs. %s",
                            attempt,
                            self.max_retries,
                            sleep_s,
                            last_error,
                        )
                        time.sleep(sleep_s)
                        continue

                    logger.error("Azure error on iteration %s: %s", iteration + 1, last_error)
                    return AgentResult(
                        action=None,
                        window_start=0,
                        window_end=0,
                        tool_calls_count=tool_calls_count,
                        model=model,
                        error=last_error,
                        usage=total_usage,
                    )

            if response is None or not getattr(response, "choices", None):
                return AgentResult(
                    action=None,
                    window_start=0,
                    window_end=0,
                    tool_calls_count=tool_calls_count,
                    model=model,
                    error=last_error or "No choices returned by Azure",
                    usage=total_usage,
                )

            usage = getattr(response, "usage", None)
            if usage is not None:
                total_usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                total_usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
                total_usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

            message = response.choices[0].message
            tool_calls, assistant_content = self._extract_tool_calls(message)

            if not tool_calls:
                no_tool_turns += 1
                logger.warning("Azure: no tool calls in iteration %s (count %s)", iteration + 1, no_tool_turns)

                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})

                if no_tool_turns > max_no_tool_turns:
                    return AgentResult(
                        action=None,
                        window_start=0,
                        window_end=0,
                        tool_calls_count=tool_calls_count,
                        model=model,
                        error="Model did not call tools after repeated prompts",
                        usage=total_usage,
                    )

                messages.append({"role": "user", "content": "You must call one tool now. Do not answer with plain text."})
                continue

            no_tool_turns = 0

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if assistant_content:
                assistant_msg["content"] = assistant_content
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)

            for tc in tool_calls:
                tool_calls_count += 1
                try:
                    result = tool_executor(tc.name, tc.arguments)
                except Exception as e:
                    tool_error = {"success": False, "error": f"Tool execution error: {e}"}
                    result = ToolResult(output=json.dumps(tool_error), terminal=False)

                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.output})

                if result.terminal:
                    return AgentResult(
                        action=result.terminal_data.get("action"),
                        window_start=int(result.terminal_data.get("window_start", 0)),
                        window_end=int(result.terminal_data.get("window_end", 0)),
                        tool_calls_count=tool_calls_count,
                        model=model,
                        error=None,
                        usage=total_usage,
                    )

        return AgentResult(
            action=None,
            window_start=0,
            window_end=0,
            tool_calls_count=tool_calls_count,
            model=model,
            error=f"Max iterations ({max_iterations}) reached",
            usage=total_usage,
        )
