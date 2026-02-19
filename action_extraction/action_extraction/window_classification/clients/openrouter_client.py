"""OpenRouter API client with tool calling agent support."""

from __future__ import annotations

import json
import random
import time
from typing import Any, Optional

import requests

from action_extraction.extraction import logger
from action_extraction.window_classification.clients.client_protocol import (
    AgentResult,
    ClientProtocol,
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
)


class OpenRouterClient(ClientProtocol):
    """Client for OpenRouter Chat Completions with tool calling support."""

    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, *, api_key: str, max_retries: int = 3, timeout_s: int = 600):
        """
        Args:
            api_key: OpenRouter API key.
            max_retries: Max attempts for retryable failures.
            timeout_s: Request timeout in seconds.
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout_s = timeout_s

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def _build_tools_payload(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert tool definitions to OpenAI style tool payload."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _safe_json_loads(value: Any) -> dict[str, Any]:
        """
        Tool call arguments can arrive as:
          1) a dict
          2) a JSON string
          3) garbage or empty

        Returns a dict in all cases.
        """
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
    def _is_retryable_status(status_code: int) -> bool:
        """Return True for errors worth retrying."""
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504}

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
            model: Model identifier.
            prompt: Either a string prompt or a list of message dicts.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens to generate.
            providers: Optional list of preferred providers.

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
        
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if providers:
            payload["provider"] = {"order": providers}
        
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    self.API_URL,
                    json=payload,
                    timeout=self.timeout_s,
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise ValueError("No choices in OpenRouter response")
                    
                    message = choices[0].get("message", {})
                    return message.get("content", "")
                
                # Raise on 400 - context may have been exceeded
                if resp.status_code == 400:
                    error_body = resp.text[:500] if resp.text else ""
                    logger.warning("OpenRouter 400 Bad Request - context may have been exceeded: %s", error_body)
                    raise ValueError(error_body or "400 Bad Request")
                
                if self._is_retryable_status(resp.status_code) and attempt < self.max_retries:
                    sleep_s = (2**attempt) + random.random()
                    logger.warning(
                        "OpenRouter generate retryable status %s (attempt %s/%s). Sleeping %.2fs.",
                        resp.status_code,
                        attempt,
                        self.max_retries,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                
                resp.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    sleep_s = (2**attempt) + random.random()
                    logger.warning(
                        "OpenRouter generate request error (attempt %s/%s). Sleeping %.2fs. %s",
                        attempt,
                        self.max_retries,
                        sleep_s,
                        str(e),
                    )
                    time.sleep(sleep_s)
                    continue
                raise
        
        raise RuntimeError("Failed to generate response after retries")

    def _send_chat_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float,
        providers: Optional[list[str]],
        max_tokens: int,
    ) -> tuple[dict[str, Any], str]:
        """
        Send a chat completion request.

        Returns:
            (response_json, error_message). error_message is empty on success.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if providers:
            payload["provider"] = {"only": providers, "allow_fallbacks": False}

        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(self.API_URL, json=payload, timeout=self.timeout_s)

                if resp.status_code < 200 or resp.status_code >= 300:
                    body_preview = resp.text[:800] if resp.text else ""
                    last_error = f"HTTP {resp.status_code}: {body_preview}"

                    if self._is_retryable_status(resp.status_code) and attempt < self.max_retries:
                        sleep_s = (2**attempt) + random.random()
                        logger.warning(
                            "OpenRouter retryable HTTP error (attempt %s/%s). Sleeping %.2fs. %s",
                            attempt,
                            self.max_retries,
                            sleep_s,
                            last_error,
                        )
                        time.sleep(sleep_s)
                        continue

                    return {}, last_error

                try:
                    result = resp.json()
                except ValueError:
                    body_preview = resp.text[:800] if resp.text else ""
                    last_error = f"Non JSON response: {body_preview}"

                    if attempt < self.max_retries:
                        sleep_s = (2**attempt) + random.random()
                        logger.warning(
                            "OpenRouter non JSON response (attempt %s/%s). Sleeping %.2fs.",
                            attempt,
                            self.max_retries,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue

                    return {}, last_error

                if "error" in result:
                    last_error = str(result["error"])
                    if attempt < self.max_retries:
                        sleep_s = (2**attempt) + random.random()
                        logger.warning(
                            "OpenRouter API error (attempt %s/%s). Sleeping %.2fs. %s",
                            attempt,
                            self.max_retries,
                            sleep_s,
                            last_error,
                        )
                        time.sleep(sleep_s)
                        continue
                    return {}, last_error

                return result, ""

            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
                if attempt < self.max_retries:
                    sleep_s = (2**attempt) + random.random()
                    logger.warning(
                        "OpenRouter request exception (attempt %s/%s). Sleeping %.2fs. %s",
                        attempt,
                        self.max_retries,
                        sleep_s,
                        last_error,
                    )
                    time.sleep(sleep_s)
                    continue
                return {}, last_error

        return {}, last_error or "Max retries exceeded"

    @staticmethod
    def _extract_tool_calls(response: dict[str, Any]) -> tuple[list[ToolCall], Optional[str]]:
        """
        Extract tool calls and assistant content from OpenAI compatible response.
        """
        choices = response.get("choices") or []
        if not choices:
            return [], None

        message = choices[0].get("message") or {}
        content = message.get("content")
        tool_calls_raw = message.get("tool_calls") or []

        tool_calls: list[ToolCall] = []
        for tc in tool_calls_raw:
            if tc.get("type") != "function":
                continue

            func = tc.get("function") or {}
            name = func.get("name", "") or ""
            args = func.get("arguments", {})

            tool_calls.append(
                ToolCall(
                    id=tc.get("id", "") or "",
                    name=name,
                    arguments=OpenRouterClient._safe_json_loads(args),
                )
            )

        return tool_calls, content

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

        Failure modes are returned as AgentResult with action=None.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        tools_payload = self._build_tools_payload(tools)

        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_calls_count = 0

        max_tokens_per_turn = 1024
        max_no_tool_turns = 2
        no_tool_turns = 0

        for iteration in range(max_iterations):
            response, error = self._send_chat_request(
                model=model,
                messages=messages,
                tools=tools_payload,
                temperature=temperature,
                providers=providers,
                max_tokens=max_tokens_per_turn,
            )

            if error:
                logger.error("Agent iteration %s failed: %s", iteration + 1, error)
                return AgentResult(
                    action=None,
                    window_start=0,
                    window_end=0,
                    tool_calls_count=tool_calls_count,
                    model=model,
                    error=error,
                    usage=total_usage,
                )

            usage = response.get("usage") or {}
            total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            total_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            total_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

            tool_calls, assistant_content = self._extract_tool_calls(response)

            if not tool_calls:
                no_tool_turns += 1
                logger.warning("No tool calls in iteration %s (count %s)", iteration + 1, no_tool_turns)

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

                messages.append(
                    {
                        "role": "user",
                        "content": "You must call one tool now. Do not answer with plain text.",
                    }
                )
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

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.output,
                    }
                )

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
