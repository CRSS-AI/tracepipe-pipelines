from __future__ import annotations

import base64
import json
import os
import random
import time
from typing import Any, Optional

from google import genai
from google.genai import types
from google.oauth2 import service_account

from action_extraction.extraction import logger
from action_extraction.window_classification.clients.client_protocol import (
    AgentResult,
    ClientProtocol,
    ToolCall,
    ToolDefinition,
    ToolExecutor,
    ToolResult,
)


class VertexClient(ClientProtocol):
    """Client for Vertex AI (Gemini) with tool calling agent loop."""

    def __init__(self, *, api_key: str | None = None, max_retries: int = 3, timeout_s: int = 600):
        """
        Args:
            api_key: Unused for Vertex (kept for protocol compatibility).
            max_retries: Max attempts for retryable failures.
            timeout_s: Not all transports expose timeout; kept for parity and future use.
        """
        self.max_retries = max_retries
        self.timeout_s = timeout_s

        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")

        credentials = None
        if sa_path:
            if not os.path.exists(sa_path):
                raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS path does not exist: {sa_path}")
            credentials = service_account.Credentials.from_service_account_file(
                sa_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )

        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            credentials=credentials,
        )

    @staticmethod
    def _content_part_to_vertex(part: dict[str, Any]) -> types.Part:
        """Convert an OpenAI-style content part (text/image_url) into a Vertex Part."""
        ptype = part.get("type")

        if ptype == "text":
            return types.Part.from_text(text=str(part.get("text", "")))

        if ptype == "image_url":
            image_url = (part.get("image_url") or {}).get("url", "")
            if not image_url:
                raise ValueError("image_url part missing url")

            # Expect data URI: data:<mime>;base64,<...>
            if not image_url.startswith("data:"):
                raise ValueError("VertexClient only supports data URI images")

            meta, data = image_url.split(",", 1)
            mime_type = meta.split(":")[1].split(";")[0]
            image_bytes = base64.b64decode(data)
            return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        return types.Part.from_text(text=str(part))

    @staticmethod
    def _build_tools(tools: list[ToolDefinition]) -> list[types.Tool]:
        """
        Convert tool definitions to Vertex tool declarations.

        Vertex tool format expects a Tool with function_declarations. Each declaration follows
        a JSON-schema-like parameters object (your ToolDefinition already uses this format).
        """
        declarations = [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in tools
        ]
        return [types.Tool(function_declarations=declarations)]

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
            providers: Ignored for Vertex.

        Returns:
            Generated text response.
        """
        contents: list[types.Content] = []
        
        if isinstance(prompt, str):
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
            )
        else:
            for msg in prompt:
                role = msg.get("role", "user")
                if role == "system":
                    continue
                
                vertex_role = "model" if role == "assistant" else "user"
                content_val = msg.get("content")
                
                if isinstance(content_val, str):
                    contents.append(
                        types.Content(role=vertex_role, parts=[types.Part.from_text(text=content_val)])
                    )
                elif isinstance(content_val, list):
                    parts = [self._content_part_to_vertex(part) for part in content_val]
                    contents.append(types.Content(role=vertex_role, parts=parts))
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_prompt,
        )
        
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                
                candidates = getattr(response, "candidates", None) or []
                if not candidates:
                    raise ValueError("No candidates in Vertex response")
                
                content = getattr(candidates[0], "content", None)
                if not content:
                    raise ValueError("No content in Vertex candidate")
                
                parts = getattr(content, "parts", None) or []
                if not parts:
                    return ""
                
                text_parts = [getattr(p, "text", "") for p in parts if hasattr(p, "text")]
                return "".join(text_parts)
                
            except Exception as e:
                error_msg = str(e).lower()
                # Check for context/token limit errors - raise immediately
                if any(kw in error_msg for kw in ["context", "token limit", "invalid_argument"]):
                    logger.warning("Vertex error - context may have been exceeded: %s", str(e))
                    raise
                
                if self._is_retryable_exception(e) and attempt < self.max_retries:
                    sleep_s = (2**attempt) + random.random()
                    logger.warning(
                        "Vertex generate retryable error (attempt %s/%s). Sleeping %.2fs. %s",
                        attempt,
                        self.max_retries,
                        sleep_s,
                        str(e),
                    )
                    time.sleep(sleep_s)
                    continue
                raise
        
        raise RuntimeError("Failed to generate response after retries")

    @staticmethod
    def _extract_tool_calls(response: Any) -> tuple[list[ToolCall], Optional[str]]:
        """Extract function calls and any text from a Vertex response."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return [], None

        content = getattr(candidates[0], "content", None)
        if content is None:
            return [], None

        parts = getattr(content, "parts", None) or []

        tool_calls: list[ToolCall] = []
        text_chunks: list[str] = []

        for p in parts:
            t = getattr(p, "text", None)
            if isinstance(t, str) and t:
                text_chunks.append(t)

            fc = getattr(p, "function_call", None)
            if fc is None:
                continue

            name = getattr(fc, "name", "") or ""
            args = dict(getattr(fc, "args", {}) or {})

            tool_calls.append(
                ToolCall(
                    id=f"call_{name}_{len(tool_calls)}",
                    name=name,
                    arguments=args,
                )
            )

        text_content = "".join(text_chunks).strip() if text_chunks else None
        return tool_calls, text_content

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return {}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
            "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
            "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
        }

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        """
        Conservative retry policy for transient failures.

        The google-genai SDK typically raises transport / HTTP exceptions that
        include status codes or messages (429/5xx). We keep this heuristic simple.
        """
        msg = str(exc).lower()
        retry_markers = [
            "429",
            "rate limit",
            "resource exhausted",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "unavailable",
            "internal",
            "500",
            "502",
            "503",
            "504",
            "connection reset",
            "connection aborted",
        ]
        return any(m in msg for m in retry_markers)

    @staticmethod
    def _make_tool_config_any_if_available() -> Any:
        """
        Best-effort: force function calling when supported by the SDK.

        Some versions expose:
          types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
        If not available, returns None.
        """
        tool_config_cls = getattr(types, "ToolConfig", None)
        fn_cfg_cls = getattr(types, "FunctionCallingConfig", None)
        if tool_config_cls is None or fn_cfg_cls is None:
            return None
        try:
            return tool_config_cls(function_calling_config=fn_cfg_cls(mode="ANY"))
        except Exception:
            return None

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
        Tool calling loop until a terminal ToolResult is returned.

        Notes:
          - Vertex does not use `providers`.
          - To avoid oversized prompts, the agent should explore via tools (get_traces pagination, get_screenshot, etc).
        """
        vertex_tools = self._build_tools(tools)

        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_calls_count = 0

        # Conversation state: keep it minimal. The system instruction is in config.
        contents: list[types.Content] = []

        max_output_tokens = 1024
        max_no_tool_turns = 2
        no_tool_turns = 0

        tool_config = self._make_tool_config_any_if_available()

        for iteration in range(max_iterations):
            # Build a fresh config each iteration to avoid accidental mutation.
            config = types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                system_instruction=system_prompt,
                tools=vertex_tools,
            )
            if tool_config is not None:
                # Best-effort: encourage tool calling strongly.
                setattr(config, "tool_config", tool_config)

            # A small “kick” user message helps some models start calling tools.
            if not contents:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="Begin by using tools to inspect the current window.")],
                    )
                )

            last_error = ""
            response = None

            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    break
                except Exception as e:
                    last_error = str(e)
                    retryable = self._is_retryable_exception(e)
                    if retryable and attempt < self.max_retries:
                        sleep_s = (2**attempt) + random.random()
                        logger.warning(
                            "Vertex retryable error (attempt %s/%s). Sleeping %.2fs. %s",
                            attempt,
                            self.max_retries,
                            sleep_s,
                            last_error,
                        )
                        time.sleep(sleep_s)
                        continue

                    logger.error("Vertex error on iteration %s: %s", iteration + 1, last_error)
                    return AgentResult(
                        action=None,
                        window_start=0,
                        window_end=0,
                        tool_calls_count=tool_calls_count,
                        model=model,
                        error=last_error,
                        usage=total_usage,
                    )

            if response is None:
                return AgentResult(
                    action=None,
                    window_start=0,
                    window_end=0,
                    tool_calls_count=tool_calls_count,
                    model=model,
                    error="No response received from Vertex",
                    usage=total_usage,
                )

            usage = self._extract_usage(response)
            total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            total_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            total_usage["total_tokens"] += int(usage.get("total_tokens", 0) or 0)

            tool_calls, assistant_text = self._extract_tool_calls(response)

            if not tool_calls:
                no_tool_turns += 1
                logger.warning("Vertex: no tool calls in iteration %s (count %s)", iteration + 1, no_tool_turns)

                if assistant_text:
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=assistant_text)]))

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

                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text="You must call one tool now. Do not answer with plain text.")],
                    )
                )
                continue

            no_tool_turns = 0

            # Append model content (including the function calls) to the conversation.
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                model_content = getattr(candidates[0], "content", None)
                if model_content is not None:
                    contents.append(model_content)

            # Execute tools; feed results back as function responses.
            for tc in tool_calls:
                tool_calls_count += 1
                try:
                    result = tool_executor(tc.name, tc.arguments)
                except Exception as e:
                    tool_error = {"success": False, "error": f"Tool execution error: {e}"}
                    result = ToolResult(output=json.dumps(tool_error), terminal=False)

                # Function response should be returned as a function_response part.
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=tc.name,
                                response={"result": result.output},
                            )
                        ],
                    )
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
