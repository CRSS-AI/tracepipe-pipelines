from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
import json


@dataclass
class ToolDefinition:
    """Schema definition for a tool callable by the LLM.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of tool functionality.
        parameters: JSON Schema defining the tool's input parameters.
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    """Represents a single tool invocation requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool to invoke.
        arguments: Parsed arguments to pass to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AgentResult:
    """Result of an agentic classification run.

    Attributes:
        action: The classified action label, or None if classification failed.
        window_start: Starting index of the classified window.
        window_end: Ending index (exclusive) of the classified window.
        tool_calls_count: Total number of tool calls made during classification.
        model: Model identifier used for classification.
        error: Error message if classification failed, None otherwise.
        usage: Token usage statistics from the LLM.
    """

    action: str | None
    window_start: int
    window_end: int
    tool_calls_count: int
    model: str
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        output: JSON string output to return to the LLM.
        terminal: Whether this result terminates the agent loop.
        terminal_data: Data to extract when terminal is True.
    """

    output: str
    terminal: bool = False
    terminal_data: dict[str, Any] = field(default_factory=dict)


ToolExecutor = Callable[[str, dict[str, Any]], ToolResult]


def load_json_string(json_string: str) -> dict[str, Any]:
    """Load a JSON string into a Python dictionary, handling code block formatting.

    Args:
        json_string: A string in JSON format.

    Returns:
        A dictionary representation of the JSON string.
    """
    raw_json = json_string.strip()
    if raw_json.startswith("```json"):
        raw_json = raw_json[7:]
    if raw_json.startswith("```"):
        raw_json = raw_json[3:]
    if raw_json.endswith("```"):
        raw_json = raw_json[:-3]
    raw_json = raw_json.strip()
    return json.loads(raw_json)


class ClientProtocol(Protocol):
    """Protocol for LLM clients supporting tool-based agentic classification."""

    def __init__(self, *, api_key: str | None, max_retries: int = 3):
        """Initialize ClientProtocol with API key and retry configuration.

        Args:
            api_key: API key for authentication.
            max_retries: Maximum number of retry attempts for failed requests.
        """
        ...

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
            providers: Optional list of provider names (client-specific).

        Returns:
            Generated text response.
        """
        ...

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
        """Run an agentic loop with tool calling until a terminal action is reached.

        The loop continues until the tool executor returns a ToolResult with
        terminal=True, which signals the agent should stop.

        Args:
            model: Model identifier.
            system_prompt: System instructions for the agent.
            tools: List of available tool definitions.
            tool_executor: Callable that executes tools and returns ToolResult.
                Returns ToolResult with terminal=True when done.
            max_iterations: Maximum tool call iterations before forced termination.
            temperature: Sampling temperature for generation.
            providers: Optional list of provider names (client-specific).

        Returns:
            AgentResult containing the final action and metadata.
        """
        ...