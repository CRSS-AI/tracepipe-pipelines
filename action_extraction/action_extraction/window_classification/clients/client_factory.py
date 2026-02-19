from action_extraction.window_classification.clients.client_protocol import ClientProtocol
from action_extraction.window_classification.clients.openrouter_client import OpenRouterClient
from action_extraction.window_classification.clients.azure_client import AzureClient
from action_extraction.window_classification.clients.vertex_client import VertexClient


def get_client(
    client_type: str,
    *,
    api_key: str | None = None,
    max_retries: int = 3,
) -> ClientProtocol:
    """Factory function to get the appropriate client based on client_type.

    Args:
        client_type: The type of client to create ('openrouter', 'azure', 'vertex').
        api_key: The API key for authentication (not required for Vertex).
        max_retries: Maximum number of retry attempts for failed requests.

    Returns:
        An instance of a ClientProtocol implementation.

    Raises:
        ValueError: If client_type is not supported.
    """
    match client_type:
        case "openrouter":
            if not api_key:
                raise ValueError("API key required for OpenRouter client")
            return OpenRouterClient(api_key=api_key, max_retries=max_retries)
        case "azure":
            if not api_key:
                raise ValueError("API key required for Azure client")
            return AzureClient(api_key=api_key, max_retries=max_retries)
        case "vertex" | "google-vertex":
            return VertexClient(api_key=api_key, max_retries=max_retries)
        case _:
            raise ValueError(f"Unsupported client type: {client_type}")

