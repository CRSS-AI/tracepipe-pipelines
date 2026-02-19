import argparse
import base64
import json
import os
from dataclasses import dataclass, field
from tempfile import template
from typing import Any, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from action_extraction.extraction import logger
from action_extraction.window_classification.clients.client_factory import get_client
from action_extraction.window_classification.clients.client_protocol import ToolDefinition, ToolResult

load_dotenv()

INITIAL_WINDOW_SIZE = 50
MAX_ITERATIONS = 25

# Default intermediate directory for storing chunks and responses
DEFAULT_INTERMEDIATE_DIR = "../../data/intermediate/window_classification/intermediate"

# Cost and context safety knobs
DEFAULT_TRACES_LIMIT = 25
MAX_TRACES_LIMIT = 60
MAX_WINDOW_DELTA_PER_CALL = 200
MAX_IMAGE_BYTES = 1_200_000  # 1.2MB guardrail per screenshot payload

ALLOWED_ACTIONS = [
    "send_email",
    "create_draft",
    "read_email",
    "download_attachment",
    "search_emails",
    "modify_email_labels",
    "delete_email",
    "list_labels",
    "create_label",
    "get_or_create_label",
    "update_label",
    "delete_label",
    "batch_modify_emails",
    "batch_delete_emails",
    "create_filter",
    "delete_filter",
    "get_filter",
    "create_filter_from_template",
]

@dataclass
class WindowState:
    """Mutable state for the current window."""

    df: pd.DataFrame
    start_idx: int
    end_idx: int
    total_rows: int

    @property
    def size(self) -> int:
        return self.end_idx - self.start_idx

    @property
    def traces_before(self) -> int:
        return self.start_idx

    @property
    def traces_after(self) -> int:
        return self.total_rows - self.end_idx


@dataclass
class ClassifiedAction:
    """A single classified action with window boundaries and metadata."""

    start_seq: Any
    end_seq: Any
    action: str

def load_prompt() -> str:
    """Load the window classification prompt from prompts/window_classification_prompt.md."""
    prompt_path = os.path.join(os.path.dirname(__file__), "window_classification", "window_classification_prompt.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def get_system_prompt(**kwargs) -> str:
    """Load the window classification prompt from prompts/window_classification_prompt.md."""
    prompt_path = os.path.join(os.path.dirname(__file__), "window_classification", "window_classification_prompt.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    return template.format(**kwargs)

def generate_chunks(*, num_chunks: int, chunk_margin: int, df: pd.DataFrame) -> list[pd.DataFrame]:
    """Generate chunks of the dataframe for processing.

    Args:
        num_chunks: Number of chunks to divide the dataframe into.
        chunk_margin: Number of overlapping rows between consecutive chunks.
        df: The full dataframe to chunk.

    Returns:
        A list of dataframe chunks.
    """
    n_rows = len(df)
    if n_rows == 0:
        return []

    chunk_size = n_rows // num_chunks
    chunks: list[pd.DataFrame] = []
    start = 0
    end = chunk_size

    for _ in range(max(1, num_chunks)):
        if start >= n_rows:
            break
        end = min(start + chunk_size, n_rows)
        chunks.append(df.iloc[start:end])
        start = max(0, end - chunk_margin)

    return chunks

def df_to_string(df: pd.DataFrame) -> str:
    """Convert a dataframe to a string representation for prompting."""
    str_df = df.to_csv(index=True)
    return str_df


def _is_bad_request_error(error: Exception) -> bool:
    """Return True when the error looks like a 400 Bad Request.

    Args:
        error: Exception raised by the client.

    Returns:
        True if the error indicates a 400 Bad Request.
    """
    message = str(error).lower()
    return "400" in message or "bad request" in message


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove image_bytes field from dataframe to reduce size.

    Args:
        df: Input dataframe with optional image_bytes column.

    Returns:
        A copy of the dataframe with image_bytes column removed (if present).
    """
    if "image_bytes" not in df.columns:
        return df
    return df.drop(columns=["image_bytes"])


def extract_actions_from_response(response: str) -> list[ClassifiedAction]:
    """Extract classified actions from the agent's final response.

    This function assumes the response is a JSON array of action classifications.
    Each classification should include start_seq, end_seq, and action fields.

    Args:
        response: The raw string response from the agent.
    Returns:
        A list of ClassifiedAction instances extracted from the response.
    """
    try:
        data = json.loads(response)
        if not isinstance(data, list):
            logger.error("Expected a list of classifications in the response")
            return []
        classified_actions: list[ClassifiedAction] = []
        for item in data:
            if not all(k in item for k in ("start_seq", "end_seq", "action")):
                logger.warning("Skipping invalid classification item: %s", item)
                continue
            classified_actions.append(
                ClassifiedAction(
                    start_seq=item["start_seq"],
                    end_seq=item["end_seq"],
                    action=item["action"],
                )
            )
        return classified_actions
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON from response: %s", e)
        return []


def classify_traces(
    df: pd.DataFrame,
    *,
    client_type: str,
    model_name: str,
    api_key: str | None,
    providers: list[str] | None = None,
    intermediate_dir: str | None = None,
) -> list[ClassifiedAction]:
    """
    Classify all traces into actions using chunked prompting.

    The dataframe is split into chunks. Each chunk is converted to a string and
    sent to the client for classification. If any request fails with a 400 error,
    the chunks are subdivided and retried.

    Args:
        df: Input dataframe of traces.
        client_type: LLM client type.
        model_name: Model identifier.
        api_key: API key for the selected client.
        providers: Optional provider override list.
        intermediate_dir: Directory to store intermediate chunks and responses.

    Returns:
        List of classified actions.
    """
    client = get_client(client_type, api_key=api_key, max_retries=2)
    classified_actions: list[ClassifiedAction] = []
    n_rows = len(df)
    if n_rows == 0:
        return classified_actions

    system_prompt = load_prompt()
    num_chunks = 1
    chunk_margin = 10

    while True:

        chunks = generate_chunks(num_chunks=num_chunks, chunk_margin=chunk_margin, df=df)
        logger.info("Classifying %s chunks", len(chunks))

        retry_with_smaller_chunks = False

        for idx, chunk_df in enumerate(chunks):
            if chunk_df.empty:
                logger.warning("Skipping empty chunk %s/%s", idx + 1, len(chunks))
                continue
            chunk_string = df_to_string(chunk_df)

            if intermediate_dir:
                chunk_file = os.path.join(intermediate_dir, f"chunk_{num_chunks}_{idx:03d}.csv")
                os.makedirs(intermediate_dir, exist_ok=True)
                with open(chunk_file, "w", encoding="utf-8") as f:
                    f.write(chunk_string)
                logger.info("Saved chunk to %s", chunk_file)

            prompt = chunk_string

            try:
                response = client.generate(
                    model=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    providers=providers,
                )

                if intermediate_dir:
                    response_file = os.path.join(intermediate_dir, f"response_{num_chunks}_{idx:03d}.json")
                    with open(response_file, "w", encoding="utf-8") as f:
                        json.dump({"response": response}, f, indent=2)
                    logger.info("Saved response to %s", response_file)

                classified_actions = extract_actions_from_response(response)
            except Exception as exc:
                if _is_bad_request_error(exc):
                    logger.warning(
                        "Chunk %s/%s failed with 400; dividing further",
                        idx + 1,
                        len(chunks),
                    )
                    retry_with_smaller_chunks = True
                    break
                raise

        if retry_with_smaller_chunks:
            num_chunks = num_chunks*2
            if num_chunks == n_rows:
                logger.error("Reached maximum chunk division without success")
                return classified_actions
            classified_actions.clear()
            continue
        break

    return classified_actions


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Classify trace windows into actions")
    parser.add_argument("--input", required=True, help="Input directory with parquet files")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--intermediate", default=DEFAULT_INTERMEDIATE_DIR, help="Intermediate directory for chunks and responses")
    args = parser.parse_args()

    client_type = os.getenv("LLM_CLIENT_TYPE", "openrouter")
    model_providers: list[str] | None = None

    match client_type:
        case "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            model_name = os.getenv("OPENROUTER_MODEL_NAME", "")
            providers_env = os.getenv("OPENROUTER_MODEL_PROVIDERS")
            if providers_env:
                model_providers = [p.strip() for p in providers_env.split(",") if p.strip()]
        case "azure":
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "")
        case "vertex":
            api_key = None
            model_name = os.getenv("VERTEX_MODEL_NAME", "")
        case _:
            logger.error("Unknown client type: %s", client_type)
            return

    if not api_key and client_type != "vertex":
        logger.error("API key not found for %s", client_type)
        return
    if not model_name:
        logger.error("Model name not configured")
        return

    logger.info("Window Classification reading from %s", args.input)
    os.makedirs(args.output, exist_ok=True)

    for filename in os.listdir(args.input):
        if not filename.endswith(".parquet"):
            continue

        in_path = os.path.join(args.input, filename)
        logger.info("Processing %s", in_path)

        df = pd.read_parquet(in_path)

        df = clean_df(df)

        intermediate_subdir = os.path.join(args.intermediate, os.path.splitext(filename)[0])

        classified = classify_traces(
            df,
            client_type=client_type,
            model_name=model_name,
            api_key=api_key,
            providers=model_providers,
            intermediate_dir=intermediate_subdir,
        )

        out_records = [
            {
                "start_seq": ca.start_seq,
                "end_seq": ca.end_seq,
                "action": ca.action,
            }
            for ca in classified
        ]

        out_df = pd.DataFrame(out_records)
        out_path = os.path.join(args.output, filename)
        logger.info("Writing %s actions to %s", len(out_df), out_path)
        out_df.to_parquet(out_path, index=False)
        out_df.to_csv(out_path.replace(".parquet", ".csv"), index=False)


if __name__ == "__main__":
    main()