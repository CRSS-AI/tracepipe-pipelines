"""Type definitions and schemas for the trace classification pipeline."""

from typing import TypedDict, Any
from pandera.pandas import Column, DataFrameSchema
# Metrics JSON structure
class ProcessingMetrics(TypedDict):
    """Metrics for processing errors and warnings."""
    errors: dict[str, Any]
    warnings: dict[str, Any]


class OutputMetrics(TypedDict):
    """Metrics for output file information."""
    file: str
    file_size_mb: float


class MetricsDict(TypedDict):
    """Standardized metrics dictionary structure."""
    timestamp: str
    summary: dict[str, Any]
    processing: ProcessingMetrics
    output: OutputMetrics


# Bronze layer Pandera schema - for DataFrame validation and Parquet I/O
BRONZE_TRACE_DATAFRAME_SCHEMA = DataFrameSchema(
    {
        "trace": Column(object, nullable=False, description="Complete trace object containing session_uuid, event_type, timestamp, and json_raw"),
        "instance_id": Column(str, nullable=False, description="Instance UUID"),
        "image_mime": Column(str, nullable=True, description="MIME type for image (e.g., 'image/png')"),
        "image_bytes": Column(object, nullable=True, description="Raw binary image data")
    },
    strict=True,
    coerce=False,
)

# Suite Dataset (Bronze) schema
SUITE_DATAFRAME_SCHEMA = DataFrameSchema(
    {
        "suite_id": Column(str, nullable=False, description="Suite UUID identifier"),
        "suite_name": Column(str, nullable=False, description="Suite name"),
        "action_inventory": Column(object, nullable=False, description="List of actions and their parameters as JSON object")
    },
    strict=True,
    coerce=False,
)

# Instance Dataset schema - for Instance data from Backend API
BRONZE_INSTANCE_DATAFRAME_SCHEMA = DataFrameSchema(
    {
        "instance_id": Column(str, nullable=False, description="Instance UUID"),
        "task_id": Column(str, nullable=False, description="Task UUID"),
        "suite_id": Column(str, nullable=False, description="Suite UUID"),
        "task_instruction": Column(str, nullable=False, description="Natural-language task spec"),
        "task_guardrails": Column(object, nullable=True, description="Guardrails configuration (JSON)"),
        "input_payload": Column(object, nullable=True, description="Instance input context (JSON)")
    },
    strict=True,
    coerce=False,
)

# Silver layer schemas (to be defined when implementing cleaning.py)
# SILVER_DATAFRAME_SCHEMA = DataFrameSchema({...})

# Gold layer schemas (to be defined when implementing classification.py)
# GOLD_DATAFRAME_SCHEMA = DataFrameSchema({...})
