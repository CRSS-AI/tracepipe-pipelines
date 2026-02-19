"""Common utilities shared across pipeline steps."""

import logging
from pathlib import Path
from datetime import datetime

from .schemas import MetricsDict

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging for pipeline steps.
    
    Args:
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to "INFO".
    
    Returns:
        Logger instance configured for the current module.
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def generate_timestamp() -> str:
    """
    Generate timestamp string for output files.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS (e.g., "20260115_143022").
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def create_metrics_dict(
    total_events: int,
    error_count: int,
    warning_count: int,
    output_file: Path,
    **kwargs
) -> MetricsDict:
    """
    Create standardized metrics dictionary.
    
    Args:
        total_events: Total number of events processed.
        error_count: Number of errors encountered during processing.
        warning_count: Number of warnings encountered during processing.
        output_file: Path to the output file.
        **kwargs: Additional summary metrics to include (e.g., total_sessions,
            total_screenshots).
    
    Returns:
        Dictionary containing timestamp, summary statistics, processing counts,
        and output file information.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_events": total_events,
            **kwargs
        },
        "processing": {
            "errors": {"count": error_count},
            "warnings": {"count": warning_count}
        },
        "output": {
            "file": str(output_file),
            "file_size_mb": output_file.stat().st_size / (1024*1024),
        }
    }