# Trace Classification Pipeline

A data pipeline for extracting and cleaning user session traces from storage.

## Overview

This pipeline processes session trace data containing user input events and screenshot captures, transforming raw data into classification-ready features.

- **Bronze (Extraction)**: Raw data extraction from source storage

## Pipeline Architecture

```text
data/raw/                          data/bronze/                 
(Source)                           (Raw Extract)                
    │                                   │                      
    │   ┌──────────────────┐           │   ┌──────────────────┐  
    └──▶│  extraction.py   │──────────▶│   │ traces_bronze_   │
        │  (Bronze Layer)  │           │   │  {timestamp}.    │
        └──────────────────┘           │   │    parquet       │ 
                                       └───┴──────────────────┘
```

## Installation

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip

### Setup with UV (Recommended)

```bash
# From repository root
cd action_extraction

# Install dependencies
uv sync

# Or install with development dependencies
uv sync --all-extras
```

### Setup with pip

```bash
cd action_extraction
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Run Individual Steps

```bash
# Bronze: Extract raw traces, instances, and suites
uv run python -m action_extraction.extraction

# Bronze: Extract only traces (skip instance and suite extraction)
uv run python -m action_extraction.extraction --traces-only
```

## Configuration

### Environment Variables

Create a `.env` file in the `action_extraction` directory (use `.env.example` as template):

```bash
# Backend API configuration for Suite extraction
BACKEND_API_BASE_URL=https://autoactivity-backend-dev.jollyforest-cbc46cf4.eastus.azurecontainerapps.io
API_KEY=your-api-key-here
```

**Note**: `.env` is gitignored. Never commit API keys to version control.

### YAML Configuration Files

Pipeline behavior is controlled through YAML configuration files in `config/`:

### `config/pipeline.yaml`

Main pipeline configuration defining step execution order and data paths.

### `config/extraction.yaml`

Bronze layer extraction configuration.

### `config/integration.yaml`

Silver layer integration configuration.

### `config/window_classification.yaml`

Gold layer window classification configuration.

### `config/parameter_extraction.yaml`

Gold layer parameter extraction configuration.

## Data Schema

### Bronze Layer Output

| Column        | Type           | Nullable | Description                                          |
|---------------|----------------|----------|------------------------------------------------------|
| session_uuid  | string         | No       | Session identifier (UUID from directory name)        |
| event_type    | string         | No       | Event type: "input" or "screenshot"                  |
| timestamp     | string         | No       | ISO 8601 timestamp from JSON content                 |
| json_raw      | string         | No       | Raw JSON trace data                                  |
| image_mime    | string         | Yes      | MIME type for screenshots (e.g., "image/png")        |
| image_bytes   | large_binary   | Yes      | Raw image bytes for screenshot events                |

### Source Data Structure

Expected directory structure in input directory:

```text
<input_directory>/
└── {session_uuid}/
    ├── input/
    │   └── {timestamp}-{event_uuid}.json
    └── screenshot/
        └── {timestamp}/
            ├── {event_uuid}.json
            └── {event_uuid}.png
```

## Usage Examples

### Basic Usage (Default Paths)

```bash
# Extract traces with default settings (data/raw → data/bronze)
uv run python -m action_extraction.extraction
```

### Custom Input/Output Paths

```bash
# Process traces from custom directory
uv run python -m action_extraction.extraction \
    --input /path/to/custom/raw \
    --output /path/to/custom/bronze
```

### Verbose Logging

```bash
# Enable debug-level logging
uv run python -m action_extraction.extraction --verbose
```

### Configuration Override

```bash
# Use custom configuration file
uv run python -m action_extraction.extraction \
    --config config/extraction_custom.yaml
```

## Output

Each pipeline step generates:

1. **Parquet file**: Main data output
   - Filename: `traces_{layer}_{timestamp}.parquet`
   - Location: Corresponding `data/{layer}/` directory
   - Format: Compressed Parquet (Snappy)

2. **Metrics file**: Processing statistics
   - Filename: `metrics_{timestamp}.json`
   - Contains: Event counts, error/warning stats, file sizes
   - Location: Same directory as Parquet output

### Example Metrics Output

```json
{
  "timestamp": "2026-01-09T12:00:00.123456",
  "summary": {
    "total_events": 300,
    "input_events": 263,
    "screenshot_events": 37,
    "unique_sessions": 4
  },
  "processing": {
    "errors": {
      "count": 0,
      "descriptions": []
    },
    "warnings": {
      "count": 2
    }
  },
  "output": {
    "file": "data/bronze/traces_bronze_20260109_120000.parquet",
    "file_size_mb": 7.94,
    "total_rows": 300
  }
}
```

## Development

### Project Structure

```text
action_extraction/
├── config/                          # YAML configuration files
│   ├── pipeline.yaml               # Main pipeline config
│   ├── extraction.yaml             # Bronze step config
│   ├── integration.yaml            # Silver step config
│   ├── window_classification.yaml  # Gold step config
│   └── parameter_extraction.yaml   # Gold step config
├── action_extraction/              # Python package
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Configuration loader
│   ├── common.py                  # Shared utilities and schemas
│   ├── extraction.py              # Bronze: Data extraction
│   ├── integration.py             # Silver: Data integration
│   ├── window_classification.py   # Gold: Window classification
│   └── parameter_extraction.py    # Gold: Parameter extraction
├── AGENTS.md                      # AI coding instructions
├── pyproject.toml                 # Package dependencies
└── README.md                      # This file
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=action_extraction

# Run specific test file
uv run pytest tests/unit/test_extraction.py
```

### Code Formatting

```bash
# Format code with black
uv run black action_extraction/

# Check types with mypy
uv run mypy action_extraction/
```

## Data Privacy & Security

 **Important**: This pipeline processes potentially sensitive user activity data.

- **Data Directory**: `data/` is gitignored - never commit raw data
- **Screenshots**: May contain PII or sensitive information
- **Access Control**: Ensure proper Azure credentials and access policies
- **Retention**: Follow data retention policies for your organization
- **Logging**: Logs may contain file paths - review before sharing

### Coding Standards

See [`AGENTS.md`](AGENTS.md) for detailed coding standards including:

- Python style guide (PEP 8, type hints, docstrings)
- Error handling patterns
- Logging conventions
- Schema evolution rules

## License

See [LICENSE](../LICENSE) in repository root.
