# Action Extraction Pipeline

## Purpose

Extract and normalize raw session traces (input events + screenshots) from session directories into structured Parquet format for downstream processing.

**Status**: Bronze extraction layer implemented. Provides foundational trace data for action extraction pipelines.

**See**: [Root AGENTS.md](../AGENTS.md) for workspace-level coding standards, testing guidelines, and architecture patterns.

---

## Package Structure

```text
action_extraction/
├── action_extraction/
│   ├── __init__.py
│   ├── config.py                 # Configuration loader
│   ├── common.py                 # Shared utilities
│   ├── extraction.py             # Bronze: Data extraction
│   ├── integration.py            # Silver: Data integration
│   ├── window_classification.py  # Gold: Window classification
│   └── parameter_extraction.py   # Gold: Parameter extraction
├── config/
│   ├── pipeline.yaml
│   ├── extraction.yaml
│   ├── integration.yaml
│   ├── window_classification.yaml
│   └── parameter_extraction.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── AGENTS.md
├── pyproject.toml
└── README.md
```

---

## Bronze Layer: extraction.py

### Input

Raw session directories in `data/raw/`:

```text
data/raw/{session_uuid}/
├── input/
│   └── {timestamp}-{uuid}.json
└── screenshot/
    └── {timestamp}/
        ├── {uuid}.json
        └── {uuid}.png
```

### Output

Single Parquet file: `data/bronze/traces_bronze_{timestamp}.parquet`

**Schema** (nested structure):

- `trace` (struct):
  - `session_uuid` (string, NOT NULL)
  - `event_type` (string, NOT NULL) - "input" | "screenshot"
  - `timestamp` (string, NOT NULL) - ISO 8601 from filename
  - `json_raw` (string, NOT NULL) - raw JSON payload
- `image_mime` (string, NULLABLE) - "image/png" for screenshots
- `image_bytes` (large_binary, NULLABLE) - raw PNG bytes

### Key Behaviors

1. **Timestamp extraction**: Reads timestamp from JSON content's `timestamp` field (immune to filename modifications from Azure blob storage)
2. **Partial data handling**: Includes screenshot records even if PNG is missing (warns, doesn't fail)
3. **Nested schema**: Stores trace metadata in nested `trace` column (serialized as JSON string in Parquet)
4. **Metrics output**: Generates `metrics_{timestamp}.json` alongside Parquet file

### Running

```bash
# With defaults
python -m action_extraction.extraction

# Custom paths
python -m action_extraction.extraction \
  --input /path/to/raw \
  --output /path/to/bronze

# Environment variables (optional)
export LOG_LEVEL=DEBUG
python -m action_extraction.extraction
```

### CLI Arguments

- `--input` (default: `data/raw`) - Input directory containing session subdirectories
- `--output` (default: `data/bronze`) - Output directory for Parquet files

---

## Schema Definitions

Located in `action_extraction/common.py`:

- `BRONZE_DATAFRAME_SCHEMA` - Pandera schema for DataFrame validation
- `BRONZE_PARQUET_SCHEMA` - PyArrow schema for Parquet serialization

**Important**: The `trace` column is stored as a dict in DataFrame but serialized to JSON string for Parquet storage.

---

## Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=action_extraction tests/
```

Test fixtures in `tests/fixtures/`:

- `valid_input.json` - Sample input event
- `valid_screenshot.json` - Sample screenshot event

---

## Pipeline-Specific Notes

### Timestamp Parsing

Timestamps are extracted from the JSON content's `timestamp` field rather than filenames:

- All JSON files must contain a `timestamp` field at root level
- This approach is immune to filename modifications during Azure blob downloads (which remove colons)
- Falls back gracefully: logs warning and skips trace if timestamp field is missing or JSON is malformed

**Recent improvement**: Changed from filename-based parsing to JSON content-based extraction for Azure blob compatibility.

### Error Handling

- Logs warnings for missing PNG files but continues processing
- Returns partial data with null `image_bytes` rather than failing
- Logs errors for unparseable timestamps but skips those events
- Generates metrics tracking error/warning counts

### Metrics Structure

```json
{
  "timestamp": "2026-01-12T14:30:00",
  "summary": {
    "total_events": 150,
    "input_events": 100,
    "screenshot_events": 50,
    "unique_sessions": 5
  },
  "processing": {
    "errors": {"count": 2}
  },
  "output": {
    "file": "data/bronze/traces_bronze_20260112_143000.parquet",
    "file_size_mb": 12.5,
    "total_rows": 148
  }
}
```
