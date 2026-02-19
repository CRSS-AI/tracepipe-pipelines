#!/usr/bin/env python3
"""
Extraction script for processing trace data from session directories.

This script:
1. Loads all sessions from an input directory
2. Processes input and screenshot traces for each session
3. Coerces data into a structured DataFrame
4. Saves to bronze layer as Parquet format
"""

import argparse
import json
import logging
import mimetypes
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Generator, Tuple, Any

import pandas as pd
import requests
from dotenv import load_dotenv

from action_extraction.schemas import BRONZE_TRACE_DATAFRAME_SCHEMA, SUITE_DATAFRAME_SCHEMA, BRONZE_INSTANCE_DATAFRAME_SCHEMA

# Load environment variables
load_dotenv()

# Backend API configuration
BACKEND_API_BASE_URL = os.getenv('BACKEND_API_BASE_URL')
API_KEY = os.getenv('API_KEY')
BACKEND_API_MAX_RETRIES = int(os.getenv('BACKEND_API_MAX_RETRIES', '2'))  # Default: 2 retries
BACKEND_API_RETRY_DELAY = int(os.getenv('BACKEND_API_RETRY_DELAY', '1'))  # Default: 1 second between retries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract and process trace data from session directories and Suite definitions from API.'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw',
        help='Input directory containing session data (default: data/raw)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/bronze',
        help='Output directory for bronze data (default: data/bronze)'
    )
    parser.add_argument(
        '--traces-only',
        action='store_true',
        help='Extract only traces (skip instance and suite extraction)'
    )
    return parser.parse_args()


def _process_trace_json(json_content: str, json_path: Path) -> Optional[Dict]:
    """
    Parse JSON once and extract all needed fields for trace processing.
    
    This function parses the JSON content once and extracts the timestamp,
    then removes it to avoid duplication (timestamp is stored separately).
    
    Args:
        json_content: Raw JSON string content
        json_path: Path to the JSON file (for error reporting)
    
    Returns:
        Dict with 'timestamp' and 'json_cleaned' keys, or None if extraction fails
    """
    try:
        data = json.loads(json_content)
        
        # Extract timestamp
        timestamp = data.get('timestamp')
        if not timestamp:
            logger.warning(f"No 'timestamp' field found in {json_path}")
            return None
        
        # Remove timestamp to avoid duplication (stored separately)
        data.pop('timestamp', None)
        json_cleaned = json.dumps(data, ensure_ascii=False)
        
        return {
            'timestamp': timestamp,
            'json_cleaned': json_cleaned
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {json_path}: {e}")
        return None


def _validate_session(session_dir: Path, session_uuid: str) -> Tuple[bool, Optional[str]]:
    """
    Validate session for data integrity issues.
    
    Rules:
    - If any screenshot directory has multiple JSON files -> reject entire session
    - If any screenshot directory has multiple image files -> reject entire session
    - Missing files are acceptable (will skip those traces)
    
    Args:
        session_dir: Path to the session directory
        session_uuid: The session UUID
    
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    screenshot_dir = session_dir / "screenshot"
    
    # If no screenshot directory, session is valid
    if not screenshot_dir.exists():
        return True, None
    
    # Check each timestamp directory in screenshot/
    screenshot_timestamp_dirs = [d for d in screenshot_dir.iterdir() if d.is_dir()]
    
    for timestamp_dir in screenshot_timestamp_dirs:
        # Check for multiple JSON files
        json_files = list(timestamp_dir.glob('*.json'))
        if len(json_files) > 1:
            error_msg = f"Session {session_uuid} rejected: Multiple JSON files found in {timestamp_dir.name}"
            return False, error_msg
        
        # Check for multiple image files (any format)
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.gif']
        image_files = []
        for pattern in image_extensions:
            image_files.extend(timestamp_dir.glob(pattern))
        
        if len(image_files) > 1:
            error_msg = f"Session {session_uuid} rejected: Multiple image files found in {timestamp_dir.name}"
            return False, error_msg
    
    return True, None


def _get_mime_type_from_path(file_path: Path) -> str:
    """
    Infer MIME type from file extension.
    
    Args:
        file_path: Path to the file
    
    Returns:
        MIME type string (e.g., 'image/png', 'image/jpeg')
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type is None:
        # Fallback to generic binary type if MIME type cannot be determined
        return 'application/octet-stream'
    return mime_type


def _find_image_for_screenshot(json_path: Path) -> Optional[Path]:
    """
    Find the image file associated with a screenshot JSON file.
    Supports multiple image formats (PNG, JPEG, WebP, etc.).
    
    Args:
        json_path: Path to the screenshot JSON file
    
    Returns:
        Path to image file if found, None otherwise
    """
    screenshot_dir = json_path.parent
    
    # Look for common image formats
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp', '*.gif']
    
    for pattern in image_extensions:
        image_files = list(screenshot_dir.glob(pattern))
        if image_files:
            return image_files[0]
    
    return None


def process_session(session_dir: Path, session_uuid: str) -> Tuple[Generator[Dict, None, None], Optional[Dict[str, str]]]:
    """
    Process all traces (input and screenshot) for a single session.
    Yields flat tabular records for each event and returns session metadata.
    
    Args:
        session_dir: Path to the session directory
        session_uuid: The session UUID
    
    Returns:
        Tuple of (trace_generator, session_metadata)
        - trace_generator: Generator yielding trace dictionaries
        - session_metadata: Dict with 'case_id' and 'instance_id' or None
    
    Yields:
        Dictionary with record data in nested trace structure
    """
    logger.info(f"Processing session: {session_uuid}")
    
    # Validate session first - reject entire session if validation fails
    is_valid, error_msg = _validate_session(session_dir, session_uuid)
    if not is_valid:
        logger.error(error_msg)
        # Return empty generator and no metadata
        return (x for x in []), None
    
    # Extract session metadata from API lookup
    # NOTE: This requires API credentials to be available
    session_metadata = None
    
    if API_KEY and BACKEND_API_BASE_URL:
        logger.debug(f"Looking up session {session_uuid} via API")
        session_data = lookup_session_by_id(session_uuid, API_KEY)
        
        if session_data:
            human_instance_id = session_data.get('human_instance_id')
            if human_instance_id:
                logger.debug(f"Session {session_uuid} -> instance {human_instance_id}")
                
                # Lookup instance to get case_id (returned in instance response)
                instance_data = lookup_instance_by_id(human_instance_id, API_KEY)
                if instance_data:
                    session_metadata = {
                        'case_id': instance_data.get('caseId'),
                        'instance_id': human_instance_id
                    }
                    logger.info(f"Extracted metadata for session {session_uuid} via API: case_id={session_metadata['case_id']}, instance_id={session_metadata['instance_id']}")
                else:
                    logger.error(f"Failed to lookup instance {human_instance_id} for session {session_uuid}")
            else:
                logger.error(f"Session {session_uuid} has no human_instance_id in API response")
        else:
            logger.error(f"Failed to lookup session {session_uuid} via API")
    else:
        logger.error(f"API credentials not available - cannot extract metadata for session {session_uuid}")
    
    # Track event count for logging
    event_count = 0
    
    # Extract instance_id from metadata for inclusion in traces
    # Skip sessions without instance_id to maintain schema non-nullability
    if not session_metadata or not session_metadata.get('instance_id'):
        logger.warning(f"Skipping session {session_uuid} - no instance_id in metadata")
        return iter([]), None  # Return empty generator
    
    instance_id = session_metadata.get('instance_id')
    
    def trace_generator():
        nonlocal event_count
        
        # Process both event types uniformly
        for event_type in ("input", "screenshot"):
            modality_dir = session_dir / event_type
            
            if not modality_dir.exists():
                logger.warning(f"No {event_type} directory found for session {session_uuid}")
                continue
            
            # Find all JSON files in this modality directory (recursively for screenshots)
            json_files = list(modality_dir.rglob("*.json"))
            
            if not json_files:
                logger.warning(f"No JSON files found in {event_type} directory for session {session_uuid}")
                continue
            
            logger.info(f"Found {len(json_files)} {event_type} events")
            
            for json_path in json_files:
                try:
                    # Read JSON content as raw string
                    with open(json_path, 'r', encoding='utf-8') as f:
                        json_raw = f.read()
                    
                    # Parse JSON once and extract timestamp + cleaned JSON
                    processed = _process_trace_json(json_raw, json_path)
                    if processed is None:
                        logger.warning(f"Skipping {json_path} - failed to process JSON")
                        continue
                    
                    timestamp = processed['timestamp']
                    json_raw_cleaned = processed['json_cleaned']
                    
                    # Build the flat record
                    rec = {
                        'session_uuid': session_uuid,
                        'event_type': event_type,
                        'timestamp': timestamp,
                        'json_raw': json_raw_cleaned,
                        'image_mime': None,
                        'image_bytes': None
                    }
                    
                    # Handle images for screenshot events
                    if event_type == "screenshot":
                        image_path = _find_image_for_screenshot(json_path)
                        if image_path is not None and image_path.exists():
                            rec['image_mime'] = _get_mime_type_from_path(image_path)
                            rec['image_bytes'] = image_path.read_bytes()
                        else:
                            # Missing image: skip this trace (don't include partial data)
                            logger.warning(f"No image file found for {json_path}, skipping this trace")
                            continue
                    
                    # Wrap in nested trace structure with instance_id
                    yield {
                        'trace': {
                            'session_uuid': rec['session_uuid'],
                            'event_type': rec['event_type'],
                            'timestamp': rec['timestamp'],
                            'json_raw': rec['json_raw']
                        },
                        'instance_id': instance_id,  # NEW: Add instance_id as separate column
                        'image_mime': rec['image_mime'],
                        'image_bytes': rec['image_bytes']
                    }
                    
                    event_count += 1
                    
                except Exception as e:
                    logger.error(f"[Session: {session_uuid}] Failed to process {event_type} file {json_path.name}: {e}")
                    continue
    
    return trace_generator(), session_metadata


def _get_api_headers(api_key: str) -> Dict[str, str]:
    """
    Get API request headers with authentication.
    
    Args:
        api_key: Backend API key
    
    Returns:
        Dict with authentication headers
    """
    return {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }


def lookup_session_by_id(session_id: str, api_key: str) -> Optional[Dict]:
    """
    Lookup session details directly by session ID using the lookup API.
    
    This endpoint doesn't require parent case_id or instance_id,
    making it simpler than the nested API structure.
    
    Args:
        session_id: UUID of the session to lookup
        api_key: Backend API key
    
    Returns:
        Dict with keys: id, human_instance_id, status, started_at, 
        completed_at, storage_path, storage_url, human_instance (optional)
        None if API call fails
    """
    url = f"{BACKEND_API_BASE_URL}/api/v1/lookup/sessions/{session_id}"
    headers = _get_api_headers(api_key)
    
    for attempt in range(BACKEND_API_MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=40)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < BACKEND_API_MAX_RETRIES:
                logger.warning(f"Attempt {attempt + 1} failed for session lookup {session_id}: {e}. Retrying in {BACKEND_API_RETRY_DELAY}s...")
                time.sleep(BACKEND_API_RETRY_DELAY)
            else:
                logger.error(f"Failed to lookup session {session_id} after {BACKEND_API_MAX_RETRIES + 1} attempts: {e}")
                return None
    
    return None


def lookup_instance_by_id(instance_id: str, api_key: str) -> Optional[Dict]:
    """
    Lookup instance details directly by instance ID using the lookup API.
    
    This endpoint returns complete nested data including task, suite, and action
    information, eliminating the need for separate API calls.
    
    Args:
        instance_id: UUID of the instance to lookup
        api_key: Backend API key
    
    Returns:
        Dict with complete nested structure:
        {
            id, caseId, taskId, status, inputPayload, outputPayload,
            actions, createdAt, updatedAt, userId,
            task: {
                id, activityId, suiteId, instruction, guardrails,
                suite: {
                    suite_id, name,
                    actions: [{action_id, name, parameters}]
                }
            },
            case: {...},
            sessions: [...]
        }
        None if API call fails
    """
    url = f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}"
    headers = _get_api_headers(api_key)
    
    for attempt in range(BACKEND_API_MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=40)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < BACKEND_API_MAX_RETRIES:
                logger.warning(f"Attempt {attempt + 1} failed for instance lookup {instance_id}: {e}. Retrying in {BACKEND_API_RETRY_DELAY}s...")
                time.sleep(BACKEND_API_RETRY_DELAY)
            else:
                logger.error(f"Failed to lookup instance {instance_id} after {BACKEND_API_MAX_RETRIES + 1} attempts: {e}")
                return None
    
    return None

def lookup_action_by_id(action_id: str, api_key: str) -> Optional[Dict]:
    """
    Lookup action details directly by action ID using the lookup API.
    
    This endpoint returns complete action information including parameters,
    which are not included in the nested suite.actions array from instance lookup.
    
    Args:
        action_id: UUID of the action to lookup
        api_key: Backend API key
    
    Returns:
        Dict with complete action structure:
        {
            action_id, name, description, type,
            parameters: [{name, type, required, default, description}]
        }
        None if API call fails
    """
    url = f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}"
    headers = _get_api_headers(api_key)
    
    for attempt in range(BACKEND_API_MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=40)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt < BACKEND_API_MAX_RETRIES:
                logger.warning(f"Attempt {attempt + 1} failed for action lookup {action_id}: {e}. Retrying in {BACKEND_API_RETRY_DELAY}s...")
                time.sleep(BACKEND_API_RETRY_DELAY)
            else:
                logger.error(f"Failed to lookup action {action_id} after {BACKEND_API_MAX_RETRIES + 1} attempts: {e}")
                return None
    
    return None

def collect_instance_data(session_metadata_list: List[Dict[str, str]], api_key: str) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Collect instance, task, suite, and action data for all unique instances.
    
    Uses lookup API to get nested data, then enriches action_inventory with
    complete action details (including parameters) via separate action lookups.
    
    Args:
        session_metadata_list: List of dicts with 'case_id' and 'instance_id'
        api_key: Backend API key
    
    Returns:
        Tuple of (instance_records, suite_records, collection_stats)
        - instance_records: List of dicts with complete instance data ready for DataFrame
        - suite_records: List of suite dicts for Suite DataFrame with enriched action_inventory
        - collection_stats: Dict with keys 'total_attempted', 'instance_api_failures', 
                           'missing_task_data', 'missing_suite_data'
    """
    # Initialize statistics tracking
    collection_stats = {
        'total_attempted': 0,
        'instance_api_failures': 0,
        'missing_task_data': 0,
        'missing_suite_data': 0,
    }
    
    # Deduplicate instances (only need instance_id now, case_id not used for lookup)
    unique_instance_ids = set()
    for metadata in session_metadata_list:
        if metadata:  # Skip None values
            unique_instance_ids.add(metadata['instance_id'])
    
    logger.info(f"Found {len(unique_instance_ids)} unique instances across all sessions")
    
    # Collect instance data and extract suite data
    instance_records = []
    suite_map = {}  # Deduplicate suites by suite_id
    
    for instance_id in unique_instance_ids:
        collection_stats['total_attempted'] += 1
        
        # Fetch complete instance details (includes nested task/suite/actions)
        instance_data = lookup_instance_by_id(instance_id, api_key)
        if not instance_data:
            collection_stats['instance_api_failures'] += 1
            logger.warning(f"Skipping instance {instance_id} - API lookup failed")
            continue
        
        # Extract task data from nested response
        task_data = instance_data.get('task')
        if not task_data:
            collection_stats['missing_task_data'] += 1
            logger.warning(f"Skipping instance {instance_id} - no task data in response")
            continue
        
        # Extract suite data from nested task
        suite_data = task_data.get('suite')
        if not suite_data:
            collection_stats['missing_suite_data'] += 1
            logger.warning(f"Skipping instance {instance_id} - no suite data in task response")
            continue
        
        # Extract suite while we have the nested response (deduplicate by suite_id)
        suite_id = suite_data.get('suite_id')
        if not suite_id:
            collection_stats['missing_suite_data'] += 1
            logger.warning(f"Skipping instance {instance_id} - suite_id is missing in suite data")
            continue
        
        if suite_id not in suite_map:
            # Validate suite_name is present (required by schema)
            suite_name = suite_data.get('name')
            if not suite_name:
                collection_stats['missing_suite_data'] += 1
                logger.warning(f"Skipping instance {instance_id} - suite_name is missing in suite data")
                continue
            
            # Get actions from suite (without parameters)
            actions_basic = suite_data.get('actions', [])
            
            # Enrich each action with full details including parameters
            enriched_actions = []
            for action in actions_basic:
                action_id = action.get('action_id')
                if action_id:
                    # Lookup full action details including parameters
                    action_details = lookup_action_by_id(action_id, api_key)
                    if action_details:
                        enriched_actions.append(action_details)
                        logger.debug(f"Enriched action {action_id} with parameters")
                    else:
                        # Fallback: use basic action if lookup fails
                        enriched_actions.append(action)
                        logger.warning(f"Failed to enrich action {action_id}, using basic data")
                else:
                    # Action without ID, keep as-is
                    enriched_actions.append(action)
            
            suite_map[suite_id] = {
                'suite_id': suite_id,
                'suite_name': suite_name,
                'action_inventory': enriched_actions
            }
            logger.info(f"Enriched suite {suite_id} with {len(enriched_actions)} action(s) including parameters")
        
        # Validate required task fields before building instance record
        task_id = task_data.get('id')
        task_instruction = task_data.get('instruction')
        
        if not task_id:
            collection_stats['missing_task_data'] += 1
            logger.warning(f"Skipping instance {instance_id} - task_id is missing in task data")
            continue
        
        if not task_instruction:
            collection_stats['missing_task_data'] += 1
            logger.warning(f"Skipping instance {instance_id} - task_instruction is missing in task data")
            continue
        
        # Build complete instance record (no additional API calls needed!)
        record = {
            'instance_id': instance_id,
            'task_id': task_id,
            'suite_id': suite_id,
            'task_instruction': task_instruction,
            'task_guardrails': task_data.get('guardrails'),  # From nested task (can be None)
            'input_payload': instance_data.get('inputPayload'),  # From instance (can be None)
        }
        
        instance_records.append(record)
        logger.info(f"Collected data for instance {instance_id} (with nested task/suite/actions)")
    
    suite_records = list(suite_map.values())
    
    logger.info(f"Successfully collected data for {len(instance_records)} instances")
    logger.info(f"Extracted {len(suite_records)} unique suite(s) from nested data")
    logger.info(f"Collection stats - Attempted: {collection_stats['total_attempted']}, "
                f"Instance API failures: {collection_stats['instance_api_failures']}, "
                f"Missing task data: {collection_stats['missing_task_data']}, "
                f"Missing suite data: {collection_stats['missing_suite_data']}")
    
    return instance_records, suite_records, collection_stats


def create_traces_dataframe(rows: List[Dict]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the collected rows.
    
    Args:
        rows: List of row dictionaries
    
    Returns:
        pandas DataFrame with the specified schema
    """
    if not rows:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=['trace', 'instance_id', 'image_mime', 'image_bytes'])
    else:
        df = pd.DataFrame(rows)
        
        # Ensure correct column order
        df = df[['trace', 'instance_id', 'image_mime', 'image_bytes']]
        
        # Explicitly set types
        df['instance_id'] = df['instance_id'].astype('str')  # instance_id is required (not nullable)
        df['image_mime'] = df['image_mime'].astype('object')  # Allow None
        df['image_bytes'] = df['image_bytes'].astype('object')  # Allow None
    
    # Validate DataFrame schema with Pandera
    try:
        df = BRONZE_TRACE_DATAFRAME_SCHEMA.validate(df, lazy=True)
        logger.info("DataFrame schema validation passed")
    except Exception as e:
        logger.warning(f"DataFrame schema validation failed: {e}")
    
    return df


def create_instance_dataframe(instance_records: List[Dict]) -> pd.DataFrame:
    """
    Create a pandas DataFrame for Instance Dataset.
    
    Args:
        instance_records: List of instance record dictionaries
    
    Returns:
        pandas DataFrame with Instance Dataset schema
    """
    if not instance_records:
        # Create empty DataFrame with correct columns
        df = pd.DataFrame(columns=[
            'instance_id', 'task_id', 'suite_id', 
            'task_instruction', 'task_guardrails', 'input_payload'
        ])
    else:
        df = pd.DataFrame(instance_records)
        
        # Ensure correct column order
        df = df[[
            'instance_id', 'task_id', 'suite_id',
            'task_instruction', 'task_guardrails', 'input_payload'
        ]]
    
    # Validate DataFrame schema with Pandera
    try:
        df = BRONZE_INSTANCE_DATAFRAME_SCHEMA.validate(df, lazy=True)
        logger.info("Instance DataFrame schema validation passed")
    except Exception as e:
        logger.warning(f"Instance DataFrame schema validation failed: {e}")
    
    return df


def save_to_bronze(
    df: pd.DataFrame,
    output_dir: Path,
    timestamp: str,
    dataset_name: str,
    json_columns: List[str]
) -> Path:
    """
    Save DataFrame to bronze layer as Parquet.
    
    Args:
        df: DataFrame to save
        output_dir: Base output directory path
        timestamp: Timestamp string for filename
        dataset_name: Name of dataset
        json_columns: List of column names containing JSON data to serialize
    
    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f'{dataset_name}_bronze_{timestamp}.parquet'
    
    # Create a copy for Parquet serialization
    df_parquet = df.copy()
    
    # Serialize JSON columns to strings for Parquet
    for col in json_columns:
        if col in df_parquet.columns:
            df_parquet[col] = df_parquet[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )
    
    # Write to Parquet
    df_parquet.to_parquet(
        output_file,
        engine='pyarrow',
        compression='snappy',
        index=False
    )
    
    # Log results
    logger.info(f"Saved {dataset_name} data to: {output_file}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    return output_file


def save_unified_metrics(
    output_dir: Path,
    timestamp: str,
    *,
    trace_stats: dict[str, Any] | None = None,
    suite_stats: dict[str, Any] | None = None,
    instance_stats: dict[str, Any] | None = None,
) -> Path:
    """
    Save unified extraction metrics to a JSON file.
    
    Combines statistics from trace, suite, and instance extraction into a single metrics file.
    
    Args:
        output_dir: Output directory path
        timestamp: Timestamp string for filename
        trace_stats: Optional trace extraction statistics
        suite_stats: Optional suite extraction statistics
        instance_stats: Optional instance extraction statistics
    
    Returns:
        Path to the saved metrics file
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / f'metrics_{timestamp}.json'
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'processing': {
            'errors': {
                'count': 0
            }
        },
        'output': {}
    }
    
    if trace_stats:
        metrics['summary']['total_events'] = trace_stats['total_events']
        metrics['summary']['input_events'] = trace_stats['input_events']
        metrics['summary']['screenshot_events'] = trace_stats['screenshot_events']
        metrics['summary']['unique_sessions'] = trace_stats['unique_sessions']
        metrics['processing']['errors']['count'] += trace_stats['errors']
        
        # Only add output file info if file was actually created
        if trace_stats.get('output_file'):
            metrics['output']['traces'] = {
                'file': str(trace_stats['output_file']),
                'file_size_mb': trace_stats['file_size_mb']
            }
    
    if suite_stats and suite_stats.get('total_suites', 0) > 0:
        metrics['summary']['total_suites'] = suite_stats['total_suites']
        metrics['summary']['total_actions'] = suite_stats.get('total_actions', 0)
        
        if suite_stats.get('output_file'):
            metrics['output']['suites'] = {
                'file': str(suite_stats['output_file']),
                'file_size_mb': suite_stats['file_size_mb']
            }
    
    if instance_stats:
        if instance_stats.get('total_instances', 0) > 0:
            metrics['summary']['total_instances'] = instance_stats['total_instances']
            metrics['summary']['unique_tasks'] = instance_stats['unique_tasks']
            metrics['summary']['unique_suites'] = instance_stats['unique_suites']
            
            if instance_stats.get('output_file'):
                metrics['output']['instances'] = {
                    'file': str(instance_stats['output_file']),
                    'file_size_mb': instance_stats['file_size_mb']
                }
        
        # Track instance-specific errors
        instance_errors = (
            instance_stats.get('instance_api_failures', 0) +
            instance_stats.get('missing_task_data', 0) +
            instance_stats.get('missing_suite_data', 0) +
            instance_stats.get('catastrophic_error', 0)
        )
        metrics['processing']['errors']['count'] += instance_errors
        
        # Add detailed instance error breakdown if any errors occurred
        if instance_errors > 0:
            metrics['processing']['errors']['instances'] = {
                'instance_api_failures': instance_stats.get('instance_api_failures', 0),
                'missing_task_data': instance_stats.get('missing_task_data', 0),
                'missing_suite_data': instance_stats.get('missing_suite_data', 0),
                'catastrophic_error': instance_stats.get('catastrophic_error', 0)
            }
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Unified metrics saved to: {metrics_file}")
    return metrics_file


def main():
    """Main execution function."""
    args = parse_arguments()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Generate timestamp once for consistent file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    trace_stats = None
    suite_stats = None
    
    # Validate input directory for session trace extraction
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    logger.info(f"Starting session trace extraction from: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find all session directories
    session_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(session_dirs)} session directories")
    
    if not session_dirs:
        logger.warning("No session directories found!")
        # Create trace stats to indicate no sessions were found
        trace_stats = {
            'total_events': 0,
            'input_events': 0,
            'screenshot_events': 0,
            'unique_sessions': 0,
            'errors': 0,
            'output_file': None,
            'file_size_mb': 0,
        }
        save_unified_metrics(output_dir, timestamp, trace_stats=trace_stats, suite_stats=suite_stats)
        sys.exit(0)
    
    # Process all sessions - collect rows from generator
    all_rows = []
    session_metadata_list = []  # Collect instance metadata from all sessions
    total_errors = 0
    
    for session_dir in session_dirs:
        session_uuid = session_dir.name
        try:
            # Process session - now returns tuple with metadata
            session_generator, session_metadata = process_session(session_dir, session_uuid)
            session_rows = list(session_generator)
            all_rows.extend(session_rows)
            
            # Log session processing results
            if len(session_rows) == 0:
                logger.warning(f"Session {session_uuid} produced no valid records")
            else:
                logger.info(f"Session {session_uuid} produced {len(session_rows)} records")
            
            # Collect metadata for instance extraction
            if session_metadata:
                session_metadata_list.append(session_metadata)
            else:
                logger.debug(f"No instance metadata found for session {session_uuid} (may be old data format)")
                
        except Exception as e:
            logger.error(f"Failed to process session {session_uuid}: {e}")
            total_errors += 1
            continue
    
    if not all_rows:
        logger.error("No data extracted from any session!")
        # Create trace stats to indicate extraction was attempted but failed
        trace_stats = {
            'total_events': 0,
            'input_events': 0,
            'screenshot_events': 0,
            'unique_sessions': len(session_dirs),  # Sessions were found but produced no data
            'errors': total_errors,  # Capture error count
            'output_file': None,
            'file_size_mb': 0,
        }
        save_unified_metrics(output_dir, timestamp, trace_stats=trace_stats, suite_stats=suite_stats)
        sys.exit(1)
    
    # Create DataFrame - pandas handles tabular operations from here
    logger.info(f"Creating DataFrame with {len(all_rows)} rows")
    df = create_traces_dataframe(all_rows)
    
    # Extract event_type and session_uuid from nested trace for statistics
    df_stats = df.copy()
    df_stats['event_type'] = df['trace'].apply(lambda x: x.get('event_type'))
    df_stats['session_uuid'] = df['trace'].apply(lambda x: x.get('session_uuid'))
    
    # Log statistics
    logger.info(f"\nData Statistics:")
    logger.info(f"Total events: {len(df)}")
    logger.info(f"Input events: {len(df_stats[df_stats['event_type'] == 'input'])}")
    logger.info(f"Screenshot events: {len(df_stats[df_stats['event_type'] == 'screenshot'])}")
    logger.info(f"Unique sessions: {df_stats['session_uuid'].nunique()}")
    logger.info(f"Errors encountered: {total_errors}")
    
    # Save traces to bronze
    output_file = save_to_bronze(df, output_dir, timestamp, 'traces', ['trace'])
    
    # Create trace statistics
    trace_stats = {
        'total_events': len(df),
        'input_events': len(df_stats[df_stats['event_type'] == 'input']),
        'screenshot_events': len(df_stats[df_stats['event_type'] == 'screenshot']),
        'unique_sessions': df_stats['session_uuid'].nunique(),
        'errors': total_errors,
        'output_file': output_file,
        'file_size_mb': output_file.stat().st_size / (1024*1024),
    }
    
    # Instance extraction
    instance_df = None
    instance_stats = None
    
    if not args.traces_only:
        logger.info("\n" + "="*50)
        logger.info("Starting Instance Dataset Extraction")
        logger.info("="*50)
        
        # Check API key and backend URL from module-level configuration
        if not API_KEY:
            logger.error("API_KEY not found in environment variables!")
            logger.warning("Skipping instance extraction")
        elif not BACKEND_API_BASE_URL:
            logger.error("BACKEND_API_BASE_URL not found in environment variables!")
            logger.warning("Skipping instance extraction")
        elif not session_metadata_list:
            logger.warning("No instance metadata found in any session")
            logger.warning("This may be old data format - skipping instance extraction")
        else:
            try:
                # Collect instance and suite data from API (now returns 3-tuple)
                instance_records, suite_records, collection_stats = collect_instance_data(session_metadata_list, API_KEY)
                
                if instance_records:
                    # Create DataFrame
                    instance_df = create_instance_dataframe(instance_records)
                    
                    # Save to bronze
                    instance_output_file = save_to_bronze(
                        instance_df, output_dir, timestamp, 'instances', 
                        ['task_guardrails', 'input_payload']
                    )
                    
                    # Calculate statistics
                    instance_stats = {
                        'total_instances': len(instance_df),
                        'unique_tasks': instance_df['task_id'].nunique() if len(instance_df) > 0 else 0,
                        'unique_suites': instance_df['suite_id'].nunique() if len(instance_df) > 0 else 0,
                        'total_attempted': collection_stats['total_attempted'],
                        'instance_api_failures': collection_stats['instance_api_failures'],
                        'missing_task_data': collection_stats['missing_task_data'],
                        'missing_suite_data': collection_stats['missing_suite_data'],
                        'output_file': instance_output_file,
                        'file_size_mb': instance_output_file.stat().st_size / (1024*1024),
                    }
                    
                    logger.info(f"\nInstance Statistics:")
                    logger.info(f"Total instances: {instance_stats['total_instances']}")
                    logger.info(f"Unique tasks: {instance_stats['unique_tasks']}")
                    logger.info(f"Unique suites: {instance_stats['unique_suites']}")
                else:
                    logger.warning("No instance data collected from API")
                    # Still capture collection stats even if no successful records
                    instance_stats = {
                        'total_instances': 0,
                        'unique_tasks': 0,
                        'unique_suites': 0,
                        'total_attempted': collection_stats['total_attempted'],
                        'instance_api_failures': collection_stats['instance_api_failures'],
                        'missing_task_data': collection_stats['missing_task_data'],
                        'missing_suite_data': collection_stats['missing_suite_data'],
                    }
                
                # Process suite records (extracted during instance collection)
                if suite_records:
                    logger.info("\n" + "="*50)
                    logger.info("Processing Suite Data from Nested Responses")
                    logger.info("="*50)
                    logger.info(f"Creating Suite DataFrame from {len(suite_records)} suite(s)")
                    
                    # Create DataFrame from suite records
                    suite_df = pd.DataFrame(suite_records)
                    suite_df = suite_df[["suite_id", "suite_name", "action_inventory"]]
                    
                    # Validate against schema
                    try:
                        SUITE_DATAFRAME_SCHEMA.validate(suite_df, lazy=True)
                        logger.info("Suite DataFrame schema validation passed")
                    except Exception as e:
                        logger.warning(f"Suite DataFrame schema validation failed: {e}")
                    
                    # Save to bronze
                    suite_output_file = save_to_bronze(
                        suite_df, output_dir, timestamp, 'suites', ['action_inventory']
                    )
                    
                    # Calculate suite statistics
                    total_actions = sum(len(s.get('action_inventory', [])) for s in suite_records)
                    suite_stats = {
                        'total_suites': len(suite_df),
                        'total_actions': total_actions,
                        'output_file': suite_output_file,
                        'file_size_mb': suite_output_file.stat().st_size / (1024*1024),
                    }
                    
                    logger.info(f"Suite Statistics:")
                    logger.info(f"Total suites: {suite_stats['total_suites']}")
                    logger.info(f"Total actions: {suite_stats['total_actions']}")
                else:
                    logger.info("No suite data extracted (all instances missing suite information)")
                    
            except Exception as e:
                logger.error(f"Instance extraction failed: {e}")
                instance_stats = {'catastrophic_error': 1}
    else:
        logger.info("\nSkipping instance extraction (--traces-only flag set)")
    
    # Save unified metrics
    save_unified_metrics(
        output_dir, 
        timestamp, 
        trace_stats=trace_stats, 
        suite_stats=suite_stats,
        instance_stats=instance_stats
    )
    
    logger.info("\nExtraction completed successfully!")


if __name__ == '__main__':
    main()
