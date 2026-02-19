"""
Unit tests for action_extraction.extraction module.

Tests cover:
- _process_trace_json: Parse JSON once and extract timestamp + cleaned JSON
- _find_image_for_screenshot: Image file discovery (PNG, JPEG, etc.)
- _get_mime_type_from_path: MIME type inference from file extension
- process_session: Session-level processing with generator
- create_traces_dataframe: DataFrame creation and schema
- save_to_bronze: Unified Parquet file saving (traces and suites)
- save_unified_metrics: Unified metrics JSON generation
- fetch_suites_from_api: API requests for suite data
- transform_suite_data: Suite data transformation
- extract_suites: End-to-end suite extraction
"""

import os
import pandas as pd
import json
import requests
import pytest
from dotenv import load_dotenv

from action_extraction.extraction import (
    _process_trace_json,
    _find_image_for_screenshot,
    _get_mime_type_from_path,
    process_session,
    create_traces_dataframe,
    save_to_bronze,
    save_unified_metrics,
    lookup_session_by_id,
    lookup_instance_by_id,
    lookup_action_by_id,
    collect_instance_data,
    create_instance_dataframe,
    BACKEND_API_BASE_URL
)


# Load environment variables for tests
load_dotenv()


@pytest.fixture(autouse=True)
def mock_backend_url(monkeypatch):
    """Set BACKEND_API_BASE_URL from .env for all tests."""
    backend_url = os.getenv('BACKEND_API_BASE_URL')
    if backend_url:
        monkeypatch.setattr(
            'action_extraction.extraction.BACKEND_API_BASE_URL',
            backend_url
        )


# ==============================================================================
# Tests for helper functions
# ==============================================================================

class TestHelperFunctions:
    """Test JSON processing and image file discovery."""
    
    def test_extract_timestamp_from_json_valid(self, tmp_path):
        """Extract timestamp from JSON content with timezone and clean JSON."""
        json_content = json.dumps({"timestamp": "2026-01-06T20:30:01.337745+00:00"})
        json_path = tmp_path / "test.json"
        
        result = _process_trace_json(json_content, json_path)
        
        assert result is not None
        assert result['timestamp'] == '2026-01-06T20:30:01.337745+00:00'
        # Verify timestamp is removed from cleaned JSON
        cleaned_data = json.loads(result['json_cleaned'])
        assert 'timestamp' not in cleaned_data
    
    def test_extract_timestamp_from_json_without_timezone(self, tmp_path):
        """Extract timestamp from JSON content without timezone."""
        json_content = json.dumps({"timestamp": "2026-01-06T20:30:01.337745"})
        json_path = tmp_path / "test.json"
        
        result = _process_trace_json(json_content, json_path)
        
        assert result is not None
        assert result['timestamp'] == '2026-01-06T20:30:01.337745'
    
    def test_extract_timestamp_from_json_missing_field(self, tmp_path):
        """Return None when timestamp field is missing."""
        json_content = json.dumps({"other_field": "value"})
        json_path = tmp_path / "test.json"
        
        result = _process_trace_json(json_content, json_path)
        
        assert result is None
    
    def test_extract_timestamp_from_json_invalid_json(self, tmp_path):
        """Return None when JSON is malformed."""
        json_content = "invalid json {{"
        json_path = tmp_path / "test.json"
        
        result = _process_trace_json(json_content, json_path)
        
        assert result is None
    
    def test_extract_timestamp_from_json_with_extra_fields(self, tmp_path):
        """Extract timestamp and preserve other fields in cleaned JSON."""
        json_content = json.dumps({
            "timestamp": "2026-01-06T20:30:01.844649+00:00",
            "action": "click",
            "selector": {"tag": "button"}
        })
        json_path = tmp_path / "test.json"
        
        result = _process_trace_json(json_content, json_path)
        
        assert result is not None
        assert result['timestamp'] == '2026-01-06T20:30:01.844649+00:00'
        # Verify other fields are preserved in cleaned JSON
        cleaned_data = json.loads(result['json_cleaned'])
        assert cleaned_data['action'] == 'click'
        assert cleaned_data['selector'] == {"tag": "button"}
        assert 'timestamp' not in cleaned_data
    
    def test_find_image_exists_returns_path(self, tmp_path, sample_png):
        """Find image file returns a path object."""
        screenshot_dir = tmp_path / "timestamp"
        screenshot_dir.mkdir()
        
        json_path = screenshot_dir / "screenshot.json"
        json_path.write_text("{}")
        
        png_path = screenshot_dir / "screenshot.png"
        png_path.write_bytes(sample_png)
        
        found_image = _find_image_for_screenshot(json_path)
        
        assert found_image is not None
    
    def test_find_image_exists_file_exists(self, tmp_path, sample_png):
        """Found image file exists on filesystem."""
        screenshot_dir = tmp_path / "timestamp"
        screenshot_dir.mkdir()
        
        json_path = screenshot_dir / "screenshot.json"
        json_path.write_text("{}")
        
        png_path = screenshot_dir / "screenshot.png"
        png_path.write_bytes(sample_png)
        
        found_image = _find_image_for_screenshot(json_path)
        
        assert found_image.exists()
    
    def test_find_image_exists_correct_name(self, tmp_path, sample_png):
        """Found image file has correct filename."""
        screenshot_dir = tmp_path / "timestamp"
        screenshot_dir.mkdir()
        
        json_path = screenshot_dir / "screenshot.json"
        json_path.write_text("{}")
        
        png_path = screenshot_dir / "screenshot.png"
        png_path.write_bytes(sample_png)
        
        found_image = _find_image_for_screenshot(json_path)
        
        assert found_image.name == "screenshot.png"
    
    def test_find_image_missing(self, tmp_path):
        """Return None when no image file exists."""
        screenshot_dir = tmp_path / "timestamp"
        screenshot_dir.mkdir()
        
        json_path = screenshot_dir / "screenshot.json"
        json_path.write_text("{}")
        
        found_image = _find_image_for_screenshot(json_path)
        
        assert found_image is None
    
    def test_find_image_multiple_returns_path(self, tmp_path, sample_png):
        """Multiple image files returns a path object."""
        screenshot_dir = tmp_path / "timestamp"
        screenshot_dir.mkdir()
        
        json_path = screenshot_dir / "screenshot.json"
        json_path.write_text("{}")
        
        png_path_1 = screenshot_dir / "aaa.png"
        png_path_1.write_bytes(sample_png)
        
        png_path_2 = screenshot_dir / "bbb.png"
        png_path_2.write_bytes(sample_png)
        
        found_image = _find_image_for_screenshot(json_path)
        
        assert found_image is not None
    
    def test_find_image_multiple_returns_valid_name(self, tmp_path, sample_png):
        """Multiple image files returns one of the valid filenames."""
        screenshot_dir = tmp_path / "timestamp"
        screenshot_dir.mkdir()
        
        json_path = screenshot_dir / "screenshot.json"
        json_path.write_text("{}")
        
        png_path_1 = screenshot_dir / "aaa.png"
        png_path_1.write_bytes(sample_png)
        
        png_path_2 = screenshot_dir / "bbb.png"
        png_path_2.write_bytes(sample_png)
        
        found_image = _find_image_for_screenshot(json_path)
        
        assert found_image.name in ["aaa.png", "bbb.png"]
    
    def test_get_mime_type_png(self, tmp_path):
        """Get MIME type for PNG file."""
        png_path = tmp_path / "test.png"
        png_path.write_bytes(b'')
        
        mime_type = _get_mime_type_from_path(png_path)
        
        assert mime_type == 'image/png'
    
    def test_get_mime_type_jpeg(self, tmp_path):
        """Get MIME type for JPEG file."""
        jpg_path = tmp_path / "test.jpg"
        jpg_path.write_bytes(b'')
        
        mime_type = _get_mime_type_from_path(jpg_path)
        
        assert mime_type == 'image/jpeg'
    
    def test_get_mime_type_webp(self, tmp_path):
        """Get MIME type for WebP file."""
        webp_path = tmp_path / "test.webp"
        webp_path.write_bytes(b'')
        
        mime_type = _get_mime_type_from_path(webp_path)
        
        # On Windows, WebP may not be recognized and returns 'application/octet-stream'
        # Both are acceptable since the function handles unknown types gracefully
        assert mime_type in ('image/webp', 'application/octet-stream')


# ==============================================================================
# Tests for API lookup functions
# ==============================================================================

class TestLookupSessionById:
    """Test lookup_session_by_id() function."""
    
    def test_lookup_session_success(self, requests_mock):
        """Successfully lookup session by ID."""
        session_id = "test-session-123"
        api_key = "test-api-key"
        
        expected_response = {
            "id": session_id,
            "human_instance_id": "instance-456",
            "status": "completed",
            "started_at": "2026-01-20T10:00:00Z",
            "completed_at": "2026-01-20T10:05:00Z",
            "storage_path": "/path/to/session",
            "storage_url": "https://example.com/session"
        }
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/sessions/{session_id}",
            json=expected_response
        )
        
        result = lookup_session_by_id(session_id, api_key)
        
        assert result is not None
        assert result["id"] == session_id
        assert result["human_instance_id"] == "instance-456"
        assert result["status"] == "completed"
    
    def test_lookup_session_with_nested_instance(self, requests_mock):
        """Lookup session returns optional nested human_instance data."""
        session_id = "test-session-123"
        api_key = "test-api-key"
        
        expected_response = {
            "id": session_id,
            "human_instance_id": "instance-456",
            "status": "completed",
            "human_instance": {
                "id": "instance-456",
                "caseId": "case-789"
            }
        }
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/sessions/{session_id}",
            json=expected_response
        )
        
        result = lookup_session_by_id(session_id, api_key)
        
        assert result is not None
        assert "human_instance" in result
        assert result["human_instance"]["id"] == "instance-456"
    
    def test_lookup_session_404_not_found(self, requests_mock):
        """Return None when session not found (404)."""
        session_id = "nonexistent-session"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/sessions/{session_id}",
            status_code=404
        )
        
        result = lookup_session_by_id(session_id, api_key)
        
        assert result is None
    
    def test_lookup_session_500_server_error(self, requests_mock):
        """Return None when server returns 500 error."""
        session_id = "test-session-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/sessions/{session_id}",
            status_code=500
        )
        
        result = lookup_session_by_id(session_id, api_key)
        
        assert result is None
    
    def test_lookup_session_connection_error(self, requests_mock):
        """Return None when connection fails."""
        session_id = "test-session-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/sessions/{session_id}",
            exc=requests.exceptions.ConnectionError("Connection failed")
        )
        
        result = lookup_session_by_id(session_id, api_key)
        
        assert result is None
    
    def test_lookup_session_timeout(self, requests_mock):
        """Return None when request times out."""
        session_id = "test-session-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/sessions/{session_id}",
            exc=requests.exceptions.Timeout("Request timed out")
        )
        
        result = lookup_session_by_id(session_id, api_key)
        
        assert result is None
    
    def test_lookup_session_retry_succeeds_on_second_attempt(self, requests_mock):
        """Retry logic succeeds on second attempt after first failure."""
        session_id = "test-session-123"
        api_key = "test-api-key"
        
        expected_response = {
            "id": session_id,
            "human_instance_id": "instance-456",
            "status": "completed"
        }
        
        # First call fails, second succeeds
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/sessions/{session_id}",
            [
                {"status_code": 500},
                {"json": expected_response, "status_code": 200}
            ]
        )
        
        result = lookup_session_by_id(session_id, api_key)
        
        assert result is not None
        assert result["id"] == session_id
        assert requests_mock.call_count == 2


class TestLookupInstanceById:
    """Test lookup_instance_by_id() function."""
    
    def test_lookup_instance_success(self, requests_mock):
        """Successfully lookup instance by ID with nested data."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        expected_response = {
            "id": instance_id,
            "caseId": "case-456",
            "taskId": "task-789",
            "status": "completed",
            "inputPayload": {"query": "test"},
            "outputPayload": {"result": "success"},
            "task": {
                "id": "task-789",
                "activityId": "activity-001",
                "suiteId": "suite-999",
                "instruction": "Test instruction",
                "guardrails": {"max_retries": 3}
            }
        }
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json=expected_response
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is not None
        assert result["id"] == instance_id
        assert result["caseId"] == "case-456"
        assert result["taskId"] == "task-789"
        assert result["task"]["id"] == "task-789"
    
    def test_lookup_instance_with_complete_nested_structure(self, requests_mock):
        """Lookup instance returns complete nested task/suite/actions."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        expected_response = {
            "id": instance_id,
            "caseId": "case-456",
            "taskId": "task-789",
            "inputPayload": {},
            "task": {
                "id": "task-789",
                "suiteId": "suite-999",
                "instruction": "Test",
                "guardrails": {"max_retries": 3},
                "suite": {
                    "suite_id": "suite-999",
                    "name": "Test Suite",
                    "actions": [
                        {
                            "action_id": "action-1",
                            "name": "click",
                            "parameters": [{"name": "selector", "type": "string"}]
                        },
                        {
                            "action_id": "action-2",
                            "name": "type",
                            "parameters": [{"name": "text", "type": "string"}]
                        }
                    ]
                }
            }
        }
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json=expected_response
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is not None
        assert "task" in result
        assert "suite" in result["task"]
        assert "actions" in result["task"]["suite"]
        assert len(result["task"]["suite"]["actions"]) == 2
        assert result["task"]["suite"]["actions"][0]["action_id"] == "action-1"
    
    def test_lookup_instance_minimal_response(self, requests_mock):
        """Lookup instance handles minimal response without optional fields."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        # Minimal response with only required fields
        expected_response = {
            "id": instance_id,
            "caseId": "case-456",
            "taskId": "task-789",
            "task": {
                "id": "task-789",
                "suiteId": "suite-999",
                "instruction": "Test"
            }
        }
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json=expected_response
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is not None
        assert result["id"] == instance_id
        assert result["task"]["id"] == "task-789"
    
    def test_lookup_instance_404_not_found(self, requests_mock):
        """Return None when instance not found (404)."""
        instance_id = "nonexistent-instance"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            status_code=404
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is None
    
    def test_lookup_instance_500_server_error(self, requests_mock):
        """Return None when server returns 500 error."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            status_code=500
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is None
    
    def test_lookup_instance_401_unauthorized(self, requests_mock):
        """Return None when API key is invalid (401)."""
        instance_id = "instance-123"
        api_key = "invalid-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            status_code=401
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is None
    
    def test_lookup_instance_connection_error(self, requests_mock):
        """Return None when connection fails."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            exc=requests.exceptions.ConnectionError("Connection failed")
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is None
    
    def test_lookup_instance_timeout(self, requests_mock):
        """Return None when request times out."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            exc=requests.exceptions.Timeout("Request timed out")
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is None
    
    def test_lookup_instance_retry_succeeds_on_second_attempt(self, requests_mock):
        """Retry logic succeeds on second attempt after first failure."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        expected_response = {
            "id": instance_id,
            "caseId": "case-456",
            "taskId": "task-789",
            "task": {
                "id": "task-789",
                "suiteId": "suite-999",
                "instruction": "Test"
            }
        }
        
        # First call fails, second succeeds
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            [
                {"status_code": 500},
                {"json": expected_response, "status_code": 200}
            ]
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is not None
        assert result["id"] == instance_id
        assert requests_mock.call_count == 2
    
    def test_lookup_instance_retry_fails_all_attempts(self, requests_mock):
        """Return None when all retry attempts fail."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        # All attempts return 500
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            status_code=500
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is None
        # Should try 3 times (1 initial + 2 retries)
        assert requests_mock.call_count == 3
    
    def test_lookup_instance_null_payloads(self, requests_mock):
        """Handle instance with null input/output payloads."""
        instance_id = "instance-123"
        api_key = "test-api-key"
        
        expected_response = {
            "id": instance_id,
            "caseId": "case-456",
            "taskId": "task-789",
            "inputPayload": None,
            "outputPayload": None,
            "task": {
                "id": "task-789",
                "suiteId": "suite-999",
                "instruction": "Test",
                "guardrails": None
            }
        }
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json=expected_response
        )
        
        result = lookup_instance_by_id(instance_id, api_key)
        
        assert result is not None
        assert result["inputPayload"] is None
        assert result["outputPayload"] is None
        assert result["task"]["guardrails"] is None


class TestLookupActionById:
    """Test action lookup by ID via lookup API."""
    
    def test_lookup_action_success(self, requests_mock):
        """Lookup action returns complete action with parameters."""
        action_id = "action-123"
        api_key = "test-api-key"
        
        expected_response = {
            "action_id": action_id,
            "name": "click",
            "description": "Click on an element",
            "type": "user_interaction",
            "parameters": [
                {
                    "name": "selector",
                    "type": "string",
                    "required": True,
                    "description": "CSS selector"
                },
                {
                    "name": "timeout",
                    "type": "number",
                    "required": False,
                    "default": 5000
                }
            ]
        }
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}",
            json=expected_response
        )
        
        result = lookup_action_by_id(action_id, api_key)
        
        assert result is not None
        assert result["action_id"] == action_id
        assert result["name"] == "click"
        assert len(result["parameters"]) == 2
        assert result["parameters"][0]["name"] == "selector"
    
    def test_lookup_action_404_not_found(self, requests_mock):
        """Return None when action not found (404)."""
        action_id = "nonexistent-action"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}",
            status_code=404
        )
        
        result = lookup_action_by_id(action_id, api_key)
        
        assert result is None
    
    def test_lookup_action_500_server_error(self, requests_mock):
        """Return None when server returns 500 error."""
        action_id = "action-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}",
            status_code=500
        )
        
        result = lookup_action_by_id(action_id, api_key)
        
        assert result is None
    
    def test_lookup_action_401_unauthorized(self, requests_mock):
        """Return None when API key is invalid (401)."""
        action_id = "action-123"
        api_key = "invalid-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}",
            status_code=401
        )
        
        result = lookup_action_by_id(action_id, api_key)
        
        assert result is None
    
    def test_lookup_action_connection_error(self, requests_mock):
        """Return None when connection fails."""
        action_id = "action-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}",
            exc=requests.exceptions.ConnectionError("Connection failed")
        )
        
        result = lookup_action_by_id(action_id, api_key)
        
        assert result is None
    
    def test_lookup_action_timeout(self, requests_mock):
        """Return None when request times out."""
        action_id = "action-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}",
            exc=requests.exceptions.Timeout("Request timed out")
        )
        
        result = lookup_action_by_id(action_id, api_key)
        
        assert result is None
    
    def test_lookup_action_retry_succeeds_on_second_attempt(self, requests_mock):
        """Retry logic succeeds on second attempt after first failure."""
        action_id = "action-123"
        api_key = "test-api-key"
        
        expected_response = {
            "action_id": action_id,
            "name": "click",
            "parameters": []
        }
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}",
            [
                {"status_code": 500},
                {"json": expected_response, "status_code": 200}
            ]
        )
        
        result = lookup_action_by_id(action_id, api_key)
        
        assert result is not None
        assert result["action_id"] == action_id
        assert requests_mock.call_count == 2
    
    def test_lookup_action_retry_fails_all_attempts(self, requests_mock):
        """Return None when all retry attempts fail."""
        action_id = "action-123"
        api_key = "test-api-key"
        
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/{action_id}",
            status_code=500
        )
        
        result = lookup_action_by_id(action_id, api_key)
        
        assert result is None
        assert requests_mock.call_count == 3


# ==============================================================================
# Tests for process_session()
# ==============================================================================

class TestProcessSession:
    """Test session-level processing with generator."""
    
    @pytest.fixture(autouse=True)
    def mock_api_for_metadata(self):
        """Auto-mock API credentials and lookups for all tests in this class."""
        from unittest.mock import patch
        
        with patch('action_extraction.extraction.API_KEY', 'test-api-key'):
            with patch('action_extraction.extraction.BACKEND_API_BASE_URL', 'http://test-backend.com'):
                with patch('action_extraction.extraction.lookup_session_by_id') as mock_session:
                    with patch('action_extraction.extraction.lookup_instance_by_id') as mock_instance:
                        # Mock successful API lookups
                        mock_session.return_value = {
                            'id': 'test-session',
                            'human_instance_id': 'instance-123'
                        }
                        mock_instance.return_value = {
                            'id': 'instance-123',
                            'caseId': 'case-456'
                        }
                        yield
    
    def test_process_session_complete_row_count(self, create_valid_session):
        """Process session returns correct number of rows."""
        session_dir = create_valid_session(num_inputs=2, num_screenshots=1)
        session_uuid = session_dir.name
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert len(rows) == 3
    
    def test_process_session_complete_same_session_uuid(self, create_valid_session):
        """Process session rows all have same session_uuid."""
        session_dir = create_valid_session(num_inputs=2, num_screenshots=1)
        session_uuid = session_dir.name
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        session_uuids = [row['trace']['session_uuid'] for row in rows]
        
        assert all(uuid == session_uuid for uuid in session_uuids)
    
    def test_process_session_complete_json_raw_is_string(self, create_valid_session):
        """Process session stores JSON as raw string."""
        session_dir = create_valid_session(num_inputs=2, num_screenshots=1)
        session_uuid = session_dir.name
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert all(isinstance(row['trace']['json_raw'], str) for row in rows)
    
    def test_process_session_input_only_has_rows(self, tmp_path, sample_input_json):
        """Process session with input only returns rows."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        input_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        input_file.write_text(sample_input_json)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert len(rows) > 0
    
    def test_process_session_input_only_correct_event_type(self, tmp_path, sample_input_json):
        """Process session with input only has correct event type."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        input_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        input_file.write_text(sample_input_json)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert rows[0]['trace']['event_type'] == 'input'
    
    def test_process_session_input_only_json_raw_is_string(self, tmp_path, sample_input_json):
        """Process session with input only stores JSON as string."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        input_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        input_file.write_text(sample_input_json)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert isinstance(rows[0]['trace']['json_raw'], str)
    
    def test_process_session_screenshot_only_has_rows(self, tmp_path, sample_screenshot_json, sample_png):
        """Process session with screenshot only returns rows."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        screenshot_dir = session_dir / "screenshot" / "2026-01-06T20_30_01.844649+00_00"
        screenshot_dir.mkdir(parents=True)
        
        json_file = screenshot_dir / "screenshot.json"
        json_file.write_text(sample_screenshot_json)
        
        png_file = screenshot_dir / "screenshot.png"
        png_file.write_bytes(sample_png)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert len(rows) > 0
    
    def test_process_session_screenshot_only_correct_event_type(self, tmp_path, sample_screenshot_json, sample_png):
        """Process session with screenshot only has correct event type."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        screenshot_dir = session_dir / "screenshot" / "2026-01-06T20_30_01.844649+00_00"
        screenshot_dir.mkdir(parents=True)
        
        json_file = screenshot_dir / "screenshot.json"
        json_file.write_text(sample_screenshot_json)
        
        png_file = screenshot_dir / "screenshot.png"
        png_file.write_bytes(sample_png)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert rows[0]['trace']['event_type'] == 'screenshot'
    
    def test_process_session_screenshot_only_has_image_mime(self, tmp_path, sample_screenshot_json, sample_png):
        """Process session with screenshot only includes image MIME type."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        screenshot_dir = session_dir / "screenshot" / "2026-01-06T20_30_01.844649+00_00"
        screenshot_dir.mkdir(parents=True)
        
        json_file = screenshot_dir / "screenshot.json"
        json_file.write_text(sample_screenshot_json)
        
        png_file = screenshot_dir / "screenshot.png"
        png_file.write_bytes(sample_png)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert rows[0]['image_mime'] == 'image/png'
    
    def test_process_session_screenshot_only_json_raw_is_string(self, tmp_path, sample_screenshot_json, sample_png):
        """Process session with screenshot only stores JSON as string."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        screenshot_dir = session_dir / "screenshot" / "2026-01-06T20_30_01.844649+00_00"
        screenshot_dir.mkdir(parents=True)
        
        json_file = screenshot_dir / "screenshot.json"
        json_file.write_text(sample_screenshot_json)
        
        png_file = screenshot_dir / "screenshot.png"
        png_file.write_bytes(sample_png)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert isinstance(rows[0]['trace']['json_raw'], str)
    
    def test_process_session_empty(self, tmp_path):
        """Handle session without input or screenshot dirs."""
        session_uuid = "empty-session"
        session_dir = tmp_path / session_uuid
        session_dir.mkdir()
        
        # Process empty session
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        # Assertions
        assert len(rows) == 0, "Should return no rows for empty session"
    
    def test_process_session_mixed_valid_invalid_row_count(self, tmp_path, sample_input_json):
        """Process session with mixed valid and different format files."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        # File with timezone format
        valid_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        valid_file.write_text(sample_input_json)
        
        # File without timezone (forward compatible)
        other_file = input_dir / "2026-01-06T20_31_00-input.json"
        other_file.write_text(sample_input_json)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert len(rows) == 2
    
    def test_process_session_mixed_valid_invalid_correct_timestamp(self, tmp_path, sample_input_json):
        """Process session extracts correct timestamp from timezone format."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        valid_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        valid_file.write_text(sample_input_json)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert rows[0]['trace']['timestamp'] == '2026-01-06T20:30:01.337745+00:00'
    
    def test_process_session_unicode_content_has_rows(self, tmp_path):
        """Process session with Unicode content returns rows."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        unicode_json = '{"text": "Hello 世界", "emoji": "🎉", "timestamp": "2026-01-06T20:30:01.337745+00:00"}'
        input_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        input_file.write_text(unicode_json, encoding='utf-8')
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        
        assert len(rows) == 1
    
    def test_process_session_unicode_content_json_raw_is_string(self, tmp_path):
        """Process session with Unicode stores JSON as string."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        unicode_json = '{"text": "Hello 世界", "emoji": "🎉", "timestamp": "2026-01-06T20:30:01.337745+00:00"}'
        input_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        input_file.write_text(unicode_json, encoding='utf-8')
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        json_raw = rows[0]['trace']['json_raw']
        
        assert isinstance(json_raw, str)
    
    def test_process_session_unicode_content_preserves_unicode(self, tmp_path):
        """Process session preserves Unicode characters and removes timestamp."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        unicode_json = '{"text": "Hello 世界", "emoji": "🎉", "timestamp": "2026-01-06T20:30:01.337745+00:00"}'
        input_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        input_file.write_text(unicode_json, encoding='utf-8')
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        json_raw = rows[0]['trace']['json_raw']
        
        # Verify unicode is preserved
        assert "世界" in json_raw
        # Verify timestamp is removed from json_raw
        parsed_json = json.loads(json_raw)
        assert 'timestamp' not in parsed_json
        # Verify timestamp is stored separately in trace
        assert rows[0]['trace']['timestamp'] == '2026-01-06T20:30:01.337745+00:00'
    
    def test_process_session_unicode_content_preserves_emoji(self, tmp_path):
        """Process session preserves emoji characters and removes timestamp."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        unicode_json = '{"text": "Hello 世界", "emoji": "🎉", "timestamp": "2026-01-06T20:30:01.337745+00:00"}'
        input_file = input_dir / "2026-01-06T20_30_01.337745+00_00-test-uuid.json"
        input_file.write_text(unicode_json, encoding='utf-8')
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)
        json_raw = rows[0]['trace']['json_raw']
        
        # Verify emoji is preserved
        assert "🎉" in json_raw
        # Verify timestamp is removed from json_raw
        parsed_json = json.loads(json_raw)
        assert 'timestamp' not in parsed_json
        # Verify timestamp is stored separately in trace
        assert rows[0]['trace']['timestamp'] == '2026-01-06T20:30:01.337745+00:00'


# ==============================================================================
# Tests for create_traces_dataframe()
# ==============================================================================

class TestCreateDataframe:
    """Test DataFrame creation and schema."""
    
    def test_create_traces_dataframe_valid_rows_row_count(self):
        """Create DataFrame has correct number of rows."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "scroll"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            },
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'screenshot',
                    'timestamp': '2026-01-06T20:30:02.000000+00:00',
                    'json_raw': '{"action": "click"}'
                },
                'instance_id': 'instance-123',
                'image_mime': 'image/png',
                'image_bytes': b'\x89PNG\r\n\x1a\n'
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert len(df) == 2
    
    def test_create_traces_dataframe_valid_rows_column_order(self):
        """Create DataFrame has correct column order."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "scroll"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        expected_columns = ['trace', 'instance_id', 'image_mime', 'image_bytes']
        
        assert list(df.columns) == expected_columns
    
    def test_create_traces_dataframe_valid_rows_trace_dtype(self):
        """Create DataFrame trace column has object dtype."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "scroll"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert df['trace'].dtype == 'object'
    
    def test_create_traces_dataframe_valid_rows_json_raw_is_string(self):
        """Create DataFrame stores json_raw as string."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "scroll"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert isinstance(df['trace'].iloc[0]['json_raw'], str)
    
    def test_create_traces_dataframe_empty_has_no_rows(self):
        """Create empty DataFrame has no rows."""
        rows = []
        
        df = create_traces_dataframe(rows)
        
        assert len(df) == 0
    
    def test_create_traces_dataframe_empty_has_correct_columns(self):
        """Create empty DataFrame has correct column structure."""
        rows = []
        
        df = create_traces_dataframe(rows)
        expected_columns = ['trace', 'instance_id', 'image_mime', 'image_bytes']
        
        assert list(df.columns) == expected_columns
    
    def test_create_traces_dataframe_null_handling_image_mime_nullable(self):
        """DataFrame allows None in image_mime column."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "test"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert pd.isna(df['image_mime'].iloc[0]) or df['image_mime'].iloc[0] is None
    
    def test_create_traces_dataframe_null_handling_image_bytes_nullable(self):
        """DataFrame allows None in image_bytes column."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "test"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert pd.isna(df['image_bytes'].iloc[0]) or df['image_bytes'].iloc[0] is None
    
    def test_create_traces_dataframe_null_handling_trace_not_null(self):
        """DataFrame trace column is not null."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "test"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert df['trace'].iloc[0] is not None
    
    def test_create_traces_dataframe_null_handling_session_uuid_not_null(self):
        """DataFrame trace.session_uuid is not null."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "test"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert df['trace'].iloc[0]['session_uuid'] is not None
    
    def test_create_traces_dataframe_null_handling_event_type_not_null(self):
        """DataFrame trace.event_type is not null."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "test"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert df['trace'].iloc[0]['event_type'] is not None
    
    def test_create_traces_dataframe_null_handling_timestamp_not_null(self):
        """DataFrame trace.timestamp is not null."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "test"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert df['trace'].iloc[0]['timestamp'] is not None
    
    def test_create_traces_dataframe_null_handling_json_raw_not_null_and_string(self):
        """DataFrame trace.json_raw is not null and is string type."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"data": "value"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        
        assert df['trace'].iloc[0]['json_raw'] is not None
        assert isinstance(df['trace'].iloc[0]['json_raw'], str)


# ==============================================================================
# Tests for Suite extraction functions - REMOVED (suites now extracted inline from nested API responses)
# ==============================================================================

# TestSuiteExtraction class removed - obsolete functions:
# - fetch_suites_from_api() - deleted
# - transform_suite_data() - deleted  
# - extract_suites() - deleted

# ==============================================================================
# Tests for action detail fetching and enrichment - REMOVED (actions come nested in suite)
# ==============================================================================

# TestActionDetailFetching class removed - obsolete functions:
# - fetch_action_details() - deleted
# - collect_action_data() - deleted

# ==============================================================================
# Tests for new instance extraction functionality - REMOVED (no JSON fallback)
# ==============================================================================

# TestExtractMetadataFromJson class removed - obsolete function:
# - _extract_metadata_from_json() - deleted (API-only now)

# ==============================================================================
# Tests for deprecated fetch functions - REMOVED
# ==============================================================================

# TestFetchInstanceDetails class removed - replaced by TestLookupInstanceById
# TestFetchTaskDetails class removed - task data comes nested in instance lookup

# ==============================================================================
# TODO: Add new tests for refactored lookup API functions
# ==============================================================================
# - TestLookupSessionById: Test new lookup_session_by_id() function
# - TestLookupInstanceById: Test new lookup_instance_by_id() function  
# - Update TestCollectInstanceData to use new 3-tuple return and mock lookup functions

class TestCollectInstanceData:
    """Test instance data collection orchestration."""
    
    def test_collect_instance_data_single_instance(self, requests_mock):
        """Collect data for single instance successfully."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        instance_id = "770e8400-e29b-41d4-a716-446655440001"
        task_id = "880e8400-e29b-41d4-a716-446655440002"
        suite_id = "aa0e8400-e29b-41d4-a716-446655440004"
        
        session_metadata_list = [
            {"case_id": case_id, "instance_id": instance_id}
        ]
        
        # Mock lookup instance API with nested task/suite/actions
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json={
                "id": instance_id,
                "caseId": case_id,
                "taskId": task_id,
                "inputPayload": {"query": "test"},
                "task": {
                    "id": task_id,
                    "suiteId": suite_id,
                    "instruction": "Test instruction",
                    "guardrails": {"max_retries": 3},
                    "suite": {
                        "suite_id": suite_id,
                        "name": "Test Suite",
                        "actions": [
                            {"action_id": "act-1", "name": "click"}
                        ]
                    }
                }
            }
        )
        
        # Mock action lookup for enrichment
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/act-1",
            json={
                "action_id": "act-1",
                "name": "click",
                "parameters": [
                    {"name": "selector", "type": "string", "required": True}
                ]
            }
        )
        
        # NEW: Returns 3-tuple now
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        assert len(instance_records) == 1
        assert instance_records[0]['instance_id'] == instance_id
        assert instance_records[0]['task_id'] == task_id
        assert instance_records[0]['suite_id'] == suite_id
        
        # NEW: Verify suite extraction
        assert len(suite_records) == 1
        assert suite_records[0]['suite_id'] == suite_id
        assert suite_records[0]['suite_name'] == "Test Suite"
        assert len(suite_records[0]['action_inventory']) == 1
        assert suite_records[0]['action_inventory'][0]['parameters'][0]['name'] == "selector"
        
        # Verify statistics (updated stat names)
        assert stats['total_attempted'] == 1
        assert stats['instance_api_failures'] == 0
        assert stats['missing_task_data'] == 0
        assert stats['missing_suite_data'] == 0
    
    def test_collect_instance_data_deduplication(self, requests_mock):
        """Deduplicate instances across multiple sessions."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        instance_id = "770e8400-e29b-41d4-a716-446655440001"
        task_id = "880e8400-e29b-41d4-a716-446655440002"
        suite_id = "aa0e8400-e29b-41d4-a716-446655440004"
        
        # Same instance appears in 3 sessions
        session_metadata_list = [
            {"case_id": case_id, "instance_id": instance_id},
            {"case_id": case_id, "instance_id": instance_id},
            {"case_id": case_id, "instance_id": instance_id}
        ]
        
        # Mock lookup instance API with nested data
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json={
                "id": instance_id,
                "caseId": case_id,
                "taskId": task_id,
                "inputPayload": {},
                "task": {
                    "id": task_id,
                    "suiteId": suite_id,
                    "instruction": "Test",
                    "guardrails": None,
                    "suite": {
                        "suite_id": suite_id,
                        "name": "Test Suite",
                        "actions": []
                    }
                }
            }
        )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        # Should only have 1 instance despite 3 sessions
        assert len(instance_records) == 1
        assert len(suite_records) == 1  # Suite also deduplicated
        assert stats['total_attempted'] == 1
    
    def test_collect_instance_data_multiple_instances(self, requests_mock):
        """Collect data for multiple different instances."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        instance_id_1 = "770e8400-e29b-41d4-a716-446655440001"
        instance_id_2 = "770e8400-e29b-41d4-a716-446655440002"
        task_id = "880e8400-e29b-41d4-a716-446655440003"
        suite_id = "aa0e8400-e29b-41d4-a716-446655440004"
        
        session_metadata_list = [
            {"case_id": case_id, "instance_id": instance_id_1},
            {"case_id": case_id, "instance_id": instance_id_2}
        ]
        
        # Mock lookup instance API for both instances with nested data
        for instance_id in [instance_id_1, instance_id_2]:
            requests_mock.get(
                f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
                json={
                    "id": instance_id,
                    "caseId": case_id,
                    "taskId": task_id,
                    "inputPayload": {},
                    "task": {
                        "id": task_id,
                        "suiteId": suite_id,
                        "instruction": "Test",
                        "guardrails": None,
                        "suite": {
                            "suite_id": suite_id,
                            "name": "Test Suite",
                            "actions": []
                        }
                    }
                }
            )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        assert len(instance_records) == 2
        assert instance_records[0]['instance_id'] != instance_records[1]['instance_id']
        assert len(suite_records) == 1  # Same suite for both instances (deduplicated)
        assert stats['total_attempted'] == 2
        assert stats['instance_api_failures'] == 0
    
    def test_collect_instance_data_skip_none_metadata(self, requests_mock):
        """Skip sessions with None metadata."""
        from action_extraction.extraction import collect_instance_data
        
        session_metadata_list = [
            {"case_id": "case-1", "instance_id": "instance-1"},
            None,  # Old data format
            None
        ]
        
        # Mock lookup API
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/instance-1",
            json={
                "id": "instance-1",
                "caseId": "case-1",
                "taskId": "task-1",
                "inputPayload": {},
                "task": {
                    "id": "task-1",
                    "suiteId": "suite-1",
                    "instruction": "Test",
                    "guardrails": None,
                    "suite": {
                        "suite_id": "suite-1",
                        "name": "Test Suite",
                        "actions": []
                    }
                }
            }
        )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        # Should only process the 1 valid metadata entry
        assert len(instance_records) == 1
        assert len(suite_records) == 1
        assert stats['total_attempted'] == 1
    
    def test_collect_instance_data_skip_missing_suite_id(self, requests_mock):
        """Skip instance when suite_id is None in suite data."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        instance_id = "770e8400-e29b-41d4-a716-446655440001"
        task_id = "880e8400-e29b-41d4-a716-446655440002"
        
        session_metadata_list = [
            {"case_id": case_id, "instance_id": instance_id}
        ]
        
        # Mock API response with suite data but no suite_id
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json={
                "id": instance_id,
                "caseId": case_id,
                "taskId": task_id,
                "inputPayload": {},
                "task": {
                    "id": task_id,
                    "suiteId": None,
                    "instruction": "Test",
                    "guardrails": None,
                    "suite": {
                        "suite_id": None,  # Missing suite_id
                        "name": "Test Suite",
                        "actions": []
                    }
                }
            }
        )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        # Instance should be skipped due to missing suite_id
        assert len(instance_records) == 0
        assert len(suite_records) == 0
        assert stats['total_attempted'] == 1
        assert stats['missing_suite_data'] == 1  # Counted as missing suite data
    
    def test_collect_instance_data_task_caching(self, requests_mock):
        """Verify instance lookup is called once per unique instance (no task caching needed - data is nested)."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        task_id = "880e8400-e29b-41d4-a716-446655440003"
        suite_id = "aa0e8400-e29b-41d4-a716-446655440004"
        
        # 3 instances with same task
        session_metadata_list = [
            {"case_id": case_id, "instance_id": f"instance-{i}"}
            for i in range(3)
        ]
        
        # Mock lookup API for each instance (all return nested task/suite)
        for i in range(3):
            requests_mock.get(
                f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/instance-{i}",
                json={
                    "id": f"instance-{i}",
                    "caseId": case_id,
                    "taskId": task_id,
                    "inputPayload": {},
                    "task": {
                        "id": task_id,
                        "suiteId": suite_id,
                        "instruction": "Test",
                        "guardrails": None,
                        "suite": {
                            "suite_id": suite_id,
                            "name": "Test Suite",
                            "actions": []
                        }
                    }
                }
            )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        # All 3 instances collected
        assert len(instance_records) == 3
        # Suite deduplicated (only 1 unique suite_id)
        assert len(suite_records) == 1
        assert stats['total_attempted'] == 3
        assert stats['instance_api_failures'] == 0
    
    def test_collect_instance_data_action_enrichment_success(self, requests_mock):
        """Verify actions are enriched with parameters from action lookup."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        instance_id = "770e8400-e29b-41d4-a716-446655440001"
        task_id = "880e8400-e29b-41d4-a716-446655440002"
        suite_id = "aa0e8400-e29b-41d4-a716-446655440004"
        
        session_metadata_list = [
            {"case_id": case_id, "instance_id": instance_id}
        ]
        
        # Mock instance lookup with basic actions (no parameters)
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json={
                "id": instance_id,
                "caseId": case_id,
                "taskId": task_id,
                "inputPayload": {},
                "task": {
                    "id": task_id,
                    "suiteId": suite_id,
                    "instruction": "Test",
                    "guardrails": None,
                    "suite": {
                        "suite_id": suite_id,
                        "name": "Test Suite",
                        "actions": [
                            {"action_id": "action-1", "name": "click"},
                            {"action_id": "action-2", "name": "type"}
                        ]
                    }
                }
            }
        )
        
        # Mock action lookups with full details including parameters
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/action-1",
            json={
                "action_id": "action-1",
                "name": "click",
                "parameters": [
                    {"name": "selector", "type": "string", "required": True}
                ]
            }
        )
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/action-2",
            json={
                "action_id": "action-2",
                "name": "type",
                "parameters": [
                    {"name": "text", "type": "string", "required": True},
                    {"name": "delay", "type": "number", "required": False}
                ]
            }
        )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        assert len(suite_records) == 1
        actions = suite_records[0]['action_inventory']
        assert len(actions) == 2
        assert len(actions[0]['parameters']) == 1
        assert actions[0]['parameters'][0]['name'] == "selector"
        assert len(actions[1]['parameters']) == 2
        assert actions[1]['parameters'][0]['name'] == "text"
    
    def test_collect_instance_data_action_enrichment_failure_fallback(self, requests_mock):
        """Fallback to basic action data when action lookup fails."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        instance_id = "770e8400-e29b-41d4-a716-446655440001"
        task_id = "880e8400-e29b-41d4-a716-446655440002"
        suite_id = "aa0e8400-e29b-41d4-a716-446655440004"
        
        session_metadata_list = [
            {"case_id": case_id, "instance_id": instance_id}
        ]
        
        # Mock instance lookup
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json={
                "id": instance_id,
                "caseId": case_id,
                "taskId": task_id,
                "inputPayload": {},
                "task": {
                    "id": task_id,
                    "suiteId": suite_id,
                    "instruction": "Test",
                    "guardrails": None,
                    "suite": {
                        "suite_id": suite_id,
                        "name": "Test Suite",
                        "actions": [
                            {"action_id": "action-1", "name": "click"}
                        ]
                    }
                }
            }
        )
        
        # Mock action lookup failure
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/actions/action-1",
            status_code=500
        )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        assert len(suite_records) == 1
        actions = suite_records[0]['action_inventory']
        assert len(actions) == 1
        assert actions[0]['action_id'] == "action-1"
        assert actions[0]['name'] == "click"
    
    def test_collect_instance_data_action_without_id(self, requests_mock):
        """Handle actions missing action_id."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        instance_id = "770e8400-e29b-41d4-a716-446655440001"
        task_id = "880e8400-e29b-41d4-a716-446655440002"
        suite_id = "aa0e8400-e29b-41d4-a716-446655440004"
        
        session_metadata_list = [
            {"case_id": case_id, "instance_id": instance_id}
        ]
        
        # Mock instance lookup with action missing action_id
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json={
                "id": instance_id,
                "caseId": case_id,
                "taskId": task_id,
                "inputPayload": {},
                "task": {
                    "id": task_id,
                    "suiteId": suite_id,
                    "instruction": "Test",
                    "guardrails": None,
                    "suite": {
                        "suite_id": suite_id,
                        "name": "Test Suite",
                        "actions": [
                            {"name": "unknown_action"}
                        ]
                    }
                }
            }
        )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        assert len(suite_records) == 1
        actions = suite_records[0]['action_inventory']
        assert len(actions) == 1
        assert actions[0]['name'] == "unknown_action"
    
    def test_collect_instance_data_multiple_actions_per_suite(self, requests_mock):
        """Verify all actions in suite are enriched."""
        from action_extraction.extraction import collect_instance_data
        
        case_id = "550e8400-e29b-41d4-a716-446655440000"
        instance_id = "770e8400-e29b-41d4-a716-446655440001"
        task_id = "880e8400-e29b-41d4-a716-446655440002"
        suite_id = "aa0e8400-e29b-41d4-a716-446655440004"
        
        session_metadata_list = [
            {"case_id": case_id, "instance_id": instance_id}
        ]
        
        # Mock instance lookup with 5 actions
        requests_mock.get(
            f"{BACKEND_API_BASE_URL}/api/v1/lookup/instances/{instance_id}",
            json={
                "id": instance_id,
                "caseId": case_id,
                "taskId": task_id,
                "inputPayload": {},
                "task": {
                    "id": task_id,
                    "suiteId": suite_id,
                    "instruction": "Test",
                    "guardrails": None,
                    "suite": {
                        "suite_id": suite_id,
                        "name": "Test Suite",
                        "actions": [
                            {"action_id": f"action-{i}", "name": f"action_{i}"}
                            for i in range(5)
                        ]
                    }
                }
            }
        )
        
        # Mock action lookups for all 5 actions
        for i in range(5):
            requests_mock.get(
                f"{BACKEND_API_BASE_URL}/api/v1/actions/action-{i}",
                json={
                    "action_id": f"action-{i}",
                    "name": f"action_{i}",
                    "parameters": [
                        {"name": f"param_{i}", "type": "string", "required": True}
                    ]
                }
            )
        
        instance_records, suite_records, stats = collect_instance_data(session_metadata_list, "test-api-key")
        
        assert len(suite_records) == 1
        actions = suite_records[0]['action_inventory']
        assert len(actions) == 5
        for i, action in enumerate(actions):
            assert action['action_id'] == f"action-{i}"
            assert len(action['parameters']) == 1
            assert action['parameters'][0]['name'] == f"param_{i}"


class TestCreateInstanceDataframe:
    """Test instance DataFrame creation."""
    
    def test_create_instance_dataframe_valid_records(self):
        """Create instance DataFrame with valid records."""
        from action_extraction.extraction import create_instance_dataframe
        
        instance_records = [
            {
                'instance_id': 'instance-1',
                'task_id': 'task-1',
                'suite_id': 'suite-1',
                'task_instruction': 'Test instruction 1',
                'task_guardrails': {'max_retries': 3},
                'input_payload': {'query': 'test'}
            },
            {
                'instance_id': 'instance-2',
                'task_id': 'task-2',
                'suite_id': 'suite-2',
                'task_instruction': 'Test instruction 2',
                'task_guardrails': None,
                'input_payload': None
            }
        ]
        
        df = create_instance_dataframe(instance_records)
        
        assert len(df) == 2
        assert list(df.columns) == [
            'instance_id', 'task_id', 'suite_id',
            'task_instruction', 'task_guardrails', 'input_payload'
        ]
    
    def test_create_instance_dataframe_empty(self):
        """Create empty instance DataFrame."""
        from action_extraction.extraction import create_instance_dataframe
        
        instance_records = []
        
        df = create_instance_dataframe(instance_records)
        
        assert len(df) == 0
        assert list(df.columns) == [
            'instance_id', 'task_id', 'suite_id',
            'task_instruction', 'task_guardrails', 'input_payload'
        ]
    
    def test_create_instance_dataframe_nullable_fields(self):
        """Instance DataFrame allows None in nullable fields."""
        from action_extraction.extraction import create_instance_dataframe
        
        instance_records = [
            {
                'instance_id': 'instance-1',
                'task_id': 'task-1',
                'suite_id': 'suite-1',
                'task_instruction': 'Test',
                'task_guardrails': None,
                'input_payload': None
            }
        ]
        
        df = create_instance_dataframe(instance_records)
        
        assert pd.isna(df['task_guardrails'].iloc[0]) or df['task_guardrails'].iloc[0] is None
        assert pd.isna(df['input_payload'].iloc[0]) or df['input_payload'].iloc[0] is None


class TestProcessSessionWithMetadata:
    """Test process_session returns metadata tuple."""
    
    def test_process_session_returns_tuple(self, tmp_path):
        """Process session returns tuple of (generator, metadata)."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        json_with_metadata = json.dumps({
            "case": "case-1",
            "instance": "instance-1",
            "timestamp": "2026-01-06T20:30:01.337745+00:00",
            "action": "click"
        })
        
        input_file = input_dir / "trace.json"
        input_file.write_text(json_with_metadata)
        
        result = process_session(session_dir, session_uuid)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_process_session_extracts_metadata(self, tmp_path):
        """Process session extracts metadata via API lookup."""
        from unittest.mock import patch
        
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        json_content = json.dumps({
            "timestamp": "2026-01-06T20:30:01.337745+00:00",
            "action": "click"
        })
        
        input_file = input_dir / "trace.json"
        input_file.write_text(json_content)
        
        # Mock API credentials (required for API lookup to trigger)
        with patch('action_extraction.extraction.API_KEY', 'test-api-key'):
            with patch('action_extraction.extraction.BACKEND_API_BASE_URL', 'http://test-backend.com'):
                # Mock the NEW lookup functions
                with patch('action_extraction.extraction.lookup_session_by_id') as mock_session_lookup:
                    with patch('action_extraction.extraction.lookup_instance_by_id') as mock_instance_lookup:
                        # Mock session lookup returns instance ID
                        mock_session_lookup.return_value = {
                            'id': session_uuid,
                            'human_instance_id': 'instance-456'
                        }
                        
                        # Mock instance lookup returns case ID
                        mock_instance_lookup.return_value = {
                            'id': 'instance-456',
                            'caseId': 'case-123',
                            'taskId': 'task-789'
                        }
                        
                        generator, metadata = process_session(session_dir, session_uuid)
                        list(generator)  # Consume generator
                        
                        # Verify mocked functions were actually called
                        mock_session_lookup.assert_called_once_with(session_uuid, 'test-api-key')
                        mock_instance_lookup.assert_called_once_with('instance-456', 'test-api-key')
        
        assert metadata is not None
        assert metadata['case_id'] == "case-123"
        assert metadata['instance_id'] == "instance-456"
    
    def test_process_session_no_metadata_skips_session(self, tmp_path):
        """Process session skips sessions without instance_id metadata."""
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        json_without_metadata = json.dumps({
            "timestamp": "2026-01-06T20:30:01.337745+00:00",
            "action": "click"
        })
        
        input_file = input_dir / "trace.json"
        input_file.write_text(json_without_metadata)
        
        generator, metadata = process_session(session_dir, session_uuid)
        rows = list(generator)  # Consume generator
        
        # Session without instance_id should be skipped (empty generator)
        assert len(rows) == 0
        assert metadata is None
    
    def test_process_session_api_lookup_failure_skips_session(self, tmp_path):
        """Process session skips sessions where API lookup fails."""
        from unittest.mock import patch
        
        session_uuid = "test-session"
        session_dir = tmp_path / session_uuid
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        json_content = json.dumps({
            "timestamp": "2026-01-06T20:30:01.337745+00:00",
            "action": "click"
        })
        
        input_file = input_dir / "trace.json"
        input_file.write_text(json_content)
        
        # Mock API credentials but make lookup fail
        with patch('action_extraction.extraction.API_KEY', 'test-api-key'):
            with patch('action_extraction.extraction.BACKEND_API_BASE_URL', 'http://test-backend.com'):
                with patch('action_extraction.extraction.lookup_session_by_id') as mock_session_lookup:
                    # Mock session lookup returns None (failed lookup)
                    mock_session_lookup.return_value = None
                    
                    generator, metadata = process_session(session_dir, session_uuid)
                    rows = list(generator)  # Consume generator
                    
                    # Session with failed API lookup should be skipped
                    assert len(rows) == 0
                    assert metadata is None


class TestSaveTracesToBronze:
    """Test saving trace DataFrame to Parquet."""
    
    def test_save_traces_to_bronze_creates_file(self, tmp_path):
        """Save traces creates Parquet file."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "scroll"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        output_file = save_to_bronze(df, tmp_path, "20260121_120000", 'traces', ['trace'])
        
        assert output_file.exists()
        assert output_file.name == "traces_bronze_20260121_120000.parquet"
    
    def test_save_traces_to_bronze_creates_subdirectory(self, tmp_path):
        """Save traces creates traces subdirectory."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "test"}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        output_file = save_to_bronze(df, tmp_path, "20260121_120000", 'traces', ['trace'])
        
        assert output_file.parent.name == "traces"
    
    def test_save_traces_to_bronze_readable(self, tmp_path):
        """Saved trace Parquet file is readable."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'screenshot',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "click"}'
                },
                'instance_id': 'instance-123',
                'image_mime': 'image/png',
                'image_bytes': b'\x89PNG\r\n\x1a\n'
            }
        ]
        
        df = create_traces_dataframe(rows)
        output_file = save_to_bronze(df, tmp_path, "20260121_120000", 'traces', ['trace'])
        
        # Read back and verify
        df_read = pd.read_parquet(output_file)
        assert len(df_read) == 1
        # Trace column is serialized as JSON string
        trace = json.loads(df_read['trace'].iloc[0])
        assert trace['session_uuid'] == 'session-1'
        # Instance ID is preserved
        assert df_read['instance_id'].iloc[0] == 'instance-123'
    
    def test_save_traces_to_bronze_json_serialization(self, tmp_path):
        """Saved trace Parquet serializes trace column."""
        rows = [
            {
                'trace': {
                    'session_uuid': 'session-1',
                    'event_type': 'input',
                    'timestamp': '2026-01-06T20:30:01.337745+00:00',
                    'json_raw': '{"action": "scroll", "selector": {"tag": "div"}}'
                },
                'instance_id': 'instance-123',
                'image_mime': None,
                'image_bytes': None
            }
        ]
        
        df = create_traces_dataframe(rows)
        output_file = save_to_bronze(df, tmp_path, "20260121_120000", 'traces', ['trace'])
        
        # Read back
        df_read = pd.read_parquet(output_file)
        
        # Trace column should be serialized as JSON string in Parquet
        trace = json.loads(df_read['trace'].iloc[0])
        assert trace['session_uuid'] == 'session-1'
        assert trace['event_type'] == 'input'
        assert trace['timestamp'] == '2026-01-06T20:30:01.337745+00:00'
        assert isinstance(trace['json_raw'], str)
        # Instance ID preserved
        assert df_read['instance_id'].iloc[0] == 'instance-123'


class TestSaveInstancesToBronze:
    """Test saving instance DataFrame to Parquet."""
    
    def test_save_instances_to_bronze_creates_file(self, tmp_path):
        """Save instances creates Parquet file."""
        from action_extraction.extraction import save_to_bronze, create_instance_dataframe
        
        instance_records = [
            {
                'instance_id': 'instance-1',
                'task_id': 'task-1',
                'suite_id': 'suite-1',
                'task_instruction': 'Test',
                'task_guardrails': {'max_retries': 3},
                'input_payload': {'query': 'test'}
            }
        ]
        
        df = create_instance_dataframe(instance_records)
        output_file = save_to_bronze(df, tmp_path, "20260121_120000", 'instances', ['task_guardrails', 'input_payload'])
        
        assert output_file.exists()
        assert output_file.name == "instances_bronze_20260121_120000.parquet"
    
    def test_save_instances_to_bronze_creates_subdirectory(self, tmp_path):
        """Save instances creates instances subdirectory."""
        from action_extraction.extraction import save_to_bronze, create_instance_dataframe
        
        instance_records = [
            {
                'instance_id': 'instance-1',
                'task_id': 'task-1',
                'suite_id': 'suite-1',
                'task_instruction': 'Test',
                'task_guardrails': None,
                'input_payload': None
            }
        ]
        
        df = create_instance_dataframe(instance_records)
        output_file = save_to_bronze(df, tmp_path, "20260121_120000", 'instances', ['task_guardrails', 'input_payload'])
        
        assert output_file.parent.name == "instances"
    
    def test_save_instances_to_bronze_readable(self, tmp_path):
        """Saved instance Parquet file is readable."""
        from action_extraction.extraction import save_to_bronze, create_instance_dataframe
        
        instance_records = [
            {
                'instance_id': 'instance-1',
                'task_id': 'task-1',
                'suite_id': 'suite-1',
                'task_instruction': 'Test instruction',
                'task_guardrails': {'max_retries': 3},
                'input_payload': {'query': 'test'}
            }
        ]
        
        df = create_instance_dataframe(instance_records)
        output_file = save_to_bronze(df, tmp_path, "20260121_120000", 'instances', ['task_guardrails', 'input_payload'])
        
        # Read back and verify
        df_read = pd.read_parquet(output_file)
        assert len(df_read) == 1
        assert df_read['instance_id'].iloc[0] == 'instance-1'
    
    def test_save_instances_to_bronze_json_serialization(self, tmp_path):
        """Saved instance Parquet serializes JSON columns."""
        from action_extraction.extraction import save_to_bronze, create_instance_dataframe
        
        instance_records = [
            {
                'instance_id': 'instance-1',
                'task_id': 'task-1',
                'suite_id': 'suite-1',
                'task_instruction': 'Test',
                'task_guardrails': {'max_retries': 3, 'timeout': 30},
                'input_payload': {'query': 'test', 'filters': ['a', 'b']}
            }
        ]
        
        df = create_instance_dataframe(instance_records)
        output_file = save_to_bronze(df, tmp_path, "20260121_120000", 'instances', ['task_guardrails', 'input_payload'])
        
        # Read back
        df_read = pd.read_parquet(output_file)
        
        # JSON columns should be strings in Parquet
        guardrails = json.loads(df_read['task_guardrails'].iloc[0])
        assert guardrails['max_retries'] == 3
        
        payload = json.loads(df_read['input_payload'].iloc[0])
        assert payload['query'] == 'test'