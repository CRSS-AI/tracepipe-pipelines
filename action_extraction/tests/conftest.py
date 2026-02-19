"""Shared pytest fixtures for tests."""

import pytest
from PIL import Image
import io
import requests_mock as rm_module


@pytest.fixture
def sample_input_json():
    """Sample input trace JSON content."""
    return """{
  "suite": "gmail",
  "tab": 1,
  "url": "https://mail.google.com",
  "timestamp": "2026-01-06T20:30:01.337745+00:00",
  "action": "scroll"
}"""


@pytest.fixture
def sample_screenshot_json():
    """Sample screenshot metadata JSON content."""
    return """{
  "suite": "gmail",
  "tab": 1,
  "url": "https://mail.google.com",
  "timestamp": "2026-01-06T20:30:01.844649+00:00",
  "action": "click",
  "image_format": "png"
}"""


@pytest.fixture
def sample_png():
    """Generate a small valid PNG image (1x1 pixel)."""
    img = Image.new('RGB', (1, 1), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with test data."""
    from action_extraction.extraction import create_dataframe
    
    rows = [
        {
            'trace': {
                'session_uuid': 'test-uuid',
                'event_type': 'input',
                'timestamp': '2026-01-06T20:30:01.337745+00:00',
                'json_raw': '{"action": "test"}',
            },
            'image_mime': None,
            'image_bytes': None,
        },
        {
            'trace': {
                'session_uuid': 'test-uuid',
                'event_type': 'screenshot',
                'timestamp': '2026-01-06T20:30:02.000000+00:00',
                'json_raw': '{"action": "click"}',
            },
            'image_mime': 'image/png',
            'image_bytes': b'\x89PNG\r\n\x1a\n',  # PNG header
        },
    ]
    return create_dataframe(rows)


@pytest.fixture
def create_valid_session(tmp_path, sample_input_json, sample_screenshot_json, sample_png):
    """Helper to create a valid session directory."""
    def _create_session(session_uuid="test-session-uuid", num_inputs=2, num_screenshots=1):
        session_dir = tmp_path / session_uuid
        
        # Create input directory
        input_dir = session_dir / "input"
        input_dir.mkdir(parents=True)
        
        for i in range(num_inputs):
            # Use Windows-compatible filename (colons replaced with underscores)
            # Actual timestamp is in JSON content, not filename
            input_file = input_dir / f"2026-01-06T20_30_0{i}.337745+00_00-test-uuid-{i}.json"
            input_file.write_text(sample_input_json)
        
        # Create screenshot directory
        for i in range(num_screenshots):
            # Use Windows-compatible directory name
            timestamp_dir = f"2026-01-06T20_30_0{i}.844649+00_00"
            screenshot_dir = session_dir / "screenshot" / timestamp_dir
            screenshot_dir.mkdir(parents=True)
            
            json_file = screenshot_dir / f"screenshot-uuid-{i}.json"
            json_file.write_text(sample_screenshot_json)
            
            png_file = screenshot_dir / f"screenshot-uuid-{i}.png"
            png_file.write_bytes(sample_png)
        
        return session_dir
    
    return _create_session


@pytest.fixture
def requests_mock():
    """Provide requests_mock adapter for API testing."""
    with rm_module.Mocker() as m:
        yield m
