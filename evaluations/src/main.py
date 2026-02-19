"""
Main entrypoint for evaluation pipeline: downloads dataset manifest from Azure Blob Storage.
Sensitive values are loaded from .env via Settings.
"""
import os
import re
import subprocess
import sys
from pathlib import Path
import pandas as pd
from typing import List, Dict
from config.settings import Settings
from azure.storage.blob import BlobServiceClient


def download_manifest(settings: Settings, download_dir: str) -> str:
    """
    Download the latest manifest file from Azure Blob Storage.
    Lists all blobs in the manifests directory and downloads the latest
    one following the naming convention: dataset-manifest-00n.csv
    """
    blob_service_client = BlobServiceClient(
        account_url=f"https://{settings.azure_storage_account_name}.blob.core.windows.net",
        credential=settings.azure_storage_account_key
    )
    container_client = blob_service_client.get_container_client(settings.azure_blob_container)
    
    # List all blobs in the manifests directory
    manifest_pattern = re.compile(r"dataset-manifest-(\d+)\.csv$")
    manifests = []
    
    # Use name_starts_with to filter blobs in the directory
    prefix = settings.manifests_directory.rstrip('/') + '/'
    blob_list = container_client.list_blobs(name_starts_with=prefix)
    
    for blob in blob_list:
        blob_name = blob.name
        # Extract just the filename
        filename = os.path.basename(blob_name)
        match = manifest_pattern.match(filename)
        if match:
            sequence_number = int(match.group(1))
            manifests.append((sequence_number, blob_name))
    
    if not manifests:
        raise FileNotFoundError(
            f"No manifest files matching pattern 'dataset-manifest-*.csv' found in {prefix}"
        )
    
    # Sort by sequence number and get the latest
    manifests.sort(reverse=True)
    latest_manifest_blob_path = manifests[0][1]
    
    print(f"Found {len(manifests)} manifest(s). Latest: {latest_manifest_blob_path}")
    
    # Download the latest manifest
    blob_client = container_client.get_blob_client(latest_manifest_blob_path)
    download_path = os.path.join(download_dir, os.path.basename(latest_manifest_blob_path))
    
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    with open(download_path, "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())
    
    return download_path


def download_sessions_from_manifest(
    settings: Settings,
    manifest_path: str,
    download_dir: str
) -> List[str]:
    """
    Read the dataset manifest CSV and download blob directories indicated by session_id.
    Each session_id corresponds to a directory in the sessions container.
    
    Args:
        settings: Settings object with Azure credentials and container info
        manifest_path: Path to the local manifest CSV file
        download_dir: Local directory to download session blobs to
        
    Returns:
        List of paths to downloaded session directories
    """
    # Read manifest CSV using pandas
    df = pd.read_csv(manifest_path)
    
    if 'session_id' not in df.columns:
        raise ValueError(f"Manifest must contain 'session_id' column. Found columns: {df.columns.tolist()}")
    
    # Get unique session IDs
    session_ids = df['session_id'].dropna().unique()
    
    if len(session_ids) == 0:
        raise ValueError("No session_id values found in manifest")
    
    print(f"Found {len(session_ids)} unique session(s) in manifest")
    
    # Initialize blob service client
    blob_service_client = BlobServiceClient(
        account_url=f"https://{settings.azure_storage_account_name}.blob.core.windows.net",
        credential=settings.azure_storage_account_key
    )
    container_client = blob_service_client.get_container_client(settings.sessions_container)
    
    downloaded_paths = []
    
    # Download each session directory
    for session_id in session_ids:
        print(f"Downloading session: {session_id}")
        
        # List all blobs in the session directory
        prefix = str(session_id).rstrip('/') + '/'
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        
        session_blobs = list(blob_list)
        if not session_blobs:
            print(f"  Warning: No blobs found for session {session_id}")
            continue
        
        print(f"  Found {len(session_blobs)} blob(s) in session directory")
        
        # Download each blob in the session directory
        for blob in session_blobs:
            blob_name = blob.name
            
            # Create local path maintaining directory structure
            relative_path = blob_name
            local_path = os.path.join(download_dir, relative_path)
            
            # Create parent directories
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download blob
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            
            print(f"  Downloaded: {blob_name}")
        
        session_local_path = os.path.join(download_dir, str(session_id))
        downloaded_paths.append(session_local_path)
    
    print(f"\nSuccessfully downloaded {len(downloaded_paths)} session(s)")
    return downloaded_paths


def run_action_extraction_pipeline(
    sessions_dir: str,
    output_base_dir: str
) -> Dict[str, str]:
    """
    Run the action_extraction pipeline for each session directory.
    
    Args:
        sessions_dir: Directory containing downloaded session folders
        output_base_dir: Base directory for processing results (e.g., artifacts/processing_results)
        
    Returns:
        Dictionary mapping session_id to output path
    """
    sessions_path = Path(sessions_dir)
    results = {}
    
    if not sessions_path.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")
    
    # Find all session directories
    session_dirs = [d for d in sessions_path.iterdir() if d.is_dir()]
    
    if not session_dirs:
        print(f"Warning: No session directories found in {sessions_dir}")
        return results
    
    print(f"\nProcessing {len(session_dirs)} session(s) through action_extraction pipeline")
    
    # Get path to action_extraction module (relative to evaluations directory)
    action_extraction_path = Path(__file__).parent.parent.parent / "action_extraction"
    
    for session_dir in session_dirs:
        session_id = session_dir.name
        print(f"\nProcessing session: {session_id}")
        
        # Create output directory for this session
        session_output_dir = Path(output_base_dir) / session_id
        session_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run extraction pipeline with session as input
        # The action_extraction pipeline expects the input to be a directory containing session folders
        # So we need to pass the parent directory (sessions_dir) not the individual session
        # But we only want to process one session at a time
        
        # Create a temporary input structure for this session
        temp_input_dir = session_output_dir / "temp_input"
        temp_input_dir.mkdir(exist_ok=True)
        
        # Create symlink to session directory
        temp_session_link = temp_input_dir / session_id
        if temp_session_link.exists():
            temp_session_link.unlink()
        temp_session_link.symlink_to(session_dir.absolute())
        
        try:
            # Run the extraction pipeline using uv run to use action_extraction's environment
            # Use absolute paths since we're changing cwd to action_extraction
            cmd = [
                "uv", "run", "python", "-m", "action_extraction.extraction",
                "--input", str(temp_input_dir.absolute()),
                "--output", str(session_output_dir.absolute()),
                "--traces-only"
            ]
            
            print(f"  Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(action_extraction_path),
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                print(f"  ✓ Successfully processed session {session_id}")
                results[session_id] = str(session_output_dir)
            else:
                print(f"  ✗ Failed to process session {session_id}")
                print(f"  Error: {result.stderr}")
                
        except Exception as e:
            print(f"  ✗ Exception processing session {session_id}: {e}")
        finally:
            # Clean up temp input directory
            if temp_session_link.exists():
                temp_session_link.unlink()
            if temp_input_dir.exists():
                temp_input_dir.rmdir()
    
    print(f"\n✓ Completed processing {len(results)}/{len(session_dirs)} session(s)")
    return results


def run_window_classification_pipeline(
    processing_results_dir: str,
    classification_output_dir: str
) -> str:
    """
    Run window_classification on processed sessions and aggregate results.
    
    Args:
        processing_results_dir: Directory containing processing results by session_id
        classification_output_dir: Directory to store classification results CSV
        
    Returns:
        Path to the aggregated classification results CSV
    """
    processing_path = Path(processing_results_dir)
    output_path = Path(classification_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not processing_path.exists():
        raise FileNotFoundError(f"Processing results directory not found: {processing_results_dir}")
    
    # Find all session result directories
    session_dirs = [d for d in processing_path.iterdir() if d.is_dir()]
    
    if not session_dirs:
        print(f"Warning: No session directories found in {processing_results_dir}")
        return ""
    
    print(f"\nRunning window_classification on {len(session_dirs)} session(s)")
    
    # Get path to action_extraction module
    action_extraction_path = Path(__file__).parent.parent.parent / "action_extraction"
    
    # Store all classification results
    all_results = []
    
    for session_dir in session_dirs:
        session_id = session_dir.name
        
        # Skip temp directories
        if session_id.startswith("temp_"):
            continue
            
        print(f"\nClassifying session: {session_id}")
        
        # Find parquet files in session directory
        parquet_files = list(session_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"  Warning: No parquet files found for session {session_id}")
            continue
        
        # Create temporary output directory for this session's classification
        temp_classification_output = session_dir / "classification_temp"
        temp_classification_output.mkdir(exist_ok=True)
        
        try:
            # Run window_classification using uv run to use action_extraction's environment
            # Use absolute paths since we're changing cwd to action_extraction
            cmd = [
                "uv", "run", "python", "-m", "action_extraction.window_classification",
                "--input", str(session_dir.absolute()),
                "--output", str(temp_classification_output.absolute())
            ]
            
            print(f"  Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(action_extraction_path),
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                print(f"  ✓ Successfully classified session {session_id}")
                
                # Read the classification CSV output
                csv_files = list(temp_classification_output.glob("*.csv"))
                
                if csv_files:
                    for csv_file in csv_files:
                        # Read the classification results
                        classification_df = pd.read_csv(csv_file)
                        
                        # Convert the entire classification result to a JSON string
                        results_json = classification_df.to_json(orient='records')
                        
                        # Add session_id and results to our aggregated list
                        all_results.append({
                            'session_id': session_id,
                            'results': results_json
                        })
                        
                        print(f"  Found {len(classification_df)} classified actions")
                else:
                    print(f"  Warning: No CSV output found for session {session_id}")
            else:
                print(f"  ✗ Failed to classify session {session_id}")
                print(f"  Error: {result.stderr}")
                
        except Exception as e:
            print(f"  ✗ Exception classifying session {session_id}: {e}")
    
    # Create aggregated DataFrame
    if not all_results:
        print("\n⚠ No classification results to aggregate")
        return ""
    
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_csv_path = output_path / "classification_results.csv"
    results_df.to_csv(output_csv_path, index=False)
    
    print(f"\n✓ Aggregated classification results for {len(results_df)} session(s)")
    print(f"✓ Results saved to: {output_csv_path}")
    
    return str(output_csv_path)


def main():
    settings = Settings()
    manifest_dir = "artifacts/manifest"
    sessions_dir = "artifacts/sessions"
    processing_results_dir = "artifacts/processing_results"
    classification_results_dir = "artifacts/classification_results"
    
    print(f"Fetching latest manifest from blob storage directory: {settings.manifests_directory}")
    manifest_path = download_manifest(settings, manifest_dir)
    print(f"Manifest downloaded to: {manifest_path}")
    
    print(f"\nDownloading sessions from manifest to: {sessions_dir}")
    session_paths = download_sessions_from_manifest(settings, manifest_path, sessions_dir)
    print(f"Downloaded {len(session_paths)} session(s)")
    
    print(f"\nRunning action_extraction pipeline on downloaded sessions")
    processing_results = run_action_extraction_pipeline(sessions_dir, processing_results_dir)
    print(f"\nPipeline results stored in: {processing_results_dir}")
    for session_id, output_path in processing_results.items():
        print(f"  {session_id}: {output_path}")
    
    print(f"\nRunning window_classification on processed sessions")
    classification_csv = run_window_classification_pipeline(processing_results_dir, classification_results_dir)
    if classification_csv:
        print(f"\n✓ Classification results saved to: {classification_csv}")


if __name__ == "__main__":
    main()
