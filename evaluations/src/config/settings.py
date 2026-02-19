from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field


class Settings(BaseSettings):
    """Settings for the evaluations pipeline."""

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra='ignore',
    ) # type: ignore

    # Azure Blob Storage settings
    azure_storage_account_name: str = Field(..., description="Azure Storage Account Name")
    azure_storage_account_key: str = Field(..., description="Azure Storage Account Key")
    azure_blob_container: str = Field(default="evaluations", description="Azure Blob Container Name")
    manifests_directory: str = Field(default="dataset-manifests", description="Path to manifest blob in container")
    sessions_container: str = Field(default="traces-test", description="Container name for session blob directories")


    
