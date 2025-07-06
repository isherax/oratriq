"""Configuration settings for Oratriq application."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-3.5-turbo", env="OPENAI_MODEL")

    # Audio Configuration
    audio_sample_rate: int = Field(16000, env="AUDIO_SAMPLE_RATE")
    audio_channels: int = Field(1, env="AUDIO_CHANNELS")
    audio_chunk_size: int = Field(1024, env="AUDIO_CHUNK_SIZE")
    audio_format: str = Field("wav", env="AUDIO_FORMAT")

    # Speech Recognition Configuration
    stt_provider: str = Field("google", env="STT_PROVIDER")
    stt_language: str = Field("en-US", env="STT_LANGUAGE")

    # Application Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    max_recording_duration: int = Field(300, env="MAX_RECORDING_DURATION")
    cache_enabled: bool = Field(True, env="CACHE_ENABLED")

    # File Paths
    recordings_dir: Path = Field(Path("data/recordings"), env="RECORDINGS_DIR")
    transcripts_dir: Path = Field(Path("data/transcripts"), env="TRANSCRIPTS_DIR")
    cache_dir: Path = Field(Path("data/cache"), env="CACHE_DIR")

    # Performance Configuration
    max_concurrent_requests: int = Field(5, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    retry_attempts: int = Field(3, env="RETRY_ATTEMPTS")

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.recordings_dir,
            self.transcripts_dir,
            self.cache_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug or os.getenv("ENVIRONMENT") == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.is_development


# Global settings instance
settings = Settings() 
