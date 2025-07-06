"""Helper utility functions for Oratriq application."""

import os
from pathlib import Path
from typing import Union


def validate_audio_file(file_path: Union[str, Path]) -> bool:
    """Validate if the given file is a valid audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if valid audio file, False otherwise
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.exists():
        return False
    
    # Check if it's a file (not directory)
    if not file_path.is_file():
        return False
    
    # Check file size (must be > 0)
    if file_path.stat().st_size == 0:
        return False
    
    # Check file extension
    valid_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
    if file_path.suffix.lower() not in valid_extensions:
        return False
    
    return True


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    if not isinstance(directory, Path):
        directory = Path(directory)
    
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in megabytes
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    if not file_path.exists():
        return 0.0
    
    return file_path.stat().st_size / (1024 * 1024)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = "untitled"
    
    return filename 
