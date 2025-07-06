"""Tests for helper utility functions."""

import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.utils.helpers import (
    validate_audio_file,
    format_duration,
    ensure_directory_exists,
    get_file_size_mb,
    sanitize_filename,
)


class TestValidateAudioFile:
    """Test audio file validation functionality."""

    def test_valid_audio_file(self):
        """Test validation of a valid audio file."""
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(b"fake audio data")
            temp_file_path = temp_file.name
        
        try:
            assert validate_audio_file(temp_file_path) is True
        finally:
            Path(temp_file_path).unlink()

    def test_nonexistent_file(self):
        """Test validation of a non-existent file."""
        assert validate_audio_file("nonexistent.wav") is False

    def test_invalid_extension(self):
        """Test validation of file with invalid extension."""
        with NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"text data")
            temp_file_path = temp_file.name
        
        try:
            assert validate_audio_file(temp_file_path) is False
        finally:
            Path(temp_file_path).unlink()

    def test_empty_file(self):
        """Test validation of an empty file."""
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            assert validate_audio_file(temp_file_path) is False
        finally:
            Path(temp_file_path).unlink()


class TestFormatDuration:
    """Test duration formatting functionality."""

    def test_seconds_only(self):
        """Test formatting of duration less than 60 seconds."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(0.1) == "0.1s"

    def test_minutes_and_seconds(self):
        """Test formatting of duration between 1 minute and 1 hour."""
        assert format_duration(90) == "1m 30.0s"
        assert format_duration(125.7) == "2m 5.7s"

    def test_hours_minutes_seconds(self):
        """Test formatting of duration over 1 hour."""
        assert format_duration(3661) == "1h 1m 1.0s"
        assert format_duration(7325.5) == "2h 2m 5.5s"


class TestEnsureDirectoryExists:
    """Test directory creation functionality."""

    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "test_dir"
        result = ensure_directory_exists(new_dir)
        
        assert result.exists()
        assert result.is_dir()
        assert result == new_dir

    def test_existing_directory(self, tmp_path):
        """Test with existing directory."""
        existing_dir = tmp_path / "existing_dir"
        existing_dir.mkdir()
        
        result = ensure_directory_exists(existing_dir)
        assert result == existing_dir


class TestGetFileSizeMB:
    """Test file size calculation functionality."""

    def test_file_size_calculation(self):
        """Test calculating file size in megabytes."""
        with NamedTemporaryFile(delete=False) as temp_file:
            # Write 1MB of data
            temp_file.write(b"0" * (1024 * 1024))
            temp_file_path = temp_file.name
        
        try:
            size_mb = get_file_size_mb(temp_file_path)
            assert size_mb == 1.0
        finally:
            Path(temp_file_path).unlink()

    def test_nonexistent_file(self):
        """Test size calculation for non-existent file."""
        assert get_file_size_mb("nonexistent.txt") == 0.0


class TestSanitizeFilename:
    """Test filename sanitization functionality."""

    def test_valid_filename(self):
        """Test sanitization of already valid filename."""
        filename = "valid_filename.txt"
        assert sanitize_filename(filename) == filename

    def test_invalid_characters(self):
        """Test removal of invalid characters."""
        filename = "file<name>with:invalid/chars"
        expected = "file_name_with_invalid_chars"
        assert sanitize_filename(filename) == expected

    def test_leading_trailing_spaces(self):
        """Test removal of leading and trailing spaces."""
        filename = "  filename.txt  "
        assert sanitize_filename(filename) == "filename.txt"

    def test_empty_filename(self):
        """Test handling of empty filename."""
        assert sanitize_filename("") == "untitled"
        assert sanitize_filename("   ") == "untitled" 
