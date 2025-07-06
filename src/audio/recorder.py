"""Audio recording functionality for microphone input and file operations."""

import os
import wave
from typing import Optional, List, Tuple
import pyaudio
import numpy as np

from src.audio.exceptions import (
    AudioDeviceError,
    AudioRecordingError,
    AudioFormatError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioRecorder:
    """Handles audio recording from microphone and file operations."""

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_size: int = 1024,
        format_type: int = pyaudio.paInt16,
    ) -> None:
        """Initialize the audio recorder.

        Args:
            sample_rate: Sample rate in Hz (default: 44100)
            channels: Number of audio channels (default: 1 for mono)
            chunk_size: Size of audio chunks to process (default: 1024)
            format_type: Audio format type (default: 16-bit integer)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format_type = format_type
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self._validate_audio_settings()

    def _validate_audio_settings(self) -> None:
        """Validate audio settings and check device availability."""
        try:
            # Check if default input device is available
            device_info = self.audio.get_default_input_device_info()
            logger.info(f"Using audio device: {device_info['name']}")
        except OSError as e:
            raise AudioDeviceError(f"No default audio input device found: {e}")

        # Validate sample rate
        if self.sample_rate <= 0:
            raise AudioFormatError("Sample rate must be positive")

        # Validate channels
        if self.channels not in [1, 2]:
            raise AudioFormatError("Channels must be 1 (mono) or 2 (stereo)")

        # Validate chunk size
        if self.chunk_size <= 0:
            raise AudioFormatError("Chunk size must be positive")

    def list_audio_devices(self) -> List[Tuple[int, str]]:
        """List available audio input devices.

        Returns:
            List of tuples containing (device_index, device_name)
        """
        devices = []
        for i in range(self.audio.get_device_count()):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info["maxInputChannels"] > 0:
                    devices.append((i, device_info["name"]))
            except OSError:
                continue
        return devices

    def start_recording(self) -> None:
        """Start recording audio from the microphone."""
        try:
            self.stream = self.audio.open(
                format=self.format_type,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            logger.info("Started audio recording")
        except OSError as e:
            raise AudioRecordingError(f"Failed to start recording: {e}")

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            logger.info("Stopped audio recording")

    def record_for_duration(self, duration_seconds: float) -> bytes:
        """Record audio for a specified duration.

        Args:
            duration_seconds: Duration to record in seconds

        Returns:
            Raw audio data as bytes

        Raises:
            AudioRecordingError: If recording fails
        """
        if duration_seconds <= 0:
            raise AudioRecordingError("Duration must be positive")

        try:
            self.start_recording()
            frames = []
            num_chunks = int(self.sample_rate / self.chunk_size * duration_seconds)

            logger.info(f"Recording for {duration_seconds} seconds...")
            for _ in range(num_chunks):
                data = self.stream.read(self.chunk_size)
                frames.append(data)

            self.stop_recording()
            logger.info("Recording completed successfully")
            return b"".join(frames)

        except Exception as e:
            self.stop_recording()
            raise AudioRecordingError(f"Recording failed: {e}")

    def record_until_stopped(self) -> bytes:
        """Record audio until manually stopped.

        Returns:
            Raw audio data as bytes

        Raises:
            AudioRecordingError: If recording fails
        """
        try:
            self.start_recording()
            frames = []
            logger.info("Recording started. Press Ctrl+C to stop...")

            while True:
                data = self.stream.read(self.chunk_size)
                frames.append(data)

        except KeyboardInterrupt:
            self.stop_recording()
            logger.info("Recording stopped by user")
            return b"".join(frames)
        except Exception as e:
            self.stop_recording()
            raise AudioRecordingError(f"Recording failed: {e}")

    def save_audio(self, audio_data: bytes, filepath: str) -> None:
        """Save audio data to a WAV file.

        Args:
            audio_data: Raw audio data as bytes
            filepath: Path where to save the audio file

        Raises:
            AudioFormatError: If file format is not supported
        """
        if not filepath.lower().endswith(".wav"):
            raise AudioFormatError("Only WAV format is currently supported")

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with wave.open(filepath, "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.format_type))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data)

            logger.info(f"Audio saved to: {filepath}")

        except Exception as e:
            raise AudioFormatError(f"Failed to save audio file: {e}")

    def load_audio(self, filepath: str) -> bytes:
        """Load audio data from a WAV file.

        Args:
            filepath: Path to the audio file

        Returns:
            Raw audio data as bytes

        Raises:
            AudioFormatError: If file cannot be loaded or format is not supported
        """
        if not filepath.lower().endswith(".wav"):
            raise AudioFormatError("Only WAV format is currently supported")

        try:
            with wave.open(filepath, "rb") as wav_file:
                audio_data = wav_file.readframes(wav_file.getnframes())
            logger.info(f"Audio loaded from: {filepath}")
            return audio_data

        except Exception as e:
            raise AudioFormatError(f"Failed to load audio file: {e}")

    def get_audio_info(self, filepath: str) -> dict:
        """Get information about an audio file.

        Args:
            filepath: Path to the audio file

        Returns:
            Dictionary containing audio file information

        Raises:
            AudioFormatError: If file cannot be read
        """
        try:
            with wave.open(filepath, "rb") as wav_file:
                info = {
                    "channels": wav_file.getnchannels(),
                    "sample_width": wav_file.getsampwidth(),
                    "sample_rate": wav_file.getframerate(),
                    "frames": wav_file.getnframes(),
                    "duration": wav_file.getnframes() / wav_file.getframerate(),
                }
            return info

        except Exception as e:
            raise AudioFormatError(f"Failed to get audio info: {e}")

    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self.stream:
            self.stop_recording()
        self.audio.terminate()
        logger.info("Audio recorder cleaned up")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup() 
