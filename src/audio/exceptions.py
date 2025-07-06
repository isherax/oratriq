"""Audio module exceptions."""


class AudioError(Exception):
    """Base exception for audio-related errors."""

    pass


class AudioDeviceError(AudioError):
    """Raised when there are issues with audio devices."""

    pass


class AudioFormatError(AudioError):
    """Raised when there are issues with audio formats."""

    pass


class AudioRecordingError(AudioError):
    """Raised when there are issues during audio recording."""

    pass


class AudioProcessingError(AudioError):
    """Raised when there are issues during audio processing."""

    pass 
