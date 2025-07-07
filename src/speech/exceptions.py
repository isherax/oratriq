"""Custom exceptions for speech module."""


class SpeechError(Exception):
    """Base exception for speech-related errors."""
    pass


class STTError(SpeechError):
    """Exception raised for speech-to-text conversion errors."""
    pass


class STTProviderError(STTError):
    """Exception raised when STT provider is not available or fails."""
    pass


class STTAudioError(STTError):
    """Exception raised when audio input is invalid or corrupted."""
    pass


class STTTimeoutError(STTError):
    """Exception raised when STT conversion times out."""
    pass


class SpeechAnalysisError(SpeechError):
    """Exception raised for speech analysis errors."""
    pass


class SpeechAnalysisProviderError(SpeechAnalysisError):
    """Exception raised when speech analysis provider fails."""
    pass


class SpeechAnalysisDataError(SpeechAnalysisError):
    """Exception raised when speech data is invalid for analysis."""
    pass 
