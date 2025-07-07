"""Speech-to-text and analysis module."""

from src.speech.analyzer import PresentationFeedback, SpeechAnalyzer, SpeechMetrics
from src.speech.exceptions import (
    STTAudioError,
    STTError,
    STTProviderError,
    STTTimeoutError,
    SpeechAnalysisDataError,
    SpeechAnalysisError,
    SpeechAnalysisProviderError,
)
from src.speech.stt import SpeechToText

__all__ = [
    "SpeechToText",
    "SpeechAnalyzer",
    "SpeechMetrics",
    "PresentationFeedback",
    "STTError",
    "STTProviderError",
    "STTAudioError",
    "STTTimeoutError",
    "SpeechAnalysisError",
    "SpeechAnalysisDataError",
    "SpeechAnalysisProviderError",
] 
