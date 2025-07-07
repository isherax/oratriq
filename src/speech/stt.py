"""Speech-to-text conversion module."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import speech_recognition as sr
from config.settings import settings
from src.speech.exceptions import (
    STTAudioError,
    STTError,
    STTProviderError,
    STTTimeoutError,
)
from src.utils.logger import get_logger


class SpeechToText:
    """Speech-to-text conversion with multiple provider support."""

    def __init__(self, provider: Optional[str] = None) -> None:
        """Initialize SpeechToText with specified provider.
        
        Args:
            provider: STT provider to use (google, whisper, etc.)
        """
        self.provider = provider or settings.stt_provider
        self.language = settings.stt_language
        self.logger = get_logger("speech.stt")
        self.recognizer = sr.Recognizer()
        self._setup_recognizer()
        
    def _setup_recognizer(self) -> None:
        """Configure the speech recognizer settings."""
        # Adjust for ambient noise
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
    def convert_audio_file(
        self, 
        audio_file_path: Union[str, Path],
        language: Optional[str] = None,
        cache_result: bool = True
    ) -> str:
        """Convert audio file to text.
        
        Args:
            audio_file_path: Path to the audio file
            language: Language code for recognition
            cache_result: Whether to cache the result
            
        Returns:
            Transcribed text
            
        Raises:
            STTAudioError: If audio file is invalid or corrupted
            STTProviderError: If STT provider fails
            STTTimeoutError: If conversion times out
        """
        audio_path = Path(audio_file_path)
        
        if not audio_path.exists():
            raise STTAudioError(f"Audio file not found: {audio_path}")
            
        if not audio_path.is_file():
            raise STTAudioError(f"Path is not a file: {audio_path}")
            
        # Check cache first
        if cache_result and settings.cache_enabled:
            cached_result = self._get_cached_result(audio_path)
            if cached_result:
                self.logger.info(f"Using cached transcription for {audio_path}")
                return cached_result
        
        self.logger.info(f"Converting audio file: {audio_path}")
        
        try:
            with sr.AudioFile(str(audio_path)) as source:
                audio = self.recognizer.record(source)
                
            text = self._recognize_speech(audio, language or self.language)
            
            # Cache the result
            if cache_result and settings.cache_enabled:
                self._cache_result(audio_path, text)
                
            self.logger.info(f"Successfully transcribed {audio_path}")
            return text
            
        except sr.UnknownValueError as e:
            error_msg = f"Could not understand audio in {audio_path}"
            self.logger.warning(error_msg)
            raise STTAudioError(error_msg) from e
            
        except sr.RequestError as e:
            error_msg = f"STT provider error for {audio_path}: {str(e)}"
            self.logger.error(error_msg)
            raise STTProviderError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error transcribing {audio_path}: {str(e)}"
            self.logger.error(error_msg)
            raise STTError(error_msg) from e
            
    def convert_audio_data(
        self, 
        audio_data: bytes,
        language: Optional[str] = None
    ) -> str:
        """Convert audio data to text.
        
        Args:
            audio_data: Raw audio data
            language: Language code for recognition
            
        Returns:
            Transcribed text
            
        Raises:
            STTAudioError: If audio data is invalid
            STTProviderError: If STT provider fails
        """
        self.logger.info("Converting audio data to text")
        
        try:
            audio = sr.AudioData(audio_data, settings.audio_sample_rate, 2)
            text = self._recognize_speech(audio, language or self.language)
            
            self.logger.info("Successfully transcribed audio data")
            return text
            
        except sr.UnknownValueError as e:
            error_msg = "Could not understand audio data"
            self.logger.warning(error_msg)
            raise STTAudioError(error_msg) from e
            
        except sr.RequestError as e:
            error_msg = f"STT provider error: {str(e)}"
            self.logger.error(error_msg)
            raise STTProviderError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error transcribing audio data: {str(e)}"
            self.logger.error(error_msg)
            raise STTError(error_msg) from e
            
    def _recognize_speech(self, audio: sr.AudioData, language: str) -> str:
        """Recognize speech using the configured provider.
        
        Args:
            audio: Audio data to recognize
            language: Language code for recognition
            
        Returns:
            Transcribed text
            
        Raises:
            STTProviderError: If provider is not supported
            STTTimeoutError: If recognition times out
        """
        start_time = time.time()
        
        try:
            if self.provider.lower() == "google":
                text = self.recognizer.recognize_google(
                    audio, 
                    language=language
                )
            elif self.provider.lower() == "whisper":
                # Note: Whisper requires additional setup
                raise STTProviderError("Whisper provider not yet implemented")
            else:
                raise STTProviderError(f"Unsupported STT provider: {self.provider}")
                
            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > settings.request_timeout:
                raise STTTimeoutError(f"STT conversion timed out after {elapsed_time}s")
                
            return text
            
        except sr.UnknownValueError:
            raise STTAudioError("Could not understand audio")
        except sr.RequestError as e:
            raise STTProviderError(f"STT provider error: {str(e)}")
            
    def _get_cached_result(self, audio_path: Path) -> Optional[str]:
        """Get cached transcription result.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Cached transcription text or None if not found
        """
        cache_file = self._get_cache_file_path(audio_path)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is still valid (based on file modification time)
                if cache_data.get('mtime') == audio_path.stat().st_mtime:
                    return cache_data.get('text')
                    
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Failed to read cache file {cache_file}: {e}")
                
        return None
        
    def _cache_result(self, audio_path: Path, text: str) -> None:
        """Cache transcription result.
        
        Args:
            audio_path: Path to the audio file
            text: Transcribed text
        """
        cache_file = self._get_cache_file_path(audio_path)
        
        try:
            cache_data = {
                'text': text,
                'provider': self.provider,
                'language': self.language,
                'mtime': audio_path.stat().st_mtime,
                'timestamp': time.time()
            }
            
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except OSError as e:
            self.logger.warning(f"Failed to cache result to {cache_file}: {e}")
            
    def _get_cache_file_path(self, audio_path: Path) -> Path:
        """Get cache file path for audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Path to the cache file
        """
        cache_name = f"{audio_path.stem}_{self.provider}_{self.language}.json"
        return settings.cache_dir / "stt" / cache_name
        
    def get_supported_providers(self) -> List[str]:
        """Get list of supported STT providers.
        
        Returns:
            List of supported provider names
        """
        return ["google", "whisper"]
        
    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported languages for each provider.
        
        Returns:
            Dictionary mapping provider names to supported language codes
        """
        return {
            "google": ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT"],
            "whisper": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
        } 
