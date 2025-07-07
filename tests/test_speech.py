"""Unit tests for speech module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest
import speech_recognition as sr

from src.speech.analyzer import SpeechAnalyzer, SpeechMetrics, PresentationFeedback
from src.speech.exceptions import (
    STTAudioError,
    STTError,
    STTProviderError,
    STTTimeoutError,
    SpeechAnalysisDataError,
    SpeechAnalysisError,
    SpeechAnalysisProviderError,
    SpeechError,
)
from src.speech.stt import SpeechToText


class TestSpeechToText:
    """Test cases for SpeechToText class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.stt = SpeechToText(provider="google")
        self.test_audio_path = Path("test_audio.wav")

    def test_init_with_default_provider(self):
        """Test initialization with default provider."""
        stt = SpeechToText()
        assert stt.provider == "google"
        assert stt.language == "en-US"

    def test_init_with_custom_provider(self):
        """Test initialization with custom provider."""
        stt = SpeechToText(provider="whisper")
        assert stt.provider == "whisper"

    def test_convert_audio_file_file_not_found(self):
        """Test convert_audio_file with non-existent file."""
        with pytest.raises(STTAudioError, match="Audio file not found"):
            self.stt.convert_audio_file("nonexistent.wav")

    def test_convert_audio_file_not_a_file(self, tmp_path):
        """Test convert_audio_file with directory path."""
        directory = tmp_path / "test_dir"
        directory.mkdir()
        
        with pytest.raises(STTAudioError, match="Path is not a file"):
            self.stt.convert_audio_file(directory)

    @patch('speech_recognition.AudioFile')
    @patch('speech_recognition.Recognizer')
    def test_convert_audio_file_success(self, mock_recognizer, mock_audio_file, tmp_path):
        """Test successful audio file conversion."""
        # Create a temporary audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        
        # Mock the recognizer
        mock_recognizer_instance = Mock()
        mock_recognizer_instance.record.return_value = Mock()
        mock_recognizer_instance.recognize_google.return_value = "Hello world"
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Mock AudioFile context manager
        mock_source = Mock()
        mock_audio_file.return_value.__enter__.return_value = mock_source
        
        # Mock the entire convert_audio_file method to avoid actual file processing
        with patch.object(self.stt, '_recognize_speech') as mock_recognize:
            mock_recognize.return_value = "Hello world"
            
            # Mock the cache check to return None
            with patch.object(self.stt, '_get_cached_result') as mock_cache:
                mock_cache.return_value = None
                
                # Mock the AudioFile context manager to avoid FLAC issues
                with patch('speech_recognition.AudioFile') as mock_audio_file_inner:
                    mock_audio_file_inner.return_value.__enter__.return_value = mock_source
                    mock_audio_file_inner.return_value.__exit__.return_value = None
                    
                    result = self.stt.convert_audio_file(audio_file)
                    
                    assert result == "Hello world"
                    mock_recognize.assert_called_once()

    @patch('speech_recognition.Recognizer')
    def test_convert_audio_file_unknown_value_error(self, mock_recognizer, tmp_path):
        """Test audio file conversion with unknown value error."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        
        mock_recognizer_instance = Mock()
        mock_recognizer_instance.record.side_effect = sr.UnknownValueError()
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Mock the cache check to return None
        with patch.object(self.stt, '_get_cached_result') as mock_cache:
            mock_cache.return_value = None
            
            # Mock the _recognize_speech method to raise the expected error
            with patch.object(self.stt, '_recognize_speech') as mock_recognize:
                mock_recognize.side_effect = sr.UnknownValueError()
                
                # Mock the AudioFile context manager to avoid FLAC issues
                with patch('speech_recognition.AudioFile') as mock_audio_file:
                    mock_source = Mock()
                    mock_audio_file.return_value.__enter__.return_value = mock_source
                    mock_audio_file.return_value.__exit__.return_value = None
                    
                    with pytest.raises(STTAudioError, match="Could not understand audio"):
                        self.stt.convert_audio_file(audio_file)

    @patch('speech_recognition.Recognizer')
    def test_convert_audio_file_request_error(self, mock_recognizer, tmp_path):
        """Test audio file conversion with request error."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        
        mock_recognizer_instance = Mock()
        mock_recognizer_instance.record.side_effect = sr.RequestError("API Error")
        mock_recognizer.return_value = mock_recognizer_instance
        
        # Mock the cache check to return None
        with patch.object(self.stt, '_get_cached_result') as mock_cache:
            mock_cache.return_value = None
            
            # Mock the _recognize_speech method to raise the expected error
            with patch.object(self.stt, '_recognize_speech') as mock_recognize:
                mock_recognize.side_effect = sr.RequestError("API Error")
                
                # Mock the AudioFile context manager to avoid FLAC issues
                with patch('speech_recognition.AudioFile') as mock_audio_file:
                    mock_source = Mock()
                    mock_audio_file.return_value.__enter__.return_value = mock_source
                    mock_audio_file.return_value.__exit__.return_value = None
                    
                    with pytest.raises(STTProviderError, match="STT provider error"):
                        self.stt.convert_audio_file(audio_file)

    def test_convert_audio_data_success(self):
        """Test successful audio data conversion."""
        with patch.object(self.stt, '_recognize_speech') as mock_recognize:
            mock_recognize.return_value = "Test transcript"
            
            result = self.stt.convert_audio_data(b"fake audio data")
            
            assert result == "Test transcript"
            mock_recognize.assert_called_once()

    def test_recognize_speech_unsupported_provider(self):
        """Test speech recognition with unsupported provider."""
        self.stt.provider = "unsupported"
        
        with pytest.raises(STTProviderError, match="Unsupported STT provider"):
            self.stt._recognize_speech(Mock(), "en-US")

    def test_recognize_speech_whisper_not_implemented(self):
        """Test speech recognition with whisper provider (not implemented)."""
        self.stt.provider = "whisper"
        
        with pytest.raises(STTProviderError, match="Whisper provider not yet implemented"):
            self.stt._recognize_speech(Mock(), "en")

    @patch('time.time')
    def test_recognize_speech_timeout(self, mock_time):
        """Test speech recognition timeout."""
        mock_time.side_effect = [0, 35]  # 35 seconds elapsed
        
        # Mock the recognizer to avoid actual API calls
        with patch.object(self.stt.recognizer, 'recognize_google') as mock_recognize:
            mock_recognize.return_value = "test"
            
            with pytest.raises(STTTimeoutError, match="STT conversion timed out"):
                self.stt._recognize_speech(Mock(), "en-US")

    def test_get_supported_providers(self):
        """Test getting supported providers."""
        providers = self.stt.get_supported_providers()
        assert "google" in providers
        assert "whisper" in providers

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        languages = self.stt.get_supported_languages()
        assert "google" in languages
        assert "whisper" in languages
        assert "en-US" in languages["google"]

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_get_cached_result_success(self, mock_exists, mock_file):
        """Test successful cache retrieval."""
        mock_exists.return_value = True
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps({
            'text': 'Cached transcript',
            'mtime': 123456789
        })
        
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_mtime = 123456789
            
            result = self.stt._get_cached_result(Path("test.wav"))
            assert result == "Cached transcript"

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_cache_result(self, mock_exists, mock_file):
        """Test caching result."""
        mock_exists.return_value = False
        
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_mtime = 123456789
            
            # Mock mkdir to avoid file system issues
            with patch.object(Path, 'mkdir') as mock_mkdir:
                self.stt._cache_result(Path("test.wav"), "Test transcript")
                mock_file.assert_called_once()


class TestSpeechAnalyzer:
    """Test cases for SpeechAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SpeechAnalyzer()
        self.test_transcript = "Hello world. This is a test presentation. It has multiple sentences."

    def test_init_with_default_ai_provider(self):
        """Test initialization with default AI provider."""
        analyzer = SpeechAnalyzer()
        assert analyzer.ai_provider == "openai"

    def test_init_with_custom_ai_provider(self):
        """Test initialization with custom AI provider."""
        analyzer = SpeechAnalyzer(ai_provider="custom")
        assert analyzer.ai_provider == "custom"

    def test_analyze_speech_empty_transcript(self):
        """Test speech analysis with empty transcript."""
        with pytest.raises(SpeechAnalysisDataError, match="Empty or invalid transcript"):
            self.analyzer.analyze_speech("")

    def test_analyze_speech_whitespace_transcript(self):
        """Test speech analysis with whitespace-only transcript."""
        with pytest.raises(SpeechAnalysisDataError, match="Empty or invalid transcript"):
            self.analyzer.analyze_speech("   \n\t   ")

    def test_analyze_speech_basic_metrics(self):
        """Test basic speech analysis metrics."""
        metrics = self.analyzer.analyze_speech(self.test_transcript)
        
        assert metrics.word_count == 11  # "Hello world. This is a test presentation. It has multiple sentences."
        assert metrics.sentence_count == 3
        assert metrics.unique_words == 11
        assert metrics.vocabulary_diversity == 11 / 11

    def test_analyze_speech_with_audio_duration(self):
        """Test speech analysis with audio duration."""
        metrics = self.analyzer.analyze_speech(self.test_transcript, audio_duration=60.0)
        
        # 11 words in 60 seconds = 11 words per minute
        assert metrics.speaking_rate == 11.0

    def test_analyze_speech_filler_words(self):
        """Test speech analysis with filler words."""
        transcript_with_fillers = "Um, hello world. You know, this is a test. Like, it has um filler words."
        
        metrics = self.analyzer.analyze_speech(transcript_with_fillers)
        
        assert metrics.filler_word_count > 0
        assert metrics.filler_word_ratio > 0

    def test_extract_words(self):
        """Test word extraction."""
        text = "Hello, world! This is a test."
        words = self.analyzer._extract_words(text)
        
        expected = ["hello", "world", "this", "is", "a", "test"]
        assert words == expected

    def test_extract_sentences(self):
        """Test sentence extraction."""
        text = "Hello world. This is a test! How are you?"
        sentences = self.analyzer._extract_sentences(text)
        
        expected = ["Hello world.", "This is a test!", "How are you?"]
        assert sentences == expected

    def test_count_filler_words(self):
        """Test filler word counting."""
        text = "um hello world you know this is basically a test"
        count = self.analyzer._count_filler_words(text)
        
        assert count >= 3  # um, you know, basically

    def test_generate_basic_feedback_good_metrics(self):
        """Test basic feedback generation with good metrics."""
        metrics = SpeechMetrics(
            word_count=100,
            speaking_rate=140.0,
            pause_count=5,
            average_pause_duration=1.0,
            filler_word_count=2,
            filler_word_ratio=0.02,
            sentence_count=10,
            average_sentence_length=10.0,
            unique_words=80,
            vocabulary_diversity=0.8
        )
        
        feedback = self.analyzer._generate_basic_feedback(metrics)
        
        assert feedback.overall_score > 80
        assert "Appropriate speaking pace" in feedback.strengths
        assert "Clear speech with few filler words" in feedback.strengths

    def test_generate_basic_feedback_poor_metrics(self):
        """Test basic feedback generation with poor metrics."""
        metrics = SpeechMetrics(
            word_count=100,
            speaking_rate=200.0,
            pause_count=2,
            average_pause_duration=0.5,
            filler_word_count=15,
            filler_word_ratio=0.15,
            sentence_count=5,
            average_sentence_length=20.0,
            unique_words=30,
            vocabulary_diversity=0.3
        )
        
        feedback = self.analyzer._generate_basic_feedback(metrics)
        
        assert feedback.overall_score < 70
        assert "Speaking rate" in feedback.areas_for_improvement
        assert "Filler word usage" in feedback.areas_for_improvement

    def test_generate_feedback_basic_only(self):
        """Test feedback generation without AI."""
        metrics = self.analyzer.analyze_speech(self.test_transcript)
        feedback = self.analyzer.generate_feedback(self.test_transcript, metrics)
        
        assert isinstance(feedback, PresentationFeedback)
        assert feedback.overall_score >= 0
        assert feedback.overall_score <= 100

    @patch('openai.ChatCompletion.create')
    def test_generate_ai_feedback_success(self, mock_openai):
        """Test AI feedback generation success."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Strengths: Good pace\nImprovements: Reduce filler words"
        mock_openai.return_value = mock_response
        
        metrics = SpeechMetrics(
            word_count=100,
            speaking_rate=140.0,
            pause_count=5,
            average_pause_duration=1.0,
            filler_word_count=5,
            filler_word_ratio=0.05,
            sentence_count=10,
            average_sentence_length=10.0,
            unique_words=80,
            vocabulary_diversity=0.8
        )
        
        feedback = self.analyzer._generate_ai_feedback(self.test_transcript, metrics)
        
        assert isinstance(feedback, PresentationFeedback)
        mock_openai.assert_called_once()

    @patch('openai.ChatCompletion.create')
    def test_generate_ai_feedback_failure(self, mock_openai):
        """Test AI feedback generation failure."""
        mock_openai.side_effect = Exception("API Error")
        
        metrics = SpeechMetrics(
            word_count=100,
            speaking_rate=140.0,
            pause_count=5,
            average_pause_duration=1.0,
            filler_word_count=5,
            filler_word_ratio=0.05,
            sentence_count=10,
            average_sentence_length=10.0,
            unique_words=80,
            vocabulary_diversity=0.8
        )
        
        with pytest.raises(SpeechAnalysisProviderError, match="AI feedback generation failed"):
            self.analyzer._generate_ai_feedback(self.test_transcript, metrics)

    def test_build_ai_prompt(self):
        """Test AI prompt building."""
        metrics = SpeechMetrics(
            word_count=100,
            speaking_rate=140.0,
            pause_count=5,
            average_pause_duration=1.0,
            filler_word_count=5,
            filler_word_ratio=0.05,
            sentence_count=10,
            average_sentence_length=10.0,
            unique_words=80,
            vocabulary_diversity=0.8
        )
        
        prompt = self.analyzer._build_ai_prompt(self.test_transcript, metrics, "Test context")
        
        assert "TRANSCRIPT:" in prompt
        assert "SPEECH METRICS:" in prompt
        assert "CONTEXT: Test context" in prompt
        assert "100" in prompt  # word count
        assert "140.0" in prompt  # speaking rate

    def test_extract_ai_section(self):
        """Test AI response section extraction."""
        response = """
        Strengths:
        - Good pace
        - Clear articulation
        
        Improvements:
        - Reduce filler words
        - More pauses
        """
        
        strengths = self.analyzer._extract_ai_section(response, "strengths")
        improvements = self.analyzer._extract_ai_section(response, "improvements")
        
        assert "Good pace" in strengths
        assert "Clear articulation" in strengths
        assert "Reduce filler words" in improvements
        assert "More pauses" in improvements

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_get_cached_metrics_success(self, mock_exists, mock_file):
        """Test successful metrics cache retrieval."""
        mock_exists.return_value = True
        cache_data = {
            'word_count': 100,
            'speaking_rate': 140.0,
            'pause_count': 5,
            'average_pause_duration': 1.0,
            'filler_word_count': 5,
            'filler_word_ratio': 0.05,
            'sentence_count': 10,
            'average_sentence_length': 10.0,
            'unique_words': 80,
            'vocabulary_diversity': 0.8
        }
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(cache_data)
        
        result = self.analyzer._get_cached_metrics(self.test_transcript)
        
        assert isinstance(result, SpeechMetrics)
        assert result.word_count == 100
        assert result.speaking_rate == 140.0

    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists')
    def test_cache_metrics(self, mock_exists, mock_file):
        """Test metrics caching."""
        mock_exists.return_value = False
        
        metrics = SpeechMetrics(
            word_count=100,
            speaking_rate=140.0,
            pause_count=5,
            average_pause_duration=1.0,
            filler_word_count=5,
            filler_word_ratio=0.05,
            sentence_count=10,
            average_sentence_length=10.0,
            unique_words=80,
            vocabulary_diversity=0.8
        )
        
        self.analyzer._cache_metrics(self.test_transcript, metrics)
        mock_file.assert_called_once()


class TestSpeechExceptions:
    """Test cases for speech exceptions."""

    def test_speech_error_inheritance(self):
        """Test speech error inheritance."""
        assert issubclass(STTError, SpeechError)
        assert issubclass(STTProviderError, STTError)
        assert issubclass(STTAudioError, STTError)
        assert issubclass(SpeechAnalysisError, SpeechError)

    def test_exception_messages(self):
        """Test exception message handling."""
        stt_error = STTError("Test STT error")
        assert str(stt_error) == "Test STT error"
        
        provider_error = STTProviderError("Provider failed")
        assert str(provider_error) == "Provider failed"
        
        analysis_error = SpeechAnalysisError("Analysis failed")
        assert str(analysis_error) == "Analysis failed" 
