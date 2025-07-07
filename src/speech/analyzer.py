"""Speech analysis and presentation feedback module."""

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import openai
from config.settings import settings
from src.speech.exceptions import (
    SpeechAnalysisDataError,
    SpeechAnalysisError,
    SpeechAnalysisProviderError,
)
from src.utils.logger import get_logger


@dataclass
class SpeechMetrics:
    """Speech analysis metrics."""
    
    word_count: int
    speaking_rate: float  # words per minute
    pause_count: int
    average_pause_duration: float
    filler_word_count: int
    filler_word_ratio: float
    sentence_count: int
    average_sentence_length: float
    unique_words: int
    vocabulary_diversity: float  # unique words / total words


@dataclass
class PresentationFeedback:
    """Presentation feedback and recommendations."""
    
    overall_score: float  # 0-100
    strengths: List[str]
    areas_for_improvement: List[str]
    specific_recommendations: List[str]
    speaking_rate_feedback: str
    clarity_feedback: str
    engagement_feedback: str
    confidence_indicators: List[str]


class SpeechAnalyzer:
    """Analyze speech patterns and provide presentation feedback."""

    def __init__(self, ai_provider: Optional[str] = None) -> None:
        """Initialize SpeechAnalyzer.
        
        Args:
            ai_provider: AI provider for advanced analysis (openai, etc.)
        """
        self.ai_provider = ai_provider or "openai"
        self.logger = get_logger("speech.analyzer")
        self._setup_ai_client()
        
        # Common filler words to detect
        self.filler_words = {
            "um", "uh", "ah", "er", "like", "you know", "basically", 
            "actually", "literally", "sort of", "kind of", "right",
            "so", "well", "i mean", "you see", "okay", "alright"
        }
        
    def _setup_ai_client(self) -> None:
        """Setup AI client for advanced analysis."""
        if self.ai_provider == "openai" and settings.openai_api_key:
            openai.api_key = settings.openai_api_key
        else:
            self.logger.warning("OpenAI API key not configured, AI analysis disabled")
            
    def analyze_speech(
        self, 
        transcript: str,
        audio_duration: Optional[float] = None,
        cache_result: bool = True
    ) -> SpeechMetrics:
        """Analyze speech patterns from transcript.
        
        Args:
            transcript: Speech transcript text
            audio_duration: Duration of audio in seconds (optional)
            cache_result: Whether to cache the analysis result
            
        Returns:
            Speech analysis metrics
            
        Raises:
            SpeechAnalysisDataError: If transcript is invalid
        """
        if not transcript or not transcript.strip():
            raise SpeechAnalysisDataError("Empty or invalid transcript provided")
            
        # Check cache first
        if cache_result and settings.cache_enabled:
            cached_metrics = self._get_cached_metrics(transcript)
            if cached_metrics:
                self.logger.info("Using cached speech metrics")
                return cached_metrics
                
        self.logger.info("Analyzing speech patterns")
        
        # Basic text analysis
        words = self._extract_words(transcript)
        sentences = self._extract_sentences(transcript)
        
        # Calculate metrics
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Speaking rate (words per minute)
        speaking_rate = 0.0
        if audio_duration and audio_duration > 0:
            speaking_rate = (word_count / audio_duration) * 60
            
        # Pause analysis (simplified - based on punctuation)
        pause_count = len(re.findall(r'[.!?]+', transcript))
        average_pause_duration = 0.0  # Would need audio analysis for accurate timing
        
        # Filler word analysis
        filler_word_count = self._count_filler_words(transcript.lower())
        filler_word_ratio = filler_word_count / word_count if word_count > 0 else 0.0
        
        # Sentence length analysis
        sentence_lengths = [len(self._extract_words(sentence)) for sentence in sentences]
        average_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths) 
            if sentence_lengths else 0.0
        )
        
        # Vocabulary diversity
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / word_count if word_count > 0 else 0.0
        
        metrics = SpeechMetrics(
            word_count=word_count,
            speaking_rate=speaking_rate,
            pause_count=pause_count,
            average_pause_duration=average_pause_duration,
            filler_word_count=filler_word_count,
            filler_word_ratio=filler_word_ratio,
            sentence_count=sentence_count,
            average_sentence_length=average_sentence_length,
            unique_words=unique_words,
            vocabulary_diversity=vocabulary_diversity
        )
        
        # Cache the result
        if cache_result and settings.cache_enabled:
            self._cache_metrics(transcript, metrics)
            
        self.logger.info("Speech analysis completed")
        return metrics
        
    def generate_feedback(
        self, 
        transcript: str,
        metrics: SpeechMetrics,
        context: Optional[str] = None
    ) -> PresentationFeedback:
        """Generate presentation feedback and recommendations.
        
        Args:
            transcript: Speech transcript
            metrics: Speech analysis metrics
            context: Additional context (e.g., presentation topic)
            
        Returns:
            Presentation feedback and recommendations
            
        Raises:
            SpeechAnalysisProviderError: If AI provider fails
        """
        self.logger.info("Generating presentation feedback")
        
        # Basic feedback based on metrics
        basic_feedback = self._generate_basic_feedback(metrics)
        
        # AI-enhanced feedback if available
        if self.ai_provider == "openai" and settings.openai_api_key:
            try:
                ai_feedback = self._generate_ai_feedback(transcript, metrics, context)
                return ai_feedback
            except Exception as e:
                self.logger.warning(f"AI feedback generation failed: {e}")
                return basic_feedback
        else:
            return basic_feedback
            
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text, excluding punctuation.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Remove punctuation and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 0]
        
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Split on sentence endings, but preserve abbreviations
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
        
    def _count_filler_words(self, text: str) -> int:
        """Count filler words in text.
        
        Args:
            text: Input text (should be lowercase)
            
        Returns:
            Number of filler words found
        """
        count = 0
        for filler_word in self.filler_words:
            # Count exact matches and word boundaries
            pattern = r'\b' + re.escape(filler_word) + r'\b'
            count += len(re.findall(pattern, text))
        return count
        
    def _generate_basic_feedback(self, metrics: SpeechMetrics) -> PresentationFeedback:
        """Generate basic feedback based on metrics.
        
        Args:
            metrics: Speech analysis metrics
            
        Returns:
            Basic presentation feedback
        """
        strengths = []
        areas_for_improvement = []
        specific_recommendations = []
        
        # Speaking rate analysis
        if metrics.speaking_rate > 0:
            if 120 <= metrics.speaking_rate <= 160:
                speaking_rate_feedback = "Good speaking rate - well-paced and clear"
                strengths.append("Appropriate speaking pace")
            elif metrics.speaking_rate < 120:
                speaking_rate_feedback = "Speaking rate is slow - consider picking up the pace"
                areas_for_improvement.append("Speaking rate")
                specific_recommendations.append("Practice speaking at a slightly faster pace")
            else:
                speaking_rate_feedback = "Speaking rate is fast - slow down for better clarity"
                areas_for_improvement.append("Speaking rate")
                specific_recommendations.append("Practice pausing between sentences")
        else:
            speaking_rate_feedback = "Unable to determine speaking rate"
            
        # Filler word analysis
        if metrics.filler_word_ratio < 0.05:
            clarity_feedback = "Excellent clarity - minimal filler words"
            strengths.append("Clear speech with few filler words")
        elif metrics.filler_word_ratio < 0.1:
            clarity_feedback = "Good clarity - some filler words detected"
            areas_for_improvement.append("Filler word usage")
            specific_recommendations.append("Practice pausing instead of using filler words")
        else:
            clarity_feedback = "High filler word usage - focus on reducing 'um' and 'uh'"
            areas_for_improvement.append("Filler word usage")
            specific_recommendations.append("Record yourself and identify filler word patterns")
            
        # Vocabulary diversity
        if metrics.vocabulary_diversity > 0.7:
            engagement_feedback = "Rich vocabulary - engaging and varied language"
            strengths.append("Diverse vocabulary")
        elif metrics.vocabulary_diversity > 0.5:
            engagement_feedback = "Moderate vocabulary diversity"
            areas_for_improvement.append("Vocabulary variety")
            specific_recommendations.append("Expand vocabulary for more engaging presentations")
        else:
            engagement_feedback = "Limited vocabulary - consider using more varied language"
            areas_for_improvement.append("Vocabulary variety")
            specific_recommendations.append("Practice using synonyms and varied expressions")
            
        # Overall score calculation
        score = 100.0
        
        # Deduct points for issues
        if metrics.filler_word_ratio > 0.1:
            score -= 20
        if metrics.speaking_rate > 0 and (metrics.speaking_rate < 120 or metrics.speaking_rate > 160):
            score -= 15
        if metrics.vocabulary_diversity < 0.5:
            score -= 10
            
        # Add points for strengths
        if metrics.filler_word_ratio < 0.05:
            score += 10
        if metrics.vocabulary_diversity > 0.7:
            score += 10
            
        score = max(0, min(100, score))
        
        confidence_indicators = []
        if metrics.filler_word_ratio < 0.05:
            confidence_indicators.append("Low filler word usage suggests confidence")
        if metrics.speaking_rate > 0 and 120 <= metrics.speaking_rate <= 160:
            confidence_indicators.append("Consistent speaking pace indicates preparation")
            
        return PresentationFeedback(
            overall_score=score,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            specific_recommendations=specific_recommendations,
            speaking_rate_feedback=speaking_rate_feedback,
            clarity_feedback=clarity_feedback,
            engagement_feedback=engagement_feedback,
            confidence_indicators=confidence_indicators
        )
        
    def _generate_ai_feedback(
        self, 
        transcript: str, 
        metrics: SpeechMetrics,
        context: Optional[str] = None
    ) -> PresentationFeedback:
        """Generate AI-enhanced feedback using OpenAI.
        
        Args:
            transcript: Speech transcript
            metrics: Speech analysis metrics
            context: Additional context
            
        Returns:
            AI-enhanced presentation feedback
            
        Raises:
            SpeechAnalysisProviderError: If AI provider fails
        """
        try:
            prompt = self._build_ai_prompt(transcript, metrics, context)
            
            response = openai.ChatCompletion.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert presentation coach analyzing speech patterns and providing constructive feedback."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI response and combine with basic feedback
            basic_feedback = self._generate_basic_feedback(metrics)
            
            # Enhance with AI insights
            if "strengths:" in ai_response.lower():
                # Extract AI-identified strengths
                ai_strengths = self._extract_ai_section(ai_response, "strengths")
                if ai_strengths:
                    basic_feedback.strengths.extend(ai_strengths)
                    
            if "improvements:" in ai_response.lower():
                # Extract AI-identified improvements
                ai_improvements = self._extract_ai_section(ai_response, "improvements")
                if ai_improvements:
                    basic_feedback.areas_for_improvement.extend(ai_improvements)
                    
            return basic_feedback
            
        except Exception as e:
            error_msg = f"AI feedback generation failed: {str(e)}"
            self.logger.error(error_msg)
            raise SpeechAnalysisProviderError(error_msg) from e
            
    def _build_ai_prompt(
        self, 
        transcript: str, 
        metrics: SpeechMetrics,
        context: Optional[str] = None
    ) -> str:
        """Build AI prompt for feedback generation.
        
        Args:
            transcript: Speech transcript
            metrics: Speech analysis metrics
            context: Additional context
            
        Returns:
            Formatted AI prompt
        """
        prompt = f"""
        Analyze this presentation transcript and provide feedback:
        
        TRANSCRIPT:
        {transcript}
        
        SPEECH METRICS:
        - Word count: {metrics.word_count}
        - Speaking rate: {metrics.speaking_rate:.1f} words per minute
        - Filler word ratio: {metrics.filler_word_ratio:.2%}
        - Vocabulary diversity: {metrics.vocabulary_diversity:.2%}
        - Sentence count: {metrics.sentence_count}
        - Average sentence length: {metrics.average_sentence_length:.1f} words
        
        CONTEXT: {context or "General presentation"}
        
        Please provide:
        1. Key strengths of the presentation
        2. Areas for improvement
        3. Specific actionable recommendations
        4. Overall assessment of clarity, engagement, and confidence
        
        Focus on practical, actionable feedback that can help improve future presentations.
        """
        return prompt
        
    def _extract_ai_section(self, response: str, section: str) -> List[str]:
        """Extract specific section from AI response.
        
        Args:
            response: AI response text
            section: Section to extract (strengths, improvements, etc.)
            
        Returns:
            List of items from the section
        """
        items = []
        lines = response.split('\n')
        in_section = False
        
        for line in lines:
            line = line.strip()
            if section.lower() in line.lower():
                in_section = True
                continue
            elif in_section and line and not line.startswith('-') and not line.startswith('*'):
                break
            elif in_section and line:
                # Extract item (remove bullet points)
                item = line.lstrip('- ').lstrip('* ').strip()
                if item:
                    items.append(item)
                    
        return items
        
    def _get_cached_metrics(self, transcript: str) -> Optional[SpeechMetrics]:
        """Get cached speech metrics.
        
        Args:
            transcript: Speech transcript
            
        Returns:
            Cached metrics or None if not found
        """
        cache_file = self._get_metrics_cache_path(transcript)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                # Reconstruct SpeechMetrics from cache
                return SpeechMetrics(**cache_data)
                
            except (json.JSONDecodeError, OSError, TypeError) as e:
                self.logger.warning(f"Failed to read metrics cache: {e}")
                
        return None
        
    def _cache_metrics(self, transcript: str, metrics: SpeechMetrics) -> None:
        """Cache speech metrics.
        
        Args:
            transcript: Speech transcript
            metrics: Speech metrics to cache
        """
        cache_file = self._get_metrics_cache_path(transcript)
        
        try:
            cache_data = asdict(metrics)
            cache_data['transcript_hash'] = hash(transcript)
            cache_data['timestamp'] = time.time()
            
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
                
        except OSError as e:
            self.logger.warning(f"Failed to cache metrics: {e}")
            
    def _get_metrics_cache_path(self, transcript: str) -> Path:
        """Get cache file path for metrics.
        
        Args:
            transcript: Speech transcript
            
        Returns:
            Path to the cache file
        """
        transcript_hash = hash(transcript)
        cache_name = f"metrics_{transcript_hash}.json"
        return settings.cache_dir / "speech_analysis" / cache_name 
