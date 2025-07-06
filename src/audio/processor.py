"""Audio processing and analysis functionality."""

import wave
from typing import Optional, Tuple, List
import numpy as np
from scipy import signal
from scipy.stats import entropy

from src.audio.exceptions import AudioProcessingError, AudioFormatError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """Handles audio processing, analysis, and quality validation."""

    def __init__(self, sample_rate: int = 44100) -> None:
        """Initialize the audio processor.

        Args:
            sample_rate: Sample rate in Hz (default: 44100)
        """
        self.sample_rate = sample_rate

    def bytes_to_numpy(self, audio_data: bytes, channels: int = 1) -> np.ndarray:
        """Convert raw audio bytes to numpy array.

        Args:
            audio_data: Raw audio data as bytes
            channels: Number of audio channels

        Returns:
            Audio data as numpy array

        Raises:
            AudioProcessingError: If conversion fails
        """
        try:
            # Convert bytes to 16-bit integers
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Reshape for stereo if needed
            if channels == 2:
                audio_array = audio_array.reshape(-1, 2)
            
            return audio_array.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

        except Exception as e:
            raise AudioProcessingError(f"Failed to convert audio to numpy array: {e}")

    def numpy_to_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy array back to raw audio bytes.

        Args:
            audio_array: Audio data as numpy array

        Returns:
            Raw audio data as bytes

        Raises:
            AudioProcessingError: If conversion fails
        """
        try:
            # Denormalize and convert to 16-bit integers
            audio_array = (audio_array * 32768.0).astype(np.int16)
            return audio_array.tobytes()

        except Exception as e:
            raise AudioProcessingError(f"Failed to convert numpy array to bytes: {e}")

    def validate_audio_quality(self, audio_array: np.ndarray) -> dict:
        """Validate audio quality and provide metrics.

        Args:
            audio_array: Audio data as numpy array

        Returns:
            Dictionary containing quality metrics

        Raises:
            AudioProcessingError: If validation fails
        """
        try:
            # Calculate RMS (Root Mean Square) for volume level
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Calculate peak amplitude
            peak = np.max(np.abs(audio_array))
            
            # Calculate signal-to-noise ratio (simplified)
            signal_power = np.mean(audio_array**2)
            noise_estimate = np.var(audio_array - np.mean(audio_array))
            snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
            
            # Calculate dynamic range
            dynamic_range = 20 * np.log10(peak / (rms + 1e-10))
            
            # Check for clipping
            clipping_percentage = np.sum(np.abs(audio_array) >= 0.99) / len(audio_array) * 100
            
            # Calculate frequency content
            freqs, psd = signal.welch(audio_array.flatten(), self.sample_rate)
            dominant_freq = freqs[np.argmax(psd)]
            
            quality_metrics = {
                "rms_level": float(rms),
                "peak_amplitude": float(peak),
                "signal_to_noise_ratio": float(snr),
                "dynamic_range": float(dynamic_range),
                "clipping_percentage": float(clipping_percentage),
                "dominant_frequency": float(dominant_freq),
                "duration_seconds": len(audio_array) / self.sample_rate,
                "is_acceptable": (
                    rms > 0.01 and  # Not too quiet
                    rms < 0.8 and   # Not too loud
                    clipping_percentage < 1.0 and  # Minimal clipping
                    snr > 10  # Good signal-to-noise ratio
                )
            }
            
            logger.info(f"Audio quality validation completed: {quality_metrics}")
            return quality_metrics

        except Exception as e:
            raise AudioProcessingError(f"Failed to validate audio quality: {e}")

    def apply_noise_reduction(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction using spectral gating.

        Args:
            audio_array: Audio data as numpy array

        Returns:
            Processed audio array with reduced noise

        Raises:
            AudioProcessingError: If noise reduction fails
        """
        try:
            # Simple spectral gating noise reduction
            # This is a basic implementation - more sophisticated methods could be added
            
            # Apply FFT
            fft = np.fft.fft(audio_array)
            magnitude = np.abs(fft)
            
            # Calculate noise floor (simplified)
            noise_floor = np.percentile(magnitude, 10)
            threshold = noise_floor * 2
            
            # Apply spectral gate
            magnitude_filtered = np.where(magnitude > threshold, magnitude, 0)
            
            # Reconstruct signal
            phase = np.angle(fft)
            fft_filtered = magnitude_filtered * np.exp(1j * phase)
            audio_filtered = np.real(np.fft.ifft(fft_filtered))
            
            logger.info("Noise reduction applied successfully")
            return audio_filtered

        except Exception as e:
            raise AudioProcessingError(f"Failed to apply noise reduction: {e}")

    def normalize_audio(self, audio_array: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """Normalize audio to target RMS level.

        Args:
            audio_array: Audio data as numpy array
            target_rms: Target RMS level (default: 0.1)

        Returns:
            Normalized audio array

        Raises:
            AudioProcessingError: If normalization fails
        """
        try:
            current_rms = np.sqrt(np.mean(audio_array**2))
            if current_rms > 0:
                gain = target_rms / current_rms
                # Limit gain to prevent excessive amplification
                gain = min(gain, 10.0)
                normalized_audio = audio_array * gain
            else:
                normalized_audio = audio_array
            
            logger.info(f"Audio normalized to RMS level: {target_rms}")
            return normalized_audio

        except Exception as e:
            raise AudioProcessingError(f"Failed to normalize audio: {e}")

    def trim_silence(self, audio_array: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from the beginning and end of audio.

        Args:
            audio_array: Audio data as numpy array
            threshold: Amplitude threshold for silence detection

        Returns:
            Audio array with silence trimmed

        Raises:
            AudioProcessingError: If trimming fails
        """
        try:
            # Find non-silent regions
            energy = np.abs(audio_array)
            silent_regions = energy < threshold
            
            # Find first and last non-silent samples
            non_silent_indices = np.where(~silent_regions)[0]
            
            if len(non_silent_indices) == 0:
                logger.warning("No non-silent regions found in audio")
                return audio_array
            
            start_idx = non_silent_indices[0]
            end_idx = non_silent_indices[-1] + 1
            
            trimmed_audio = audio_array[start_idx:end_idx]
            
            logger.info(f"Trimmed silence: {len(audio_array)} -> {len(trimmed_audio)} samples")
            return trimmed_audio

        except Exception as e:
            raise AudioProcessingError(f"Failed to trim silence: {e}")

    def resample_audio(self, audio_array: np.ndarray, target_sample_rate: int) -> np.ndarray:
        """Resample audio to a different sample rate.

        Args:
            audio_array: Audio data as numpy array
            target_sample_rate: Target sample rate in Hz

        Returns:
            Resampled audio array

        Raises:
            AudioProcessingError: If resampling fails
        """
        try:
            if target_sample_rate == self.sample_rate:
                return audio_array
            
            # Calculate resampling ratio
            ratio = target_sample_rate / self.sample_rate
            
            # Resample using scipy
            resampled_audio = signal.resample(audio_array, int(len(audio_array) * ratio))
            
            logger.info(f"Resampled audio: {self.sample_rate}Hz -> {target_sample_rate}Hz")
            return resampled_audio

        except Exception as e:
            raise AudioProcessingError(f"Failed to resample audio: {e}")

    def get_audio_statistics(self, audio_array: np.ndarray) -> dict:
        """Get comprehensive audio statistics.

        Args:
            audio_array: Audio data as numpy array

        Returns:
            Dictionary containing audio statistics

        Raises:
            AudioProcessingError: If analysis fails
        """
        try:
            # Basic statistics
            mean_amplitude = np.mean(audio_array)
            std_amplitude = np.std(audio_array)
            min_amplitude = np.min(audio_array)
            max_amplitude = np.max(audio_array)
            
            # Energy statistics
            energy = np.sum(audio_array**2)
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Zero crossing rate (measure of frequency content)
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            zero_crossing_rate = zero_crossings / len(audio_array)
            
            # Spectral statistics
            freqs, psd = signal.welch(audio_array.flatten(), self.sample_rate)
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
            
            stats = {
                "mean_amplitude": float(mean_amplitude),
                "std_amplitude": float(std_amplitude),
                "min_amplitude": float(min_amplitude),
                "max_amplitude": float(max_amplitude),
                "energy": float(energy),
                "rms": float(rms),
                "zero_crossing_rate": float(zero_crossing_rate),
                "spectral_centroid": float(spectral_centroid),
                "spectral_bandwidth": float(spectral_bandwidth),
                "duration_seconds": len(audio_array) / self.sample_rate,
                "sample_count": len(audio_array),
            }
            
            logger.info("Audio statistics calculated successfully")
            return stats

        except Exception as e:
            raise AudioProcessingError(f"Failed to calculate audio statistics: {e}")

    def detect_speech_segments(self, audio_array: np.ndarray, min_segment_duration: float = 0.5) -> List[Tuple[float, float]]:
        """Detect speech segments in audio using energy-based detection.

        Args:
            audio_array: Audio data as numpy array
            min_segment_duration: Minimum duration for a speech segment in seconds

        Returns:
            List of tuples containing (start_time, end_time) for each speech segment

        Raises:
            AudioProcessingError: If speech detection fails
        """
        try:
            # Calculate energy over time
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop
            
            energy = []
            for i in range(0, len(audio_array) - frame_length, hop_length):
                frame = audio_array[i:i + frame_length]
                energy.append(np.sum(frame**2))
            
            energy = np.array(energy)
            
            # Calculate adaptive threshold
            threshold = np.mean(energy) + 2 * np.std(energy)
            
            # Find speech regions
            speech_frames = energy > threshold
            
            # Group consecutive speech frames into segments
            segments = []
            start_frame = None
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and start_frame is None:
                    start_frame = i
                elif not is_speech and start_frame is not None:
                    end_frame = i
                    duration = (end_frame - start_frame) * hop_length / self.sample_rate
                    
                    if duration >= min_segment_duration:
                        start_time = start_frame * hop_length / self.sample_rate
                        end_time = end_frame * hop_length / self.sample_rate
                        segments.append((start_time, end_time))
                    
                    start_frame = None
            
            # Handle case where speech continues to the end
            if start_frame is not None:
                duration = (len(speech_frames) - start_frame) * hop_length / self.sample_rate
                if duration >= min_segment_duration:
                    start_time = start_frame * hop_length / self.sample_rate
                    end_time = len(audio_array) / self.sample_rate
                    segments.append((start_time, end_time))
            
            logger.info(f"Detected {len(segments)} speech segments")
            return segments

        except Exception as e:
            raise AudioProcessingError(f"Failed to detect speech segments: {e}") 
