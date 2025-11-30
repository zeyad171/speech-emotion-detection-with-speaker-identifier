"""
Speaker-Optimized Feature Extraction
Extracts features specifically designed for speaker identification:
- Formants (F1-F4) - vocal tract shape
- Jitter/Shimmer - voice quality
- Pitch statistics - speaker-specific but not emotional
- MFCCs - still useful for timbre
- Spectral features - voice characteristics
"""
import os
os.environ['NPY_DISABLE_CPU_FEATURES'] = 'AVX512F,AVX512CD,AVX512_SKX'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import librosa
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class SpeakerFeatureExtractor:
    """Extract features optimized for speaker identification."""
    
    def __init__(self, sr: int = 22050, n_mfcc: int = 13):
        """
        Initialize speaker feature extractor.
        
        Args:
            sr: Sample rate
            n_mfcc: Number of MFCC coefficients
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_formants(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract formant frequencies (F1-F4).
        Formants represent vocal tract resonances - unique per speaker.
        
        Returns: [F1_mean, F1_std, F2_mean, F2_std, F3_mean, F3_std, F4_mean, F4_std] = 8 features
        """
        if sr is None:
            sr = self.sr
        
        try:
            # Pre-emphasis to boost higher frequencies
            pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
            
            # Frame the signal
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            frames = librosa.util.frame(pre_emphasized, frame_length=frame_length, 
                                       hop_length=hop_length)
            
            # Estimate formants using LPC (Linear Predictive Coding)
            formants = []
            for frame in frames.T:
                if len(frame) > 0 and np.any(frame != 0):
                    # LPC analysis (order 12 for 4 formants)
                    lpc_coeffs = librosa.lpc(frame, order=12)
                    
                    # Find roots of LPC polynomial
                    roots = np.roots(lpc_coeffs)
                    roots = roots[np.imag(roots) >= 0]  # Keep positive frequencies
                    
                    # Convert to Hz
                    angz = np.arctan2(np.imag(roots), np.real(roots))
                    freqs = angz * (sr / (2 * np.pi))
                    
                    # Filter formants (typical range: 200-4000 Hz)
                    formant_freqs = freqs[(freqs > 200) & (freqs < 4000)]
                    formant_freqs = np.sort(formant_freqs)[:4]  # Take first 4
                    
                    if len(formant_freqs) == 4:
                        formants.append(formant_freqs)
            
            if len(formants) > 0:
                formants = np.array(formants)
                # Return mean and std for each formant
                features = []
                for i in range(4):
                    features.append(np.mean(formants[:, i]))
                    features.append(np.std(formants[:, i]))
                return np.array(features)
            else:
                # Return zeros if formant extraction failed
                return np.zeros(8)
        except:
            return np.zeros(8)
    
    def extract_jitter_shimmer(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract jitter (pitch variation) and shimmer (amplitude variation).
        Both are voice quality measures unique to each speaker.
        
        Returns: [jitter, shimmer, jitter_local, shimmer_local] = 4 features
        """
        if sr is None:
            sr = self.sr
        
        try:
            # Extract pitch using YIN algorithm (more accurate for jitter)
            f0 = librosa.yin(audio, fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'), sr=sr)
            
            # Remove unvoiced frames (f0 == 0)
            voiced_f0 = f0[f0 > 0]
            
            if len(voiced_f0) > 10:
                # Jitter: cycle-to-cycle variation in pitch
                periods = 1.0 / voiced_f0
                period_diffs = np.abs(np.diff(periods))
                jitter = np.mean(period_diffs) / np.mean(periods)
                jitter_local = np.std(period_diffs) / np.mean(periods)
            else:
                jitter = 0.0
                jitter_local = 0.0
            
            # Shimmer: cycle-to-cycle variation in amplitude
            # Extract amplitude envelope
            amplitude = np.abs(librosa.stft(audio))
            amplitude = np.mean(amplitude, axis=0)
            
            if len(amplitude) > 10:
                amp_diffs = np.abs(np.diff(amplitude))
                shimmer = np.mean(amp_diffs) / np.mean(amplitude)
                shimmer_local = np.std(amp_diffs) / np.mean(amplitude)
            else:
                shimmer = 0.0
                shimmer_local = 0.0
            
            return np.array([jitter, shimmer, jitter_local, shimmer_local])
        except:
            return np.zeros(4)
    
    def extract_speaker_pitch_stats(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract speaker-specific pitch statistics (not emotional variation).
        
        Returns: [pitch_mean, pitch_std, pitch_min, pitch_max, pitch_range, pitch_median] = 6 features
        """
        if sr is None:
            sr = self.sr
        
        try:
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            
            # Select pitch values with high magnitude
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 0:
                pitch_values = np.array(pitch_values)
                return np.array([
                    np.mean(pitch_values),
                    np.std(pitch_values),
                    np.min(pitch_values),
                    np.max(pitch_values),
                    np.max(pitch_values) - np.min(pitch_values),
                    np.median(pitch_values)
                ])
            else:
                return np.zeros(6)
        except:
            return np.zeros(6)
    
    def extract_mfcc(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract MFCCs (still useful for speaker timbre).
        
        Returns: [mfcc_mean (13), mfcc_std (13)] = 26 features
        """
        if sr is None:
            sr = self.sr
        
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            return np.hstack([mfcc_mean, mfcc_std])
        except:
            return np.zeros(self.n_mfcc * 2)
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract spectral features (voice timbre characteristics).
        
        Returns: [
            spectral_centroid_mean, spectral_centroid_std,
            spectral_bandwidth_mean, spectral_bandwidth_std,
            spectral_rolloff_mean, spectral_rolloff_std,
            spectral_contrast_mean (7), spectral_contrast_std (7),
            spectral_flatness_mean, spectral_flatness_std
        ] = 20 features
        """
        if sr is None:
            sr = self.sr
        
        try:
            # Spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            
            # Spectral contrast (peaks vs valleys)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            # Spectral flatness (tonality)
            flatness = librosa.feature.spectral_flatness(y=audio)
            
            return np.hstack([
                np.mean(centroid), np.std(centroid),
                np.mean(bandwidth), np.std(bandwidth),
                np.mean(rolloff), np.std(rolloff),
                np.mean(contrast, axis=1), np.std(contrast, axis=1),
                np.mean(flatness), np.std(flatness)
            ])
        except:
            return np.zeros(20)
    
    def extract_voice_onset_offset(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract voice onset/offset characteristics (speaking style).
        
        Returns: [onset_strength_mean, onset_strength_std, onset_rate, voice_activity_ratio] = 4 features
        """
        if sr is None:
            sr = self.sr
        
        try:
            # Onset strength (how sharply voice starts)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onset_strength_mean = np.mean(onset_env)
            onset_strength_std = np.std(onset_env)
            
            # Onset rate (speaking tempo)
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
            onset_rate = len(onset_frames) / (len(audio) / sr) if len(audio) > 0 else 0
            
            # Voice activity ratio (how much of the audio is voiced)
            rms = librosa.feature.rms(y=audio)[0]
            voice_activity_ratio = np.sum(rms > np.mean(rms)) / len(rms) if len(rms) > 0 else 0
            
            return np.array([onset_strength_mean, onset_strength_std, onset_rate, voice_activity_ratio])
        except:
            return np.zeros(4)
    
    def extract_zcr_energy(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract zero crossing rate and energy (still useful for speaker ID).
        
        Returns: [zcr_mean, zcr_std, energy_mean, energy_std] = 4 features
        """
        if sr is None:
            sr = self.sr
        
        try:
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            
            # RMS energy
            energy = librosa.feature.rms(y=audio)
            
            return np.hstack([
                np.mean(zcr), np.std(zcr),
                np.mean(energy), np.std(energy)
            ])
        except:
            return np.zeros(4)
    
    def extract_all_features(self, audio: np.ndarray, sr: int = None) -> np.ndarray:
        """
        Extract all speaker-optimized features.
        
        Feature breakdown:
        - Formants: 8 (F1-F4 mean/std)
        - Jitter/Shimmer: 4
        - Pitch stats: 6
        - MFCCs: 26 (13 mean + 13 std)
        - Spectral: 20
        - Voice onset/offset: 4
        - ZCR/Energy: 4
        Total: 72 features
        
        Returns: numpy array of shape (72,)
        """
        if sr is None:
            sr = self.sr
        
        if len(audio) == 0:
            return np.zeros(72)
        
        features = []
        
        # Extract each feature type
        features.append(self.extract_formants(audio, sr))           # 8
        features.append(self.extract_jitter_shimmer(audio, sr))     # 4
        features.append(self.extract_speaker_pitch_stats(audio, sr)) # 6
        features.append(self.extract_mfcc(audio, sr))               # 26
        features.append(self.extract_spectral_features(audio, sr))  # 20
        features.append(self.extract_voice_onset_offset(audio, sr)) # 4
        features.append(self.extract_zcr_energy(audio, sr))         # 4
        
        return np.hstack(features)
    
    def extract_features_batch(self, audio_list: list, sr: int = None) -> np.ndarray:
        """
        Extract features from a batch of audio samples.
        
        Args:
            audio_list: List of audio numpy arrays
            sr: Sample rate
        
        Returns: numpy array of shape (n_samples, 72)
        """
        if sr is None:
            sr = self.sr
        
        features = []
        for i, audio in enumerate(audio_list):
            try:
                feat = self.extract_all_features(audio, sr)
                features.append(feat)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Extracted features from {i + 1}/{len(audio_list)} audio samples")
            except Exception as e:
                print(f"  Error extracting features from sample {i}: {e}")
                features.append(np.zeros(72))
        
        return np.array(features)


if __name__ == '__main__':
    # Test the speaker feature extractor
    extractor = SpeakerFeatureExtractor()
    
    # Generate a test signal
    duration = 3  # seconds
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simulated voice (fundamental + harmonics)
    f0 = 150  # Hz (typical male voice)
    audio = np.sin(2 * np.pi * f0 * t) + \
            0.5 * np.sin(2 * np.pi * 2 * f0 * t) + \
            0.25 * np.sin(2 * np.pi * 3 * f0 * t)
    
    # Add some noise
    audio += 0.05 * np.random.randn(len(audio))
    
    features = extractor.extract_all_features(audio)
    
    print(f"Extracted {len(features)} features for speaker identification")
    print(f"Feature breakdown:")
    print(f"  Formants (F1-F4): 8 features")
    print(f"  Jitter/Shimmer: 4 features")
    print(f"  Pitch statistics: 6 features")
    print(f"  MFCCs: 26 features")
    print(f"  Spectral: 20 features")
    print(f"  Voice onset/offset: 4 features")
    print(f"  ZCR/Energy: 4 features")
    print(f"  Total: {len(features)} features")
