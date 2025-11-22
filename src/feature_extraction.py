"""
Feature extraction module for Speech Emotion Detection.
Extracts various audio features using Librosa.
"""
import numpy as np
import librosa
from typing import List, Tuple, Dict


class FeatureExtractor:
    """Extract audio features for emotion detection."""
    
    def __init__(self, n_mfcc: int = 13, n_mels: int = 128, hop_length: int = 512):
        """
        Initialize feature extractor.
        
        Args:
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            hop_length: Hop length for STFT
        """
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
    
    def extract_mfcc(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Extract MFCC features."""
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        return mfccs
    
    def extract_chroma(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Extract chroma features."""
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=self.hop_length)
        return chroma
    
    def extract_pitch(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Extract pitch (fundamental frequency)."""
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=self.hop_length)
        # Get the pitch values
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        return np.array(pitch_values) if pitch_values else np.array([0.0])
    
    def extract_energy(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract energy features."""
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)[0]
        
        return {
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'rms_max': np.max(rms),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr)
        }
    
    def extract_spectral_features(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, float]:
        """Extract spectral features."""
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=self.hop_length)[0]
        
        return {
            'centroid_mean': np.mean(spectral_centroids),
            'centroid_std': np.std(spectral_centroids),
            'rolloff_mean': np.mean(spectral_rolloff),
            'rolloff_std': np.std(spectral_rolloff),
            'bandwidth_mean': np.mean(spectral_bandwidth),
            'bandwidth_std': np.std(spectral_bandwidth)
        }
    
    def extract_mel_spectrogram(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """Extract mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_all_features(self, audio: np.ndarray, sr: int = 22050, 
                           use_statistical: bool = True) -> np.ndarray:
        """
        Extract all features and return as a feature vector.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            use_statistical: If True, use statistical aggregation for variable-length features
            
        Returns:
            Feature vector
        """
        features = []
        
        # MFCC features (statistical aggregation)
        mfccs = self.extract_mfcc(audio, sr)
        if use_statistical:
            features.extend(np.mean(mfccs, axis=1).tolist())
            features.extend(np.std(mfccs, axis=1).tolist())
        else:
            features.extend(mfccs.flatten().tolist())
        
        # Chroma features (statistical aggregation)
        chroma = self.extract_chroma(audio, sr)
        if use_statistical:
            features.extend(np.mean(chroma, axis=1).tolist())
            features.extend(np.std(chroma, axis=1).tolist())
        else:
            features.extend(chroma.flatten().tolist())
        
        # Pitch features
        pitch = self.extract_pitch(audio, sr)
        features.append(np.mean(pitch))
        features.append(np.std(pitch))
        features.append(np.max(pitch))
        features.append(np.min(pitch))
        
        # Energy features
        energy = self.extract_energy(audio)
        features.extend(list(energy.values()))
        
        # Spectral features
        spectral = self.extract_spectral_features(audio, sr)
        features.extend(list(spectral.values()))
        
        return np.array(features)
    
    def extract_features_batch(self, audio_list: List[np.ndarray], sr: int = 22050) -> np.ndarray:
        """
        Extract features for a batch of audio files.
        
        Args:
            audio_list: List of audio signals
            sr: Sample rate
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features_list = []
        for i, audio in enumerate(audio_list):
            if len(audio) > 0:
                features = self.extract_all_features(audio, sr)
                features_list.append(features)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(audio_list)} files")
        
        return np.array(features_list)


if __name__ == '__main__':
    # Test feature extraction
    import librosa
    
    # Create a test audio signal
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 440 * t)  # A4 note
    
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(test_audio, sr)
    
    print(f"Extracted {len(features)} features")
    print(f"Feature vector shape: {features.shape}")

