"""
Data loading and preprocessing module for Speech Emotion Detection.
Handles multiple datasets: TESS, SAVEE, RAVDESS, and CREMA-D.
"""
import re
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class EmotionDatasetLoader:
    """Unified dataset loader for multiple emotion datasets."""
    
    # Emotion mapping to standard labels
    EMOTION_MAP = {
        # Standard emotions
        'angry': 'angry',
        'anger': 'angry',
        'ang': 'angry',
        'a': 'angry',
        
        'disgust': 'disgust',
        'dis': 'disgust',
        'd': 'disgust',
        
        'fear': 'fear',
        'fea': 'fear',
        'f': 'fear',
        
        'happy': 'happy',
        'happiness': 'happy',
        'hap': 'happy',
        'h': 'happy',
        
        'neutral': 'neutral',
        'neu': 'neutral',
        'n': 'neutral',
        
        'sad': 'sad',
        'sadness': 'sad',
        'sa': 'sad',
        's': 'sad',
        
        'surprise': 'surprise',
        'surprised': 'surprise',
        'pleasant_surprise': 'surprise',
        'pleasant_surprised': 'surprise',
        'su': 'surprise',
    }
    
    # Standard emotion list
    STANDARD_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def __init__(self, data_root: str = None, target_sr: int = 22050):
        """
        Initialize the dataset loader.
        
        Args:
            data_root: Root directory containing dataset folder
            target_sr: Target sample rate for audio resampling
        """
        if data_root is None:
            # Assume data is in dataset folder in parent directory
            self.data_root = Path(__file__).parent.parent / 'dataset'
        else:
            self.data_root = Path(data_root)
        
        self.target_sr = target_sr
        self.audio_files = []
        self.labels = []
        self.dataset_info = {}
        
    def load_tess_dataset(self) -> Tuple[List[str], List[str]]:
        """Load TESS dataset (folder-based structure)."""
        tess_path = self.data_root / 'Tess'
        if not tess_path.exists():
            tess_path = self.data_root / 'TESS'
        
        if not tess_path.exists():
            print(f"TESS dataset not found at {tess_path}")
            print(f"Looking in: {self.data_root}")
            return [], []
        
        audio_files = []
        labels = []
        
        # TESS structure: OAF_angry/, YAF_happy/, etc.
        for folder in tess_path.iterdir():
            if folder.is_dir():
                folder_name = folder.name.lower()
                
                # Extract emotion from folder name
                emotion = None
                for key, value in self.EMOTION_MAP.items():
                    if key in folder_name:
                        emotion = value
                        break
                
                if emotion and emotion in self.STANDARD_EMOTIONS:
                    for audio_file in folder.glob('*.wav'):
                        audio_files.append(str(audio_file))
                        labels.append(emotion)
        
        print(f"Loaded {len(audio_files)} files from TESS dataset")
        return audio_files, labels
    
    def load_savee_dataset(self) -> Tuple[List[str], List[str]]:
        """Load SAVEE dataset (filename-based structure)."""
        savee_path = self.data_root / 'Savee'
        if not savee_path.exists():
            savee_path = self.data_root / 'SAVEE'
        
        if not savee_path.exists():
            print(f"SAVEE dataset not found at {savee_path}")
            print(f"Looking in: {self.data_root}")
            return [], []
        
        audio_files = []
        labels = []
        
        # SAVEE naming format: DC_a01.wav, JE_h01.wav, etc.
        # Emotion codes: 'a'=anger, 'd'=disgust, 'f'=fear, 'h'=happiness, 'n'=neutral, 'sa'=sadness, 'su'=surprise
        # Note: Check 'sa' and 'su' before 'a' and 's' to avoid partial matches
        emotion_codes = {
            'sa': 'sad',      # Check before 'a' to avoid partial match
            'su': 'surprise',  # Check before 's' to avoid partial match
            'a': 'angry',      # anger -> angry
            'd': 'disgust',
            'f': 'fear',
            'h': 'happy',      # happiness -> happy
            'n': 'neutral'
        }
        
        for audio_file in savee_path.glob('*.wav'):
            filename = audio_file.stem.lower()
            # Extract emotion code (e.g., 'a' from 'DC_a01' or 'sa' from 'DC_sa01')
            parts = filename.split('_')
            if len(parts) >= 2:
                code_part = parts[1]  # Get the emotion code part (e.g., 'a01' or 'sa01')
                # Extract the emotion code by removing digits
                code = re.sub(r'\d+', '', code_part)  # Remove all digits, leaving just the emotion code
                
                # Check for emotion code in our mapping
                if code in emotion_codes:
                    emotion = emotion_codes[code]
                    audio_files.append(str(audio_file))
                    labels.append(emotion)
        
        print(f"Loaded {len(audio_files)} files from SAVEE dataset")
        return audio_files, labels
    
    def load_ravdess_dataset(self) -> Tuple[List[str], List[str]]:
        """Load RAVDESS dataset (filename-encoded structure)."""
        ravdess_path = self.data_root / 'Ravdess'
        if not ravdess_path.exists():
            ravdess_path = self.data_root / 'RAVDESS'
        
        if not ravdess_path.exists():
            print(f"RAVDESS dataset not found at {ravdess_path}")
            print(f"Looking in: {self.data_root}")
            return [], []
        
        audio_files = []
        labels = []
        
        # RAVDESS naming: 03-01-01-01-01-01-01.wav
        # Format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor
        # Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
        
        emotion_codes = {
            '01': 'neutral',
            '02': 'neutral',  # calm -> neutral
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fear',
            '07': 'disgust',
            '08': 'surprise'
        }
        
        # Search in audio_speech_actors_01-24 folder
        actors_folder = ravdess_path / 'audio_speech_actors_01-24'
        if actors_folder.exists():
            for actor_folder in actors_folder.iterdir():
                if actor_folder.is_dir() and actor_folder.name.startswith('Actor_'):
                    for audio_file in actor_folder.glob('*.wav'):
                        filename = audio_file.stem
                        parts = filename.split('-')
                        if len(parts) >= 3:
                            emotion_code = parts[2]
                            if emotion_code in emotion_codes:
                                emotion = emotion_codes[emotion_code]
                                audio_files.append(str(audio_file))
                                labels.append(emotion)
        
        print(f"Loaded {len(audio_files)} files from RAVDESS dataset")
        return audio_files, labels
    
    def load_crema_dataset(self) -> Tuple[List[str], List[str]]:
        """Load CREMA-D dataset (filename-encoded structure)."""
        crema_path = self.data_root / 'Crema'
        if not crema_path.exists():
            crema_path = self.data_root / 'CREMA'
        
        if not crema_path.exists():
            print(f"CREMA-D dataset not found at {crema_path}")
            print(f"Looking in: {self.data_root}")
            return [], []
        
        audio_files = []
        labels = []
        
        # CREMA-D naming: 1001_DFA_ANG_XX.wav
        # Format: ActorID_Sentence_Emotion_Intensity
        # Emotion codes: ANG=angry, DIS=disgust, FEA=fear, HAP=happy, NEU=neutral, SAD=sad
        
        emotion_codes = {
            'ANG': 'angry',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        
        for audio_file in crema_path.glob('*.wav'):
            filename = audio_file.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in emotion_codes:
                    emotion = emotion_codes[emotion_code]
                    audio_files.append(str(audio_file))
                    labels.append(emotion)
        
        print(f"Loaded {len(audio_files)} files from CREMA-D dataset")
        return audio_files, labels
    
    def load_all_datasets(self) -> Tuple[List[str], List[str]]:
        """Load all available datasets."""
        all_files = []
        all_labels = []
        
        # Load each dataset
        datasets = [
            ('TESS', self.load_tess_dataset),
            ('SAVEE', self.load_savee_dataset),
            ('RAVDESS', self.load_ravdess_dataset),
            ('CREMA-D', self.load_crema_dataset)
        ]
        
        for name, loader_func in datasets:
            files, labels = loader_func()
            all_files.extend(files)
            all_labels.extend(labels)
            self.dataset_info[name] = len(files)
        
        print(f"\nTotal files loaded: {len(all_files)}")
        print(f"Emotion distribution:")
        unique, counts = np.unique(all_labels, return_counts=True)
        for emotion, count in zip(unique, counts):
            print(f"  {emotion}: {count}")
        
        return all_files, all_labels
    
    def preprocess_audio(self, audio_path: str, trim_silence: bool = True) -> np.ndarray:
        """
        Preprocess a single audio file.
        
        Args:
            audio_path: Path to audio file
            trim_silence: Whether to trim silence from beginning/end
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # Normalize volume (amplitude normalization)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Trim silence
            if trim_silence:
                audio, _ = librosa.effects.trim(audio, top_db=20)
            
            return audio
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.array([])
    
    def get_dataset_statistics(self, audio_files: List[str]) -> Dict:
        """Get statistics about the dataset."""
        durations = []
        sample_rates = []
        
        for audio_file in audio_files[:100]:  # Sample first 100 files
            try:
                info = sf.info(audio_file)
                sample_rates.append(info.samplerate)
                duration = info.frames / info.samplerate
                durations.append(duration)
            except:
                pass
        
        stats = {
            'mean_duration': np.mean(durations) if durations else 0,
            'std_duration': np.std(durations) if durations else 0,
            'min_duration': np.min(durations) if durations else 0,
            'max_duration': np.max(durations) if durations else 0,
            'sample_rates': list(set(sample_rates))
        }
        
        return stats
    
    def explore_datasets(self):
        """Explore and analyze datasets."""
        print("="*60)
        print("Dataset Exploration")
        print("="*60)
        
        files, labels = self.load_all_datasets()
        
        if len(files) > 0:
            print(f"\nTotal files: {len(files)}")
            print(f"Unique emotions: {len(set(labels))}")
            
            # Get statistics
            stats = self.get_dataset_statistics(files)
            print(f"\nDataset Statistics:")
            print(f"  Mean duration: {stats['mean_duration']:.2f}s")
            print(f"  Duration range: {stats['min_duration']:.2f}s - {stats['max_duration']:.2f}s")
            print(f"  Sample rates: {stats['sample_rates']}")
            
            # Emotion distribution
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\nEmotion Distribution:")
            for emotion, count in zip(unique, counts):
                print(f"  {emotion}: {count} ({count/len(labels)*100:.1f}%)")
        else:
            print("No files found. Please check dataset paths.")


if __name__ == '__main__':
    # Test the data loader
    loader = EmotionDatasetLoader()
    loader.explore_datasets()

