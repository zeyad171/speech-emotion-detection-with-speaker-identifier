"""
Utility functions for the Speech Emotion Detection project.
"""
import numpy as np
from pathlib import Path


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


def save_features(features, labels, filepath):
    """Save extracted features and labels to a file."""
    ensure_dir(Path(filepath).parent)
    np.savez_compressed(filepath, features=features, labels=labels)
    print(f"Features saved to {filepath}")


def load_features(filepath):
    """Load extracted features and labels from a file."""
    data = np.load(filepath, allow_pickle=True)
    return data['features'], data['labels']


def print_dataset_info(dataset_name, file_count, emotions, sample_rate=None):
    """Print dataset information."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Total files: {file_count}")
    print(f"Emotions: {', '.join(sorted(emotions))}")
    if sample_rate:
        print(f"Sample rate: {sample_rate} Hz")
    print(f"{'='*60}\n")


def save_preprocessed_audio(audio_list, labels, filepath):
    """
    Save preprocessed audio arrays and labels to a file.
    
    Args:
        audio_list: List of numpy arrays (preprocessed audio)
        labels: Corresponding labels
        filepath: Path to save the file
    """
    ensure_dir(Path(filepath).parent)
    # Save each audio array individually since they have variable lengths
    # Use a dictionary to store arrays with unique keys
    save_dict = {'labels': labels, 'count': len(audio_list)}
    for i, audio in enumerate(audio_list):
        save_dict[f'audio_{i}'] = audio
    
    np.savez_compressed(filepath, **save_dict)
    print(f"Preprocessed audio saved to {filepath}")


def load_preprocessed_audio(filepath):
    """
    Load preprocessed audio arrays and labels from a file.
    
    Args:
        filepath: Path to the saved file
        
    Returns:
        Tuple of (audio_list, labels)
    """
    data = np.load(filepath, allow_pickle=True)
    keys = list(data.keys())
    
    # Get labels if they exist
    if 'labels' in keys:
        labels = data['labels']
    else:
        # If labels don't exist, this is an incompatible old format
        # Close the file before raising error so it can be deleted
        data.close()
        raise ValueError(
            f"Cache file {filepath} is in an incompatible format (missing 'labels'). "
            f"Please delete this file to regenerate it: {filepath}"
        )
    
    # Check if it's the new format (with individual audio_X keys) or old format
    if 'count' in keys:
        # New format: individual audio arrays with keys audio_0, audio_1, etc.
        count = int(data['count'])
        audio_list = []
        for i in range(count):
            if f'audio_{i}' in keys:
                audio_list.append(data[f'audio_{i}'])
            else:
                raise ValueError(f"Missing audio_{i} in cache file. File may be corrupted.")
    elif 'audio_arrays' in keys:
        # Old format: try to load as array/list
        audio_arrays = data['audio_arrays']
        # Convert to list if it's a numpy array
        if isinstance(audio_arrays, np.ndarray):
            audio_list = list(audio_arrays)
        else:
            audio_list = audio_arrays
    else:
        # Try to find audio keys manually
        audio_list = []
        audio_keys = [k for k in keys if k.startswith('audio_')]
        if audio_keys:
            # Sort by index to maintain order
            audio_keys.sort(key=lambda x: int(x.split('_')[1]))
            for key in audio_keys:
                audio_list.append(data[key])
        else:
            raise ValueError(
                f"Could not find audio data in {filepath}. "
                f"Available keys: {keys}. "
                f"Please delete this file to regenerate it: {filepath}"
            )
    
    # Close the file handle
    data.close()
    print(f"Preprocessed audio loaded from {filepath} ({len(audio_list)} files)")
    return audio_list, labels


def save_dl_data(spectrograms, sequences, labels, filepath):
    """
    Save DL data (spectrograms, sequences, and labels) to a file.
    
    Args:
        spectrograms: Mel spectrograms array
        sequences: MFCC sequences array
        labels: Corresponding labels
        filepath: Path to save the file
    """
    ensure_dir(Path(filepath).parent)
    np.savez_compressed(filepath,
                       spectrograms=spectrograms,
                       sequences=sequences,
                       labels=labels)
    print(f"DL data saved to {filepath}")


def load_dl_data(filepath):
    """
    Load DL data (spectrograms, sequences, and labels) from a file.
    
    Args:
        filepath: Path to the saved file
        
    Returns:
        Tuple of (spectrograms, sequences, labels)
    """
    data = np.load(filepath, allow_pickle=True)
    spectrograms = data['spectrograms']
    sequences = data['sequences']
    labels = data['labels']
    print(f"DL data loaded from {filepath}")
    return spectrograms, sequences, labels