"""
Main execution script for Speech Emotion Recognition.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import EmotionDatasetLoader
from src.models.emotion_ml import train as train_emotion_ml
from src.models.emotion_dl import train as train_emotion_dl
from src.models.speaker_ml import train as train_speaker_ml
from src.models.speaker_dl import train as train_speaker_dl


def explore_datasets():
    """Explore and analyze datasets."""
    loader = EmotionDatasetLoader()
    loader.explore_datasets()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Speech Emotion Detection')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['explore', 'train', 'train_emotion_ml', 'train_emotion_dl', 
                               'train_speaker_ml', 'train_speaker_dl'],
                       help='Mode: explore datasets or train models')
    
    args = parser.parse_args()
    
    if args.mode == 'explore':
        explore_datasets()
    elif args.mode == 'train':
        # Train all models
        print("Training all emotion detection models...")
        train_emotion_ml()
        train_emotion_dl()
    elif args.mode == 'train_emotion_ml':
        train_emotion_ml()
    elif args.mode == 'train_emotion_dl':
        train_emotion_dl()
    elif args.mode == 'train_speaker_ml':
        train_speaker_ml()
    elif args.mode == 'train_speaker_dl':
        train_speaker_dl()


if __name__ == '__main__':
    main()
