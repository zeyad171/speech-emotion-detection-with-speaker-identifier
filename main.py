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
from src.models.dl_param_config import DLParamConfig
import json


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
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (float between 0 and 1)')
    parser.add_argument('--cfg', type=str, default=None,
                       help='Path to DL JSON config produced by DLParamConfig.save()')

    args = parser.parse_args()

    if args.mode == 'explore':
        explore_datasets()
    elif args.mode == 'train':
        # Train all models
        print("Training all emotion detection models...")
        train_emotion_ml(test_size=args.test_size)
        # Load cfg if provided
        cfg = None
        if args.cfg:
            try:
                cfg = DLParamConfig.load(args.cfg)
            except Exception as e:
                print(f"Warning: could not load cfg {args.cfg}: {e}")
                cfg = None
        train_emotion_dl(test_size=args.test_size, cfg=cfg)
    elif args.mode == 'train_emotion_ml':
        train_emotion_ml(test_size=args.test_size)
    elif args.mode == 'train_emotion_dl':
        cfg = None
        if args.cfg:
            try:
                cfg = DLParamConfig.load(args.cfg)
            except Exception as e:
                print(f"Warning: could not load cfg {args.cfg}: {e}")
                cfg = None
        train_emotion_dl(test_size=args.test_size, cfg=cfg)
    elif args.mode == 'train_speaker_ml':
        train_speaker_ml(test_size=args.test_size)
    elif args.mode == 'train_speaker_dl':
        cfg = None
        if args.cfg:
            try:
                cfg = DLParamConfig.load(args.cfg)
            except Exception as e:
                print(f"Warning: could not load cfg {args.cfg}: {e}")
                cfg = None
        train_speaker_dl(test_size=args.test_size, cfg=cfg)


if __name__ == '__main__':
    main()
