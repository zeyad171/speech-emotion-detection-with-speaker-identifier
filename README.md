# Speech Emotion Detection Project

A machine learning system for detecting emotions from human speech using multiple publicly available datasets.

## Project Overview

This project implements a complete pipeline for speech emotion detection, including:
- Data loading and preprocessing from multiple datasets (TESS, SAVEE, RAVDESS, CREMA-D)
- Feature extraction (MFCCs, chroma, pitch, energy, spectral features)
- Multiple ML/DL models (Logistic Regression, Random Forest, SVM, XGBoost, CNN, LSTM)
- Model evaluation with comprehensive metrics
- Web interface for real-time emotion detection

## Datasets

The project uses four emotion datasets located in the `dataset/` folder:
- **TESS**: 2,800 files (7 emotions, 200 files each)
- **SAVEE**: ~480 files (4 speakers, 7 emotions)
- **RAVDESS**: ~1,440 files (24 actors, 8 emotions)
- **CREMA-D**: ~7,442 files (91 actors, 6 emotions)

**Note:** All datasets should be placed in the `dataset/` folder with their respective names (Tess, Savee, Ravdess, Crema).

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

   **For GPU support (recommended):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   See [INSTALL_GPU_PYTORCH.md](INSTALL_GPU_PYTORCH.md) for detailed GPU setup instructions.

3. (Optional) Check GPU availability:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Start

### 1. Explore Datasets

First, explore the datasets to understand the data:
```bash
python main.py --mode explore
```

This will:
- Load all datasets (TESS, SAVEE, RAVDESS, CREMA-D)
- Show dataset statistics
- Display emotion distribution

### 2. Train Models

Train emotion detection models:
```bash
python main.py --mode train
```

This will:
- Preprocess all audio files
- Extract features (MFCCs, chroma, pitch, energy, etc.) - **65 features total**
- Prepare spectrograms and sequences for deep learning models
- Train multiple ML models (Logistic Regression, Random Forest, SVM, XGBoost)
- Train deep learning models (CNN, LSTM)
- Evaluate all models and save results
- Save trained models to `models/` directory

**Note:** 
- Training time: ~30-60 minutes with GPU, ~2-4 hours on CPU
- GPU acceleration provides 10-50× speedup (see [INSTALL_GPU_PYTORCH.md](INSTALL_GPU_PYTORCH.md))

### 3. Run Web Interface

Launch the web interface:
```bash
streamlit run app/app.py
```

Then:
- Open your browser to the URL shown (usually http://localhost:8501)
- Upload an audio file
- See emotion prediction with confidence scores

### 4. Optional: Train Speaker Identification

Train the bonus speaker identification models:
```bash
# Train ML models (Logistic Regression, Random Forest, SVM, XGBoost)
python src/models/speaker_ml.py

# Train DL models (CNN, LSTM, RNN)
python src/models/speaker_dl.py
```

## Detailed Usage

See the [Quick Start](#quick-start) section above for basic usage. Additional details:

### Feature Extraction

The system extracts **65 features** from each audio file:
- **MFCCs**: 13 coefficients × 2 (mean + std) = 26 features
- **Chroma**: 12 features × 2 (mean + std) = 24 features
- **Pitch**: 4 features (mean, std, max, min)
- **Energy**: 5 features (RMS mean/std/max, ZCR mean/std)
- **Spectral**: 6 features (centroid, rolloff, bandwidth - each with mean/std)

Features are cached to avoid recomputation during training.

## Project Structure

```
project/
├── dataset/                 # Dataset folders (TESS, SAVEE, RAVDESS, CREMA)
├── src/
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── feature_extraction.py  # Feature extraction
│   ├── models/
│   │   ├── ml_models.py    # Traditional ML models
│   │   ├── dl_models.py   # Deep learning models
│   │   ├── speaker_ml.py   # Speaker identification ML models
│   │   └── speaker_dl.py   # Speaker identification DL models
│   ├── evaluation.py       # Evaluation metrics
│   └── utils.py            # Utility functions
├── models/                  # Saved models and features
├── results/                 # Evaluation results and plots
├── app/
│   └── app.py             # Streamlit web interface
├── notebooks/              # Jupyter notebooks for exploration
├── main.py                 # Main entry point (explore & train modes)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Features

### Emotion Classes
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

### Extracted Features
- **MFCCs**: Mel-frequency cepstral coefficients
- **Chroma**: Pitch class profile
- **Pitch**: Fundamental frequency
- **Energy**: RMS energy, zero-crossing rate
- **Spectral**: Centroid, rolloff, bandwidth
- **Mel Spectrogram**: For deep learning models

### Models
**Machine Learning Models:**
- Logistic Regression (baseline)
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

**Deep Learning Models:**
- CNN (for spectrograms) - Integrated in training pipeline
- LSTM (for sequences) - Integrated in training pipeline
- RNN (for sequences) - Integrated in training pipeline

## Evaluation Metrics

Models are evaluated using:
- Accuracy
- F1-Score (macro, micro, weighted)
- Per-class F1-scores
- Confusion Matrix
- Classification Report

## Results

Evaluation results are saved in the `results/` directory:
- Confusion matrices (PNG images)
- Evaluation reports (text files)
- Model performance metrics

## Web Interface Features

- Upload audio files (WAV, MP3, FLAC, M4A)
- View audio waveform
- See emotion prediction with confidence
- View probability distribution across all emotions

## Requirements

- Python 3.10+ (recommended)
- See `requirements.txt` for full list of dependencies
- **Deep Learning Framework**: PyTorch (with CUDA 12.4 for GPU support)
- **GPU**: NVIDIA GPU recommended for faster training (RTX 20xx series or newer)

## Troubleshooting

### Import Errors
If you get import errors, make sure all packages are installed:
```bash
pip install -r requirements.txt
```

### Dataset Not Found
Make sure your dataset folders are in the `dataset/` folder:
- `dataset/Tess/` or `dataset/TESS/`
- `dataset/Savee/` or `dataset/SAVEE/`
- `dataset/Ravdess/` or `dataset/RAVDESS/`
- `dataset/Crema/` or `dataset/CREMA/`

### Memory Issues
If you run out of memory:
- Process datasets one at a time
- Reduce batch size in training
- Use feature caching (already implemented)

### GPU Not Detected
If GPU is not detected during training:
- Check NVIDIA drivers: `nvidia-smi`
- Verify PyTorch CUDA: `python -c "import torch; print('CUDA available:', torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA: See [INSTALL_GPU_PYTORCH.md](INSTALL_GPU_PYTORCH.md)

## Notes

- All code is original work (no plagiarism)
- Models are trained on combined datasets for better generalization
- Features are cached to avoid recomputation
- Best model is automatically selected based on accuracy
- **65 features** are extracted per audio file (see Feature Extraction section)

## Future Enhancements

- Real-time audio recording in web interface
- Speaker identification (bonus feature)
- Advanced deep learning models (Transformers)
- Model ensemble methods
- Real-time emotion detection from microphone

## License

This project is for educational purposes.

## Authors

Speech Emotion Detection Team

