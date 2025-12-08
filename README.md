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

### Prerequisites
- Python 3.10 or higher (recommended: 3.10 or 3.11)
- pip (Python package manager)
- (Optional) NVIDIA GPU with CUDA 12.4 support for faster training

### Installation Steps

1. **Clone or download this repository**

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **For GPU support (recommended for deep learning):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   See [INSTALL_GPU_PYTORCH.md](INSTALL_GPU_PYTORCH.md) for detailed GPU setup instructions.

4. **(Optional) Verify installation:**
   ```bash
   # Check GPU availability
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   
   # Verify key libraries
   python -c "import librosa, sklearn, streamlit; print('All libraries loaded successfully')"
   ```

5. **(If needed) Handle LLVM/NumPy issues:**
   If you encounter LLVM symbol errors, the code handles this automatically. See [Troubleshooting](#troubleshooting) for details.

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

After a DL speaker training run completes, a consolidated metadata file `models/speaker_dl_metadata.json` is generated. It includes:
- Trained model types and best model
- Speaker index → label mapping
- Per-model configuration (dimensions, num_classes)
- Finetune configuration blocks (`*_finetune`)
- Core run parameters (epochs, batch_size, test/validation sizes)
Use this JSON for downstream analysis or reproducible inference setups.
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
│   ├── feature_extraction.py  # Feature extraction (65 features)
│   ├── evaluation.py       # Model evaluation metrics
│   ├── utils.py            # Utility functions (caching, saving/loading)
│   ├── visualization.py    # Visualization helpers (plots, comparisons)
│   └── models/
│       ├── emotion_ml.py   # Emotion detection ML models + training
│       ├── emotion_dl.py   # Emotion detection DL models + training
│       ├── speaker_ml.py   # Speaker identification ML + training
│       ├── speaker_dl.py   # Speaker identification DL + training
│       └── dl_param_config.py  # DL parameter configuration helper
├── scripts/
│   └── generate_dl_config.py  # DL config generator script
├── models/                  # Saved models, features, and cache files
├── results/                 # Evaluation results and plots
├── app/
│   └── app.py             # Streamlit web interface (multi-tab)
├── notebooks/              # Jupyter notebooks for exploration
├── main.py                 # Main CLI entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file (user documentation)
└── PROJECT_DOCUMENTATION.md  # Technical documentation
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
- Logistic Regression (baseline with L2 regularization)
- Random Forest (100-200 estimators, optimized for speed)
- Support Vector Machine (SVM with RBF kernel, probability estimates enabled)
- XGBoost (gradient boosting with hyperparameter tuning)

**Deep Learning Models (PyTorch):**
- **CNN** - 4 convolutional layers (32→64→128→256 filters) with batch normalization, for mel spectrograms (128×128)
- **LSTM** - Bidirectional multi-layer LSTM (256→128 hidden units) for MFCC sequences
- **RNN** - Multi-layer RNN (256→128 hidden units) for MFCC sequences, simpler and faster than LSTM

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

### LLVM Symbol Error (NumPy/Intel MKL)

**Error:** `LLVM ERROR: Symbol not found: __svml_cosf8_ha` or similar LLVM symbol errors.

**Cause:** This is a known issue with NumPy's Intel MKL library and certain CPU features (AVX-512).

**Solution:** The project automatically handles this by setting environment variables before importing NumPy in `src/feature_extraction.py`. However, if you still encounter issues:

```bash
# Windows PowerShell
$env:NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX"
$env:OPENBLAS_NUM_THREADS="1"
python main.py --mode train

# Linux/Mac
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX"
export OPENBLAS_NUM_THREADS="1"
python main.py --mode train
```

**Note:** These environment variables are automatically set in the code, but you can set them manually if needed. This may slightly reduce numerical performance but ensures stability.

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

## Recent Changes (Today)

- Added an editable DL configuration helper `DLParamConfig` at `src/models/dl_param_config.py`.
- Added a simple, hand-editable HPO runner script `scripts/run_manual_dl_hyperparams.py` (edit the `MANUAL_CONFIGS` dictionary, then run the script or use `main.py --cfg ...`).
- Added a lightweight HPO helper module `src/models/dl_hpo.py` (Optuna wrapper + random-search fallback) — optional, not required for manual tuning.
- Wired `DLParamConfig` into DL trainers (`src/models/emotion_dl.py` and `src/models/speaker_dl.py`) so you can pass a JSON config via `main.py --cfg`.
- Added overfitting-detection visualizations and integrated them into the ML trainers. New helper: `src/visualization.py::plot_overfitting_detection()` which writes `results/overfitting_*.png`.
- Added JSON evaluation exporters (`save_eval_json`) and collectors so visualization helpers can load `results/evaluation_*.json` produced by training.
- Added CLI `--test_size` support in `main.py` and `--cfg` to pass DL configs to trainers.

These changes were implemented to make manual DL hyperparameter experimentation reproducible, script-driven, and easy to run from the existing `main.py` entrypoint.

### Code changes (files modified today)

- `src/visualization.py`:
   - Added multiple plotting helpers used by both ML and DL flows: `plot_model_comparison()`, `plot_confusion_matrix_grid()`, `plot_calibration_curves()`, `plot_per_class_heatmap()`, `plot_embeddings_tsne()` and speaker-oriented visualizers like `plot_speaker_probabilities()` and `plot_feature_comparison()`.
   - Implemented `plot_overfitting_detection(metrics, histories, results_dir)` which writes ML train-vs-test bar charts and DL learning curves (`overfitting_train_test.png`, `overfitting_loss_curves.png`, `overfitting_accuracy_curves.png`).
   - Fixed seaborn heatmap annotation formatting to safely handle float confusion matrices (`fmt = '.2f' if normalize else '.0f'`) to prevent annotation errors.

- `src/models/emotion_ml.py`:
   - Added `save_eval_json(results, model_name, labels, results_dir)` to export evaluation payloads as JSON (`results/evaluation_{model}.json`) for downstream visualization.
   - Added `collect_evaluations_for_visualization(results_dir)` to scan those JSON files and build compact metric/confusion-matrix structures used by the visualizer.
   - Added `retrain_on_full_data(model_name, X, y, save_suffix)` to retrain the selected model on the entire dataset after HPO and save final model/scaler/metadata.
   - Integrated calls to `save_eval_json()` after each model evaluation and to `viz.plot_overfitting_detection(...)` at the end of the training pipeline.

- `src/models/speaker_ml.py`:
   - Added JSON exporters `save_eval_json()` and `collect_evaluations_for_visualization()` analogous to the emotion ML trainer so speaker evaluation artifacts are consumable by the visualizer.
   - Introduced conservative HPO support (`param_grids` + `RandomizedSearchCV` / `GridSearchCV`) in `train_model()` with best-params saving to `models/best_hyperparameters_speaker.json`.
   - Added safety checks and fixes (e.g., top-k accuracy handling when class count < k) and `retrain_on_full_data()` for speaker models.
   - Integrated overfitting visualization call `viz.plot_overfitting_detection(metrics=..., histories=None, results_dir='results')` after training.

These per-file notes document the concrete helper functions, HPO additions, JSON output conventions, and visualization wiring added today.

## How to run using a saved DL config

1. Create or edit a DL config JSON (example path `models/my_emotion_cfg.json`). You can either edit the JSON directly or create it programmatically using `DLParamConfig`:

```powershell
.\.myenv\Scripts\Activate.ps1

$python_script = @'
from src.models.dl_param_config import DLParamConfig
cfg = DLParamConfig()
cfg.update_from_dict({
   'epochs': 60,
   'batch_size': 64,
   'learning_rate': 5e-4,
   'dropout': 0.35,
})
cfg.save('models/my_emotion_cfg.json')
print('Saved config to models/my_emotion_cfg.json')
'@

python -c $python_script
```

2. Run emotion DL training with that config (example):

```powershell
.\.myenv\Scripts\Activate.ps1
python .\main.py --mode train_emotion_dl --cfg models/my_emotion_cfg.json --test_size 0.2
```

Generating DL JSON configs (script)
----------------------------------

You can also create config JSONs using the helper script `scripts/generate_dl_config.py`. This is useful for quickly producing configs for `emotion` or `speaker` runs and for applying small overrides from the command line.

Examples (PowerShell):

Step 1 — Create a combined config (recommended)

This produces a single JSON with general defaults plus the three per-model blocks `cnn_finetune`, `lstm_finetune`, and `rnn_finetune`.

```powershell
python .\scripts\generate_dl_config.py --target both --output models/my_full_dl_cfg.json
```

Tip: You can also use `--target both` (it only affects the default file name). With no `--output`, this writes `models/my_full_dl_cfg.json`:

```powershell
python .\scripts\generate_dl_config.py --target both
```

Step 2 — Train using the generated file

Emotion DL (trains CNN, LSTM, RNN; each uses its own `*_finetune` block):

```powershell
python .\main.py --mode train_emotion_dl --cfg models/my_full_dl_cfg.json --test_size 0.2
```

Speaker DL (same config file works):

```powershell
python .\main.py --mode train_speaker_dl --cfg models/my_full_dl_cfg.json --test_size 0.2
```

Optional — Include HPO candidates for one model (adds `hpo_space` to the JSON):

```powershell
python .\scripts\generate_dl_config.py --target emotion --model-type cnn --include-hpo --output models/my_emotion_cfg.json
```

Optional — Override nested keys from the CLI (dot notation):

```powershell
python .\scripts\generate_dl_config.py --target emotion --output models/my_full_dl_cfg.json `
   --set cnn_finetune.head_lr=0.0005 `
   --set lstm_finetune.finetune_head_epochs=0
```

Create a default CNN emotion config and include the HPO search space in the JSON:

```powershell
python .\scripts\generate_dl_config.py --target emotion --model-type cnn --include-hpo --output models/my_emotion_cfg.json
```

Create a speaker LSTM config and override nested finetune field `cnn_finetune.head_lr`:

```powershell
python .\scripts\generate_dl_config.py --target speaker --model-type lstm --set cnn_finetune.head_lr=0.0005 --output models/my_speaker_cfg.json
```

Notes:
- Use `--set key=value` to override values; nested keys use dot notation (e.g. `cnn_finetune.head_lr=0.0005`).
- Use `--include-hpo` to include the `hpo_space` candidates (useful for HPO runners).
- The generated JSON is compatible with `main.py --cfg` and the DL trainers which will save a copy to `models/dl_config_emotion.json` or `models/dl_config_speaker.json` when training.
- The `model_arch` field is not required and is omitted; trainers read the per-model blocks directly.

3. Run speaker DL training with a config (example):

```powershell
.\.myenv\Scripts\Activate.ps1
python .\main.py --mode train_speaker_dl --cfg models/my_speaker_cfg.json --test_size 0.2
```

4. Quick manual hyperparam script (edit `scripts/run_manual_dl_hyperparams.py` `MANUAL_CONFIGS` then run):

```powershell
.\.myenv\Scripts\Activate.ps1
python .\scripts\run_manual_dl_hyperparams.py --target emotion --model_type cnn
```

5. Where outputs land:
- Trained DL models: `models/{model}.pth` or `models/{model}_best.pth`
- Saved DL config used for the run: `models/dl_config_emotion.json` or `models/dl_config_speaker.json`
- Evaluation JSONs: `results/evaluation_{model}.json`
- Visualization images: `results/overfitting_train_test.png`, `results/overfitting_loss_curves.png`, `results/overfitting_accuracy_curves.png`

If you'd like, I can also wire additional `cfg` fields (learning rate, optimizer, weight decay) directly into the trainer optimizer/scheduler logic so the JSON controls the full training behavior. Say the word and I'll implement that.

