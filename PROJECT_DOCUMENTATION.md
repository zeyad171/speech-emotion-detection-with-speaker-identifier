# Speech Emotion Detection Project - Complete Documentation

## Project Overview

This is a comprehensive machine learning system for analyzing human speech with two main capabilities:

1. **Emotion Detection**: Detects emotions (angry, disgust, fear, happy, neutral, sad, surprise) from speech
2. **Speaker Identification**: Identifies speakers from voice characteristics

The project implements complete pipelines from data loading to model deployment, supporting multiple datasets and both traditional ML and deep learning approaches. It features:

- **Multiple ML Models**: Logistic Regression, Random Forest, SVM, XGBoost with hyperparameter tuning
- **Deep Learning Models**: CNN, LSTM, and RNN architectures with GPU acceleration
- **Comprehensive Web Interface**: Multi-tab Streamlit app with detailed analysis and visualizations
- **Advanced Features**: Model comparison, speaker metadata, similarity analysis, and rich visualizations

## Project Structure

```
project/
├── dataset/                    # Audio datasets (TESS, SAVEE, RAVDESS, CREMA-D)
│   ├── Tess/                  # 2,800 files (7 emotions)
│   ├── Savee/                 # 480 files (4 speakers, 7 emotions)
│   ├── Ravdess/               # 1,440 files (24 actors, 8 emotions)
│   └── Crema/                 # 7,442 files (91 actors, 6 emotions)
│
├── src/                       # Core source code
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── feature_extraction.py  # Feature extraction (65 features)
│   ├── evaluation.py          # Model evaluation metrics
│   ├── utils.py               # Utility functions
│   ├── visualization.py        # Visualization helpers for speaker comparison
│   └── models/
│       ├── emotion_ml.py      # Emotion detection ML models and training (LR, RF, SVM, XGBoost)
│       ├── emotion_dl.py      # Emotion detection DL models and training (CNN, LSTM, RNN)
│       ├── speaker_ml.py      # Speaker identification ML models and training (LR, RF, SVM, XGBoost)
│       └── speaker_dl.py      # Speaker identification DL models and training (CNN, LSTM, RNN)
│
├── app/
│   └── app.py                # Streamlit web interface
│
├── models/                    # Saved models and features (generated)
├── results/                   # Evaluation results (generated)
├── notebooks/                 # Jupyter notebooks (optional)
│
├── main.py                    # Main entry point with CLI
├── requirements.txt          # Python dependencies
└── README.md                 # User documentation
```

## Core Components

### 1. Data Loading (`src/data_loader.py`)

**Class:** `EmotionDatasetLoader`

**Key Methods:**
- `load_tess_dataset()` - Loads TESS dataset (folder-based structure)
- `load_savee_dataset()` - Loads SAVEE dataset (filename-based)
- `load_ravdess_dataset()` - Loads RAVDESS dataset (encoded filenames)
- `load_crema_dataset()` - Loads CREMA-D dataset (encoded filenames)
- `load_all_datasets()` - Loads all available datasets
- `preprocess_audio()` - Normalizes volume, trims silence, resamples to 22.05kHz
- `get_dataset_statistics()` - Calculates dataset statistics
- `explore_datasets()` - Comprehensive dataset exploration

**Emotion Mapping:**
- Standardizes emotions across datasets: angry, disgust, fear, happy, neutral, sad, surprise
- Handles variations: 'anger'→'angry', 'happiness'→'happy', 'surprised'→'surprise', etc.

**Preprocessing:**
- Sample rate: 22,050 Hz (standardized)
- Volume normalization: Amplitude normalization
- Silence trimming: Top DB = 20

### 2. Feature Extraction (`src/feature_extraction.py`)

**Class:** `FeatureExtractor`

**Total Features: 65**

**Feature Breakdown:**
1. **MFCCs (26 features)**
   - 13 coefficients × 2 (mean + std)
   - Extracted using librosa.feature.mfcc()

2. **Chroma (24 features)**
   - 12 pitch classes × 2 (mean + std)
   - Extracted using librosa.feature.chroma_stft()

3. **Pitch (4 features)**
   - Mean, std, max, min of fundamental frequency
   - Extracted using librosa.piptrack()

4. **Energy (5 features)**
   - RMS: mean, std, max
   - Zero Crossing Rate: mean, std
   - Extracted using librosa.feature.rms() and librosa.feature.zero_crossing_rate()

5. **Spectral (6 features)**
   - Spectral Centroid: mean, std
   - Spectral Rolloff: mean, std
   - Spectral Bandwidth: mean, std

**Key Methods:**
- `extract_all_features()` - Extracts all 65 features from audio
- `extract_features_batch()` - Batch processing for multiple audio files
- `extract_mel_spectrogram()` - For deep learning models (optional)

**Feature Caching:**
- Features saved to `models/extracted_features.npz` to avoid recomputation

### 3. Emotion Detection - ML Models (`src/models/emotion_ml.py`)

**Class:** `EmotionMLTrainer`

**Models Implemented:**
1. **Logistic Regression** - Baseline model with regularization tuning (C parameter)
2. **Random Forest** - 200-300 estimators (tunable), max_depth tuning, parallel processing
3. **Support Vector Machine (SVM)** - RBF kernel, probability=True, hyperparameter tuning (C, gamma)
4. **XGBoost** - Gradient boosting with hyperparameter tuning (learning_rate, max_depth, n_estimators, subsample)

**Hyperparameter Tuning:**
- **GridSearchCV**: Used for Logistic Regression and Random Forest (smaller parameter spaces)
- **RandomizedSearchCV**: Used for SVM and XGBoost (larger parameter spaces, faster)
- Tuning performed during training (optional, can be disabled)
- Best hyperparameters saved to `models/best_hyperparameters.json`
- Cross-validation: 5-fold CV for each model (3-fold for Random Forest to optimize training time)
- **Platform-aware**: Automatically uses `n_jobs=1` on Windows, `n_jobs=-1` on Linux/Mac to prevent multiprocessing issues

**Training Process:**
- Data split: 80% train, 20% test (stratified)
- Feature scaling: StandardScaler (fitted on training data)
- Cross-validation: 5-fold CV for each model (3-fold for Random Forest)
- Model persistence: Saved as .pkl files in `models/` directory
- Best model selection: Automatically selects best performing model based on CV accuracy
- **Random Forest Optimization**: Reduced hyperparameter search space and CV folds for faster training:
  - `n_estimators`: [100, 200] (reduced from [100, 200, 300])
  - `max_depth`: [15, 20, 25] (reduced from [10, 20, 30, None])
  - `n_iter`: 8 (reduced from 20)
  - `cv`: 2 folds (reduced from 3)

**Key Methods:**
- `train_all_models(tune_hyperparameters=True)` - Trains all 4 ML models with optional hyperparameter tuning
- `train_model()` - Trains individual model with CV and optional hyperparameter tuning
- `save_model()` - Saves trained model
- `load_model()` - Loads saved model
- `prepare_data()` - Prepares and scales data for training
- `train()` - **Complete self-contained training pipeline** (data loading, preprocessing, feature extraction, training, evaluation, saving)

**Self-Contained Training:**
- Each model file includes a `train()` function that handles the complete pipeline
- Can be run independently: `python src/models/emotion_ml.py`
- Includes data loading, preprocessing (with caching), feature extraction (with caching), training, evaluation, and model saving
- Progress indicators and detailed logging throughout the process

### 4. Emotion Detection - Deep Learning Models (`src/models/emotion_dl.py`)

**Framework:** PyTorch

**Class:** `EmotionDLTrainer`

**Models Implemented:**
1. **CNN** (`CNNModel`) - Enhanced architecture with 4 convolutional layers (32, 64, 128, 256 filters), batch normalization, deeper fully connected layers
2. **LSTM** (`LSTMModel`) - Enhanced architecture with multi-layer LSTM (256, 128 units), deeper fully connected layers
3. **RNN** (`RNNModel`) - Multi-layer RNN architecture with 2-layer RNN (256 units) → 1-layer RNN (128 units), deeper fully connected layers

**Architecture Details:**
- **CNN**: 
  - Conv2d(32) + BatchNorm → MaxPool2d → Conv2d(64) + BatchNorm → MaxPool2d → Conv2d(128) + BatchNorm → MaxPool2d → Conv2d(256) + BatchNorm → MaxPool2d → Dropout → Flatten → Linear(256) → Dropout → Linear(128) → Dropout → Linear(num_classes)
  - Input: 128×128 spectrograms (channels-first format for PyTorch)
  - Output size after 4 pools: 8×8 (256 channels = 16,384 features)
  
- **LSTM**: 
  - LSTM(256, num_layers=2, bidirectional=True) → Dropout → LSTM(128, bidirectional=True) → Dropout → Linear(256) → Dropout → Linear(64) → Dropout → Linear(num_classes)
  - Input: Variable-length MFCC sequences (padded/truncated to max_length=200)
  - Uses Bidirectional LSTMs for better context awareness (forward and backward processing)
  - Uses concatenated hidden states from both directions
  
- **RNN**: 
  - RNN(256, num_layers=2, dropout=0.3) → Dropout(0.4) → RNN(128, num_layers=1) → Dropout(0.4) → Linear(128) → Dropout(0.3) → Linear(64) → Dropout(0.2) → Linear(num_classes)
  - Input: Variable-length MFCC sequences (padded/truncated to max_length=200)
  - Uses output from last timestep of final RNN layer
  - Architecture: 2-layer RNN (256 hidden units) → 1-layer RNN (128 hidden units) → 3 fully connected layers (128 → 64 → num_classes)

**Training Features:**
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Mixed Precision Training**: Uses `torch.cuda.amp` for bfloat16 training (faster, lower memory)
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler (reduces LR by factor of 0.5 when validation loss plateaus, patience=5)
- **Optimizer**: AdamW with weight decay (0.01) for better regularization
- **Model Checkpointing**: Saves best model based on val_loss (`.pth` format)
- **Label Encoding**: Converts string labels to integers using sklearn LabelEncoder
- **Data Split**: Train/Val/Test (70%/10%/20%)
- **DataLoader Optimization**: `pin_memory=True`, `num_workers=0` (Windows compatibility)
- **Note**: Strict early stopping implemented (stops if accuracy doesn't improve by 0.2% for 30 epochs, or if performance degrades significantly)

**Key Methods:**
- `configure_gpu()` - Configures PyTorch GPU settings (cuDNN benchmark, device selection)
- `train_model()` - Custom training loop with mixed precision, checkpointing
- `train_all_models()` - Trains all DL models (CNN, LSTM, RNN)
- `prepare_data()` - Prepares data with label encoding
- `prepare_spectrograms()` - Prepares mel spectrograms for CNN
- `prepare_sequences()` - Prepares MFCC sequences for LSTM/RNN
- `save_model()` / `load_model()` - Saves/loads models in PyTorch format (`.pth`)
- `train()` - **Complete self-contained training pipeline** (data loading, preprocessing, DL data extraction, training, evaluation, saving)

**GPU Configuration:**
- Automatic GPU detection on module import
- cuDNN benchmark enabled for faster training
- Mixed precision training (autocast + GradScaler)
- Device management (CPU fallback if GPU unavailable)

**Note:** DL models require different input format (spectrograms/sequences) than ML models (feature vectors). Spectrograms are converted from channels-last (N, H, W, C) to channels-first (N, C, H, W) for PyTorch. CNN uses spectrograms, while LSTM and RNN use MFCC sequences.

**Self-Contained Training:**
- Each model file includes a `train()` function that handles the complete pipeline
- Can be run independently: `python src/models/emotion_dl.py`
- Includes data loading, preprocessing (with caching), DL data extraction (with caching), training, evaluation, and model saving
- Progress indicators during evaluation with batched predictions for large test sets
- Detailed logging for each step (predictions, metrics computation, saving)

### 5. Model Evaluation (`src/evaluation.py`)

### 6. Model Evaluation (`src/evaluation.py`)

**Class:** `ModelEvaluator`

**Metrics Calculated:**
- Accuracy
- F1-Score (macro, micro, weighted)
- Per-class F1-scores
- Confusion Matrix
- Classification Report

**Outputs:**
- Confusion matrix plots (PNG) saved to `results/`
- Evaluation reports (TXT) saved to `results/`
- Speaker Identification results use `_speaker` suffix (e.g., `evaluation_CNN_speaker.txt`) to distinguish from emotion models
- Console output with formatted results

**Key Methods:**
- `evaluate_classification()` - Calculates all metrics
- `plot_confusion_matrix()` - Creates and saves confusion matrix
- `print_evaluation_results()` - Prints formatted results
- `save_results()` - Saves evaluation to file

### 7. Web Interface (`app/app.py`)

**Framework:** Streamlit

**Structure:** Multi-tab interface with two main sections:
1. **Emotion Detection Tab** - Original emotion detection functionality
2. **Speaker Identification Tab** - New comprehensive speaker identification interface

#### Emotion Detection Tab

**Features:**
- Audio file upload (WAV, MP3, FLAC, M4A)
- Real-time emotion prediction
- Confidence scores
- Probability distribution visualization
- Audio waveform display

**Model Loading:**
- Loads best trained model (default: Random Forest)
- Loads feature scaler
- Caches model loading for performance

**Prediction Flow:**
1. User uploads audio file
2. Audio is preprocessed (normalize, trim)
3. Features are extracted (65 features)
4. Features are scaled using saved scaler
5. Model predicts emotion
6. Results displayed with confidence and probabilities

#### Speaker Identification Tab

**Features:**
- **Model Selection:**
  - ML Models: Choose from Random Forest, Logistic Regression, SVM, XGBoost
  - DL Models: Choose from CNN, LSTM, or RNN
  - Best Model: Automatically uses best performing model
  
- **Prediction Display:**
  - Predicted speaker ID with confidence score
  - Warning for low confidence predictions (<50%)
  - Top-K most likely speakers (configurable, 3-10 speakers)
  - Probability distribution bar chart
  
- **Full Data of Predicted Speaker:**
  - **Training Data Information:**
    - Number of training samples for the speaker
    - Dataset source (TESS, SAVEE, RAVDESS, CREMA-D)
    - Total dataset statistics
    
  - **Feature Statistics:**
    - Mean, standard deviation, min, max for all 65 features
    - Summary statistics table
    
  - **Model Confidence Scores:**
    - Confidence scores from all ML models (if available)
    - Comparison across different models
  
- **Comparison Section:**
  - **Probability Distribution:**
    - Bar chart showing probability for all speakers
    - Top-K speakers highlighted
    
  - **Feature Comparison:**
    - Grouped bar chart comparing predicted speaker vs. top similar speakers
    - Shows top 10 most differentiating features
    
  - **Top-K Similar Speakers:**
    - Similarity scores using cosine similarity
    - Visual bar chart of similar speakers
    - Feature comparison with most similar speaker

**Model Loading:**
- Loads all ML models for speaker identification
- Loads DL models (CNN, LSTM, RNN) if available
- Loads speaker metadata for analysis
- Caches model loading for performance

**Error Handling:**
- Checks for model availability before allowing predictions
- Displays helpful messages if models not trained
- Handles low confidence predictions with warnings
- Graceful fallback if specific model unavailable

**Auto-Launch Feature:**
- When `app.py` is executed directly with Python (e.g., `python app/app.py`), it automatically launches Streamlit
- This allows running the app from IDE run buttons without manually typing `streamlit run`
- The default run command is configured to use `streamlit run` automatically

**Prediction Flow:**
1. User selects model type and specific model (optional)
2. User uploads audio file
3. Audio is preprocessed (normalize, trim)
4. Features are extracted (65 features)
5. Model predicts speaker (ML or DL based on selection)
6. Results displayed with comprehensive analysis:
   - Prediction and confidence
   - Top-K speakers
   - Full speaker data
   - Comparison visualizations

### 8. Speaker Identification - ML Models (`src/models/speaker_ml.py`)

**Class:** `SpeakerMLTrainer`

**Purpose:** Identify speakers from voice using multiple ML algorithms

**Models Implemented:**
1. **Logistic Regression** - Baseline model
2. **Random Forest** - 200 estimators (default), parallel processing
3. **Support Vector Machine (SVM)** - RBF kernel, probability=True
4. **XGBoost** - Gradient boosting, parallel processing

**Features:**
- Uses same 65 audio features as emotion detection
- Extracts speaker IDs from filenames based on dataset (TESS, SAVEE, RAVDESS, CREMA-D)
- Evaluates using accuracy and top-k accuracy (top-3, top-5)
- Automatic best model selection based on accuracy
- Cross-validation for each model

**Key Methods:**
- `train_all_models()` - Trains all ML models and selects best performer
- `train_model()` - Trains individual model with evaluation
- `predict()` - Predicts speaker from features (can specify model or use best)
- `save_model()` - Saves all models or specific model
- `load_model()` - Loads all models or specific model
- `train()` - **Complete self-contained training pipeline** (data loading, speaker ID extraction, preprocessing, feature extraction, training, evaluation, saving)
- `extract_speaker_id_from_filename()` - Helper function to extract speaker IDs from filenames

**Model Storage:**
- Individual models: `models/speaker_{model_name}.pkl`
- Metadata: `models/speaker_models_metadata.pkl`
- Best model automatically selected and saved

**Self-Contained Training:**
- Can be run independently: `python src/models/speaker_ml.py`
- Includes speaker ID extraction from filenames, data loading, preprocessing (with caching), feature extraction (with caching), training, evaluation, and model saving

### 9. Speaker Identification - DL Models (`src/models/speaker_dl.py`)

**Class:** `SpeakerDLTrainer`

**Framework:** PyTorch

**Models Implemented:**
1. **CNN** (`SpeakerCNNModel`) - Similar architecture to emotion detection CNN, adapted for speaker classification
2. **LSTM** (`SpeakerLSTMModel`) - Similar architecture to emotion detection LSTM, adapted for speaker classification
3. **RNN** (`SpeakerRNNModel`) - Multi-layer RNN architecture adapted for speaker classification

**Architecture Details:**
- **CNN**: 4 convolutional layers with batch normalization, deeper fully connected layers
- **LSTM**: Multi-layer LSTM with enhanced hidden sizes
- **RNN**: 2-layer RNN (128 hidden units) → 1-layer RNN (64 hidden units) → 2 fully connected layers (64 → num_classes)
  - Input: Variable-length MFCC sequences (padded/truncated to max_length=200)
  - Uses output from last timestep of final RNN layer
  - Dropout regularization (0.3) applied between layers

**Training Features:**
- Same GPU acceleration and mixed precision training as emotion detection DL models
- Learning rate scheduling with ReduceLROnPlateau
- Model checkpointing (saves best model based on validation loss)
- Automatic model selection based on validation accuracy
- **Note**: Strict early stopping implemented (stops if accuracy doesn't improve by 0.2% for 30 epochs, or if performance degrades significantly)

**Key Methods:**
- `train_all_models()` - Trains CNN, LSTM, and RNN models
- `train_model()` - Trains individual model (CNN, LSTM, or RNN)
- `predict()` - Predicts speaker from input data (supports all model types)
- `prepare_spectrograms()` - Prepares mel spectrograms for CNN
- `prepare_sequences()` - Prepares MFCC sequences for LSTM and RNN
- `save_model()` / `load_model()` - Saves/loads models in PyTorch format (`.pth`)
- `train()` - **Complete self-contained training pipeline** (data loading, speaker ID extraction, preprocessing, DL data extraction, training, evaluation, saving)

**Self-Contained Training:**
- Can be run independently: `python src/models/speaker_dl.py`
- Includes speaker ID extraction from filenames, data loading, preprocessing (with caching), DL data extraction (with caching), training, evaluation, and model saving
- Progress indicators during evaluation with batched predictions for large test sets

### 10. Visualization Helpers (`src/visualization.py`)

**Purpose:** Visualization functions for speaker comparison and analysis

**Functions:**
1. **`plot_speaker_probabilities()`** - Bar chart showing probability distribution across all speakers
2. **`plot_feature_comparison()`** - Grouped bar chart comparing features between speakers
3. **`plot_radar_chart()`** - Radar/polar chart for multi-dimensional feature comparison
4. **`plot_similarity_scores()`** - Bar chart showing top-K similar speakers with similarity scores
5. **`calculate_speaker_similarity()`** - Calculates similarity between two speaker feature vectors (cosine or euclidean)
6. **`get_top_k_similar_speakers()`** - Finds top-K most similar speakers to query features
7. **`load_speaker_metadata()`** - Loads speaker metadata from JSON file
8. **`get_speaker_feature_stats()`** - Gets feature statistics for a specific speaker

**Similarity Methods:**
- **Cosine Similarity**: Measures angle between feature vectors (0-1 range)
- **Euclidean Distance**: Measures distance between feature vectors (inverted and normalized)

### 11. Speaker Metadata Storage

**File:** `models/speaker_metadata.json`

**Purpose:** Stores comprehensive information about each speaker for analysis and comparison

**Contents:**
- **Speaker Statistics:**
  - Number of training samples per speaker
  - Feature mean, std, min, max for each speaker
  - Dataset source information
  
- **Dataset Information:**
  - Total samples per dataset (TESS, SAVEE, RAVDESS, CREMA-D)
  - Total number of speakers
  - Total number of samples

**Generated During Training:**
- Created automatically when running `train_speaker_identification.py`
- Used by web interface for displaying speaker information
- Used for similarity calculations and comparisons

## Data Flow

### Emotion Detection Training Flow:
```
Audio Files → Preprocessing → Feature Extraction → Feature Matrix (n_samples × 65)
    ↓
Train/Test Split → Feature Scaling → Model Training → Model Evaluation
    ↓
Model Persistence → Results Saving
```

### Emotion Detection Inference Flow:
```
Audio File → Preprocessing → Feature Extraction → Feature Scaling
    ↓
Model Prediction → Emotion Label + Probabilities → Display Results
```

### Speaker Identification Training Flow:
```
Audio Files → Preprocessing → Extract Speaker IDs from Filenames
    ↓
Feature Extraction (65 features) → ML Model Training (4 models)
    ↓
DL Data Preparation (Spectrograms/Sequences) → DL Model Training (CNN, LSTM, RNN)
    ↓
Model Evaluation → Comparison Report → Speaker Metadata Generation
    ↓
Model Persistence → Results Saving
```

### Speaker Identification Inference Flow:
```
Audio File → Preprocessing → Feature Extraction
    ↓
Model Selection (ML/DL/Best) → Model Prediction
    ↓
Speaker ID + Probabilities → Load Speaker Metadata
    ↓
Full Analysis Display:
  - Prediction & Confidence
  - Top-K Speakers
  - Feature Statistics
  - Similarity Comparison
  - Visualizations
```

## File Execution Order

### Training Phase (Optional - Only if training new models)

**Option A: Train Individual Model Types (Recommended)**
```bash
# Train emotion detection ML models
python src/models/emotion_ml.py

# Train emotion detection DL models
python src/models/emotion_dl.py

# Train speaker identification ML models
python src/models/speaker_ml.py

# Train speaker identification DL models
python src/models/speaker_dl.py
```

**Option B: Use main.py CLI**
```bash
# Train all emotion models
python main.py --mode train

# Or train individually
python main.py --mode train_emotion_ml
python main.py --mode train_emotion_dl
python main.py --mode train_speaker_ml
python main.py --mode train_speaker_dl
```

**Option C: Explore Datasets First (Optional)**
```bash
python main.py --mode explore
```

### Running the Application

**After training (or if models already exist):**
```bash
# Run the Streamlit web interface
streamlit run app/app.py

# Or if configured to auto-run:
python app/app.py
```

### Complete Workflow

1. **First Time Setup:**
   ```bash
   # 1. Explore datasets (optional)
   python main.py --mode explore
   
   # 2. Train all models
   python src/models/emotion_ml.py
   python src/models/emotion_dl.py
   python src/models/speaker_ml.py
   python src/models/speaker_dl.py
   
   # 3. Run the application
   streamlit run app/app.py
   ```

2. **Subsequent Runs (Models Already Trained):**
   ```bash
   # Just run the app
   streamlit run app/app.py
   ```

3. **Retraining Specific Models:**
   ```bash
   # Train only what you need
   python src/models/emotion_ml.py  # or any other model file
   ```

**Note:** Training files can be run in any order since they are independent. The app requires trained models to be present in the `models/` directory.

## Key Design Decisions

1. **Feature Extraction:**
   - Uses statistical aggregation (mean, std) for variable-length features
   - Total 65 features for consistent input size
   - Features cached to avoid recomputation

2. **Model Selection:**
   - ML models use same feature vector (65 features)
   - DL models require different preprocessing (spectrograms/sequences)
   - Best model selected based on accuracy

3. **Data Handling:**
   - Handles multiple dataset formats (folder-based, filename-based, encoded)
   - Standardizes emotion labels across datasets
   - Removes invalid audio files gracefully

4. **Code Organization:**
   - Modular design with clear separation of concerns
   - Main entry point only calls functions (no implementation)
   - Reusable components across different scripts

## Dependencies

**Core Libraries:**
- numpy - Numerical operations
- pandas - Data manipulation
- librosa - Audio processing and feature extraction
- soundfile - Audio file I/O
- scikit-learn - Machine learning models and utilities
- torch (PyTorch) - Deep learning models with GPU support
- xgboost - Gradient boosting
- streamlit - Web interface
- matplotlib/seaborn - Visualization
- scipy - Scientific computing

**See `requirements.txt` for version constraints.**

## New Files and Components

### Recently Added Files:

1. **`src/models/dl_speaker_models.py`**
   - Deep learning models for speaker identification
   - CNN and LSTM architectures adapted for speaker classification
   - Full training pipeline with GPU support

2. **`src/visualization.py`**
   - Visualization helper functions for speaker analysis
   - Probability charts, feature comparisons, radar charts, similarity visualizations
   - Similarity calculation functions

3. **`models/speaker_metadata.json`** (generated)
   - Comprehensive speaker statistics and feature information
   - Used for analysis and comparison in web interface

4. **`results/speaker_identification/`** (generated)
   - Evaluation results for speaker identification models
   - Confusion matrices, classification reports, model comparison JSON

## Important Notes

1. **Feature Caching:** Extracted features are saved to avoid recomputation. Delete `models/extracted_features.npz` to re-extract.

2. **Model Formats:**
   - ML models: Saved as .pkl files (joblib)
   - DL models: Saved as .pth files (PyTorch state_dict format)
   - Scaler: Saved separately as scaler.pkl
   - Speaker ML models: `models/speaker_{model_name}.pkl`
   - Speaker DL models: `models/speaker_{model_type}_best.pth` and `speaker_{model_type}_final.pth`
   - Speaker metadata: `models/speaker_models_metadata.pkl` (for loading all models)
   - Hyperparameters: `models/best_hyperparameters.json` (for ML models)

3. **Label Handling:**
   - ML models: Use string labels directly
   - DL models: Use LabelEncoder for integer encoding
   - App interface: Handles both string and integer predictions

4. **Dataset Paths:**
   - Default: `dataset/` folder in project root
   - Supports case variations: TESS/Tess, SAVEE/Savee, etc.

5. **Memory Considerations:**
   - Feature extraction processes in batches
   - Progress indicators every 100-500 files
   - Features cached to disk

6. **Platform-Specific Considerations:**
   - Windows: `n_jobs=1` is automatically used for scikit-learn parallel processing to avoid multiprocessing issues
   - Linux/Mac: `n_jobs=-1` is used for full parallel processing
   - DataLoader: `num_workers=0` is used for Windows compatibility (can be increased on Linux/Mac)

7. **Training Execution:**
   - Each model file is self-contained and can be run independently
   - Training functions include progress indicators and detailed logging
   - Evaluation uses batched predictions for large test sets to prevent memory issues

## Common Issues and Solutions

1. **Import Errors:**
   - Run: `pip install -r requirements.txt`

2. **Dataset Not Found:**
   - Ensure datasets are in `dataset/` folder
   - Check folder names match expected format

3. **Memory Issues:**
   - Process datasets one at a time
   - Reduce batch size in DL training
   - Use feature caching (already implemented)

4. **Model Loading Errors:**
   - Ensure models are trained first
   - Check model files exist in `models/` directory

5. **Windows Multiprocessing Issues:**
   - On Windows, `n_jobs=-1` in scikit-learn can cause duplicate output or script re-execution
   - The code automatically detects Windows and sets `n_jobs=1` for GridSearchCV, RandomizedSearchCV, and cross_val_score
   - This is handled automatically in `src/models/ml_models.py` using platform detection

6. **Repeated Training Output:**
   - If you see models being trained multiple times, check if you're on Windows
   - The platform detection should prevent this, but if it persists, ensure `n_jobs=1` is set for hyperparameter tuning

7. **PyTorch Deprecation Warnings:**
   - `verbose=True` parameter in `ReduceLROnPlateau` has been removed (deprecated in PyTorch 2.0+)
   - The code has been updated to remove this parameter

## Extension Points

1. **Add New Datasets:**
   - Add loader method in `EmotionDatasetLoader`
   - Update `load_all_datasets()` to include new dataset

2. **Add New Features:**
   - Add extraction method in `FeatureExtractor`
   - Update `extract_all_features()` to include new feature
   - Update feature count (currently 65)

3. **Add New Models:**
   - ML Emotion: Add to `EmotionMLTrainer.models` dictionary in `src/models/emotion_ml.py`
   - DL Emotion: Add model class and update `train_all_models()` in `src/models/emotion_dl.py`
   - ML Speaker: Add to `SpeakerMLTrainer.models` dictionary in `src/models/speaker_ml.py`
   - DL Speaker: Add model class and update `train_model()` in `src/models/speaker_dl.py`

4. **Custom Evaluation:**
   - Extend `ModelEvaluator` class
   - Add new metric calculation methods

## Performance Characteristics

- **Feature Extraction:** ~0.1-0.5 seconds per file
- **ML Training (Emotion):** ~5-15 minutes for all 4 models (without hyperparameter tuning)
- **ML Training (Emotion with Tuning):** ~30-60 minutes for all 4 models (with hyperparameter tuning)
- **DL Training (Emotion):** ~30-60 minutes per model (CNN, LSTM, RNN - depends on epochs, GPU availability)
- **ML Training (Speaker):** ~10-20 minutes for all 4 models
- **DL Training (Speaker):** ~30-60 minutes per model (CNN, LSTM, RNN - depends on epochs, GPU availability)
- **Inference (Emotion):** <1 second per audio file
- **Inference (Speaker):** <1 second per audio file (ML), ~1-2 seconds (DL with GPU)

## Testing

- `main.py --mode explore` - Verifies dataset loading and statistics
- Can be extended with unit tests for each component

## Self-Contained Training Architecture

**Design Philosophy:**
Each model file (`emotion_ml.py`, `emotion_dl.py`, `speaker_ml.py`, `speaker_dl.py`) is fully self-contained with its own `train()` function that handles the complete pipeline from data loading to model saving.

**Benefits:**
- **Independence**: Each model type can be trained independently
- **Clarity**: All training logic for a model type is in one place
- **Flexibility**: Easy to train specific models without running everything
- **Maintainability**: Clear separation of concerns

**Training Function Structure:**
Each `train()` function includes:
1. **Data Loading** - Loads datasets using `EmotionDatasetLoader`
2. **Preprocessing** - Normalizes and trims audio (with caching support)
3. **Feature/DL Data Extraction** - Extracts features or DL data (with caching support)
4. **Model Training** - Trains all models of that type
5. **Evaluation** - Evaluates models with progress indicators
6. **Model Saving** - Saves all trained models and results

**Caching Strategy:**
- Preprocessed audio: `models/preprocessed_audio.npz` (emotion) or `models/preprocessed_audio_speakers.npz` (speaker)
- ML features: `models/extracted_features.npz` (emotion) or `models/extracted_features_speakers.npz` (speaker)
- DL data: `models/dl_data.npz` (emotion) or `models/dl_data_speakers.npz` (speaker)

**Progress Indicators:**
- Step-by-step progress messages (e.g., "[1/5] Loading datasets...")
- Model-specific progress during evaluation
- Batched predictions with progress updates for large test sets
- Completion messages for each model evaluation

## Enhanced Features Summary

### Hyperparameter Tuning
- **ML Models**: GridSearchCV/RandomizedSearchCV for optimal hyperparameters
- **DL Models**: Learning rate scheduling, improved architectures, AdamW optimizer
- Best hyperparameters saved for reproducibility

### Model Architecture Improvements
- **CNN**: Deeper architecture (4 conv layers), batch normalization, additional FC layers
- **LSTM**: Multi-layer LSTM, increased hidden sizes, deeper FC layers
- **RNN**: Multi-layer RNN architecture (2-layer → 1-layer), deeper FC layers, dropout regularization
- Better regularization and generalization across all DL models

### Speaker Identification Enhancements
- **Multiple ML Models**: 4 different algorithms for comparison
- **Deep Learning Support**: CNN, LSTM, and RNN models for speaker identification
- **Comprehensive Analysis**: Full speaker data, feature statistics, comparisons
- **Visualization Tools**: Multiple chart types for analysis

### Web Interface Enhancements
- **Multi-Tab Design**: Separate tabs for emotion detection and speaker identification
- **Model Selection**: Choose specific models or use best model
- **Rich Visualizations**: Probability distributions, feature comparisons, similarity scores
- **Comprehensive Display**: Full speaker information, training data stats, model confidence

## Deep Learning Model Details

### Input Data Formats

**CNN Models:**
- Input: Mel spectrograms (128×128 pixels)
- Format: Channels-first (N, C, H, W) for PyTorch
- Preprocessing: Converted from channels-last format, normalized. **Key Improvement:** Uses **padding/cropping** instead of resizing to preserve temporal rhythm and speech rate, which is critical for emotion and speaker recognition.

**LSTM Models:**
- Input: MFCC sequences (variable length, padded/truncated to max_length=200)
- Format: (batch_size, sequence_length, num_features)
- Features: 13 MFCC coefficients per timestep
- Preprocessing: Padding/truncation to fixed length, normalization

**RNN Models:**
- Input: MFCC sequences (variable length, padded/truncated to max_length=200)
- Format: (batch_size, sequence_length, num_features)
- Features: 13 MFCC coefficients per timestep
- Preprocessing: Same as LSTM (padding/truncation, normalization)
- Architecture: Uses standard RNN cells (tanh activation) instead of LSTM gates

### Model Training Configuration

**Common Training Parameters:**
- Optimizer: AdamW (weight_decay=0.01)
- Learning Rate: 0.001 (initial), scheduled with ReduceLROnPlateau (factor=0.5, patience=5)
- Batch Size: 32 (default, configurable)
- Epochs: 30-100 (default varies by model type, configurable)
- Loss Function: CrossEntropyLoss
- Mixed Precision: Enabled (bfloat16) for GPU training
- Device: Automatic GPU detection with CPU fallback
- **Note**: Strict early stopping implemented (stops if accuracy doesn't improve by 0.2% for 30 epochs, or if performance degrades significantly)

**Model-Specific Details:**
- **CNN**: Uses 2D convolutions, batch normalization, max pooling
  - Best for: Spectrogram-based feature extraction, spatial pattern recognition
  - Input: Fixed-size spectrograms (128×128)
  
- **LSTM**: Uses LSTM cells with forget gates, handles long-term dependencies
  - Best for: Sequential data with long-term dependencies, variable-length sequences
  - Input: Variable-length MFCC sequences
  - Advantages: Better memory retention, handles vanishing gradient problem
  
- **RNN**: Uses standard RNN cells, simpler architecture, faster training
  - Best for: Sequential data with shorter dependencies, faster inference
  - Input: Variable-length MFCC sequences (same as LSTM)
  - Advantages: Simpler architecture, faster training and inference, lower memory usage
  - Trade-offs: May struggle with very long sequences due to vanishing gradients

**Evaluation Improvements:**
- Batched predictions for large test sets (prevents memory issues)
- Progress indicators during evaluation (shows which model is being evaluated)
- Step-by-step feedback (predictions, metrics computation, saving)
- Completion messages for each model evaluation

### Model Saving and Loading

**File Naming Convention:**
- Emotion Detection: `{model_name}.pth` (e.g., `cnn.pth`, `lstm.pth`) - Saves the best performing model based on validation loss
- Speaker Identification: `speaker_{model_type}.pth` (e.g., `speaker_cnn.pth`) - Saves the best performing model
- Model types: 'cnn', 'lstm', 'rnn'

**Saved Components:**
- Model state_dict (weights and biases)
- Model configuration (input_size, hidden_sizes, num_classes, etc.)
- Label encoder (for converting integer predictions to string labels)
- Model type identifier

**Loading Process:**
1. Load state_dict from .pth file
2. Reconstruct model architecture using saved configuration
3. Load weights into model
4. Load label encoder for predictions
5. Set model to evaluation mode

## Recent Refactoring (Self-Contained Training)

**Changes Made:**
- **Removed centralized training files**: `src/training.py` and `train_speaker_identification.py` have been removed
- **Self-contained model files**: Each model file now includes its own complete `train()` function
- **New file structure**:
  - `src/models/emotion_ml.py` - Emotion detection ML models and training
  - `src/models/emotion_dl.py` - Emotion detection DL models and training
  - `src/models/speaker_ml.py` - Speaker identification ML models and training
  - `src/models/speaker_dl.py` - Speaker identification DL models and training
- **Updated class names**:
  - `MLModelTrainer` → `EmotionMLTrainer`
  - `DLModelTrainer` → `EmotionDLTrainer`
  - `SpeakerIdentifier` → `SpeakerMLTrainer`
  - `DLSpeakerIdentifier` → `SpeakerDLTrainer`
- **Updated main.py**: Now provides CLI options to train individual model types or all models
- **Evaluation improvements**: Added progress indicators and batched predictions for large test sets

**Benefits:**
- Each model file is fully independent and executable
- Clearer separation of concerns
- Easier to maintain and extend
- Can train specific model types without running everything
- Better progress visibility during training and evaluation

## Future Enhancements

- Real-time microphone recording in web interface
- Model ensemble methods (voting/averaging across models)
- Advanced DL models (Transformers, Attention mechanisms, GRU variants)
- Real-time emotion detection from microphone
- Multi-language support
- Emotion intensity prediction
- Speaker verification (1:1 matching) in addition to identification (1:N matching)
- Interactive model comparison dashboard
- Export analysis results to PDF/CSV
- Hyperparameter optimization for DL models (Optuna, Ray Tune)
- Model quantization for faster inference

