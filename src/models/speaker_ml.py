"""
Machine Learning models and training for Speaker Identification.
"""
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import top_k_accuracy_score, accuracy_score
import xgboost as xgb
import joblib
import os
import time
from typing import Tuple, Dict, List, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import EmotionDatasetLoader
from src.speaker_feature_extraction import SpeakerFeatureExtractor  # Speaker-optimized features
from src.evaluation import ModelEvaluator
from src.utils import save_features, load_features, save_preprocessed_audio, load_preprocessed_audio
import glob
import src.visualization as viz


def _make_json_serializable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return o


def save_eval_json(results: Dict, model_name: str, labels: list = None, results_dir: str = 'results') -> str:
    """
    Save an evaluator results dict as JSON for visualization.
    """
    os.makedirs(results_dir, exist_ok=True)
    out = {}
    for k, v in results.items():
        try:
            out[k] = _make_json_serializable(v)
        except Exception:
            out[k] = str(v)

    payload = {
        'meta': {'model_name': model_name, 'labels': labels or []},
        'results': out
    }
    path = os.path.join(results_dir, f'evaluation_{model_name}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Evaluation JSON saved to {path}")
    return path


def collect_evaluations_for_visualization(results_dir: str = 'results') -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    metrics = {}
    confusion_matrices = {}
    for p in glob.glob(os.path.join(results_dir, 'evaluation_*.json')):
        try:
            with open(p, 'r') as f:
                payload = json.load(f)
            model_name = payload.get('meta', {}).get('model_name') or Path(p).stem.replace('evaluation_', '')
            results = payload.get('results', {})
            metrics[model_name] = {
                'accuracy': float(results.get('accuracy', np.nan)),
                'f1_macro': float(results.get('f1_macro', np.nan)),
                'f1_micro': float(results.get('f1_micro', np.nan)),
                'f1_weighted': float(results.get('f1_weighted', np.nan))
            }
            cm = results.get('confusion_matrix')
            if cm is not None:
                confusion_matrices[model_name] = np.array(cm)
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
    return metrics, confusion_matrices


class SpeakerMLTrainer:
    """Speaker identification model with multiple ML algorithms."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize speaker identifier.
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Base class references (will be instantiated during tuning)
        self.base_models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'xgboost': xgb.XGBClassifier
        }

        # Hyperparameter grids for tuning (kept conservative for speaker ID)
        self.param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf']
            },
            'xgboost': {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6],
                'n_estimators': [100, 200]
            }
        }

        # Store best params when tuning
        self.best_params = {}

        # Initialize multiple ML models
        # Note: SVC with probability=True can be very slow for many speakers
        self.models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'svm': SVC(kernel='rbf', probability=True, random_state=42), 
            'xgboost': xgb.XGBClassifier(random_state=42, n_jobs=-1)
        }
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')  # Handle NaN values
        self.label_encoder = LabelEncoder()
        self.model = None  # For backward compatibility (best model)
        self.trained_models = {}  # Store all trained models
        self.best_model_name = None  # Track best performing model
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple:
        """
        Prepare data for training.
        """
        # Check for NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"  Warning: Found {nan_count} NaN values in features. Replacing with mean values.")
        
        # Encode labels first
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Handle NaN values (impute with mean)
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _get_base_estimator(self, model_name: str):
        """
        Instantiate a base estimator for tuning with sensible defaults.
        """
        if model_name == 'svm':
            return self.base_models[model_name](random_state=42, probability=True)
        elif model_name == 'logistic_regression':
            return self.base_models[model_name](random_state=42, n_jobs=-1, max_iter=1000)
        else:
            return self.base_models[model_name](random_state=42, n_jobs=-1)


    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray, tune_hyperparameters: bool = False) -> Dict:
        """
        Train a single model.
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"\nTraining {model_name}...")

        # Platform-aware n_jobs
        n_jobs_value = 1 if platform.system() == 'Windows' else -1

        if tune_hyperparameters and model_name in self.param_grids:
            print(f"  Tuning hyperparameters for {model_name}...")
            estimator = self._get_base_estimator(model_name)

            # Use RandomizedSearchCV for heavier models
            if model_name in ['svm', 'xgboost', 'random_forest']:
                n_iter = 12
                cv_folds = 3
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=self.param_grids[model_name],
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=n_jobs_value,
                    random_state=42,
                    verbose=0
                )
            else:
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=self.param_grids[model_name],
                    cv=3,
                    scoring='accuracy',
                    n_jobs=n_jobs_value,
                    verbose=0
                )

            search.fit(X_train, y_train)
            model = search.best_estimator_
            self.best_params[model_name] = search.best_params_
            print(f"  Best parameters: {search.best_params_}")
            print(f"  Best CV score: {search.best_score_:.4f}")
        else:
            model = self.models[model_name]
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model

            # Evaluate
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            results = {
                'model': model,
                'accuracy': accuracy,
                'y_pred': y_pred,
                'cv_mean': 0.0,
                'top3_accuracy': 0.0,
                'top5_accuracy': 0.0
            }

        # Calculate Probabilities and Top-K (Only if applicable)
        # If tuning was performed and model wasn't added yet, ensure trained_models updated
        if model_name not in self.trained_models:
            self.trained_models[model_name] = model

        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
            results['y_pred_proba'] = y_pred_proba
            
            # --- CRITICAL FIX: Prevent crash if n_classes < k ---
            n_classes = len(self.label_encoder.classes_)
            
            if n_classes >= 3:
                results['top3_accuracy'] = top_k_accuracy_score(y_test, y_pred_proba, k=3)
            else:
                results['top3_accuracy'] = accuracy # If < 3 classes, top-3 is just 100% or regular accuracy
                
            if n_classes >= 5:
                results['top5_accuracy'] = top_k_accuracy_score(y_test, y_pred_proba, k=5)
            else:
                results['top5_accuracy'] = accuracy
        
        # Cross-validation score (Skip for heavy models if needed)
        if model_name != 'svm': 
            cv_jobs = 1 if platform.system() == 'Windows' else -1
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=cv_jobs)
            results['cv_mean'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
            print(f"  CV Accuracy: {results['cv_mean']:.4f}")
        
        print(f"  Accuracy: {results.get('accuracy', accuracy):.4f}")
        if results['top3_accuracy'] > 0:
            print(f"  Top-3 Accuracy: {results['top3_accuracy']:.4f}")
        
        return results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, tune_hyperparameters: bool = False) -> Dict:
        """
        Train all ML models and select best performer.
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, test_size)
        
        results = {
            'X_test': X_test,
            'y_test': y_test,
            'models': {},
            'best_model': None,
            'best_score': 0
        }
        
        # Train each model
        for model_name in self.models.keys():
            try:
                model_results = self.train_model(model_name, X_train, y_train, X_test, y_test, tune_hyperparameters=tune_hyperparameters)
                results['models'][model_name] = model_results
                
                # Track best model (based on accuracy)
                if model_results['accuracy'] > results['best_score']:
                    results['best_score'] = model_results['accuracy']
                    results['best_model'] = model_name
                    self.best_model_name = model_name
                    self.model = self.trained_models[model_name]
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        # Print summary
        if results['best_model']:
            print("\n" + "="*60)
            print("Speaker ID Model Comparison")
            print("="*60)
            for model_name, model_results in results['models'].items():
                marker = " <-- BEST" if model_name == results['best_model'] else ""
                print(f"{model_name:20s} Accuracy: {model_results['accuracy']:.4f}{marker}")
            print("="*60)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler_speakers.pkl'))

        # Save best parameters if tuning was performed
        if tune_hyperparameters and self.best_params:
            params_file = os.path.join(self.models_dir, 'best_hyperparameters_speaker.json')
            # Convert numpy types to native Python types for JSON serialization
            params_serializable = {}
            for k, v in self.best_params.items():
                params_serializable[k] = {k2: (float(v2) if isinstance(v2, (np.integer, np.floating)) else v2) for k2, v2 in v.items()}
            with open(params_file, 'w') as f:
                json.dump(params_serializable, f, indent=2)
            print(f"\nBest hyperparameters saved to {params_file}")

        return results

    def retrain_on_full_data(self, model_name: str, X: np.ndarray, y: np.ndarray, save_suffix: str = '_final'):
        """
        Retrain the selected speaker model on the full dataset (no train/test split).
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model for retraining: {model_name}")

        print(f"\nRetraining {model_name} on full dataset ({len(X)} samples)...")

        estimator = self._get_base_estimator(model_name)
        if model_name in self.best_params:
            try:
                estimator.set_params(**self.best_params[model_name])
            except Exception:
                print("  [WARNING] Could not apply some best_params to estimator; using estimator defaults.")

        # Refit scaler on full data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        estimator.fit(X_scaled, y)

        final_name = f"{model_name}{save_suffix}"
        self.trained_models[final_name] = estimator

        model_path = os.path.join(self.models_dir, f"{final_name}.pkl")
        scaler_path = os.path.join(self.models_dir, f"scaler_speakers{save_suffix}.pkl")
        joblib.dump(estimator, model_path)
        joblib.dump(self.scaler, scaler_path)

        metadata = {
            'model_name': final_name,
            'base_model': model_name,
            'best_params': self.best_params.get(model_name, {}),
            'training_samples': int(len(X)),
            'feature_count': int(X.shape[1]) if len(X.shape) > 1 else 0,
            'saved_model': model_path,
            'saved_scaler': scaler_path,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        meta_path = os.path.join(self.models_dir, f"{final_name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Final model saved to {model_path}")
        print(f"Final scaler saved to {scaler_path}")
        print(f"Metadata saved to {meta_path}\n")
    
    def predict(self, X: np.ndarray, model_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict speaker from features."""
        if model_name is None:
            if self.model is None:
                raise ValueError("Model not trained yet.")
            model = self.model
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained yet")
            model = self.trained_models[model_name]
        
        # Handle NaN values and scale features
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded, probabilities
    
    def save_model(self, model_name: str = None, save_all: bool = False, filepath: str = None):
        """Save trained model(s)."""
        if save_all:
            # Save all models
            metadata = {
                'models': {},
                'scaler': self.scaler,
                'imputer': self.imputer,
                'label_encoder': self.label_encoder,
                'best_model_name': self.best_model_name
            }
            for name, model in self.trained_models.items():
                metadata['models'][name] = model
            
            if filepath is None:
                filepath = os.path.join(self.models_dir, 'speaker_models_metadata.pkl')
            joblib.dump(metadata, filepath)
            print(f"All speaker models saved to {filepath}")
        else:
            # Default to saving the best model if none specified
            if model_name is None:
                model_name = self.best_model_name
                
            if model_name is None or model_name not in self.trained_models:
                 raise ValueError("No model specified or model not trained.")

            if filepath is None:
                filepath = os.path.join(self.models_dir, f'speaker_{model_name}.pkl')
            
            model_data = {
                'model': self.trained_models[model_name],
                'scaler': self.scaler,
                'imputer': self.imputer,
                'label_encoder': self.label_encoder,
                'model_name': model_name
            }
            
            joblib.dump(model_data, filepath)
            print(f"Speaker model ({model_name}) saved to {filepath}")
    
    def load_model(self, filepath: str = None, model_name: str = None):
        """Load trained model."""
        if filepath is None:
            if model_name:
                filepath = os.path.join(self.models_dir, f'speaker_{model_name}.pkl')
            else:
                filepath = os.path.join(self.models_dir, 'speaker_models_metadata.pkl')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Handle metadata file (all models)
        if 'models' in model_data:
            self.trained_models = model_data['models']
            self.scaler = model_data['scaler']
            self.imputer = model_data.get('imputer', SimpleImputer(strategy='mean'))
            self.label_encoder = model_data['label_encoder']
            self.best_model_name = model_data.get('best_model_name')
            if self.best_model_name:
                self.model = self.trained_models[self.best_model_name]
        else:
            # Single model file
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.imputer = model_data.get('imputer', SimpleImputer(strategy='mean'))
            self.label_encoder = model_data['label_encoder']
            self.best_model_name = model_data.get('model_name')
            
            # Restore to dictionary
            if self.best_model_name:
                self.trained_models[self.best_model_name] = self.model
            
        print(f"Speaker model loaded from {filepath}")
        return self.model


def extract_speaker_id_from_filename(filepath: str, dataset: str) -> str:
    """
    Extract speaker ID from filename based on dataset.
    """
    filename = os.path.basename(filepath)
    
    if dataset == 'TESS':
        # TESS: OAF_angry/file.wav -> OAF
        parts = filepath.split(os.sep)
        if len(parts) >= 2:
            folder = parts[-2]
            if '_' in folder:
                return folder.split('_')[0]
    
    elif dataset == 'SAVEE':
        # SAVEE: DC_a01.wav -> DC
        if '_' in filename:
            return filename.split('_')[0]
    
    elif dataset == 'RAVDESS':
        # RAVDESS: 03-01-01-01-01-01-01.wav -> Actor_01
        parts = filename.split('-')
        if len(parts) >= 7:
            # FIX: parts[6] is "01.wav", we need just "01"
            actor_num = parts[6].split('.')[0]
            return f"Actor_{actor_num}"
    
    elif dataset == 'CREMA-D':
        # CREMA-D: 1001_DFA_ANG_XX.wav -> 1001
        if '_' in filename:
            return filename.split('_')[0]
    
    return "unknown"


def train(models_dir='models', test_size=0.2):
    """
    Complete training pipeline for speaker identification ML models.
    
    Args:
        models_dir: Directory to save trained models and cache files
        test_size: Proportion of data to use for testing
    """
    print("="*60)
    print("Training Speaker Identification ML Models")
    print("="*60)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Load datasets
    print("\n[1/5] Loading datasets and extracting speaker IDs...")
    loader = EmotionDatasetLoader()
    
    # Initialize variables
    all_audio = []
    all_speakers = []
    
    # Check for cached preprocessed audio with speaker IDs
    preprocessed_audio_file = os.path.join(models_dir, 'preprocessed_audio_speakers.npz')
    
    if os.path.exists(preprocessed_audio_file):
        print(f"  Loading preprocessed audio from cache...")
        try:
            all_audio, all_speakers = load_preprocessed_audio(preprocessed_audio_file)
            # Convert to list if it's a numpy array
            if isinstance(all_audio, np.ndarray):
                all_audio = list(all_audio)
            all_speakers = list(all_speakers)
        except (ValueError, KeyError) as e:
            print(f"  [WARNING] Cache file is incompatible: {e}")
            print(f"  Deleting old cache file and regenerating...")
            time.sleep(0.1)
            try:
                os.remove(preprocessed_audio_file)
            except PermissionError:
                old_file = preprocessed_audio_file + '.old'
                if os.path.exists(old_file):
                    os.remove(old_file)
                os.rename(preprocessed_audio_file, old_file)
                print(f"  Renamed incompatible file to {old_file} (will be overwritten)")
            all_audio = []
            all_speakers = []
    
    if len(all_audio) == 0:
        print(f"  Loading and preprocessing audio files (this may take a while)...")
        
        # Load all datasets
        audio_files, emotion_labels = loader.load_all_datasets()
        
        # Extract speaker IDs
        for i, audio_file in enumerate(audio_files):
            # Determine dataset from path (case-insensitive)
            audio_file_lower = audio_file.lower()
            dataset = None
            if 'tess' in audio_file_lower:
                dataset = 'TESS'
            elif 'savee' in audio_file_lower:
                dataset = 'SAVEE'
            elif 'ravdess' in audio_file_lower:
                dataset = 'RAVDESS'
            elif 'crema' in audio_file_lower:
                dataset = 'CREMA-D'
            
            if dataset:
                speaker_id = extract_speaker_id_from_filename(audio_file, dataset)
                audio = loader.preprocess_audio(audio_file)
                if len(audio) > 0:
                    all_audio.append(audio)
                    all_speakers.append(speaker_id)
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files")
        
        # Save preprocessed audio for future use
        save_preprocessed_audio(all_audio, all_speakers, preprocessed_audio_file)
    
    print(f"  Preprocessed {len(all_audio)} audio files")
    print(f"  Found {len(set(all_speakers))} unique speakers")
    
    # Extract features for ML models
    print(f"\n[2/5] Extracting features for ML models...")
    feature_file = os.path.join(models_dir, 'extracted_features_speakers.npz')
    
    if os.path.exists(feature_file):
        print(f"  Loading pre-extracted features...")
        try:
            features, speakers = load_features(feature_file)
            print(f"  Loaded features with shape: {features.shape}")
            print(f"  Loaded {len(speakers)} speaker labels")
        except Exception as e:
            print(f"  [WARNING] Failed to load feature cache: {e}")
            print(f"  Regenerating features...")
            os.remove(feature_file)
            print("  Using speaker-optimized feature extractor (72 features)...")
            extractor = SpeakerFeatureExtractor()  # 72 features vs 65 for emotion
            features = extractor.extract_features_batch(all_audio)
            speakers = np.array(all_speakers)
            save_features(features, speakers, feature_file)
    else:
        print("  Using speaker-optimized feature extractor (72 features)...")
        extractor = SpeakerFeatureExtractor()  # 72 features vs 65 for emotion
        features = extractor.extract_features_batch(all_audio)
        speakers = np.array(all_speakers)
        save_features(features, speakers, feature_file)
    
    print(f"  Feature shape: {features.shape}")
    
    # Train ML models
    print(f"\n[3/5] Training ML models...")
    speaker_identifier = SpeakerMLTrainer(models_dir=models_dir)
    ml_results = speaker_identifier.train_all_models(features, speakers, test_size=test_size)
    
    # Evaluate ML models
    print(f"\n[4/5] Evaluating ML models...")
    evaluator = ModelEvaluator(results_dir='results')
    
    # Get unique speakers present in test set
    y_test_decoded = speaker_identifier.label_encoder.inverse_transform(ml_results['y_test'])
    present_speakers = sorted(list(set(y_test_decoded)))
    
    best_model = None
    best_score = 0
    metrics = {}
    
    for model_name, model_results in ml_results['models'].items():
        y_pred = model_results['y_pred']
        y_pred_decoded = speaker_identifier.label_encoder.inverse_transform(y_pred)
        
        eval_results = evaluator.evaluate_classification(
            y_test_decoded, y_pred_decoded, labels=present_speakers
        )
        evaluator.print_evaluation_results(eval_results, model_name, labels=present_speakers)
        # Save confusion matrix with unique name for speaker ML
        evaluator.plot_confusion_matrix(
            eval_results['confusion_matrix'], present_speakers, 
            model_name, save_name=f'confusion_matrix_speaker_ml_{model_name}'
        )
        
        # Save with distinctive name and header
        # Save evaluation results with unique speaker ML naming
        evaluator.save_results(
            eval_results, 
            f"{model_name} (Speaker ID - ML)", 
            filepath=os.path.join(evaluator.results_dir, f'evaluation_speaker_ml_{model_name}.txt'),
            labels=present_speakers
        )
        # Save JSON for visualization
        try:
            save_eval_json(eval_results, model_name, labels=present_speakers)
        except Exception:
            pass

        # Collect simple train/test metrics
        try:
            metrics[model_name] = {
                'cv_mean': float(model_results.get('cv_mean', np.nan)),
                'test': float(eval_results.get('accuracy', np.nan))
            }
        except Exception:
            metrics[model_name] = {'cv_mean': np.nan, 'test': np.nan}
        
        # Save JSON for potential visualization (unique name)
        try:
            import json
            eval_json = {
                "meta": {"model_name": f"speaker_ml_{model_name}", "labels": present_speakers},
                "results": {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                           for k, v in eval_results.items()}
            }
            json_path = os.path.join(evaluator.results_dir, f'evaluation_speaker_ml_{model_name}.json')
            with open(json_path, 'w') as f:
                json.dump(eval_json, f, indent=2)
        except Exception as e:
            print(f"  Warning: Could not save JSON: {e}")
        
        speaker_identifier.save_model(model_name)
        
        if eval_results['accuracy'] > best_score:
            best_score = eval_results['accuracy']
            best_model = model_name
    
    print(f"\n[5/5] Training completed!")
    print(f"\n{'='*60}")
    print(f"Best model: {best_model} (Accuracy: {best_score:.4f})")
    print(f"{'='*60}")

    # Generate overfitting detection plots (ML: train vs test)
    try:
        viz.plot_overfitting_detection(metrics=metrics, histories=None, results_dir='results')
    except Exception as e:
        print(f"Warning: could not generate overfitting plots: {e}")

    # Save speaker metadata JSON for app visualizations
    try:
        # Compute per-speaker feature statistics
        speakers_list = list(speakers)
        features_arr = np.asarray(features)
        speaker_to_indices = {}
        for idx, spk in enumerate(speakers_list):
            speaker_to_indices.setdefault(spk, []).append(idx)

        meta = {
            'speakers': {},
            'datasets': {},
            'feature_count': int(features_arr.shape[1]) if features_arr.ndim == 2 else 65,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

        # Estimate dataset source per speaker via ID pattern
        def _detect_dataset(spk: str) -> str:
            if spk.startswith('Actor_'):
                return 'RAVDESS'
            if spk.isdigit() and len(spk) == 4:
                return 'CREMA-D'
            if spk in ['DC', 'JE', 'JK', 'KL']:
                return 'SAVEE'
            return 'TESS'

        for spk, idxs in speaker_to_indices.items():
            spk_feats = features_arr[idxs]
            # Aggregate stats
            meta['speakers'][spk] = {
                'id': spk,
                'num_samples': int(len(idxs)),
                'feature_mean': np.mean(spk_feats, axis=0).tolist(),
                'feature_std': np.std(spk_feats, axis=0).tolist(),
                'feature_min': np.min(spk_feats, axis=0).tolist(),
                'feature_max': np.max(spk_feats, axis=0).tolist(),
                'dataset': _detect_dataset(spk)
            }
            ds = meta['speakers'][spk]['dataset']
            meta['datasets'][ds] = meta['datasets'].get(ds, 0) + int(len(idxs))

        out_path = os.path.join(models_dir, 'speaker_metadata.json')
        with open(out_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Speaker metadata saved to {out_path}")
    except Exception as e:
        print(f"Warning: could not save speaker metadata: {e}")


if __name__ == '__main__':
    train()

