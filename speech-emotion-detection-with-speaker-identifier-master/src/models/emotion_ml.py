"""
Machine Learning models and training for Speech Emotion Detection.
"""
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import joblib
import os
import json
import platform
import time
from typing import Tuple, Dict, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import EmotionDatasetLoader
from src.feature_extraction import FeatureExtractor
from src.evaluation import ModelEvaluator
from src.utils import save_features, load_features, save_preprocessed_audio, load_preprocessed_audio
import src.visualization as viz
import glob


def _make_json_serializable(o):
    """Convert numpy types/arrays to Python native types for JSON."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return o


def save_eval_json(results: Dict, model_name: str, labels: list = None, results_dir: str = 'results') -> str:
    """
    Save an evaluation result dictionary (from ModelEvaluator.evaluate_classification)
    as a JSON file that is easy for visualization helpers to load.

    Returns the path to the saved JSON file.
    """
    os.makedirs(results_dir, exist_ok=True)
    out = {}
    # copy known keys and convert numpy arrays
    for k, v in results.items():
        try:
            out[k] = _make_json_serializable(v)
        except Exception:
            # fallback to string representation
            out[k] = str(v)

    out_meta = {
        'model_name': model_name,
        'labels': labels if labels is not None else [],
    }
    payload = {'meta': out_meta, 'results': out}
    path = os.path.join(results_dir, f'evaluation_emotion_ml_{model_name}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Evaluation JSON saved to {path}")
    return path


def collect_evaluations_for_visualization(results_dir: str = 'results') -> Tuple[Dict[str, Dict], Dict[str, np.ndarray]]:
    """
    Scan `results_dir` for `evaluation_*.json` files and return two structures:
      - metrics: dict model_name -> simple metrics dict (accuracy, f1_macro, ...)
      - confusion_matrices: dict model_name -> numpy array confusion matrix

    These are suitable inputs for the plotting helpers in `src/visualization.py`.
    """
    metrics = {}
    confusion_matrices = {}
    pattern = os.path.join(results_dir, 'evaluation_emotion_ml_*.json')
    for p in glob.glob(pattern):
        try:
            with open(p, 'r') as f:
                payload = json.load(f)
            model_name = payload.get('meta', {}).get('model_name') or Path(p).stem.replace('evaluation_', '')
            results = payload.get('results', {})
            # pick a compact metrics dict
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
            print(f"Warning: failed to load evaluation file {p}: {e}")
    return metrics, confusion_matrices

class EmotionMLTrainer:
    """Train and evaluate traditional ML models for emotion detection."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize ML model trainer for emotion detection.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Base class references (will be instantiated during training)
        self.base_models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'svm': SVC,
            'xgboost': xgb.XGBClassifier
        }
        
        # Hyperparameter grids for tuning
        self.param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'max_iter': [1000, 4000]
            },
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            },
            'xgboost': {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        # Default models (for quick training without tuning)
        self.models = {
            'logistic_regression': LogisticRegression(
                max_iter=1000, C=1.0, class_weight='balanced', random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=20, class_weight='balanced',
                random_state=42, n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf', probability=True, C=10.0, gamma='scale',
                class_weight='balanced', random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=42, n_jobs=-1, learning_rate=0.1,
                max_depth=5, n_estimators=200
            )
        }
        
        self.best_params = {}  # Store best hyperparameters
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')  # Handle NaN values
        self.label_encoder = LabelEncoder()  # For XGBoost label encoding
        self.label_encoder_fitted = False  # Track if encoder is fitted
        self.trained_models = {}
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple:
        """
        Prepare data for training.
        """
        # Check for NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"  Warning: Found {nan_count} NaN values in features. Replacing with mean values.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Handle NaN values (impute with mean)
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_test_imputed = self.imputer.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _get_base_estimator(self, model_name: str) -> Any:
        """
        Helper to safely instantiate base models with correct arguments.
        """
        if model_name == 'svm':
            # SVC does not accept n_jobs, and we need probability=True
            return self.base_models[model_name](random_state=42, probability=True)
        elif model_name == 'logistic_regression':
            # LR accepts n_jobs
            return self.base_models[model_name](random_state=42, n_jobs=-1)
        elif model_name == 'xgboost':
            # XGBoost requires numeric labels and multi-class objective
            n_classes = 7  # 7 emotions
            return self.base_models[model_name](
                random_state=42,
                n_jobs=-1,
                objective='multi:softmax',
                num_class=n_classes
            )
        else:
            # RF accepts n_jobs
            return self.base_models[model_name](random_state=42, n_jobs=-1)

    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                    tune_hyperparameters: bool = True) -> Dict:
        """
        Train a single model with optional hyperparameter tuning.
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"\nTraining {model_name}...")
        
        # Encode labels for XGBoost (requires numeric labels)
        if model_name == 'xgboost':
            if not self.label_encoder_fitted:
                self.label_encoder.fit(y_train)
                self.label_encoder_fitted = True
            y_train_for_model = self.label_encoder.transform(y_train)
        else:
            y_train_for_model = y_train
        
        # Platform-aware n_jobs for Windows compatibility
        n_jobs_value = 1 if platform.system() == 'Windows' else -1
        
        if tune_hyperparameters and model_name in self.param_grids:
            print(f"  Tuning hyperparameters for {model_name}...")
            
            estimator = self._get_base_estimator(model_name)
            
            # Use RandomizedSearchCV for faster tuning (especially for SVM and XGBoost)
            if model_name in ['svm', 'xgboost', 'random_forest']:
                # Reduce iterations and CV folds for Random Forest to speed up training
                n_iter = 8 if model_name == 'random_forest' else 15
                cv_folds = 2 if model_name == 'random_forest' else 3
                
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
                # Use GridSearchCV for smaller parameter spaces
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=self.param_grids[model_name],
                    cv=3,
                    scoring='accuracy',
                    n_jobs=n_jobs_value,
                    verbose=0
                )
            
            search.fit(X_train, y_train_for_model)
            model = search.best_estimator_
            self.best_params[model_name] = search.best_params_
            
            print(f"  Best parameters: {search.best_params_}")
            print(f"  Best CV score: {search.best_score_:.4f}")
        else:
            # Use default model
            model = self.models[model_name]
            model.fit(X_train, y_train_for_model)
        
        self.trained_models[model_name] = model
        
        # Cross-validation score (reduced folds for Random Forest to speed up)
        cv_folds = 3 if model_name == 'random_forest' else 5
        n_jobs_value = 1 if platform.system() == 'Windows' else -1
        cv_scores = cross_val_score(model, X_train, y_train_for_model, cv=cv_folds, scoring='accuracy', n_jobs=n_jobs_value)
        
        results = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        if tune_hyperparameters and model_name in self.best_params:
            results['best_params'] = self.best_params[model_name]
        
        print(f"  Final CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                         tune_hyperparameters: bool = True) -> Dict:
        """
        Train all models with optional hyperparameter tuning.
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
                model_results = self.train_model(model_name, X_train, y_train, tune_hyperparameters)
                results['models'][model_name] = model_results
                
                # Track best model
                if model_results['cv_mean'] > results['best_score']:
                    results['best_score'] = model_results['cv_mean']
                    results['best_model'] = model_name
            except Exception as e:
                print(f"Error training {model_name}: {e}")
        
        # Print summary
        if results['best_model']:
            print("\n" + "="*60)
            print("Model Comparison Summary")
            print("="*60)
            for model_name, model_results in results['models'].items():
                marker = " <-- BEST" if model_name == results['best_model'] else ""
                print(f"{model_name:20s} CV Accuracy: {model_results['cv_mean']:.4f}{marker}")
            print("="*60)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        
        # Save best parameters if tuning was performed
        if tune_hyperparameters and self.best_params:
            params_file = os.path.join(self.models_dir, 'best_hyperparameters.json')
            
            # Convert numpy types to native Python types for JSON serialization
            params_serializable = {}
            for k, v in self.best_params.items():
                params_serializable[k] = {
                    k2: (float(v2) if isinstance(v2, (np.integer, np.floating)) else str(v2) if isinstance(v2, (str)) else v2) 
                    for k2, v2 in v.items()
                }
                
            with open(params_file, 'w') as f:
                json.dump(params_serializable, f, indent=2)
            print(f"\nBest hyperparameters saved to {params_file}")
        
        return results
    
    def save_model(self, model_name: str, filepath: str = None):
        """Save a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        if filepath is None:
            filepath = os.path.join(self.models_dir, f'{model_name}.pkl')
        
        joblib.dump(self.trained_models[model_name], filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str = None):
        """Load a trained model."""
        if filepath is None:
            filepath = os.path.join(self.models_dir, f'{model_name}.pkl')
        
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        print(f"Model loaded from {filepath}")
        return model

    def retrain_on_full_data(self, model_name: str, X: np.ndarray, y: np.ndarray, save_suffix: str = '_final'):
        """
        Retrain the selected model on the full dataset (features X and labels y).

        This is intended to be run after hyperparameter search to produce a final
        model trained on all available data (no train/test split). The scaler is
        refit on the full feature set and saved alongside the model.

        Args:
            model_name: Name of the model to retrain (must be one of self.models keys)
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            save_suffix: Suffix to append to saved filenames (default: '_final')
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model for retraining: {model_name}")

        print(f"\nRetraining {model_name} on full dataset ({len(X)} samples)...")

        # If we have best params from hyperparameter tuning, apply them
        estimator = self._get_base_estimator(model_name)
        if model_name in self.best_params:
            try:
                estimator.set_params(**self.best_params[model_name])
            except Exception:
                # Best-effort: ignore incompatible params
                print("  [WARNING] Could not apply some best_params to estimator; using estimator defaults.")

        # Handle NaN values and refit scaler on full data
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"  Warning: Found {nan_count} NaN values in features. Replacing with mean values.")
        
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        # Fit estimator on full data
        estimator.fit(X_scaled, y)

        # Store trained final model
        final_name = f"{model_name}{save_suffix}"
        self.trained_models[final_name] = estimator

        # Save model, scaler, and imputer
        model_path = os.path.join(self.models_dir, f"{final_name}.pkl")
        scaler_path = os.path.join(self.models_dir, f"scaler{save_suffix}.pkl")
        imputer_path = os.path.join(self.models_dir, f"imputer{save_suffix}.pkl")
        joblib.dump(estimator, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.imputer, imputer_path)

        # Save metadata
        metadata = {
            'model_name': final_name,
            'base_model': model_name,
            'best_params': self.best_params.get(model_name, {}),
            'training_samples': int(len(X)),
            'feature_count': int(X.shape[1]) if len(X.shape) > 1 else 0,
            'saved_model': model_path,
            'saved_scaler': scaler_path,
            'saved_imputer': imputer_path,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        meta_path = os.path.join(self.models_dir, f"{final_name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Final model saved to {model_path}")
        print(f"Final scaler saved to {scaler_path}")
        print(f"Final imputer saved to {imputer_path}")
        print(f"Metadata saved to {meta_path}\n")


def train(models_dir='models', test_size=0.2, tune_hyperparameters=True, retrain_final: bool = True):
    """
    Complete training pipeline for emotion detection ML models.
    
    Args:
        models_dir: Directory to save trained models and cache files
        test_size: Proportion of data to use for testing
        tune_hyperparameters: Whether to perform hyperparameter tuning
    """
    print("="*60)
    print("Training Emotion Detection ML Models")
    print("="*60)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Load datasets
    print("\n[1/5] Loading datasets...")
    loader = EmotionDatasetLoader()
    audio_files, labels = loader.load_all_datasets()
    
    if len(audio_files) == 0:
        print("ERROR: No audio files found.")
        return
    
    # Preprocess
    print(f"\n[2/5] Preprocessing {len(audio_files)} audio files...")
    preprocessed_audio_file = os.path.join(models_dir, 'preprocessed_audio.npz')
    
    preprocessed_audio = []
    valid_labels = []
    
    if os.path.exists(preprocessed_audio_file):
        print(f"  Loading preprocessed audio from cache...")
        try:
            preprocessed_audio, valid_labels = load_preprocessed_audio(preprocessed_audio_file)
            # Convert to list if it's a numpy array
            if isinstance(preprocessed_audio, np.ndarray):
                preprocessed_audio = list(preprocessed_audio)
            valid_labels = list(valid_labels)
        except (ValueError, KeyError) as e:
            print(f"  [WARNING] Cache file is incompatible: {e}")
            print(f"  Deleting old cache file and regenerating...")
            time.sleep(0.1)  # Brief pause to ensure file is closed
            try:
                os.remove(preprocessed_audio_file)
            except PermissionError:
                # If still locked, rename it and delete later
                old_file = preprocessed_audio_file + '.old'
                if os.path.exists(old_file):
                    os.remove(old_file)
                os.rename(preprocessed_audio_file, old_file)
                print(f"  Renamed incompatible file to {old_file} (will be overwritten)")
            preprocessed_audio = []
            valid_labels = []
    
    if len(preprocessed_audio) == 0:  # Either file doesn't exist or was incompatible
        print(f"  Preprocessing audio files (this may take a while)...")
        preprocessed_audio = []
        valid_labels = []
        
        for i, audio_file in enumerate(audio_files):
            audio = loader.preprocess_audio(audio_file)
            if len(audio) > 0:
                preprocessed_audio.append(audio)
                valid_labels.append(labels[i])
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files")
        
        # Save preprocessed audio for future use
        save_preprocessed_audio(preprocessed_audio, valid_labels, preprocessed_audio_file)
    
    print(f"  Preprocessed {len(preprocessed_audio)} audio files")
    
    # Extract features for ML models
    print(f"\n[3/5] Extracting features for ML models...")
    feature_file = os.path.join(models_dir, 'extracted_features.npz')
    
    if os.path.exists(feature_file):
        print(f"  Loading pre-extracted features...")
        features, labels = load_features(feature_file)
    else:
        extractor = FeatureExtractor()
        features = extractor.extract_features_batch(preprocessed_audio)
        labels = np.array(valid_labels)
        save_features(features, labels, feature_file)
    
    print(f"  Feature shape: {features.shape}")
    
    # Train ML models
    print(f"\n[4/5] Training ML models...")
    ml_trainer = EmotionMLTrainer(models_dir=models_dir)
    ml_results = ml_trainer.train_all_models(features, labels, test_size=test_size, tune_hyperparameters=tune_hyperparameters)
    
    # Evaluate ML models
    print(f"\n[5/5] Evaluating ML models...")
    evaluator = ModelEvaluator(results_dir='results')
    unique_labels = sorted(list(set(labels)))
    metrics = {}
    
    best_model = None
    best_score = 0
    
    for model_name, model_results in ml_results['models'].items():
        model = model_results['model']
        y_pred = model.predict(ml_results['X_test'])
        
        # Decode XGBoost predictions back to string labels
        if model_name == 'xgboost' and ml_trainer.label_encoder_fitted:
            y_pred = ml_trainer.label_encoder.inverse_transform(y_pred.astype(int))
        
        eval_results = evaluator.evaluate_classification(
            ml_results['y_test'], y_pred, labels=unique_labels
        )
        evaluator.print_evaluation_results(eval_results, model_name, labels=unique_labels)
        # Save confusion matrix with unique emotion ML naming
        evaluator.plot_confusion_matrix(
            eval_results['confusion_matrix'], unique_labels, model_name,
            save_name=f'confusion_matrix_emotion_ml_{model_name}'
        )
        evaluator.save_results(eval_results, f"{model_name} (Emotion - ML)",
                             labels=unique_labels,
                             filepath=os.path.join(evaluator.results_dir, f'evaluation_emotion_ml_{model_name}.txt'))
        # Save JSON for visualization helpers
        try:
            save_eval_json(eval_results, model_name, labels=unique_labels)
        except Exception:
            pass

        # Collect simple train/test metrics for overfitting detection
        try:
            metrics[model_name] = {
                'cv_mean': float(model_results.get('cv_mean', np.nan)),
                'test': float(eval_results.get('accuracy', np.nan))
            }
        except Exception:
            metrics[model_name] = {'cv_mean': np.nan, 'test': np.nan}
        
        ml_trainer.save_model(model_name)
        
        if eval_results['accuracy'] > best_score:
            best_score = eval_results['accuracy']
            best_model = model_name

    # Optionally retrain best model on the full dataset and save final artifact
    if retrain_final and best_model is not None:
        try:
            print(f"\nRetraining best model '{best_model}' on full dataset...")
            ml_trainer.retrain_on_full_data(best_model, features, labels)
        except Exception as e:
            print(f"Error retraining final model: {e}")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model: {best_model} (Accuracy: {best_score:.4f})")
    print(f"{'='*60}")

    
    # Generate comprehensive visualizations
    print(f"\n{'='*60}")
    print("Generating comprehensive visualizations...")
    print(f"{'='*60}")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from visualization import generate_visualizations
        generate_visualizations()
    except Exception as e:
        print(f"Warning: could not generate visualizations: {e}")
        print(f"You can generate them manually: python src/visualization.py")



if __name__ == '__main__':
    train()

