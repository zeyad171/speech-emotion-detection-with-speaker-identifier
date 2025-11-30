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


class SpeakerMLTrainer:
    """Speaker identification model with multiple ML algorithms."""
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize speaker identifier.
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
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
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train a single model.
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        
        # Train model
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
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
            results['cv_mean'] = cv_scores.mean()
            results['cv_std'] = cv_scores.std()
            print(f"  CV Accuracy: {results['cv_mean']:.4f}")
        
        print(f"  Accuracy: {accuracy:.4f}")
        if results['top3_accuracy'] > 0:
            print(f"  Top-3 Accuracy: {results['top3_accuracy']:.4f}")
        
        return results
    
    def train_all_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
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
                model_results = self.train_model(model_name, X_train, y_train, X_test, y_test)
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
        
        return results
    
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


if __name__ == '__main__':
    train()

