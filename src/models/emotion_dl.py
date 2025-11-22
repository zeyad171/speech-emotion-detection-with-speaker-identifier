"""
Deep Learning models and training for Speech Emotion Detection using PyTorch.
"""
import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import time
from typing import Tuple, Dict, Any
from scipy.ndimage import zoom

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import EmotionDatasetLoader
from src.feature_extraction import FeatureExtractor
from src.evaluation import ModelEvaluator
from src.utils import save_preprocessed_audio, load_preprocessed_audio, save_dl_data, load_dl_data

# Global flag to prevent repeated GPU configuration prints
_gpu_config_printed = False

def configure_gpu(verbose: bool = True):
    """Configure PyTorch to use GPU if available."""
    global _gpu_config_printed
    
    if torch.cuda.is_available():
        try:
            device = torch.device('cuda')
            gpu_count = torch.cuda.device_count()
            
            # Enable cuDNN benchmark for faster training
            torch.backends.cudnn.benchmark = True
            
            if verbose and not _gpu_config_printed:
                print(f"\n[OK] GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"     Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                _gpu_config_printed = True
            
            return device, True
        except Exception as e:
            if verbose and not _gpu_config_printed:
                print(f"[ERROR] GPU setup error: {e}")
                _gpu_config_printed = True
            return torch.device('cpu'), False
    else:
        if verbose and not _gpu_config_printed:
            print("[WARNING] No GPU detected. Training will use CPU.")
            _gpu_config_printed = True
        return torch.device('cpu'), False

# Configure GPU on module import
_device, _GPU_AVAILABLE = configure_gpu(verbose=True)

# -----------------------------------------------------------------------------
# Neural Network Architectures
# -----------------------------------------------------------------------------

class CNNModel(nn.Module):
    """CNN model for spectrogram-based classification (Expects 128x128 input)."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 7):
        super(CNNModel, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # 128 -> 64
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # 64 -> 32
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) # 32 -> 16
        
        # Conv Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2) # 16 -> 8
        
        self.dropout1 = nn.Dropout(0.5)
        
        # Flatten
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        x = self.dropout1(x)
        
        x = self.flatten(x)
        x = self.dropout2(self.relu(self.fc1(x)))
        x = self.dropout3(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class LSTMModel(nn.Module):
    """LSTM model for sequential feature classification."""
    
    def __init__(self, input_size: int, hidden_size1: int = 256, 
                 hidden_size2: int = 128, num_classes: int = 7):
        super(LSTMModel, self).__init__()
        
        # Use bidirectional LSTM for better context
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, num_layers=2, dropout=0.3, bidirectional=True)
        self.dropout1 = nn.Dropout(0.4)
        
        # Input to lstm2 is hidden_size1 * 2 because of bidirectionality
        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True, num_layers=1, bidirectional=True)
        self.dropout2 = nn.Dropout(0.4)
        
        # FC input is hidden_size2 * 2
        self.fc1 = nn.Linear(hidden_size2 * 2, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x, _ = self.lstm1(x) 
        x = self.dropout1(x)
        
        x, (h_n, _) = self.lstm2(x)
        # Take the output of the last timestep
        x = x[:, -1, :] 
        
        x = self.dropout2(x)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x


class RNNModel(nn.Module):
    """RNN model for sequential feature classification."""
    
    def __init__(self, input_size: int, hidden_size1: int = 256, 
                 hidden_size2: int = 128, num_classes: int = 7):
        super(RNNModel, self).__init__()
        
        self.rnn1 = nn.RNN(input_size, hidden_size1, batch_first=True, num_layers=2, dropout=0.3)
        self.dropout1 = nn.Dropout(0.4)
        
        self.rnn2 = nn.RNN(hidden_size1, hidden_size2, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(hidden_size2, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x, _ = self.rnn1(x) 
        x = self.dropout1(x)
        
        x, _ = self.rnn2(x)
        # Take the output of the last timestep
        x = x[:, -1, :] 
        
        x = self.dropout2(x)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x

# -----------------------------------------------------------------------------
# Trainer Class
# -----------------------------------------------------------------------------

class EmotionDLTrainer:
    """Train and evaluate deep learning models for emotion detection using PyTorch."""
    
    def __init__(self, models_dir: str = 'models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        self.model_configs = {} 
        self.device = _device
        self.gpu_available = _GPU_AVAILABLE
    
    def prepare_spectrograms(self, audio_list, extractor, target_shape=(128, 128)):
        """Prepare mel spectrograms for CNN models (Padding/Cropping instead of Resizing)."""
        spectrograms = []
        for audio in audio_list:
            mel_spec = extractor.extract_mel_spectrogram(audio)
            
            # Crop or Pad to target_shape
            current_h, current_w = mel_spec.shape
            target_h, target_w = target_shape
            
            final_spec = np.zeros(target_shape)
            
            # For frequency dimension (height), we expect it to match or be close
            # We'll take the min of current and target
            h = min(current_h, target_h)
            
            # For time dimension (width), we crop if too long, pad if too short
            w = min(current_w, target_w)
            
            # Fill the final spectrogram
            final_spec[:h, :w] = mel_spec[:h, :w]

            # Normalize
            if np.max(final_spec) > np.min(final_spec):
                final_spec = (final_spec - np.min(final_spec)) / (np.max(final_spec) - np.min(final_spec))
            
            final_spec = np.expand_dims(final_spec, axis=-1)
            spectrograms.append(final_spec)
        
        return np.array(spectrograms)
    
    def prepare_sequences(self, audio_list, extractor, max_length=200):
        """Prepare MFCC sequences for LSTM models."""
        sequences = []
        for audio in audio_list:
            mfccs = extractor.extract_mfcc(audio).T 
            
            if mfccs.shape[0] > max_length:
                mfccs = mfccs[:max_length, :]
            else:
                pad_width = max_length - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
            
            sequences.append(mfccs)
        
        return np.array(sequences)
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size=0.2, 
                    validation_size=0.1, random_state=42) -> Tuple:
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=(test_size + validation_size), 
            random_state=random_state, stratify=y_encoded
        )
        
        val_prop = validation_size / (test_size + validation_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_prop),
            random_state=random_state, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _to_tensor(self, data: np.ndarray, is_label: bool = False) -> torch.Tensor:
        if is_label:
            return torch.from_numpy(data).long()
        return torch.from_numpy(data).float()
    
    def train_model(self, model: nn.Module, model_name: str, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 100, batch_size: int = 256) -> Dict:
        
        # Ensure model is on correct device
        model = model.to(self.device)
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        print(f"  Training {model_name} on {self.device}...")
        print(f"    Model device: {model_device}")
        
        if self.gpu_available:
            if model_device.type != 'cuda':
                print(f"    [WARNING] Model is not on GPU! Expected cuda, got {model_device}")
            else:
                print(f"    GPU: {torch.cuda.get_device_name(0)}")
                print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                print(f"    Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        else:
            if model_device.type != 'cpu':
                print(f"    [WARNING] Model device mismatch! Expected CPU, got {model_device}")
        
        if len(X_train.shape) == 4 and X_train.shape[-1] == 1:
            X_train = np.transpose(X_train, (0, 3, 1, 2))
            X_val = np.transpose(X_val, (0, 3, 1, 2))
        
        train_ds = TensorDataset(self._to_tensor(X_train), self._to_tensor(y_train, True))
        val_ds = TensorDataset(self._to_tensor(X_val), self._to_tensor(y_val, True))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                pin_memory=self.gpu_available, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                              pin_memory=self.gpu_available, num_workers=0)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        scaler = torch.amp.GradScaler('cuda') if self.gpu_available else None
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        best_val_loss = float('inf')
        best_state = None
        
        print(f"    Total epochs: {epochs}, Batch size: {batch_size}")
        print(f"    Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
        
        # Early Stopping parameters
        best_acc = 0.0
        best_loss = float('inf') # For monitoring only
        patience = 30
        patience_counter = 0
        min_delta = 0.002  # 0.2%
        min_epochs = 50
        
        # Safety checks
        safety_acc_drop = 0.02  # 2% drop
        safety_loss_rise = 0.5  # 0.5 rise
        bad_acc_counter = 0
        bad_loss_counter = 0
        
        # Timing for first epoch to verify GPU acceleration
        epoch_start_time = None
        
        for epoch in range(epochs):
            if epoch == 0:
                epoch_start_time = time.time()
            
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            # Show progress indicator for first epoch
            if epoch == 0:
                print(f"    Epoch [1/{epochs}] Training...", end='\r')
            
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Verify first batch is on correct device
                if epoch == 0 and batch_idx == 0:
                    batch_device = X_batch.device
                    model_device_check = next(model.parameters()).device
                    print(f"    Batch device: {batch_device}, Model device: {model_device_check}")
                    if batch_device != model_device_check:
                        print(f"    [WARNING] Device mismatch! Batch on {batch_device}, Model on {model_device_check}")
                    if self.gpu_available:
                        if batch_device.type != 'cuda':
                            print(f"    [WARNING] Batch is not on GPU! Expected cuda, got {batch_device}")
                        else:
                            print(f"    GPU Memory after first batch: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                train_correct += (pred == y_batch).sum().item()
                train_total += y_batch.size(0)
                
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    val_correct += (pred == y_batch).sum().item()
                    val_total += y_batch.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            
            scheduler.step(val_loss)
            
            # Print status
            epoch_time_str = ""
            if epoch == 0 and epoch_start_time is not None:
                epoch_time = time.time() - epoch_start_time
                epoch_time_str = f" | Time: {epoch_time:.2f}s"
                if self.gpu_available:
                    epoch_time_str += f" (GPU)"
            
            # Update best scores and check stopping criteria
            saved_str = ""
            
            # 1. Update best loss for safety check (independent of accuracy stopping)
            if val_loss < best_loss:
                best_loss = val_loss
                bad_loss_counter = 0
            else:
                if val_loss > (best_loss + safety_loss_rise):
                    bad_loss_counter += 1
                else:
                    bad_loss_counter = 0 # Reset if not rising dramatically
            
            # 2. Main Early Stopping Logic on Accuracy
            if val_acc > (best_acc + min_delta):
                best_acc = val_acc
                best_state = model.state_dict().copy()
                patience_counter = 0
                saved_str = " [Saved Best]"
                bad_acc_counter = 0
            else:
                patience_counter += 1
                
                # Safety check for accuracy drop
                if val_acc < (best_acc - safety_acc_drop):
                    bad_acc_counter += 1
                else:
                    bad_acc_counter = 0 # Reset if not dropping dramatically
            
            print(f"    Epoch {epoch+1}/{epochs} | Val Acc: {val_acc*100:.2f}% | Best: {best_acc*100:.2f}% | Patience: {patience_counter}/{patience}{saved_str}{epoch_time_str}")
            
            # Check triggers (only after min_epochs)
            if epoch >= min_epochs:
                # Main condition
                if patience_counter >= patience:
                    print(f"    Early stopping triggered: no improvement ≥{min_delta*100:.1f}% for {patience} epochs")
                    break
                
                # Safety checks
                if bad_acc_counter >= 10:
                    print(f"    Early stopping triggered: Validation accuracy dropped >{safety_acc_drop*100:.0f}% from best for 10 epochs (Possible Overfitting)")
                    break
                
                if bad_loss_counter >= 10:
                    print(f"    Early stopping triggered: Validation loss increased >{safety_loss_rise} from best for 10 epochs (Possible Overfitting)")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        
        self.trained_models[model_name] = model
        return history

    def train_all_models(self, spectrograms: np.ndarray, sequences: np.ndarray, 
                        labels: np.ndarray, epochs=100, batch_size=256) -> Dict:
        
        X_train_spec, X_val_spec, X_test_spec, y_train, y_val, y_test = \
            self.prepare_data(spectrograms, labels)
            
        X_train_seq, X_val_seq, X_test_seq, _, _, _ = \
            self.prepare_data(sequences, labels)
            
        num_classes = len(np.unique(y_train))
        results = {
            'X_test_spec': X_test_spec,
            'X_test_seq': X_test_seq,  # RNN uses same sequences as LSTM
            'y_test_encoded': y_test,
            'label_encoder': self.label_encoder,
            'device': self.device
        }

        try:
            print("\nStarting CNN Training...")
            cnn = CNNModel(input_channels=1, num_classes=num_classes).to(self.device)
            self.model_configs['cnn'] = {'input_channels': 1, 'num_classes': num_classes}
            
            hist = self.train_model(cnn, 'cnn', X_train_spec, y_train, X_val_spec, y_val, epochs, batch_size)
            results['cnn'] = {'model': cnn, 'history': hist, 'test_acc': 0}
        except Exception as e:
            print(f"CNN Failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            print("\nStarting LSTM Training...")
            input_size = sequences.shape[2] 
            lstm = LSTMModel(input_size=input_size, num_classes=num_classes).to(self.device)
            self.model_configs['lstm'] = {'input_size': input_size, 'num_classes': num_classes}
            
            hist = self.train_model(lstm, 'lstm', X_train_seq, y_train, X_val_seq, y_val, epochs, batch_size)
            results['lstm'] = {'model': lstm, 'history': hist}
        except Exception as e:
            print(f"LSTM Failed: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            print("\nStarting RNN Training...")
            input_size = sequences.shape[2]
            rnn = RNNModel(input_size=input_size, num_classes=num_classes).to(self.device)
            self.model_configs['rnn'] = {'input_size': input_size, 'num_classes': num_classes}
            
            hist = self.train_model(rnn, 'rnn', X_train_seq, y_train, X_val_seq, y_val, epochs, batch_size)
            results['rnn'] = {'model': rnn, 'history': hist}
        except Exception as e:
            print(f"RNN Failed: {e}")
            import traceback
            traceback.print_exc()
        
        return results

    def save_model(self, model_name: str, filepath: str = None):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        if filepath is None:
            filepath = os.path.join(self.models_dir, f'{model_name}.pth')
            
        # Determine model type
        if 'cnn' in model_name:
            model_type = 'cnn'
        elif 'lstm' in model_name:
            model_type = 'lstm'
        elif 'rnn' in model_name:
            model_type = 'rnn'
        else:
            model_type = 'cnn'  # default
        
        save_dict = {
            'model_state_dict': self.trained_models[model_name].state_dict(),
            'label_encoder': self.label_encoder,
            'model_config': self.model_configs.get(model_name, {}), 
            'model_type': model_type
        }
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")
            
        # weights_only=False is safe here since we're loading our own saved models
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        config = checkpoint.get('model_config', {})
        model_type = checkpoint.get('model_type', 'cnn')
        
        if model_type == 'cnn':
            model = CNNModel(
                input_channels=config.get('input_channels', 1),
                num_classes=config.get('num_classes', 7)
            )
        elif model_type == 'lstm':
            model = LSTMModel(
                input_size=config.get('input_size', 13), 
                num_classes=config.get('num_classes', 7)
            )
        elif model_type == 'rnn':
            model = RNNModel(
                input_size=config.get('input_size', 13), 
                num_classes=config.get('num_classes', 7)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.label_encoder = checkpoint['label_encoder']
        return model

def train(models_dir='models', test_size=0.2, validation_size=0.1, random_state=42, epochs=100, batch_size=256):
    """
    Complete training pipeline for emotion detection DL models.
    
    Args:
        models_dir: Directory to save trained models and cache files
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("="*60)
    print("Training Emotion Detection DL Models")
    print("="*60)
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Load datasets
    print("\n[1/6] Loading datasets...")
    loader = EmotionDatasetLoader()
    audio_files, labels = loader.load_all_datasets()
    
    if len(audio_files) == 0:
        print("ERROR: No audio files found.")
        return
    
    # Preprocess
    print(f"\n[2/6] Preprocessing {len(audio_files)} audio files...")
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
    
    # Prepare data for DL models
    print(f"\n[3/6] Preparing data for DL models...")
    dl_data_file = os.path.join(models_dir, 'dl_data.npz')
    dl_trainer = EmotionDLTrainer(models_dir=models_dir)
    
    if os.path.exists(dl_data_file):
        print(f"  Loading DL data from cache...")
        spectrograms, sequences, dl_labels = load_dl_data(dl_data_file)
        # Ensure labels match (use cached labels if they exist, otherwise use current labels)
        if len(dl_labels) == len(valid_labels):
            labels = np.array(dl_labels)
        else:
            labels = np.array(valid_labels)
    else:
        print(f"  Extracting DL data (this may take a while)...")
        extractor = FeatureExtractor()
        
        # Prepare spectrograms for CNN
        print("  Extracting spectrograms for CNN...")
        spectrograms = dl_trainer.prepare_spectrograms(preprocessed_audio, extractor)
        print(f"  Spectrogram shape: {spectrograms.shape}")
        
        # Prepare sequences for LSTM/RNN
        print("  Extracting sequences for LSTM/RNN...")
        sequences = dl_trainer.prepare_sequences(preprocessed_audio, extractor)
        print(f"  Sequence shape: {sequences.shape}")
        
        # Save DL data for future use
        save_dl_data(spectrograms, sequences, valid_labels, dl_data_file)
        labels = np.array(valid_labels)
    
    # Train DL models
    print(f"\n[4/6] Training DL models...")
    dl_results = dl_trainer.train_all_models(
        spectrograms, sequences, labels,
        epochs=epochs, batch_size=batch_size
    )
    
    # Evaluate DL models
    print(f"\n[5/6] Evaluating DL models...")
    evaluator = ModelEvaluator(results_dir='results')
    unique_labels = sorted(list(set(labels)))
    
    best_model = None
    best_score = 0
    
    model_names = ['cnn', 'lstm', 'rnn']
    available_models = [mn for mn in model_names if mn in dl_results and 'model' in dl_results[mn]]
    
    for idx, model_name in enumerate(available_models, 1):
        print(f"  Evaluating {model_name.upper()} model ({idx}/{len(available_models)})...")
        
        model_data = dl_results[model_name]
        model = model_data['model']
        device = dl_results.get('device', torch.device('cpu'))
        
        # Get appropriate test data
        if model_name == 'cnn':
            X_test = dl_results['X_test_spec']
            # Convert from (N, H, W, C) to (N, C, H, W) for PyTorch
            if len(X_test.shape) == 4 and X_test.shape[-1] == 1:
                X_test = np.transpose(X_test, (0, 3, 1, 2))
        else:  # lstm or rnn
            X_test = dl_results['X_test_seq']
        
        y_test_encoded = dl_results['y_test_encoded']
        
        # Predict using PyTorch with batching for large test sets
        model.eval()
        batch_size = 32
        test_size = X_test.shape[0]
        all_outputs = []
        
        print(f"    Making predictions on {test_size} samples (batch size: {batch_size})...")
        with torch.no_grad():
            for i in range(0, test_size, batch_size):
                end_idx = min(i + batch_size, test_size)
                X_batch = torch.from_numpy(X_test[i:end_idx]).float().to(device)
                outputs = model(X_batch)
                all_outputs.append(outputs.cpu())
                
                if (i + batch_size) % (batch_size * 10) == 0 or end_idx == test_size:
                    print(f"      Processed {end_idx}/{test_size} samples...", end='\r')
        
        print()  # New line after progress
        # Concatenate all outputs and apply softmax
        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        y_pred = torch.softmax(all_outputs_tensor, dim=1).numpy()
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Decode labels for evaluation
        y_test_decoded = dl_results['label_encoder'].inverse_transform(y_test_encoded)
        y_pred_decoded = dl_results['label_encoder'].inverse_transform(y_pred_classes)
        
        print(f"    Computing evaluation metrics...")
        # Evaluate
        eval_results = evaluator.evaluate_classification(
            y_test_decoded, y_pred_decoded, labels=unique_labels
        )
        evaluator.print_evaluation_results(eval_results, model_name.upper(), labels=unique_labels)
        
        print(f"    Saving confusion matrix...")
        evaluator.plot_confusion_matrix(
            eval_results['confusion_matrix'], unique_labels, f"{model_name.upper()}_emotion"
        )
        
        print(f"    Saving results...")
        evaluator.save_results(eval_results, model_name.upper(), labels=unique_labels)
        
        print(f"    Saving model...")
        dl_trainer.save_model(model_name)
        
        if eval_results['accuracy'] > best_score:
            best_score = eval_results['accuracy']
            best_model = model_name.upper()
        
        print(f"  ✓ {model_name.upper()} evaluation completed\n")
    
    print(f"\n[6/6] Training completed!")
    print(f"\n{'='*60}")
    print(f"Best model: {best_model} (Accuracy: {best_score:.4f})")
    print(f"{'='*60}")


if __name__ == '__main__':
    train()

