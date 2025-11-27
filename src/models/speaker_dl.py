"""
Deep Learning models and training for Speaker Identification using PyTorch.
"""
import numpy as np
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from typing import Optional
from sklearn.model_selection import train_test_split
import os
import time
from scipy.ndimage import zoom

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_loader import EmotionDatasetLoader
from src.feature_extraction import FeatureExtractor
from src.evaluation import ModelEvaluator
from src.utils import save_preprocessed_audio, load_preprocessed_audio
from src.models.speaker_ml import extract_speaker_id_from_filename
from src.models.dl_param_config import DLParamConfig
from src.models.dl_param_config import DLParamConfig

# --- 1. CONFIGURATION & HARDWARE ---

# Global flag to prevent repeated GPU configuration prints
_gpu_config_printed = False

def configure_gpu():
    """Configure PyTorch to use GPU if available."""
    global _gpu_config_printed
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if not _gpu_config_printed:
            print(f"[OK] Speaker ID using GPU: {torch.cuda.get_device_name(0)}")
            _gpu_config_printed = True
        return device, True
    else:
        if not _gpu_config_printed:
            print("[WARNING] Speaker ID using CPU.")
            _gpu_config_printed = True
        return torch.device('cpu'), False

_device, _GPU_AVAILABLE = configure_gpu()


# --- 2. MODEL DEFINITIONS ---

class SpeakerCNNModel(nn.Module):
    """CNN model for speaker identification using spectrograms (128x128)."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super(SpeakerCNNModel, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(0.5)
        
        # Flatten: 128 channels * 16 * 16 spatial dim = 32,768
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout1(x)
        
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class SpeakerLSTMModel(nn.Module):
    """LSTM model for speaker identification using sequences."""
    
    def __init__(self, input_size: int, hidden_size1: int = 128, 
                 hidden_size2: int = 64, num_classes: int = 10):
        super(SpeakerLSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, num_layers=2, dropout=0.3)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True, num_layers=1)
        
        self.fc1 = nn.Linear(hidden_size2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq, features)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Take the output of the LAST timestep
        # shape: (batch, hidden_size2)
        x = x[:, -1, :] 
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SpeakerRNNModel(nn.Module):
    """RNN model for speaker identification using sequences."""
    
    def __init__(self, input_size: int, hidden_size1: int = 128, 
                 hidden_size2: int = 64, num_classes: int = 10):
        super(SpeakerRNNModel, self).__init__()
        
        self.rnn1 = nn.RNN(input_size, hidden_size1, batch_first=True, num_layers=2, dropout=0.3)
        self.rnn2 = nn.RNN(hidden_size1, hidden_size2, batch_first=True, num_layers=1)
        
        self.fc1 = nn.Linear(hidden_size2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq, features)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        
        # Take the output of the LAST timestep
        # shape: (batch, hidden_size2)
        x = x[:, -1, :] 
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --- 3. TRAINER CLASS ---

class SpeakerDLTrainer:
    """Train and evaluate deep learning models for speaker identification."""
    
    def __init__(self, models_dir: str = 'models', cfg: Optional[DLParamConfig] = None):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        self.model_configs = {} # Store dimensions here
        self.device = _device
        self.gpu_available = _GPU_AVAILABLE
        self.cfg = cfg or DLParamConfig()

    def save_metadata(self, metadata: dict, filename: str = 'speaker_dl_metadata.json') -> str:
        """Persist speaker DL training metadata (models, speakers, config) as JSON.

        Args:
            metadata: Dictionary with structured metadata.
            filename: Output JSON filename inside models_dir.
        Returns:
            Path to saved metadata file.
        """
        import json
        out_path = os.path.join(self.models_dir, filename)
        # Convert any numpy types to native python
        def _convert(o):
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            return o
        serializable = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                serializable[k] = {sk: _convert(sv) for sk, sv in v.items()}
            else:
                serializable[k] = _convert(v)
        try:
            with open(out_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"Saved speaker DL metadata to {out_path}")
        except Exception as e:
            print(f"[WARNING] Could not save speaker DL metadata: {e}")
        return out_path

    def _freeze_backbone_head_only(self, model: nn.Module) -> None:
        """
        Freeze all parameters except those likely to be classifier/head layers.
        Heuristic: parameter names containing 'fc' or 'classifier' are treated as head.
        """
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _unfreeze_all(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True
    
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
            
            # Add channel dim
            final_spec = np.expand_dims(final_spec, axis=-1)
            spectrograms.append(final_spec)
        
        return np.array(spectrograms)
    
    def prepare_sequences(self, audio_list, extractor, sequence_length=100):
        """Prepare MFCC sequences for LSTM models."""
        sequences = []
        for audio in audio_list:
            # Check attribute name (singular vs plural in your Extractor class)
            if hasattr(extractor, 'extract_mfccs'):
                mfccs = extractor.extract_mfccs(audio)
            else:
                mfccs = extractor.extract_mfcc(audio).T
            
            # Pad or truncate
            if mfccs.shape[0] > sequence_length:
                mfccs = mfccs[:sequence_length, :]
            else:
                pad_length = sequence_length - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, pad_length), (0, 0)), mode='constant')
            
            sequences.append(mfccs)
        
        return np.array(sequences)
    
    def _to_tensor(self, data, labels=None):
        if labels is not None:
            return torch.from_numpy(data).float(), torch.from_numpy(labels).long()
        return torch.from_numpy(data).float()
    
    def train_model(self, model_type: str, X_train, y_train_encoded, 
                    X_val, y_val_encoded, epochs: int = 100, batch_size: int = 256):
        
        # Calculate Dimensions
        num_classes = len(np.unique(y_train_encoded))
        
        # Initialize Model & Save Config
        if model_type == 'cnn':
            # PyTorch expects (Batch, Channel, Height, Width)
            if len(X_train.shape) == 4 and X_train.shape[-1] == 1:
                X_train = np.transpose(X_train, (0, 3, 1, 2))
                X_val = np.transpose(X_val, (0, 3, 1, 2))
            
            model = SpeakerCNNModel(input_channels=1, num_classes=num_classes)
            self.model_configs['cnn'] = {'num_classes': num_classes}
            
        elif model_type == 'lstm':
            input_size = X_train.shape[2]
            model = SpeakerLSTMModel(input_size=input_size, num_classes=num_classes)
            self.model_configs['lstm'] = {'num_classes': num_classes, 'input_size': input_size}
        
        elif model_type == 'rnn':
            input_size = X_train.shape[2]
            model = SpeakerRNNModel(input_size=input_size, num_classes=num_classes)
            self.model_configs['rnn'] = {'num_classes': num_classes, 'input_size': input_size}
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        model = model.to(self.device)
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        print(f"\nTraining Speaker {model_type.upper()} on {self.device}...")
        print(f"  Model device: {model_device}")
        
        if self.gpu_available:
            if model_device.type != 'cuda':
                print(f"  [WARNING] Model is not on GPU! Expected cuda, got {model_device}")
            else:
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        else:
            if model_device.type != 'cpu':
                print(f"  [WARNING] Model device mismatch! Expected CPU, got {model_device}")
        
        # Setup Training with per-model finetune overrides
        criterion = nn.CrossEntropyLoss()
        cfg = self.cfg
        wd = getattr(cfg, 'weight_decay', 0.0)

        model_key = model_type.lower()
        model_finetune = {}
        if hasattr(cfg, f"{model_key}_finetune"):
            model_finetune = getattr(cfg, f"{model_key}_finetune") or {}

        freeze_flag = model_finetune.get('freeze_backbone', getattr(cfg, 'freeze_backbone', False))
        head_epochs = int(model_finetune.get('finetune_head_epochs', getattr(cfg, 'finetune_head_epochs', 0))) if freeze_flag else 0
        head_lr = model_finetune.get('head_lr', getattr(cfg, 'head_lr', getattr(cfg, 'learning_rate', 0.001)))
        full_phase_lr = model_finetune.get('full_finetune_lr', getattr(cfg, 'full_finetune_lr', getattr(cfg, 'learning_rate', 0.001)))

        # If head-phase configured, freeze backbone otherwise unfreeze all
        if head_epochs > 0:
            self._freeze_backbone_head_only(model)
            initial_lr = head_lr
        else:
            self._unfreeze_all(model)
            initial_lr = getattr(cfg, 'learning_rate', 0.001)

        if getattr(cfg, 'optimizer', 'adamw').lower().startswith('adamw'):
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=getattr(cfg, 'scheduler_patience', 5))
        
        # Use new torch.amp.GradScaler
        scaler = torch.amp.GradScaler('cuda') if self.gpu_available else None
        
        # Loaders
        train_ds = TensorDataset(*self._to_tensor(X_train, y_train_encoded))
        val_ds = TensorDataset(*self._to_tensor(X_val, y_val_encoded))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                pin_memory=self.gpu_available, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                              pin_memory=self.gpu_available, num_workers=0)
        
        best_val_loss = float('inf')
        best_state = None
        history = {'loss': [], 'val_loss': [], 'val_acc': []}
        
        print(f"  Total epochs: {epochs}, Batch size: {batch_size}")
        print(f"  Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
        
        # Early Stopping parameters
        best_acc = 0.0
        best_loss = float('inf')
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
            # Switch from head-phase to full-phase if configured
            if head_epochs > 0 and epoch == head_epochs:
                # Unfreeze whole model and recreate optimizer with full-phase LR
                self._unfreeze_all(model)
                if getattr(cfg, 'optimizer', 'adamw').lower().startswith('adamw'):
                    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=full_phase_lr, weight_decay=wd)
                else:
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=full_phase_lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=getattr(cfg, 'scheduler_patience', 5))

            if epoch == 0:
                epoch_start_time = time.time()
            
            model.train()
            train_loss = 0.0
            
            # Show progress for first epoch and every 5 epochs
            show_progress = (epoch == 0) or ((epoch + 1) % 5 == 0)
            if show_progress and epoch > 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Training...", end='\r')
            
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                # Verify first batch is on correct device
                if epoch == 0 and batch_idx == 0:
                    batch_device = batch_X.device
                    model_device_check = next(model.parameters()).device
                    print(f"  Batch device: {batch_device}, Model device: {model_device_check}")
                    if batch_device != model_device_check:
                        print(f"  [WARNING] Device mismatch! Batch on {batch_device}, Model on {model_device_check}")
                    if self.gpu_available:
                        if batch_device.type != 'cuda':
                            print(f"  [WARNING] Batch is not on GPU! Expected cuda, got {batch_device}")
                        else:
                            print(f"  GPU Memory after first batch: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
                
                optimizer.zero_grad()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                
                # Show progress during first epoch
                if epoch == 0 and batch_idx == 0:
                    print(f"  Epoch [1/{epochs}] Processing batches...", end='\r')
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_acc = correct / total
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
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
                    bad_acc_counter = 0
            
            print(f"  Epoch {epoch+1}/{epochs} | Val Acc: {val_acc*100:.2f}% | Best: {best_acc*100:.2f}% | Patience: {patience_counter}/{patience}{saved_str}{epoch_time_str}")
            
            # Check triggers (only after min_epochs)
            if epoch >= min_epochs:
                # Main condition
                if patience_counter >= patience:
                    print(f"  Early stopping triggered: no improvement ≥{min_delta*100:.1f}% for {patience} epochs")
                    break
                
                # Safety checks
                if bad_acc_counter >= 10:
                    print(f"  Early stopping triggered: Validation accuracy dropped >{safety_acc_drop*100:.0f}% from best for 10 epochs (Possible Overfitting)")
                    break
                
                if bad_loss_counter >= 10:
                    print(f"  Early stopping triggered: Validation loss increased >{safety_loss_rise} from best for 10 epochs (Possible Overfitting)")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
            
        self.trained_models[model_type] = model
        return model, history
    
    def train_all_models(self, spectrograms, sequences, labels, 
                        test_size=0.2, validation_size=0.1, random_state=42,
                        epochs=100, batch_size=256):
        """Train CNN, LSTM, and RNN models."""
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Split Data (Spectrograms)
        X_tr_spec, X_te_spec, y_tr, y_te = train_test_split(
            spectrograms, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded
        )
        X_tr_spec, X_val_spec, y_tr, y_val = train_test_split(
            X_tr_spec, y_tr, test_size=validation_size, random_state=random_state, stratify=y_tr
        )
        
        # Split Data (Sequences - use same seeds to match)
        X_tr_seq, X_te_seq, y_tr_seq, y_te_seq = train_test_split(
            sequences, labels_encoded, test_size=test_size, random_state=random_state, stratify=labels_encoded
        )
        X_tr_seq, X_val_seq, y_tr_seq, y_val_seq = train_test_split(
            X_tr_seq, y_tr_seq, test_size=validation_size, random_state=random_state, stratify=y_tr_seq
        )
        
        results = {
            'models': {}, 
            'y_test': y_te,
            'y_test_encoded': y_te,  # For compatibility
            'X_test_spec': X_te_spec,
            'X_test_seq': X_te_seq,
            'label_encoder': self.label_encoder
        }
        
        # Train CNN
        try:
            cnn, hist = self.train_model('cnn', X_tr_spec, y_tr, X_val_spec, y_val, epochs, batch_size)
            results['models']['cnn'] = {'model': cnn, 'history': hist}
        except Exception as e:
            print(f"CNN Failed: {e}")
            import traceback
            traceback.print_exc()

        # Train LSTM
        try:
            lstm, hist = self.train_model('lstm', X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, epochs, batch_size)
            results['models']['lstm'] = {'model': lstm, 'history': hist}
        except Exception as e:
            print(f"LSTM Failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Train RNN
        try:
            print("\nStarting RNN Training...")
            # Calculate num_classes from labels (same as other models)
            num_classes = len(np.unique(labels_encoded))
            input_size = sequences.shape[2]
            rnn = SpeakerRNNModel(input_size=input_size, num_classes=num_classes).to(self.device)
            self.model_configs['rnn'] = {'num_classes': num_classes, 'input_size': input_size}
            
            rnn_model, hist = self.train_model('rnn', X_tr_seq, y_tr_seq, X_val_seq, y_val_seq, epochs, batch_size)
            results['models']['rnn'] = {'model': rnn_model, 'history': hist}
        except Exception as e:
            print(f"RNN Failed: {e}")
            import traceback
            traceback.print_exc()
            
        return results
    
    def save_model(self, model_type: str):
        if model_type not in self.trained_models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        filepath = os.path.join(self.models_dir, f'speaker_{model_type}.pth')
        
        save_dict = {
            'model_state_dict': self.trained_models[model_type].state_dict(),
            'label_encoder': self.label_encoder,
            'config': self.model_configs.get(model_type, {}),
            'model_type': model_type  # Save model type for loading
        }
        
        torch.save(save_dict, filepath)
        print(f"Saved {model_type} to {filepath}")

    def load_model(self, model_type: str):
        filepath = os.path.join(self.models_dir, f'speaker_{model_type}.pth')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found")
        
        # weights_only=False is safe here since we're loading our own saved models
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        # Use saved model_type if available, otherwise use parameter
        saved_model_type = checkpoint.get('model_type', model_type)
        
        # Auto-configure model based on saved config
        if saved_model_type == 'cnn':
            model = SpeakerCNNModel(num_classes=config.get('num_classes', 10))
        elif saved_model_type == 'lstm':
            model = SpeakerLSTMModel(
                input_size=config.get('input_size', 13),
                num_classes=config.get('num_classes', 10)
            )
        elif saved_model_type == 'rnn':
            model = SpeakerRNNModel(
                input_size=config.get('input_size', 13),
                num_classes=config.get('num_classes', 10)
            )
        else:
            raise ValueError(f"Unknown model type: {saved_model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        self.label_encoder = checkpoint['label_encoder']
        self.trained_models[model_type] = model
        print(f"Loaded {model_type} model.")
        return model
    
    def predict(self, X, model_type: str = 'cnn'):
        """
        Predict speaker from input data.
        
        Args:
            X: Input data (spectrogram or sequence)
            model_type: 'cnn', 'lstm', or 'rnn'
            
        Returns:
            Predicted speaker labels and probabilities
        """
        if model_type not in self.trained_models:
            # Try to load the model
            try:
                self.load_model(model_type)
            except:
                raise ValueError(f"Model {model_type} not trained yet")
        
        model = self.trained_models[model_type]
        model.eval()
        
        # Prepare input
        if model_type == 'cnn':
            if len(X.shape) == 4 and X.shape[-1] == 1:
                X = np.transpose(X, (0, 3, 1, 2))
            X_tensor = torch.from_numpy(X).float().to(self.device)
        else:  # lstm or rnn
            X_tensor = torch.from_numpy(X).float().to(self.device)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
        
        # Decode labels
        predictions_decoded = self.label_encoder.inverse_transform(predictions)
        
        return predictions_decoded, probabilities

def train(models_dir='models', test_size=0.2, validation_size=0.1, random_state=42, epochs: int = None, batch_size: int = None, cfg: DLParamConfig = None):
    """
    Complete training pipeline for speaker identification DL models.
    
    Args:
        models_dir: Directory to save trained models and cache files
        test_size: Proportion of data to use for testing
        validation_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("="*60)
    print("Training Speaker Identification DL Models")
    print("="*60)
    
    os.makedirs(models_dir, exist_ok=True)

    # Configuration: prefer explicit args, otherwise use cfg defaults
    if cfg is None:
        cfg = DLParamConfig()
    if epochs is None:
        epochs = cfg.epochs
    if batch_size is None:
        batch_size = cfg.batch_size
    
    # Load datasets
    print("\n[1/6] Loading datasets and extracting speaker IDs...")
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
            if isinstance(all_speakers, np.ndarray):
                all_speakers = list(all_speakers)
            
            # Validate that the cache contains actual data
            if len(all_audio) == 0 or len(all_speakers) == 0:
                print(f"  [WARNING] Cache file is empty. Regenerating...")
                all_audio = []
                all_speakers = []
                try:
                    os.remove(preprocessed_audio_file)
                except PermissionError:
                    old_file = preprocessed_audio_file + '.old'
                    if os.path.exists(old_file):
                        os.remove(old_file)
                    os.rename(preprocessed_audio_file, old_file)
                    print(f"  Renamed empty cache file to {old_file}")
            elif len(all_audio) != len(all_speakers):
                print(f"  [WARNING] Cache file has mismatched lengths (audio: {len(all_audio)}, speakers: {len(all_speakers)}). Regenerating...")
                all_audio = []
                all_speakers = []
                try:
                    os.remove(preprocessed_audio_file)
                except PermissionError:
                    old_file = preprocessed_audio_file + '.old'
                    if os.path.exists(old_file):
                        os.remove(old_file)
                    os.rename(preprocessed_audio_file, old_file)
                    print(f"  Renamed invalid cache file to {old_file}")
        except (ValueError, KeyError, Exception) as e:
            print(f"  [WARNING] Cache file is incompatible or corrupted: {e}")
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
        all_audio = []
        all_speakers = []
        
        # Load all datasets
        audio_files, emotion_labels = loader.load_all_datasets()
        
        # Extract speaker IDs
        dataset_counts = {'TESS': 0, 'SAVEE': 0, 'RAVDESS': 0, 'CREMA-D': 0, 'UNKNOWN': 0}
        failed_files = 0
        
        for i, audio_file in enumerate(audio_files):
            # Determine dataset from path
            dataset = None
            audio_file_upper = audio_file.upper()
            if 'TESS' in audio_file_upper:
                dataset = 'TESS'
            elif 'SAVEE' in audio_file_upper:
                dataset = 'SAVEE'
            elif 'RAVDESS' in audio_file_upper:
                dataset = 'RAVDESS'
            elif 'CREMA' in audio_file_upper:
                dataset = 'CREMA-D'
            
            if dataset:
                try:
                    speaker_id = extract_speaker_id_from_filename(audio_file, dataset)
                    if speaker_id is None or speaker_id == '':
                        dataset_counts['UNKNOWN'] += 1
                        failed_files += 1
                        continue
                    
                    audio = loader.preprocess_audio(audio_file)
                    if len(audio) > 0:
                        all_audio.append(audio)
                        all_speakers.append(speaker_id)
                        dataset_counts[dataset] += 1
                    else:
                        failed_files += 1
                except Exception as e:
                    failed_files += 1
                    if (i + 1) % 1000 == 0:
                        print(f"  Warning: Error processing file {audio_file}: {e}")
            else:
                dataset_counts['UNKNOWN'] += 1
                failed_files += 1
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(audio_files)} files (successful: {len(all_audio)}, failed: {failed_files})")
        
        if len(all_audio) == 0:
            raise ValueError(f"No audio files were successfully processed. Failed files: {failed_files}. Please check your dataset paths and file formats.")
        
        # Save preprocessed audio for future use
        save_preprocessed_audio(all_audio, all_speakers, preprocessed_audio_file)
    
    print(f"  Preprocessed {len(all_audio)} audio files")
    print(f"  Found {len(set(all_speakers))} unique speakers")
    
    # Prepare data for DL models
    print(f"\n[2/6] Preparing data for DL models...")
    dl_data_file = os.path.join(models_dir, 'dl_data_speakers.npz')
    dl_trainer = SpeakerDLTrainer(models_dir=models_dir)
    extractor = FeatureExtractor()
    
    if os.path.exists(dl_data_file):
        print(f"  Loading DL data from cache...")
        data = np.load(dl_data_file, allow_pickle=True)
        spectrograms = data['spectrograms']
        sequences = data['sequences']
        dl_speakers = data['labels']
        # Ensure speakers match
        if len(dl_speakers) == len(all_speakers):
            all_speakers = list(dl_speakers)
        else:
            print(f"  [WARNING] Speaker count mismatch. Re-extracting DL data...")
            spectrograms = None
            sequences = None
    else:
        spectrograms = None
        sequences = None
    
    if spectrograms is None or sequences is None:
        print(f"  Extracting DL data (this may take a while)...")
        # Prepare spectrograms for CNN
        print("  Extracting spectrograms for CNN...")
        spectrograms = dl_trainer.prepare_spectrograms(all_audio, extractor)
        print(f"  Spectrogram shape: {spectrograms.shape}")
        
        # Prepare sequences for LSTM/RNN
        print("  Extracting sequences for LSTM/RNN...")
        sequences = dl_trainer.prepare_sequences(all_audio, extractor)
        print(f"  Sequence shape: {sequences.shape}")
        
        # Save DL data
        np.savez_compressed(dl_data_file, 
                           spectrograms=spectrograms,
                           sequences=sequences,
                           labels=np.array(all_speakers))
    
    # Validate that we have data before proceeding
    if len(all_audio) == 0 or len(all_speakers) == 0:
        raise ValueError("No audio files or speakers found. Please check your dataset paths and ensure files are being loaded correctly.")
    
    if len(spectrograms) == 0 or len(sequences) == 0:
        raise ValueError("No spectrograms or sequences extracted. Please check that audio files are valid and preprocessing is working correctly.")
    
    speakers = np.array(all_speakers)
    
    # Train DL models
    print(f"\n[3/6] Training DL models...")
    dl_results = dl_trainer.train_all_models(
        spectrograms, sequences, speakers,
        test_size=test_size, validation_size=validation_size, random_state=random_state,
        epochs=epochs, batch_size=batch_size
    )
    
    # Evaluate DL models
    print(f"\n[4/6] Evaluating DL models...")
    evaluator = ModelEvaluator(results_dir='results')
    
    # Get unique speakers present in test set
    y_test_decoded = dl_trainer.label_encoder.inverse_transform(dl_results['y_test_encoded'])
    present_speakers = sorted(list(set(y_test_decoded)))
    
    best_model = None
    best_score = 0
    
    model_types = ['cnn', 'lstm', 'rnn']
    available_models = [mt for mt in model_types if mt in dl_results['models']]
    
    for idx, model_type in enumerate(available_models, 1):
        print(f"  Evaluating {model_type.upper()} model ({idx}/{len(available_models)})...")
        
        model_data = dl_results['models'][model_type]
        model = model_data['model']
        device = dl_trainer.device
        
        # Get appropriate test data
        if model_type == 'cnn':
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
        y_test_decoded = dl_trainer.label_encoder.inverse_transform(y_test_encoded)
        y_pred_decoded = dl_trainer.label_encoder.inverse_transform(y_pred_classes)
        
        print(f"    Computing evaluation metrics...")
        # Evaluate
        eval_results = evaluator.evaluate_classification(
            y_test_decoded, y_pred_decoded, labels=present_speakers
        )
        evaluator.print_evaluation_results(eval_results, model_type.upper(), labels=present_speakers)
        
        print(f"    Saving confusion matrix...")
        evaluator.plot_confusion_matrix(
            eval_results['confusion_matrix'], present_speakers, f"{model_type.upper()}_speaker"
        )
        
        print(f"    Saving results...")
        # Add _speaker suffix to filename and Speaker ID to header
        model_save_name = f"{model_type.upper()} (Speaker ID)"
        evaluator.save_results(
            eval_results, 
            model_save_name, 
            filepath=os.path.join(evaluator.results_dir, f'evaluation_{model_type.upper()}_speaker.txt'),
            labels=present_speakers
        )
        
        print(f"    Saving model...")
        dl_trainer.save_model(model_type)
        
        if eval_results['accuracy'] > best_score:
            best_score = eval_results['accuracy']
            best_model = model_type.upper()
        
        print(f"  ✓ {model_type.upper()} evaluation completed\n")
    
    print(f"\n[5/6] Training completed!")
    print(f"\n[6/6] Summary")
    print(f"{'='*60}")
    print(f"Best model: {best_model} (Accuracy: {best_score:.4f})")
    print(f"{'='*60}")
    # Save used config for reproducibility
    try:
        cfg.save(os.path.join(models_dir, 'dl_config_speaker.json'))
        print(f"Saved DL config to {os.path.join(models_dir, 'dl_config_speaker.json')}")
    except Exception:
        pass

    # Build and save metadata (speaker mapping, models, config summary)
    try:
        # Speaker mapping index -> label
        speaker_mapping = {int(i): lbl for i, lbl in enumerate(dl_trainer.label_encoder.classes_)}
        model_file_paths = {mt: f"speaker_{mt}.pth" for mt in available_models}
        finetune_blocks = {}
        for key in ['cnn_finetune', 'lstm_finetune', 'rnn_finetune']:
            if hasattr(cfg, key):
                finetune_blocks[key] = getattr(cfg, key)
        metadata = {
            'timestamp': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
            'models_dir': models_dir,
            'model_types_trained': available_models,
            'best_model': best_model,
            'best_model_accuracy': best_score,
            'speaker_count': len(speaker_mapping),
            'speaker_mapping': speaker_mapping,
            'model_file_paths': model_file_paths,
            'epochs': epochs,
            'batch_size': batch_size,
            'test_size': test_size,
            'validation_size': validation_size,
            'random_state': random_state,
            'config_overrides': {k: getattr(cfg, k) for k in ['learning_rate', 'weight_decay', 'optimizer', 'scheduler_patience'] if hasattr(cfg, k)},
            'finetune': finetune_blocks,
            'model_configs': dl_trainer.model_configs
        }
        dl_trainer.save_metadata(metadata)
    except Exception as e:
        print(f"[WARNING] Failed to generate speaker DL metadata: {e}")


if __name__ == '__main__':
    train()

