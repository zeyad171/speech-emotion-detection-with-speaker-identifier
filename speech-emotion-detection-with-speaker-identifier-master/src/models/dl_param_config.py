from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional
import json
from pathlib import Path


@dataclass
class DLParamConfig:
    # General training
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: Optional[str] = "reduce_on_plateau"
    scheduler_patience: int = 5
    early_stopping_patience: int = 8
    seed: int = 42

    # Model specifics (kept generic; trainers may selectively use)
    in_channels: int = 1
    dropout: float = 0.4
    hidden_size: int = 128
    num_layers: int = 2

    # Finetuning phases
    finetune_head_epochs: int = 5
    finetune_full_epochs: int = 25
    head_lr: Optional[float] = 1e-3
    full_finetune_lr: Optional[float] = 1e-4
    freeze_backbone: bool = True
    freeze_backbone_until_epoch: int = 5

    # Augmentation / data
    augment_spec_augment: bool = True
    augment_noise: bool = False
    noise_level: float = 0.005

    # Runtime
    num_workers: int = 0
    device: Optional[str] = None

    # Misc
    notes: str = ""

    # Per-model finetune overrides (editable per-model)
    cnn_finetune: Dict[str, Any] = field(default_factory=lambda: {
        'freeze_backbone': True,
        'finetune_head_epochs': 5,
        'finetune_full_epochs': 25,
        'head_lr': 1e-3,
        'full_finetune_lr': 1e-4
    })
    lstm_finetune: Dict[str, Any] = field(default_factory=lambda: {
        'freeze_backbone': True,
        'finetune_head_epochs': 5,
        'finetune_full_epochs': 40,
        'head_lr': 1e-4,
        'full_finetune_lr': 1e-4
    })
    rnn_finetune: Dict[str, Any] = field(default_factory=lambda: {
        'freeze_backbone': True,
        'finetune_head_epochs': 5,
        'finetune_full_epochs': 30,
        'head_lr': 1e-3,
        'full_finetune_lr': 1e-4
    })

    def update_from_dict(self, overrides: Dict[str, Any]) -> None:
        for k, v in overrides.items():
            if hasattr(self, k):
                current = getattr(self, k)
                # If both current and incoming values are dict-like, merge them
                if isinstance(current, dict) and isinstance(v, dict):
                    current.update(v)
                    setattr(self, k, current)
                else:
                    setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DLParamConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        cfg = cls()
        cfg.update_from_dict(data)
        return cfg

    def hpo_space(self, model_type: str = "cnn") -> Dict[str, Any]:
        """
        Return a simple HPO search-space dictionary for the given model_type.
        Keys map to parameter names and values are lists of candidate values.
        """
        # Common parameter search candidates
        common = {
            'learning_rate': [1e-4, 3e-4, 1e-3, 3e-3],
            'weight_decay': [0.0, 1e-6, 1e-5, 1e-4],
            'batch_size': [32, 64, 128, 256],
            'dropout': [0.2, 0.3, 0.4, 0.5],
        }

        # Finetune candidate sets (used as nested dicts so callers/tools can
        # optionally expand them into flat param grids). These match the
        # per-model finetune override dicts present on the dataclass
        finetune_candidates = {
            'freeze_backbone': [True, False],
            'finetune_head_epochs': [3, 5, 8],
            'finetune_full_epochs': [10, 20, 30],
            'head_lr': [1e-4, 3e-4, 1e-3],
            'full_finetune_lr': [1e-5, 1e-4, 1e-3]
        }

        mt = model_type.lower()
        if mt == 'cnn':
            common.update({
                'backbone': ['resnet18', 'resnet34', None],
                'pretrained': [True, False],
            })
            # put finetune candidates under a model-scoped key so HPO runners
            # can target per-model overrides explicitly (e.g. cfg.cnn_finetune)
            common['cnn_finetune'] = finetune_candidates.copy()
            # small extra candidates that are often useful for CNNs
            common['cnn_finetune'].update({'backbone_lr_multiplier': [0.1, 0.5, 1.0]})
        elif mt in ('lstm', 'rnn'):
            common.update({
                'hidden_size': [128, 256, 512],
                'num_layers': [1, 2, 3, 4],
                'learning_rate': [1e-4, 5e-4, 1e-3, 3e-3],
            })
            common[f"{mt}_finetune"] = finetune_candidates.copy()

        return common
