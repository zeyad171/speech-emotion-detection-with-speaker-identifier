"""Generate DL JSON configs for emotion or speaker training.

Usage examples (PowerShell):

# Generate a default emotion config and include HPO
python .\scripts\generate_dl_config.py --target emotion --model-type cnn --include-hpo --output models/my_emotion_cfg.json

# Generate a speaker config, override head_lr, and save
python .\scripts\generate_dl_config.py --target speaker --model-type lstm --set cnn_finetune.head_lr=0.0005 --output models/my_speaker_cfg.json

The script supports multiple `--set key=value` overrides where `key` can use dot
notation for nested dicts (e.g. `cnn_finetune.head_lr=1e-3`).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Make script runnable from any CWD by adding repo root to sys.path so
# `import src...` works. This mirrors how other scripts handle imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.dl_param_config import DLParamConfig


def parse_set_arg(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"Invalid set argument '{s}', expected key=value")
    k, v = s.split("=", 1)
    # Try to interpret common literal types via json.loads
    try:
        parsed = json.loads(v)
    except Exception:
        # Fallback to string
        parsed = v
    return k, parsed


def nested_update(d: Dict[str, Any], key_path: str, value: Any):
    """Update dict `d` by walking dot-separated `key_path` and setting value."""
    parts = key_path.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def build_overrides_from_sets(sets: list[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for s in sets:
        k, v = parse_set_arg(s)
        nested_update(overrides, k, v)
    return overrides


def main():
    p = argparse.ArgumentParser(description="Generate DLParamConfig JSON for DL training")
    p.add_argument("--target", choices=["emotion", "speaker", "both"], required=True, help="Only affects default output name; use 'both' for combined configs")
    p.add_argument("--model-type", choices=["cnn", "lstm", "rnn"], help="Model architecture to set in the config")
    p.add_argument("--include-hpo", action="store_true", help="Include `hpo_space` for the selected model in the saved JSON")
    p.add_argument("--set", action="append", default=[], help="Override a config value using key=value (dot for nested keys). Can be passed multiple times.")
    p.add_argument("--output", default=None, help="Output JSON path (default: models/my_<target>_cfg.json)")
    args = p.parse_args()

    cfg = DLParamConfig()

    # Apply overrides
    if args.set:
        overrides = build_overrides_from_sets(args.set)
        cfg.update_from_dict(overrides)

    out = cfg.to_dict()

    if args.include_hpo:
        mt = args.model_type or "cnn"
        out["hpo_space"] = cfg.hpo_space(model_type=mt)

    # Ensure any legacy field like `model_arch` is not emitted
    out.pop("model_arch", None)

    if args.output:
        out_path = Path(args.output)
    else:
        default_name = "my_full_dl_cfg.json" if args.target == "both" else f"my_{args.target}_cfg.json"
        out_path = Path("models") / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote config to {out_path.resolve()}")


if __name__ == "__main__":
    main()
