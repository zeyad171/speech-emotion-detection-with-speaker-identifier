## Quick orientation for AI assistants

- Purpose: this repo implements a full pipeline for speech emotion detection and optional speaker identification (data loading → feature extraction → ML/DL training → Streamlit UI).
- Root entrypoints: `main.py` (CLI modes: `explore`, `train`, `train_emotion_ml`, `train_emotion_dl`, `train_speaker_ml`, `train_speaker_dl`) and `app/app.py` (Streamlit UI).

## High-level architecture (what to read first)
- `src/data_loader.py` — unified loader for TESS, SAVEE, RAVDESS, CREMA-D; standardizes emotion labels and preprocesses audio (resample to 22,050 Hz, normalize, trim silence).
- `src/feature_extraction.py` — FeatureExtractor builds the 65-dimensional ML feature vector and produces mel-spectrograms / MFCC sequences used by DL models.
- `src/models/` — four self-contained trainers (emotion_ml.py, emotion_dl.py, speaker_ml.py, speaker_dl.py). Each file exposes a train() function and save/load conventions.
- `app/app.py` — Streamlit front-end; loads cached models (ML `.pkl` and DL `.pth`) and uses `FeatureExtractor` + `EmotionDatasetLoader` at inference time.

## Naming & file conventions (important for edits)
- ML models: saved as `models/{model_name}.pkl` (e.g. `random_forest.pkl`); scaler at `models/scaler.pkl`.
- DL models: saved as PyTorch files `models/{model}_best.pth` or `models/{model}.pth`. The `.pth` contains `model_state_dict`, `label_encoder`, and `model_config`.
- Speaker models: ML `models/speaker_{model}.pkl`, DL `models/speaker_{type}.pth`, aggregated metadata `models/speaker_models_metadata.pkl` and `models/speaker_metadata.json`.
- Cached pipeline artifacts: `models/preprocessed_audio.npz`, `models/extracted_features.npz`, `models/dl_data.npz` (or `_speakers` variants). Delete these to force re-extraction.

## Developer workflows and exact commands
- Install: `pip install -r requirements.txt` (GPU PyTorch requires following INSTALL_GPU_PYTORCH.md instructions).
- Explore datasets: `python main.py --mode explore` (calls `EmotionDatasetLoader.explore_datasets()`).
- Train all emotion ML models: `python src/models/emotion_ml.py` or `python main.py --mode train_emotion_ml`.
- Train all emotion DL models: `python src/models/emotion_dl.py` or `python main.py --mode train_emotion_dl`.
- Train speaker ML/DL: `python src/models/speaker_ml.py` / `python src/models/speaker_dl.py` or via `main.py` modes.
- Run UI: `streamlit run app/app.py` (do NOT run `app/app.py` directly — the file will attempt to auto-launch Streamlit if invoked outside Streamlit).

Generating DL JSON configs
--------------------------
- Use the helper script `scripts/generate_dl_config.py` to create DL JSON configs for `emotion` or `speaker` training. The script supports `--model-type`, `--include-hpo`, and multiple `--set key=value` overrides using dot notation for nested keys (e.g. `cnn_finetune.head_lr=0.001`).

Examples (PowerShell):

```powershell
python .\scripts\generate_dl_config.py --target emotion --model-type cnn --include-hpo --output models/my_emotion_cfg.json

python .\scripts\generate_dl_config.py --target speaker --model-type lstm --set lstm_finetune.head_lr=0.0005 --output models/my_speaker_cfg.json
```

The produced JSON files are directly consumable by `main.py --cfg` and the DL trainers (`train_emotion_dl` / `train_speaker_dl`).

## Project-specific patterns & gotchas for code generation
- Self-contained trainers: each model file implements a full pipeline (load → preprocess → extract → train → evaluate → save). When changing training parameters, update the train() signature in that file and the `main.py` mode calls.
- Feature vector size is 65. Any change to `FeatureExtractor.extract_all_features()` must be propagated to ML model expectations and any saved model configs. Search for '65' or `extract_all_features` when modifying features.
- Caching: many pipeline steps rely on `.npz` caches. When testing changes to feature extraction or preprocessing, delete the corresponding cache files in `models/` to avoid stale data.
- Windows platform handling: code explicitly forces `n_jobs=1` and `num_workers=0` on Windows to avoid multiprocessing issues — be careful when introducing parallelism.
- DL inputs differ from ML inputs: DL uses spectrograms (CNN) or MFCC sequences (LSTM/RNN); ML uses flattened 65-d feature vectors. Converter helpers: `prepare_spectrograms()` and `prepare_sequences()` in DL trainers and `FeatureExtractor.extract_all_features()` for ML.

## Integration points (what to change to add features)
- To add a dataset: extend `EmotionDatasetLoader` with a loader function and include it in `load_all_datasets()`; keep label mapping consistent with `EMOTION_MAP`.
- To add a new ML model: add it to `src/models/emotion_ml.py`'s `self.models` and update saving logic (file name pattern `{model_name}.pkl`).
- To add a new DL model: add the model class to `emotion_dl.py` or `speaker_dl.py`, include it in `train_all_models()`, and follow existing save/load dict structure.

## Quick examples agents should use when editing
- Use `FeatureExtractor.extract_all_features(audio)` to get ML features (65-d) during inference or tests.
- Load a trained ML model: joblib.load('models/random_forest.pkl') and scaler joblib.load('models/scaler.pkl'), then scaler.transform(features.reshape(1,-1)).
- Load a trained DL model: EmotionDLTrainer().load_model('models/cnn_best.pth') which restores model and label encoder.

## Testing and debugging tips
- If predictions look wrong, check: whether features were re-extracted, whether scaler.pkl matches the model, and whether dataset caches are stale (delete `models/*.npz`).
- For Windows CI/development, keep `n_jobs=1` and `num_workers=0` when invoking GridSearchCV/DataLoader.
- For GPU debugging, check `torch.cuda.is_available()` and the module-level `configure_gpu()` prints in `src/models/emotion_dl.py` and `src/models/speaker_dl.py`.

## Where to look next (important files)
- `src/data_loader.py`, `src/feature_extraction.py`, `src/utils.py`, `src/evaluation.py`
- `src/models/emotion_ml.py`, `src/models/emotion_dl.py`, `src/models/speaker_ml.py`, `src/models/speaker_dl.py`
- `app/app.py`, `main.py`, `requirements.txt`, `PROJECT_DOCUMENTATION.md`

If any section is unclear or you'd like me to add example edits/tests (unit test stubs for a trainer, or a small script that loads a model and runs inference), tell me which area to expand and I'll iterate.

### Recent edits (today)

- Implemented `DLParamConfig` dataclass at `src/models/dl_param_config.py` for reproducible DL run configs.
- Added manual hyperparameter runner `scripts/run_manual_dl_hyperparams.py` for hand-editable experiment lists.
- Added optional HPO helper `src/models/dl_hpo.py` (Optuna wrapper + random-search fallback).
- Wired the `--cfg` CLI argument in `main.py` to load a JSON DL config and pass it to DL trainers (`train_emotion_dl`, `train_speaker_dl`).
- Added JSON evaluation exporters in ML trainers and visualization helpers in `src/visualization.py` including `plot_overfitting_detection()`.

Additional per-file notes (today's edits):

- `src/visualization.py`:
	- New helpers: `plot_model_comparison()`, `plot_confusion_matrix_grid()`, `plot_calibration_curves()`, `plot_per_class_heatmap()`, `plot_embeddings_tsne()`, plus speaker-focused helpers `plot_speaker_probabilities()` and `plot_feature_comparison()`.
	- `plot_overfitting_detection(metrics, histories, results_dir)` writes ML train-vs-test and DL learning-curve PNGs into `results/` for quick overfitting checks.
	- Heatmap annotation formatting was fixed to avoid seaborn `fmt` errors when confusion matrices contain floats.

- `src/models/emotion_ml.py`:
	- `save_eval_json()` and `collect_evaluations_for_visualization()` added to export/load evaluation JSONs (`results/evaluation_{model}.json`) used by the visualizer.
	- `retrain_on_full_data()` added: refits scaler on full data and saves final model, scaler, and metadata JSON.
	- Training flow now saves evaluation JSONs after each model eval and calls `plot_overfitting_detection()` at the end.

- `src/models/speaker_ml.py`:
	- Added `save_eval_json()`, `collect_evaluations_for_visualization()` for visualization compatibility.
	- Conservative HPO support added (`param_grids`, `RandomizedSearchCV`/`GridSearchCV`) and best-params saving to `models/best_hyperparameters_speaker.json`.
	- Safety fixes for top-k accuracy and `retrain_on_full_data()` implemented. Training flow calls `plot_overfitting_detection()` as well.

How to run a DL experiment with a saved config (PowerShell examples):

1. Activate your venv:

```powershell
.\.myenv\Scripts\Activate.ps1
```

2. Run emotion DL training with a saved config:

```powershell
python .\main.py --mode train_emotion_dl --cfg models/my_emotion_cfg.json --test_size 0.2
```

3. Run the manual hyperparam script (edit `MANUAL_CONFIGS` then run):

```powershell
python .\scripts\run_manual_dl_hyperparams.py --target emotion --model_type cnn
```

If you want me to wire more `DLParamConfig` fields into the trainer internals (optimizer, scheduler, weight decay, etc.), I can patch the trainers to accept and use those fields. Say "wire cfg fields" and I'll implement it.
