"""
Visualization helper functions for speaker comparison and analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import json
import os
import seaborn as sns

# Optional sklearn helpers used for calibration and embedding visualization
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE


def plot_speaker_probabilities(speaker_probs: Dict[str, float], 
                               top_k: int = 10,
                               title: str = "Speaker Probability Distribution") -> plt.Figure:
    """
    Plot bar chart of speaker probabilities.
    
    Args:
        speaker_probs: Dictionary mapping speaker IDs to probabilities
        top_k: Number of top speakers to display
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Sort by probability and get top K
    sorted_speakers = sorted(speaker_probs.items(), key=lambda x: x[1], reverse=True)[:top_k]
    speakers = [s[0] for s in sorted_speakers]
    probs = [s[1] * 100 for s in sorted_speakers]  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(speakers, probs, color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax.text(prob + 1, i, f'{prob:.2f}%', va='center', fontsize=9)
    
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_ylabel('Speaker ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(probs) * 1.15 if probs else 100)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_comparison(speaker_features: Dict[str, np.ndarray],
                           feature_names: List[str] = None,
                           top_features: int = 10,
                           title: str = "Feature Comparison") -> plt.Figure:
    """
    Compare features between speakers using bar charts.
    
    Args:
        speaker_features: Dictionary mapping speaker IDs to feature arrays
        feature_names: Names of features (if None, uses indices)
        top_features: Number of top differentiating features to show
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    speakers = list(speaker_features.keys())
    if len(speakers) < 2:
        raise ValueError("Need at least 2 speakers for comparison")
    
    # Calculate feature differences
    features_array = np.array([speaker_features[s] for s in speakers])
    feature_diffs = np.std(features_array, axis=0)  # Standard deviation across speakers
    
    # Get top differentiating features
    top_indices = np.argsort(feature_diffs)[-top_features:][::-1]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_diffs))]
    
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(top_feature_names))
    width = 0.35
    
    for i, speaker in enumerate(speakers):
        offset = (i - len(speakers)/2 + 0.5) * width / len(speakers)
        values = [speaker_features[speaker][idx] for idx in top_indices]
        ax.bar(x + offset, values, width/len(speakers), label=speaker, alpha=0.7)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Feature Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_radar_chart(speaker_features: Dict[str, np.ndarray],
                     feature_names: List[str] = None,
                     selected_features: List[int] = None,
                     title: str = "Speaker Feature Radar Chart") -> plt.Figure:
    """
    Create radar chart for multi-dimensional feature comparison.
    
    Args:
        speaker_features: Dictionary mapping speaker IDs to feature arrays
        feature_names: Names of features
        selected_features: Indices of features to include (if None, uses all or top 8)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    speakers = list(speaker_features.keys())
    
    # Select features
    if selected_features is None:
        if feature_names is None or len(feature_names) > 8:
            # Use top 8 most variable features
            features_array = np.array([speaker_features[s] for s in speakers])
            feature_vars = np.var(features_array, axis=0)
            selected_features = np.argsort(feature_vars)[-8:][::-1].tolist()
        else:
            selected_features = list(range(len(feature_names)))
    
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(len(speaker_features[speakers[0]]))]
    
    selected_feature_names = [feature_names[i] for i in selected_features]
    n_features = len(selected_features)
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize features to 0-1 range for each speaker
    for speaker in speakers:
        values = speaker_features[speaker][selected_features]
        # Normalize
        values_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
        values_norm = values_norm.tolist()
        values_norm += values_norm[:1]  # Complete the circle
        
        ax.plot(angles, values_norm, 'o-', linewidth=2, label=speaker, alpha=0.7)
        ax.fill(angles, values_norm, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(selected_feature_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    return fig


def plot_similarity_scores(similar_speakers: List[Tuple[str, float]],
                           title: str = "Top-K Similar Speakers") -> plt.Figure:
    """
    Visualize top-K similar speakers with similarity scores.
    
    Args:
        similar_speakers: List of (speaker_id, similarity_score) tuples
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    speakers = [s[0] for s in similar_speakers]
    scores = [s[1] * 100 for s in similar_speakers]  # Convert to percentage
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(speakers, scores, color='coral', alpha=0.7)
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2, 
                f'{score:.2f}%', va='center', fontsize=10)
    
    ax.set_xlabel('Similarity Score (%)', fontsize=12)
    ax.set_ylabel('Speaker ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(scores) * 1.15 if scores else 100)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def calculate_speaker_similarity(features1: np.ndarray, features2: np.ndarray,
                                method: str = 'cosine') -> float:
    """
    Calculate similarity between two speaker feature vectors.
    
    Args:
        features1: Feature vector for speaker 1
        features2: Feature vector for speaker 2
        method: 'cosine' or 'euclidean'
        
    Returns:
        Similarity score (0-1 for cosine, distance for euclidean)
    """
    if method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
    elif method == 'euclidean':
        # Euclidean distance (inverted and normalized)
        distance = np.linalg.norm(features1 - features2)
        # Normalize to 0-1 range (assuming max distance is around 100)
        similarity = 1.0 / (1.0 + distance / 100.0)
        return similarity
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def get_top_k_similar_speakers(query_features: np.ndarray,
                               speaker_features: Dict[str, np.ndarray],
                               k: int = 5,
                               method: str = 'cosine') -> List[Tuple[str, float]]:
    """
    Get top-K most similar speakers to query features.
    
    Args:
        query_features: Feature vector of query speaker
        speaker_features: Dictionary mapping speaker IDs to feature arrays
        k: Number of similar speakers to return
        method: 'cosine' or 'euclidean'
        
    Returns:
        List of (speaker_id, similarity_score) tuples, sorted by similarity
    """
    similarities = []
    for speaker_id, features in speaker_features.items():
        similarity = calculate_speaker_similarity(query_features, features, method)
        similarities.append((speaker_id, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:k]


def load_speaker_metadata(metadata_file: str = 'models/speaker_metadata.json',
                          dl_metadata_file: str = 'models/speaker_dl_metadata.json',
                          ml_metadata_pkl: str = 'models/speaker_models_metadata.pkl') -> Dict:
    """Load speaker metadata with graceful fallbacks.

    Resolution order:
    1. Explicit JSON (legacy): models/speaker_metadata.json containing full per-speaker stats.
    2. New DL metadata JSON: models/speaker_dl_metadata.json (build a minimal 'speakers' map from label encoder mapping if detailed stats absent).
    3. ML pickle metadata: models/speaker_models_metadata.pkl (extract label encoder classes only, create minimal structure).

    Returns:
        Dict with at least a 'speakers' key mapping speaker_id -> stats dict (possibly minimal if only mapping available).
    """
    # 1. Legacy JSON with full stats
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            # Ensure 'speakers' key
            if 'speakers' not in data:
                data['speakers'] = {}
            return data
        except Exception:
            pass

    # 2. New DL metadata JSON
    if os.path.exists(dl_metadata_file):
        try:
            with open(dl_metadata_file, 'r') as f:
                dl_meta = json.load(f)
            speaker_mapping = dl_meta.get('speaker_mapping', {})
            speakers_struct = {}
            for idx_str, spk in speaker_mapping.items():
                speakers_struct[spk] = {
                    'id': spk,
                    # Placeholders; detailed feature stats not present in DL metadata
                    'feature_mean': [],
                    'feature_std': [],
                    'feature_min': [],
                    'feature_max': [],
                    'num_samples': None
                }
            return {
                'source': 'dl_metadata',
                'speakers': speakers_struct,
                'raw': dl_meta
            }
        except Exception:
            pass

    # 3. ML pickle metadata (minimal)
    if os.path.exists(ml_metadata_pkl):
        try:
            import joblib
            meta = joblib.load(ml_metadata_pkl)
            le = meta.get('label_encoder')
            speakers_struct = {}
            if le is not None and hasattr(le, 'classes_'):
                for spk in le.classes_.tolist():
                    speakers_struct[spk] = {
                        'id': spk,
                        'feature_mean': [],
                        'feature_std': [],
                        'feature_min': [],
                        'feature_max': [],
                        'num_samples': None
                    }
            return {
                'source': 'ml_pickle',
                'speakers': speakers_struct
            }
        except Exception:
            pass

    # Fallback empty structure
    return {'speakers': {}}


def get_speaker_feature_stats(speaker_id: str, 
                              metadata: Dict = None,
                              metadata_file: str = 'models/speaker_metadata.json') -> Dict:
    """
    Get feature statistics for a specific speaker.
    
    Args:
        speaker_id: Speaker ID
        metadata: Pre-loaded metadata (if None, loads from file)
        metadata_file: Path to metadata file
        
    Returns:
        Dictionary with feature statistics
    """
    if metadata is None:
        metadata = load_speaker_metadata(metadata_file)
    
    if 'speakers' not in metadata or speaker_id not in metadata['speakers']:
        return {}
    
    return metadata['speakers'][speaker_id]


def plot_model_comparison(metrics: Dict[str, Dict[str, float]],
                          metrics_to_plot: List[str] = None,
                          title: str = "Model Comparison") -> plt.Figure:
    """
    Plot bar charts comparing overall metrics for multiple models.

    Args:
        metrics: Dict mapping model_name -> dict of metric_name -> value (e.g. accuracy, f1_macro)
        metrics_to_plot: list of metric keys to include (default: ['accuracy','f1_macro'])
        title: Plot title

    Returns:
        Matplotlib Figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'f1_macro']

    model_names = list(metrics.keys())
    n_models = len(model_names)

    # Build matrix of values
    values = np.zeros((len(metrics_to_plot), n_models))
    for j, mname in enumerate(model_names):
        for i, metric_key in enumerate(metrics_to_plot):
            values[i, j] = metrics[mname].get(metric_key, np.nan)

    x = np.arange(n_models)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.2), 5))
    for i in range(len(metrics_to_plot)):
        ax.bar(x + (i - len(metrics_to_plot)/2) * width/len(metrics_to_plot), values[i], width/len(metrics_to_plot), label=metrics_to_plot[i])

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrix_grid(confusion_matrices: Dict[str, np.ndarray],
                               labels: List[str],
                               normalize: bool = False,
                               cmap: str = 'Blues',
                               title_prefix: str = '') -> plt.Figure:
    """
    Plot a grid of confusion matrices (one per model) for side-by-side comparison.

    Args:
        confusion_matrices: Dict mapping model_name -> confusion matrix (2D numpy array)
        labels: list of class labels to show on axes
        normalize: whether to normalize rows
        cmap: colormap
        title_prefix: optional prefix for each subplot title

    Returns:
        Matplotlib Figure
    """
    model_names = list(confusion_matrices.keys())
    n = len(model_names)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).reshape(-1)

    for ax_idx, model_name in enumerate(model_names):
        ax = axes[ax_idx]
        cm = np.array(confusion_matrices[model_name], dtype=float)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, where=(row_sums != 0))

        # Choose annotation format: use 2 decimals for normalized, integer-like for raw counts
        fmt = '.2f' if normalize else '.0f'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"{title_prefix}{model_name}")
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    # Hide unused axes
    for i in range(len(model_names), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def plot_calibration_curves(y_true: np.ndarray,
                            probas_dict: Dict[str, np.ndarray],
                            n_bins: int = 10,
                            title: str = 'Calibration Curves') -> plt.Figure:
    """
    Plot reliability diagrams (calibration curves) for multiple models.

    Args:
        y_true: true labels (binary or multi-class flattened for each class vs rest not supported here)
        probas_dict: mapping model_name -> predicted probability for positive class (1D array)
        n_bins: number of bins for calibration_curve

    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, probas in probas_dict.items():
        probas = np.asarray(probas)
        if probas.ndim != 1:
            # If probabilities are (N, C), take max-prob as confidence
            probas = probas.max(axis=1)
        frac_pos, mean_pred = calibration_curve(y_true, probas, n_bins=n_bins, strategy='uniform')
        ax.plot(mean_pred, frac_pos, marker='o', label=model_name)

    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_per_class_heatmap(metrics_per_model: Dict[str, Dict[str, float]],
                          title: str = 'Per-class metric heatmap',
                          metric_name: Optional[str] = None) -> plt.Figure:
    """
    Create a heatmap of per-class metrics across multiple models.

    Args:
        metrics_per_model: dict mapping model_name -> dict[class_label -> metric_value]
        metric_name: optional label for colorbar

    Returns:
        Matplotlib Figure
    """
    model_names = list(metrics_per_model.keys())
    # collect all classes
    classes = sorted({c for m in metrics_per_model.values() for c in m.keys()})
    data = np.zeros((len(model_names), len(classes)))
    data[:] = np.nan
    for i, m in enumerate(model_names):
        for j, c in enumerate(classes):
            data[i, j] = metrics_per_model[m].get(c, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.6), max(4, len(model_names) * 0.6)))
    sns.heatmap(data, xticklabels=classes, yticklabels=model_names, annot=True, fmt='.2f', cmap='viridis', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Class')
    ax.set_ylabel('Model')
    plt.tight_layout()
    return fig


def plot_embeddings_tsne(embeddings: np.ndarray,
                         labels: List[str],
                         pred_labels: Optional[List[str]] = None,
                         title: str = 't-SNE embedding') -> plt.Figure:
    """
    Reduce embeddings to 2D via t-SNE and plot colored by true labels (and optionally predicted labels).

    Args:
        embeddings: (N, D) array of embeddings
        labels: list of true labels
        pred_labels: optional predicted labels to plot with different marker

    Returns:
        Matplotlib Figure
    """
    tsne = TSNE(n_components=2, random_state=42)
    X2 = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(list(set(labels)))
    palette = sns.color_palette('tab10', n_colors=max(2, len(unique_labels)))
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    for lab in unique_labels:
        mask = [l == lab for l in labels]
        pts = X2[np.array(mask)]
        ax.scatter(pts[:, 0], pts[:, 1], label=lab, color=color_map[lab], alpha=0.7, s=30)

    if pred_labels is not None:
        # overlay predicted label mismatches
        mismatches = [t != p for t, p in zip(labels, pred_labels)]
        if any(mismatches):
            pts = X2[np.array(mismatches)]
            ax.scatter(pts[:, 0], pts[:, 1], facecolors='none', edgecolors='k', s=80, linewidths=0.8, label='mismatch')

    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig


def plot_overfitting_detection(metrics: Dict[str, Dict] = None,
                                histories: Dict[str, Dict] = None,
                                results_dir: str = 'results') -> None:
    """
    Generate overfitting-detection visualizations.

    - For ML models: provide `metrics` mapping model_name -> {'cv_mean'|'train': float, 'test': float}
      This will produce `overfitting_train_test.png` comparing train (CV) vs test accuracy.
    - For DL models: provide `histories` mapping model_name -> history dict containing
      keys like 'loss','val_loss','accuracy','val_accuracy'. This will produce two files:
      `overfitting_loss_curves.png` and `overfitting_accuracy_curves.png`.

    Files are written into `results_dir`.
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1) ML-style train vs test bar chart
    if metrics:
        model_names = list(metrics.keys())
        train_vals = [metrics[m].get('train', metrics[m].get('cv_mean', np.nan)) for m in model_names]
        test_vals = [metrics[m].get('test', np.nan) for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 1.2), 5))
        ax.bar(x - width/2, train_vals, width, label='Train (CV)', color='skyblue')
        ax.bar(x + width/2, test_vals, width, label='Test', color='orange')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Train vs Test Accuracy (Overfitting check)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(results_dir, 'overfitting_train_test.png')
        fig.savefig(out_path)
        print(f"Saved overfitting train/test plot to {out_path}")

    # 2) DL-style learning curves (loss and accuracy)
    if histories:
        # Loss curves
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        any_loss = False
        for model_name, hist in histories.items():
            if hist is None:
                continue
            if 'loss' in hist and 'val_loss' in hist:
                ax1.plot(hist['loss'], label=f'{model_name} train loss')
                ax1.plot(hist['val_loss'], label=f'{model_name} val loss', linestyle='--')
                any_loss = True
        if any_loss:
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training vs Validation Loss')
            ax1.legend()
            ax1.grid(True)
            plt.tight_layout()
            out_loss = os.path.join(results_dir, 'overfitting_loss_curves.png')
            fig1.savefig(out_loss)
            print(f"Saved loss curves to {out_loss}")

        # Accuracy curves
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        any_acc = False
        for model_name, hist in histories.items():
            if hist is None:
                continue
            if 'accuracy' in hist and 'val_accuracy' in hist:
                ax2.plot(hist['accuracy'], label=f'{model_name} train acc')
                ax2.plot(hist['val_accuracy'], label=f'{model_name} val acc', linestyle='--')
                any_acc = True
        if any_acc:
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Training vs Validation Accuracy')
            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()
            out_acc = os.path.join(results_dir, 'overfitting_accuracy_curves.png')
            fig2.savefig(out_acc)
            print(f"Saved accuracy curves to {out_acc}")

    return

