"""
Visualization helper functions for speaker comparison and analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
import json
import os


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


def load_speaker_metadata(metadata_file: str = 'models/speaker_metadata.json') -> Dict:
    """
    Load speaker metadata from JSON file.
    
    Args:
        metadata_file: Path to metadata file
        
    Returns:
        Dictionary containing speaker metadata
    """
    if not os.path.exists(metadata_file):
        return {}
    
    with open(metadata_file, 'r') as f:
        return json.load(f)


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

