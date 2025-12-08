"""
Evaluation module for Speech Emotion Detection models.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os


class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               labels: List[str] = None) -> Dict:
        """
        Evaluate classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Filter labels to only include those present in y_true or y_pred
        if labels is not None:
            unique_true = set(y_true)
            unique_pred = set(y_pred)
            present_labels = [label for label in labels if label in unique_true or label in unique_pred]
            if len(present_labels) == 0:
                # If no labels match, use all unique labels from y_true and y_pred
                present_labels = sorted(list(unique_true.union(unique_pred)))
            labels = present_labels
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Classification report
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'f1_weighted': f1_weighted,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], 
                             model_name: str, save_path: str = None, save_name: str = None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            labels: Label names
            model_name: Name of the model (for title)
            save_path: Path to save the plot (overrides save_name)
            save_name: Custom filename without extension (e.g., 'confusion_matrix_emotion_ml_svm')
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path is None:
            if save_name is None:
                save_name = f'confusion_matrix_{model_name}'
            save_path = os.path.join(self.results_dir, f'{save_name}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {save_path}")
    
    def print_evaluation_results(self, results: Dict, model_name: str, labels: List[str] = None):
        """Print evaluation results."""
        print(f"\n{'='*60}")
        print(f"Evaluation Results - {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
        print(f"F1-Score (Micro): {results['f1_micro']:.4f}")
        print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
        
        if labels and len(labels) == len(results['f1_per_class']):
            print(f"\nPer-class F1-Scores:")
            for emotion, f1 in zip(labels, results['f1_per_class']):
                print(f"  {emotion}: {f1:.4f}")
        else:
            print(f"\nPer-class F1-Scores:")
            for i, f1 in enumerate(results['f1_per_class']):
                print(f"  Class {i}: {f1:.4f}")
        
        # Print detailed classification report
        if 'classification_report' in results:
            report = results['classification_report']
            print(f"\nDetailed Classification Report:")
            print(f"{'Emotion':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
            print("-" * 60)
            
            if labels:
                for emotion in labels:
                    if emotion in report and isinstance(report[emotion], dict):
                        metrics = report[emotion]
                        print(f"{emotion:<12} {metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} {metrics.get('f1-score', 0):<12.4f} {metrics.get('support', 0):<10.0f}")
        
        print(f"{'='*60}\n")
    
    def save_results(self, results: Dict, model_name: str, filepath: str = None, labels: List[str] = None):
        """Save evaluation results to file."""
        if filepath is None:
            filepath = os.path.join(self.results_dir, f'evaluation_{model_name}.txt')
        
        with open(filepath, 'w') as f:
            f.write(f"Evaluation Results - {model_name}\n")
            f.write("="*60 + "\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"F1-Score (Macro): {results['f1_macro']:.4f}\n")
            f.write(f"F1-Score (Micro): {results['f1_micro']:.4f}\n")
            f.write(f"F1-Score (Weighted): {results['f1_weighted']:.4f}\n")
            
            # Per-class F1 scores
            if labels and len(labels) == len(results['f1_per_class']):
                f.write("\nPer-class F1-Scores:\n")
                for emotion, f1 in zip(labels, results['f1_per_class']):
                    f.write(f"  {emotion}: {f1:.4f}\n")
            else:
                f.write("\nPer-class F1-Scores:\n")
                for i, f1 in enumerate(results['f1_per_class']):
                    f.write(f"  Class {i}: {f1:.4f}\n")
            
            # Detailed classification report
            f.write("\n" + "="*60 + "\n")
            f.write("Detailed Classification Report:\n")
            f.write("="*60 + "\n")
            
            if 'classification_report' in results:
                report = results['classification_report']
                
                # Header
                f.write(f"{'Emotion':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
                f.write("-" * 65 + "\n")
                
                # Per-emotion metrics
                if labels:
                    for emotion in labels:
                        if emotion in report and isinstance(report[emotion], dict):
                            metrics = report[emotion]
                            f.write(f"{emotion:<15} {metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} {metrics.get('f1-score', 0):<12.4f} {metrics.get('support', 0):<10.0f}\n")
                
                # Averages
                f.write("-" * 65 + "\n")
                if 'macro avg' in report:
                    metrics = report['macro avg']
                    f.write(f"{'Macro Avg':<15} {metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} {metrics.get('f1-score', 0):<12.4f} {metrics.get('support', 0):<10.0f}\n")
                if 'weighted avg' in report:
                    metrics = report['weighted avg']
                    f.write(f"{'Weighted Avg':<15} {metrics.get('precision', 0):<12.4f} {metrics.get('recall', 0):<12.4f} {metrics.get('f1-score', 0):<12.4f} {metrics.get('support', 0):<10.0f}\n")
        
        print(f"Results saved to {filepath}")


if __name__ == '__main__':
    # Test evaluation
    print("Evaluation module loaded successfully")

