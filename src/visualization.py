"""
Unified visualization module for ML emotion detection.
Auto-generates comprehensive visualizations after training.
Can also be run standalone: python src/visualization.py
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import seaborn as sns

# Optional sklearn helpers used for calibration and embedding visualization
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def generate_visualizations(results_dir='results'):
    """
    Generate all comprehensive visualizations for ML emotion detection.
    
    Args:
        results_dir: Directory containing evaluation JSON files and where to save plots
    """
    print("\n" + "="*80)
    print("🎨 GENERATING ML EMOTION DETECTION VISUALIZATIONS")
    print("="*80 + "\n")

    # Load only the 4 main ML models (exclude test files)
    model_files = {
        'logistic_regression': f'{results_dir}/evaluation_emotion_ml_logistic_regression.json',
        'random_forest': f'{results_dir}/evaluation_emotion_ml_random_forest.json',
        'svm': f'{results_dir}/evaluation_emotion_ml_svm.json',
        'xgboost': f'{results_dir}/evaluation_emotion_ml_xgboost.json'
    }

    models_data = {}
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                models_data[name] = json.load(f)

    if not models_data:
        print("❌ No model data found")
        return

    print(f"✅ Loaded {len(models_data)} models\n")

    emotions = models_data['svm']['meta']['labels']
    model_names = list(models_data.keys())

    # 1. Detailed confusion matrices
    print("1️⃣ Creating detailed confusion matrices...")
    for name in model_names:
        data = models_data[name]
        conf = np.array(data['results']['confusion_matrix'])
        conf_pct = conf.astype('float') / (conf.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
                    xticklabels=emotions, yticklabels=emotions, ax=ax1,
                    linewidths=1.5, linecolor='white',
                    annot_kws={'size': 11, 'weight': 'bold'})
        ax1.set_title(f'{name.upper()}: Raw Counts', fontsize=15, fontweight='bold')
        ax1.set_ylabel('True Emotion', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Predicted Emotion', fontsize=13, fontweight='bold')
        
        sns.heatmap(conf_pct, annot=True, fmt='.0%', cmap='RdYlGn',
                    xticklabels=emotions, yticklabels=emotions, ax=ax2,
                    vmin=0, vmax=1, linewidths=1.5, linecolor='white',
                    annot_kws={'size': 11, 'weight': 'bold'})
        ax2.set_title(f'{name.upper()}: Recall %', fontsize=15, fontweight='bold')
        ax2.set_ylabel('True Emotion', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Predicted Emotion', fontsize=13, fontweight='bold')
        
        plt.suptitle(f'📊 Detailed Confusion Analysis: {name.upper()}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = f'{results_dir}/confusion_detailed_{name}.png'
        plt.savefig(save_path, bbox_inches='tight')
        print(f"   ✅ {save_path}")
        plt.close()

    # 2. Comprehensive dashboard
    print("\n2️⃣ Creating comprehensive dashboard...")
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    accs = [models_data[m]['results']['accuracy'] for m in model_names]
    colors = ['#e74c3c' if a<0.6 else '#f39c12' if a<0.65 else '#2ecc71' for a in accs]
    bars = ax1.barh(model_names, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_xlabel('Accuracy', fontweight='bold', fontsize=12)
    ax1.set_title('Overall Accuracy', fontweight='bold', fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax1.text(acc+0.01, bar.get_y()+bar.get_height()/2,
                f'{acc:.1%}', va='center', fontweight='bold', fontsize=11)

    # F1-Macro
    ax2 = fig.add_subplot(gs[0, 1])
    f1s = [models_data[m]['results']['f1_macro'] for m in model_names]
    bars = ax2.barh(model_names, f1s, color='steelblue', alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_xlabel('F1-Macro', fontweight='bold', fontsize=12)
    ax2.set_title('Balanced Performance', fontweight='bold', fontsize=13)
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    for bar, f1 in zip(bars, f1s):
        ax2.text(f1+0.01, bar.get_y()+bar.get_height()/2,
                f'{f1:.3f}', va='center', fontweight='bold', fontsize=11)

    # Class distribution
    ax3 = fig.add_subplot(gs[0, 2])
    conf = np.array(models_data['svm']['results']['confusion_matrix'])
    counts = [int(conf[i].sum()) for i in range(len(emotions))]
    colors = ['#e74c3c' if c<500 else '#f39c12' if c<2000 else '#2ecc71' for c in counts]
    bars = ax3.barh(emotions, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_xlabel('Sample Count (Test Set)', fontweight='bold', fontsize=12)
    ax3.set_title('⚠️ Class Imbalance\n(20% Test Set - 2,433 total)', 
                 fontweight='bold', fontsize=13, color='#e74c3c')
    ax3.grid(axis='x', alpha=0.3)
    for bar, cnt in zip(bars, counts):
        ax3.text(cnt+50, bar.get_y()+bar.get_height()/2,
                f'{cnt}', va='center', fontweight='bold', fontsize=10)

    # F1 heatmap
    ax4 = fig.add_subplot(gs[1, :])
    f1_matrix = np.array([models_data[m]['results']['f1_per_class'] for m in model_names])
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=emotions, yticklabels=model_names,
                vmin=0.3, vmax=0.9, linewidths=2, linecolor='white', ax=ax4,
                annot_kws={'size': 11, 'weight': 'bold'})
    ax4.set_title('F1-Scores: Models × Emotions (Green=Good, Red=Poor)',
                 fontweight='bold', fontsize=14)
    ax4.set_xlabel('Emotion', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Model', fontweight='bold', fontsize=12)

    # Avg performance per emotion
    ax5 = fig.add_subplot(gs[2, 0])
    avg_f1 = f1_matrix.mean(axis=0)
    std_f1 = f1_matrix.std(axis=0)
    colors = ['#e74c3c' if f<0.5 else '#f39c12' if f<0.65 else '#2ecc71' for f in avg_f1]
    bars = ax5.barh(emotions, avg_f1, xerr=std_f1, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2, capsize=5)
    ax5.set_xlabel('Avg F1 (± StdDev)', fontweight='bold', fontsize=11)
    ax5.set_title('Performance by Emotion\n(Averaged Across All Models)', fontweight='bold', fontsize=12)
    ax5.set_xlim(0, 1)
    ax5.grid(axis='x', alpha=0.3)

    # Best model per emotion - expanded to span 2 columns
    ax6 = fig.add_subplot(gs[2, 1:])
    
    # Create side-by-side visualization
    x_pos = np.arange(len(emotions))
    width = 0.7
    
    best_idxs = f1_matrix.argmax(axis=0)
    best_f1s = [f1_matrix[best_idxs[i], i] for i in range(len(emotions))]
    best_models = [model_names[i] for i in best_idxs]
    
    model_colors = {'logistic_regression': '#3498db', 'random_forest': '#2ecc71',
                    'svm': '#e74c3c', 'xgboost': '#f39c12'}
    colors = [model_colors.get(m, '#95a5a6') for m in best_models]
    
    bars = ax6.bar(x_pos, best_f1s, width, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax6.set_ylabel('Best F1-Score', fontweight='bold', fontsize=12)
    ax6.set_title('Best Performing Model per Emotion\n(with model name and F1-score)', 
                 fontweight='bold', fontsize=13)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(emotions, fontsize=11)
    ax6.set_ylim(0, 1)
    ax6.grid(axis='y', alpha=0.3)
    
    # Add F1 scores and model names on bars
    for bar, f1, model in zip(bars, best_f1s, best_models):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{f1:.3f}\n{model[:3].upper()}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=model_colors[m], 
                                    edgecolor='black', linewidth=1, alpha=0.8, label=m)
                      for m in model_names]
    ax6.legend(handles=legend_elements, loc='lower right', fontsize=10,
              title='Models', title_fontsize=11)

    plt.suptitle('🎯 COMPREHENSIVE ML ANALYSIS - EMOTION DETECTION',
                fontsize=18, fontweight='bold')
    save_path = f'{results_dir}/COMPREHENSIVE_ML_ANALYSIS.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"   ✅ {save_path}")
    plt.close()

    # 3. Grid of confusion matrices
    print("\n3️⃣ Creating confusion matrices grid...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    axes = axes.flatten()

    for idx, name in enumerate(model_names):
        conf = np.array(models_data[name]['results']['confusion_matrix'])
        conf_pct = conf.astype('float') / (conf.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        sns.heatmap(conf_pct, annot=True, fmt='.0%', cmap='RdYlGn',
                   xticklabels=emotions, yticklabels=emotions, ax=axes[idx],
                   vmin=0, vmax=1, linewidths=1.5, linecolor='white',
                   annot_kws={'size': 10, 'weight': 'bold'})
        acc = models_data[name]['results']['accuracy']
        f1 = models_data[name]['results']['f1_macro']
        axes[idx].set_title(f'{name.upper()}\nAcc={acc:.1%} | F1={f1:.3f}',
                           fontweight='bold', fontsize=13)

    plt.suptitle('Confusion Matrices Comparison (Recall %)',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = f'{results_dir}/CONFUSION_MATRICES_GRID.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"   ✅ {save_path}")
    plt.close()

    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print("\n📁 Generated:")
    print("   • confusion_detailed_*.png - Detailed matrices")
    print("   • COMPREHENSIVE_ML_ANALYSIS.png - Full dashboard")
    print("   • CONFUSION_MATRICES_GRID.png - Side-by-side")
    print("="*80 + "\n")


if __name__ == '__main__':
    generate_visualizations()
