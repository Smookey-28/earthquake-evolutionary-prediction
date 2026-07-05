import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for high-quality plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150
})

def plot_ga_convergence(history, save_path):
    """Plots the Genetic Algorithm convergence curve (best and average fitness)."""
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    
    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_fitness, marker='o', linewidth=2, color='#1f77b4', label='Best Fitness ($R^2$ - penalty)')
    plt.plot(generations, avg_fitness, marker='s', linestyle='--', linewidth=1.5, color='#ff7f0e', label='Average Fitness')
    
    plt.title('Evolutionary Feature Selection Convergence', pad=15)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.xticks(generations)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {save_path}")

def plot_confusion_matrices(baseline_cm, ga_cm, labels, save_path):
    """Plots side-by-side confusion matrices for the Baseline and GA-optimized models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline Confusion Matrix
    sns.heatmap(
        baseline_cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels, 
        yticklabels=labels, 
        ax=axes[0],
        cbar=False,
        annot_kws={"size": 14}
    )
    axes[0].set_title('Baseline Model Confusion Matrix\n(All 15 Features)', pad=10)
    axes[0].set_xlabel('Predicted Magnitude Category')
    axes[0].set_ylabel('Actual Magnitude Category')
    
    # GA-Optimized Confusion Matrix
    sns.heatmap(
        ga_cm, 
        annot=True, 
        fmt='d', 
        cmap='Greens', 
        xticklabels=labels, 
        yticklabels=labels, 
        ax=axes[1],
        cbar=False,
        annot_kws={"size": 14}
    )
    axes[1].set_title('GA-Optimized Model Confusion Matrix\n(Selected Features Only)', pad=10)
    axes[1].set_xlabel('Predicted Magnitude Category')
    axes[1].set_ylabel('Actual Magnitude Category')
    
    plt.suptitle('Confusion Matrix Comparison (Regression Binned)', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")

def plot_feature_importances(baseline_importances, ga_importances, all_features, save_path):
    """Plots a comparison of feature importances between the two models."""
    # Ensure importances align with all features (for visualization)
    baseline_dict = dict(zip(baseline_importances['feature'], baseline_importances['importance']))
    ga_dict = dict(zip(ga_importances['feature'], ga_importances['importance']))
    
    baseline_vals = [baseline_dict.get(feat, 0.0) for feat in all_features]
    ga_vals = [ga_dict.get(feat, 0.0) for feat in all_features]
    
    y = np.arange(len(all_features))
    height = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    rects1 = ax.barh(y - height/2, baseline_vals, height, label='Baseline (All)', color='#3182bd')
    
    # For GA, use green for selected, grey for unselected
    ga_colors = ['#2ca02c' if ga_dict.get(feat, 0.0) > 0.0 else '#d9d9d9' for feat in all_features]
    rects2 = ax.barh(y + height/2, ga_vals, height, label='GA-Optimized', color=ga_colors)
    
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importance Comparison', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(all_features)
    ax.legend(frameon=True)
    ax.invert_yaxis()  # top-down feature view
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Feature importances plot saved to {save_path}")

def plot_metrics_comparison(baseline_metrics, ga_metrics, save_path):
    """Bar chart contrasting performance metrics."""
    metrics = ['MSE (Lower is Better)', 'R² (Higher is Better)', 'Binned Accuracy']
    
    baseline_vals = [
        baseline_metrics['regression_metrics']['test_mse'],
        baseline_metrics['regression_metrics']['test_r2'],
        baseline_metrics['classification_metrics']['accuracy']
    ]
    
    ga_vals = [
        ga_metrics['regression_metrics']['test_mse'],
        ga_metrics['regression_metrics']['test_r2'],
        ga_metrics['classification_metrics']['accuracy']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline Model', color='#a6cee3')
    rects2 = ax.bar(x + width/2, ga_vals, width, label='GA-Optimized Model', color='#b2df8a')
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics: Baseline vs GA-Optimized', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(frameon=True)
    ax.set_ylim(0, max(max(baseline_vals), max(ga_vals)) * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Metrics comparison plot saved to {save_path}")
