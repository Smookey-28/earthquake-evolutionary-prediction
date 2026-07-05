import os
import pandas as pd
import numpy as np
import json

from data_generator import generate_earthquake_data
from evolutionary_selector import GeneticAlgorithmFeatureSelector
from model_trainer import train_and_evaluate
import visualization

def run():
    # Setup working directory and ensure outputs directory
    project_dir = 'C:/Users/manoj/.gemini/antigravity/scratch/earthquake_evolutionary_prediction'
    os.makedirs(project_dir, exist_ok=True)
    
    print("="*60)
    print("EARTHQUAKE MAGNITUDE PREDICTION & EVOLUTIONARY FEATURE SELECTION")
    print("="*60)
    
    # 1. Generate earthquake dataset
    print("\n[Step 1] Generating Earthquake Dataset...")
    df = generate_earthquake_data(n_samples=3000, seed=42)
    csv_path = os.path.join(project_dir, 'earthquake_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated (3000 rows, 15 features) and saved to {csv_path}")
    
    # Prepare features and target
    target_col = 'magnitude'
    all_features = [col for col in df.columns if col != target_col]
    X = df[all_features]
    y = df[target_col]
    
    # 2. Run Genetic Algorithm for Feature Selection
    print("\n[Step 2] Running Genetic Algorithm for Feature Selection...")
    # Using 20 individuals and 15 generations to keep it fast and responsive
    ga = GeneticAlgorithmFeatureSelector(
        n_features=len(all_features),
        population_size=20,
        generations=15,
        crossover_rate=0.8,
        mutation_rate=0.1,
        random_state=42
    )
    best_chromosome, ga_history = ga.fit(X, y)
    
    # Identify selected features
    selected_features = [all_features[i] for i in range(len(all_features)) if best_chromosome[i]]
    discarded_features = [all_features[i] for i in range(len(all_features)) if not best_chromosome[i]]
    
    print("\nGenetic Algorithm Feature Selection Complete!")
    print(f"Selected ({len(selected_features)}): {selected_features}")
    print(f"Discarded ({len(discarded_features)}): {discarded_features}")
    
    # 3. Train and Evaluate Baseline Model (All Features)
    print("\n[Step 3] Training Baseline Random Forest Model (All Features)...")
    baseline_results = train_and_evaluate(df, target_col=target_col, selected_features=None, random_state=42)
    
    # 4. Train and Evaluate GA-Optimized Model (Selected Features)
    print("\n[Step 4] Training GA-Optimized Random Forest Model (Selected Features)...")
    ga_results = train_and_evaluate(df, target_col=target_col, selected_features=selected_features, random_state=42)
    
    # 5. Output metrics comparison in console
    print("\n" + "="*50)
    print("                 PERFORMANCE COMPARISON")
    print("="*50)
    
    bm_reg = baseline_results['regression_metrics']
    gm_reg = ga_results['regression_metrics']
    
    bm_cls = baseline_results['classification_metrics']
    gm_cls = ga_results['classification_metrics']
    
    print(f"{'Metric':<25} | {'Baseline (All)':<16} | {'GA-Optimized':<16}")
    print("-"*63)
    print(f"{'Number of Features':<25} | {len(all_features):<16d} | {len(selected_features):<16d}")
    print(f"{'Train MSE':<25} | {bm_reg['train_mse']:<16.4f} | {gm_reg['train_mse']:<16.4f}")
    print(f"{'Test MSE (Lower is Better)':<25} | {bm_reg['test_mse']:<16.4f} | {gm_reg['test_mse']:<16.4f}")
    print(f"{'Train R²':<25} | {bm_reg['train_r2']:<16.4f} | {gm_reg['train_r2']:<16.4f}")
    print(f"{'Test R² (Higher is Better)':<25} | {bm_reg['test_r2']:<16.4f} | {gm_reg['test_r2']:<16.4f}")
    print(f"{'Binned Accuracy (Class.)':<25} | {bm_cls['accuracy']:<16.4f} | {gm_cls['accuracy']:<16.4f}")
    print(f"{'Weighted F1-Score (Class.)':<25} | {bm_cls['f1_weighted']:<16.4f} | {gm_cls['f1_weighted']:<16.4f}")
    print("="*50)
    
    # 6. Generate and save Plots
    print("\n[Step 5] Generating and Saving Visualization Plots...")
    
    # Convergence Plot
    visualization.plot_ga_convergence(
        ga_history, 
        os.path.join(project_dir, 'ga_convergence.png')
    )
    
    # Confusion Matrices Plot
    visualization.plot_confusion_matrices(
        bm_cls['confusion_matrix'], 
        gm_cls['confusion_matrix'], 
        bm_cls['labels'], 
        os.path.join(project_dir, 'confusion_matrices.png')
    )
    
    # Feature Importances Plot
    visualization.plot_feature_importances(
        baseline_results['feature_importances'],
        ga_results['feature_importances'],
        all_features,
        os.path.join(project_dir, 'feature_importances.png')
    )
    
    # Metrics Comparison Plot
    visualization.plot_metrics_comparison(
        baseline_results,
        ga_results,
        os.path.join(project_dir, 'metrics_comparison.png')
    )
    
    # Save a JSON file with summaries for the Streamlit dashboard
    summary_data = {
        'all_features': all_features,
        'selected_features': selected_features,
        'discarded_features': discarded_features,
        'baseline_metrics': {
            'mse': bm_reg['test_mse'],
            'r2': bm_reg['test_r2'],
            'mae': bm_reg['test_mae'],
            'accuracy': bm_cls['accuracy'],
            'f1': bm_cls['f1_weighted']
        },
        'ga_metrics': {
            'mse': gm_reg['test_mse'],
            'r2': gm_reg['test_r2'],
            'mae': gm_reg['test_mae'],
            'accuracy': gm_cls['accuracy'],
            'f1': gm_cls['f1_weighted']
        },
        'ga_history': ga_history
    }
    
    with open(os.path.join(project_dir, 'summary_results.json'), 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"\nAll operations completed! Summary JSON saved to {os.path.join(project_dir, 'summary_results.json')}")
    print("Plots saved in project directory. Run the streamlit application to explore interactively.")

if __name__ == '__main__':
    run()
