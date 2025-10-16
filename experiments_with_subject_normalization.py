"""
Sleep Stage Classification with Subject-Level Normalization

This script implements subject-level feature normalization to handle
physiological differences between subjects (different HR baselines,
movement patterns, etc.)
"""

import pandas as pd
import numpy as np
import random
import warnings
import json
import os
from datetime import datetime
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from datasets import *

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set display options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 50)

# Set random seeds
random.seed(0)
np.random.seed(1)


def normalize_features_per_subject(df, features, subject_col='sid', method='zscore'):
    """
    Normalize features separately for each subject.
    
    This removes subject-specific baselines and scales, making features
    more comparable across subjects.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and subject IDs
    features : list
        List of feature names to normalize
    subject_col : str
        Column name containing subject IDs
    method : str
        Normalization method: 'zscore', 'robust', 'minmax', or 'percentile'
        
    Returns
    -------
    df_normalized : pd.DataFrame
        DataFrame with normalized features
    """
    df_normalized = df.copy()
    
    print(f"  Applying {method} normalization per subject...")
    print(f"  Number of subjects: {df[subject_col].nunique()}")
    print(f"  Number of features to normalize: {len(features)}")
    
    for sid in df[subject_col].unique():
        subject_mask = df[subject_col] == sid
        subject_data = df.loc[subject_mask, features].values
        
        if method == 'zscore':
            # Z-score: (x - mean) / std
            mean = subject_data.mean(axis=0)
            std = subject_data.std(axis=0)
            std[std == 0] = 1  # Avoid division by zero
            normalized_data = (subject_data - mean) / std
            
        elif method == 'robust':
            # Robust: (x - median) / IQR (less sensitive to outliers)
            median = np.median(subject_data, axis=0)
            q75 = np.percentile(subject_data, 75, axis=0)
            q25 = np.percentile(subject_data, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1  # Avoid division by zero
            normalized_data = (subject_data - median) / iqr
            
        elif method == 'minmax':
            # Min-Max: (x - min) / (max - min)
            min_val = subject_data.min(axis=0)
            max_val = subject_data.max(axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            normalized_data = (subject_data - min_val) / range_val
            
        elif method == 'percentile':
            # Percentile rank: convert to 0-1 based on within-subject rank
            normalized_data = np.zeros_like(subject_data)
            for i in range(subject_data.shape[1]):
                feature_col = subject_data[:, i]
                # Convert to percentile rank (0-1)
                ranks = np.argsort(np.argsort(feature_col))
                normalized_data[:, i] = ranks / (len(ranks) - 1) if len(ranks) > 1 else 0.5
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        df_normalized.loc[subject_mask, features] = normalized_data
    
    print(f"  Normalization complete!")
    
    return df_normalized


def calculate_focal_class_weights(y, gamma=1.2):
    """Calculate focal-loss-inspired class weights."""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    
    class_weights = {}
    for class_idx, count in enumerate(class_counts):
        if count > 0:
            class_weights[class_idx] = (total_samples / count) ** gamma
    
    return class_weights


def train_model(X_train, y_train, X_val, y_val, max_evals=100):
    """Train LightGBM model with hyperparameter optimization."""
    
    print(f"\n  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Class distribution (train): {np.bincount(y_train.astype(int))}")
    
    class_weights = calculate_focal_class_weights(y_train, gamma=1.2)
    
    space = {
        "max_depth": hp.quniform("max_depth", 3, 6, 1),
        "reg_alpha": hp.quniform("reg_alpha", 10, 100, 10),
        "reg_lambda": hp.uniform("reg_lambda", 2, 10),
        "num_leaves": hp.quniform("num_leaves", 20, 60, 10),
        "n_estimators": hp.quniform("n_estimators", 100, 250, 25),
        "learning_rate": hp.uniform("learning_rate", 0.03, 0.1),
        "min_child_samples": hp.quniform("min_child_samples", 30, 100, 10),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 0.95),
        "subsample": hp.uniform("subsample", 0.7, 0.95),
        "min_data_in_leaf": hp.quniform("min_data_in_leaf", 20, 80, 10),
    }
    
    def objective(params):
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=5,
            max_depth=int(params["max_depth"]),
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            n_estimators=int(params["n_estimators"]),
            learning_rate=params["learning_rate"],
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            min_data_in_leaf=int(params["min_data_in_leaf"]),
            colsample_bytree=params["colsample_bytree"],
            subsample=params["subsample"],
            class_weight=class_weights,
            verbose=-1,
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        
        y_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        return {"loss": -f1, "status": STATUS_OK}
    
    print(f"  Running hyperparameter optimization ({max_evals} trials)...")
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=False
    )
    
    # Convert params
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["num_leaves"] = int(best_params["num_leaves"])
    best_params["min_child_samples"] = int(best_params["min_child_samples"])
    best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"])
    
    print(f"  Best F1 on validation: {-trials.best_trial['result']['loss']:.4f}")
    
    # Train final model
    final_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        max_depth=best_params["max_depth"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        n_estimators=best_params["n_estimators"] + 50,
        learning_rate=best_params["learning_rate"],
        num_leaves=best_params["num_leaves"],
        min_child_samples=best_params["min_child_samples"],
        min_data_in_leaf=best_params["min_data_in_leaf"],
        colsample_bytree=best_params["colsample_bytree"],
        subsample=best_params["subsample"],
        class_weight=class_weights,
        random_state=1,
        verbose=-1,
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    return final_model, best_params


def evaluate_model(model, X, y, dataset_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    f1_weighted = f1_score(y, y_pred, average='weighted')
    f1_macro = f1_score(y, y_pred, average='macro')
    kappa = cohen_kappa_score(y, y_pred)
    f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
    
    results = {
        'Dataset': dataset_name,
        'Accuracy': acc,
        'F1_Weighted': f1_weighted,
        'F1_Macro': f1_macro,
        'Cohens_Kappa': kappa,
        'F1_Wake': f1_per_class[0] if len(f1_per_class) > 0 else 0,
        'F1_REM': f1_per_class[1] if len(f1_per_class) > 1 else 0,
        'F1_N1': f1_per_class[2] if len(f1_per_class) > 2 else 0,
        'F1_N2': f1_per_class[3] if len(f1_per_class) > 3 else 0,
        'F1_N3': f1_per_class[4] if len(f1_per_class) > 4 else 0,
    }
    
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  Cohen's Kappa: {kappa:.4f}")
    print(f"  Per-class F1: Wake={f1_per_class[0]:.3f}, REM={f1_per_class[1]:.3f}, N1={f1_per_class[2]:.3f}, N2={f1_per_class[3]:.3f}, N3={f1_per_class[4]:.3f}")
    
    return results


def plot_confusion_matrix(y_true, y_pred, title, filepath):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=['Wake', 'REM', 'N1', 'N2', 'N3'],
        yticklabels=['Wake', 'REM', 'N1', 'N2', 'N3'],
        cbar_kws={'label': 'Percentage'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {filepath}")


def main():
    """Main execution function comparing different normalization strategies."""
    
    print("="*80)
    print("SLEEP STAGE CLASSIFICATION WITH SUBJECT-LEVEL NORMALIZATION")
    print("="*80)
    
    # Data Preparation
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    quality_df_dir = './results/quality_scores_per_subject.csv'
    features_dir = "dataset_sample/features_df/"
    info_dir = "dataset_sample/participant_info.csv"
    
    clean_df, new_features, good_quality_sids = data_preparation(
        threshold=0.2,
        quality_df_dir=quality_df_dir,
        features_dir=features_dir,
        info_dir=info_dir
    )
    
    print(f"Data shape: {clean_df.shape}")
    print(f"Number of features: {len(new_features)}")
    print(f"Number of subjects: {len(good_quality_sids)}")
    
    SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)
    
    # Create train/val/test splits
    random.seed(0)
    train_sids = random.sample(good_quality_sids, 56)
    remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
    val_sids = random.sample(remaining_sids, 8)
    test_sids = [subj for subj in remaining_sids if subj not in val_sids]
    
    group_variables = ['AHI_Severity', 'Obesity']
    group_variable = get_variable(group_variables, idx=0)
    
    # Test different normalization methods
    methods = ['none', 'zscore', 'robust', 'percentile']
    all_results = []
    
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    results_dir = f'./results/subject_normalization_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    for method in methods:
        print("\n" + "="*80)
        print(f"TESTING NORMALIZATION METHOD: {method.upper()}")
        print("="*80)
        
        # Apply normalization if needed
        if method == 'none':
            df_to_use = SW_df.copy()
            print("  Using original features (no normalization)")
        else:
            print(f"\nApplying {method} normalization...")
            df_to_use = normalize_features_per_subject(
                SW_df, 
                final_features, 
                subject_col='sid', 
                method=method
            )
        
        # Prepare datasets
        X_train, y_train, group_train = train_test_split(df_to_use, train_sids, final_features, group_variable)
        X_val, y_val, group_val = train_test_split(df_to_use, val_sids, final_features, group_variable)
        X_test, y_test, group_test = train_test_split(df_to_use, test_sids, final_features, group_variable)
        
        print(f"\nDataset sizes:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val: {X_val.shape[0]} samples")
        print(f"  Test: {X_test.shape[0]} samples")
        
        # Train model
        print(f"\nTraining model with {method} normalization...")
        model, best_params = train_model(X_train, y_train, X_val, y_val, max_evals=100)
        
        # Evaluate
        print(f"\nEvaluating {method} normalization...")
        train_results = evaluate_model(model, X_train, y_train, f"{method}_train")
        val_results = evaluate_model(model, X_val, y_val, f"{method}_val")
        test_results = evaluate_model(model, X_test, y_test, f"{method}_test")
        
        # Calculate overfit gap
        overfit_gap = train_results['Accuracy'] - val_results['Accuracy']
        print(f"\nOverfitting Gap (Train - Val): {overfit_gap:.4f}")
        
        # Store results
        method_results = {
            'Method': method,
            'Train_Acc': train_results['Accuracy'],
            'Val_Acc': val_results['Accuracy'],
            'Test_Acc': test_results['Accuracy'],
            'Train_F1': train_results['F1_Weighted'],
            'Val_F1': val_results['F1_Weighted'],
            'Test_F1': test_results['F1_Weighted'],
            'Overfit_Gap': overfit_gap,
            'best_params': best_params,
        }
        all_results.append(method_results)
        
        # Plot confusion matrices
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        plot_confusion_matrix(
            y_val, y_pred_val,
            f"Validation - {method} normalization",
            os.path.join(results_dir, f'confusion_matrix_val_{method}.png')
        )
        
        plot_confusion_matrix(
            y_test, y_pred_test,
            f"Test - {method} normalization",
            os.path.join(results_dir, f'confusion_matrix_test_{method}.png')
        )
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON OF NORMALIZATION METHODS")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(os.path.join(results_dir, 'normalization_comparison.csv'), index=False)
    
    # Find best method
    best_method_idx = results_df['Val_Acc'].idxmax()
    best_method = results_df.loc[best_method_idx, 'Method']
    best_val_acc = results_df.loc[best_method_idx, 'Val_Acc']
    best_test_acc = results_df.loc[best_method_idx, 'Test_Acc']
    best_overfit_gap = results_df.loc[best_method_idx, 'Overfit_Gap']
    
    print("\n" + "="*80)
    print("BEST NORMALIZATION METHOD")
    print("="*80)
    print(f"Method: {best_method}")
    print(f"Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {best_test_acc:.4f}")
    print(f"Overfit Gap: {best_overfit_gap:.4f}")
    
    # Improvement analysis
    baseline_val = results_df[results_df['Method'] == 'none']['Val_Acc'].values[0]
    baseline_test = results_df[results_df['Method'] == 'none']['Test_Acc'].values[0]
    baseline_gap = results_df[results_df['Method'] == 'none']['Overfit_Gap'].values[0]
    
    print(f"\nImprovement over baseline (no normalization):")
    print(f"  Validation Accuracy: {best_val_acc - baseline_val:+.4f} ({(best_val_acc/baseline_val - 1)*100:+.1f}%)")
    print(f"  Test Accuracy: {best_test_acc - baseline_test:+.4f} ({(best_test_acc/baseline_test - 1)*100:+.1f}%)")
    print(f"  Overfit Gap Reduction: {baseline_gap - best_overfit_gap:.4f} ({(1 - best_overfit_gap/baseline_gap)*100:.1f}% reduction)")
    
    print(f"\nResults saved to: {results_dir}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

