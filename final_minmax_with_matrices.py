"""
Train with MinMax normalization and generate confusion matrices
Copied from working comprehensive_optimization_suite.py
"""

import pandas as pd
import numpy as np
import random
import warnings
import os
from datetime import datetime
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from datasets import *

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 50)

random.seed(0)
np.random.seed(1)


def normalize_features_per_subject(df, features, subject_col='sid', method='minmax'):
    """Normalize features separately for each subject using MinMax."""
    df_normalized = df.copy()
    
    for sid in df[subject_col].unique():
        subject_mask = df[subject_col] == sid
        subject_data = df.loc[subject_mask, features].values
        
        min_val = subject_data.min(axis=0)
        max_val = subject_data.max(axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        normalized_data = (subject_data - min_val) / range_val
        
        df_normalized.loc[subject_mask, features] = normalized_data
    
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


def train_model(X_train, y_train, X_val, y_val, max_evals=150):
    """Train LightGBM."""
    
    class_weights = calculate_focal_class_weights(y_train, gamma=1.2)
    
    space = {
        "max_depth": hp.quniform("max_depth", 3, 6, 1),
        "reg_alpha": hp.quniform("reg_alpha", 10, 120, 10),
        "reg_lambda": hp.uniform("reg_lambda", 1, 12),
        "num_leaves": hp.quniform("num_leaves", 20, 70, 5),
        "n_estimators": hp.quniform("n_estimators", 100, 300, 25),
        "learning_rate": hp.uniform("learning_rate", 0.02, 0.12),
        "min_child_samples": hp.quniform("min_child_samples", 20, 100, 10),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.65, 0.95),
        "subsample": hp.uniform("subsample", 0.65, 0.95),
        "min_data_in_leaf": hp.quniform("min_data_in_leaf", 15, 80, 5),
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
    
    print("  Running hyperparameter optimization...")
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
    for key in ['max_depth', 'n_estimators', 'num_leaves', 'min_child_samples', 'min_data_in_leaf']:
        if key in best_params:
            best_params[key] = int(best_params[key])
    
    print(f"  Best validation F1: {-trials.best_trial['result']['loss']:.4f}")
    
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


def plot_confusion_matrix(y_true, y_pred, class_names, title, filepath):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Percentage matrix
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt='.1%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'},
        ax=ax1,
        vmin=0,
        vmax=1
    )
    ax1.set_title(f'{title} - Percentages', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    
    # Count matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        ax=ax2
    )
    ax2.set_title(f'{title} - Counts', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nConfusion matrix saved to: {filepath}")
    print(f"\nConfusion Matrix (counts):")
    print(cm)
    print(f"\nPer-class recall:")
    for i, class_name in enumerate(class_names):
        recall = cm_percent[i, i]
        print(f"  {class_name}: {recall:.1%} ({cm[i, i]}/{cm[i, :].sum()} samples)")


def main():
    """Main function."""
    
    print("="*80)
    print("MINMAX NORMALIZATION WITH CONFUSION MATRICES")
    print("="*80)
    
    # Data preparation
    print("\nLoading data...")
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
    print(f"Features: {len(new_features)}, Subjects: {len(good_quality_sids)}")
    
    SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)
    
    # Apply MinMax normalization
    print("\nApplying MinMax normalization per subject...")
    df_normalized = normalize_features_per_subject(SW_df, final_features, 'sid', 'minmax')
    
    # Create splits
    random.seed(0)
    train_sids = random.sample(good_quality_sids, 56)
    remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
    val_sids = random.sample(remaining_sids, 8)
    test_sids = [subj for subj in remaining_sids if subj not in val_sids]
    
    group_variables = ['AHI_Severity', 'Obesity']
    group_variable = get_variable(group_variables, idx=0)
    
    # Prepare datasets
    print("\nPreparing datasets...")
    X_train, y_train, _ = train_test_split(df_normalized, train_sids, final_features, group_variable)
    X_val, y_val, _ = train_test_split(df_normalized, val_sids, final_features, group_variable)
    X_test, y_test, _ = train_test_split(df_normalized, test_sids, final_features, group_variable)
    
    print(f"Train: {X_train.shape[0]} samples, {np.bincount(y_train.astype(int))}")
    print(f"Val: {X_val.shape[0]} samples, {np.bincount(y_val.astype(int))}")
    print(f"Test: {X_test.shape[0]} samples, {np.bincount(y_test.astype(int))}")
    
    # Train model
    print("\nTraining model with MinMax normalization (150 trials)...")
    model, best_params = train_model(X_train, y_train, X_val, y_val, max_evals=150)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    results_dir = f'./results/final_minmax_matrices_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\n" + "="*80)
    print(f"RESULTS")
    print(f"="*80)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Overfit Gap (Train-Val): {train_acc - val_acc:.4f}")
    
    # Plot confusion matrices
    class_names = ['Wake', 'REM', 'N1', 'N2', 'N3']
    
    print(f"\n" + "="*80)
    print(f"TRAINING SET")
    print(f"="*80)
    plot_confusion_matrix(
        y_train, y_train_pred, class_names,
        'Training Set - MinMax Normalization',
        os.path.join(results_dir, 'train_confusion_matrix.png')
    )
    
    print(f"\n" + "="*80)
    print(f"VALIDATION SET")
    print(f"="*80)
    plot_confusion_matrix(
        y_val, y_val_pred, class_names,
        'Validation Set - MinMax Normalization',
        os.path.join(results_dir, 'val_confusion_matrix.png')
    )
    
    print(f"\n" + "="*80)
    print(f"TEST SET")
    print(f"="*80)
    plot_confusion_matrix(
        y_test, y_test_pred, class_names,
        'Test Set - MinMax Normalization',
        os.path.join(results_dir, 'test_confusion_matrix.png')
    )
    
    print(f"\n\n" + "="*80)
    print(f"COMPLETE - All results saved to: {results_dir}")
    print(f"="*80)


if __name__ == "__main__":
    main()

