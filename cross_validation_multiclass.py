"""
Subject-Level Cross-Validation for Sleep Stage Classification

This script runs multiple train/validation splits with different random seeds
to understand if poor generalization is consistent or split-dependent.
"""

import pandas as pd
import numpy as np
import random
import warnings
import json
import os
from datetime import datetime
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

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


def calculate_focal_class_weights(y, gamma=1.2):
    """Calculate focal-loss-inspired class weights."""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    
    class_weights = {}
    for class_idx, count in enumerate(class_counts):
        if count > 0:
            class_weights[class_idx] = (total_samples / count) ** gamma
    
    return class_weights


def train_lightgbm_fast(X_train, y_train, X_val, y_val, max_evals=50):
    """
    Train LightGBM with fewer trials for faster cross-validation.
    """
    print(f"  Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    
    class_weights = calculate_focal_class_weights(y_train, gamma=1.2)
    
    # Simplified hyperparameter space for speed
    space = {
        "max_depth": hp.quniform("max_depth", 3, 5, 1),
        "reg_alpha": hp.quniform("reg_alpha", 20, 100, 20),
        "reg_lambda": hp.uniform("reg_lambda", 2, 8),
        "num_leaves": hp.quniform("num_leaves", 20, 50, 10),
        "n_estimators": hp.quniform("n_estimators", 100, 200, 50),
        "learning_rate": hp.uniform("learning_rate", 0.03, 0.1),
        "min_child_samples": hp.quniform("min_child_samples", 30, 80, 10),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 0.9),
        "subsample": hp.uniform("subsample", 0.7, 0.9),
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
            colsample_bytree=params["colsample_bytree"],
            subsample=params["subsample"],
            class_weight=class_weights,
            verbose=-1,
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        y_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        return {"loss": -f1, "status": STATUS_OK}
    
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
    
    # Train final model
    final_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        max_depth=best_params["max_depth"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        num_leaves=best_params["num_leaves"],
        min_child_samples=best_params["min_child_samples"],
        colsample_bytree=best_params["colsample_bytree"],
        subsample=best_params["subsample"],
        class_weight=class_weights,
        random_state=1,
        verbose=-1,
    )
    
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    
    return final_model, best_params


def evaluate_split(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model on all splits."""
    results = {}
    
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        
        acc = accuracy_score(y, y_pred)
        f1_weighted = f1_score(y, y_pred, average='weighted')
        f1_macro = f1_score(y, y_pred, average='macro')
        kappa = cohen_kappa_score(y, y_pred)
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        results[name] = {
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'kappa': kappa,
            'f1_wake': f1_per_class[0] if len(f1_per_class) > 0 else 0,
            'f1_rem': f1_per_class[1] if len(f1_per_class) > 1 else 0,
            'f1_n1': f1_per_class[2] if len(f1_per_class) > 2 else 0,
            'f1_n2': f1_per_class[3] if len(f1_per_class) > 3 else 0,
            'f1_n3': f1_per_class[4] if len(f1_per_class) > 4 else 0,
        }
    
    return results


def run_cross_validation(n_splits=5):
    """
    Run multiple train/val/test splits with different random seeds.
    """
    print("="*80)
    print("SUBJECT-LEVEL CROSS-VALIDATION FOR SLEEP STAGE CLASSIFICATION")
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
    
    group_variables = ['AHI_Severity', 'Obesity']
    group_variable = get_variable(group_variables, idx=0)
    
    # Run multiple splits
    all_results = []
    
    for split_idx in range(n_splits):
        print("\n" + "="*80)
        print(f"SPLIT {split_idx + 1}/{n_splits} (Random Seed: {split_idx})")
        print("="*80)
        
        # Use different random seed for each split
        random.seed(split_idx)
        np.random.seed(split_idx)
        
        # Create train/val/test splits
        train_sids = random.sample(good_quality_sids, 56)
        remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
        val_sids = random.sample(remaining_sids, 8)
        test_sids = [subj for subj in remaining_sids if subj not in val_sids]
        
        print(f"Train subjects: {len(train_sids)}")
        print(f"Val subjects: {len(val_sids)}")
        print(f"Test subjects: {len(test_sids)}")
        
        # Prepare datasets
        X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
        X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
        X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)
        
        print(f"  Train: {X_train.shape[0]} samples, distribution: {np.bincount(y_train.astype(int))}")
        print(f"  Val: {X_val.shape[0]} samples, distribution: {np.bincount(y_val.astype(int))}")
        print(f"  Test: {X_test.shape[0]} samples, distribution: {np.bincount(y_test.astype(int))}")
        
        # Train model (fewer trials for speed)
        print("\n  Training model...")
        model, best_params = train_lightgbm_fast(X_train, y_train, X_val, y_val, max_evals=50)
        
        # Evaluate
        print("  Evaluating...")
        split_results = evaluate_split(model, X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Print results
        print(f"\n  Results for Split {split_idx + 1}:")
        print(f"    Train  - Acc: {split_results['train']['accuracy']:.4f}, F1: {split_results['train']['f1_weighted']:.4f}, Kappa: {split_results['train']['kappa']:.4f}")
        print(f"    Val    - Acc: {split_results['val']['accuracy']:.4f}, F1: {split_results['val']['f1_weighted']:.4f}, Kappa: {split_results['val']['kappa']:.4f}")
        print(f"    Test   - Acc: {split_results['test']['accuracy']:.4f}, F1: {split_results['test']['f1_weighted']:.4f}, Kappa: {split_results['test']['kappa']:.4f}")
        print(f"    Overfit Gap (Train-Val): {split_results['train']['accuracy'] - split_results['val']['accuracy']:.4f}")
        
        # Store results
        split_results['split_id'] = split_idx
        split_results['train_sids'] = train_sids
        split_results['val_sids'] = val_sids
        split_results['test_sids'] = test_sids
        split_results['best_params'] = best_params
        all_results.append(split_results)
    
    # Aggregate results
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    # Calculate statistics
    train_accs = [r['train']['accuracy'] for r in all_results]
    val_accs = [r['val']['accuracy'] for r in all_results]
    test_accs = [r['test']['accuracy'] for r in all_results]
    overfit_gaps = [r['train']['accuracy'] - r['val']['accuracy'] for r in all_results]
    
    print(f"\nAccuracy Statistics Across {n_splits} Splits:")
    print(f"  Train:        Mean={np.mean(train_accs):.4f}, Std={np.std(train_accs):.4f}, Range=[{np.min(train_accs):.4f}, {np.max(train_accs):.4f}]")
    print(f"  Validation:   Mean={np.mean(val_accs):.4f}, Std={np.std(val_accs):.4f}, Range=[{np.min(val_accs):.4f}, {np.max(val_accs):.4f}]")
    print(f"  Test:         Mean={np.mean(test_accs):.4f}, Std={np.std(test_accs):.4f}, Range=[{np.min(test_accs):.4f}, {np.max(test_accs):.4f}]")
    print(f"  Overfit Gap:  Mean={np.mean(overfit_gaps):.4f}, Std={np.std(overfit_gaps):.4f}")
    
    # Detailed results table
    print("\n" + "="*80)
    print("DETAILED RESULTS PER SPLIT")
    print("="*80)
    
    results_df = pd.DataFrame([
        {
            'Split': r['split_id'] + 1,
            'Train_Acc': r['train']['accuracy'],
            'Val_Acc': r['val']['accuracy'],
            'Test_Acc': r['test']['accuracy'],
            'Train_F1': r['train']['f1_weighted'],
            'Val_F1': r['val']['f1_weighted'],
            'Test_F1': r['test']['f1_weighted'],
            'Train_Kappa': r['train']['kappa'],
            'Val_Kappa': r['val']['kappa'],
            'Test_Kappa': r['test']['kappa'],
            'Overfit_Gap': r['train']['accuracy'] - r['val']['accuracy'],
        }
        for r in all_results
    ])
    
    print(results_df.to_string(index=False))
    
    # Save results
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    results_dir = f'./results/cross_validation_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    results_df.to_csv(os.path.join(results_dir, 'cross_validation_summary.csv'), index=False)
    
    # Save detailed results as JSON
    with open(os.path.join(results_dir, 'cross_validation_detailed.json'), 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for r in all_results:
            serializable = {
                'split_id': r['split_id'],
                'train': r['train'],
                'val': r['val'],
                'test': r['test'],
                'best_params': r['best_params'],
                'train_sids': r['train_sids'],
                'val_sids': r['val_sids'],
                'test_sids': r['test_sids'],
            }
            serializable_results.append(serializable)
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    mean_overfit_gap = np.mean(overfit_gaps)
    std_val_acc = np.std(val_accs)
    
    print(f"\n1. Overfitting Severity:")
    if mean_overfit_gap > 0.3:
        print(f"   SEVERE (Gap={mean_overfit_gap:.4f}) - Model memorizes training subjects")
    elif mean_overfit_gap > 0.2:
        print(f"   MODERATE (Gap={mean_overfit_gap:.4f}) - Some subject-specific learning")
    else:
        print(f"   MILD (Gap={mean_overfit_gap:.4f}) - Reasonable generalization")
    
    print(f"\n2. Split Consistency:")
    if std_val_acc > 0.05:
        print(f"   INCONSISTENT (Std={std_val_acc:.4f}) - Performance highly dependent on subject split")
    else:
        print(f"   CONSISTENT (Std={std_val_acc:.4f}) - Performance stable across splits")
    
    print(f"\n3. Expected Performance:")
    print(f"   On new subjects: {np.mean(val_accs):.1%} ± {1.96*std_val_acc:.1%} (95% CI)")
    
    print(f"\n4. Recommendations:")
    if mean_overfit_gap > 0.3:
        print("   - Consider subject-level normalization/calibration")
        print("   - Use more subjects for training if possible")
        print("   - Try domain adaptation techniques")
        print("   - Features may be too subject-specific")
    if std_val_acc > 0.05:
        print("   - Some subject groups are harder to predict")
        print("   - Consider analyzing which subjects cause poor performance")
        print("   - May need subject-specific models or adaptation")
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPLETE")
    print("="*80)
    
    return all_results, results_df


if __name__ == "__main__":
    # Run cross-validation with 5 different splits
    results, summary_df = run_cross_validation(n_splits=5)

