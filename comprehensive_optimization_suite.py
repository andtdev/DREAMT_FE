"""
Comprehensive 2-Hour Optimization Suite for Sleep Stage Classification

This script runs multiple experiments automatically to find the best approach:
1. Subject-level normalization (4 methods × 150 trials)
2. Feature selection (removing subject-specific features)
3. Combined normalization + feature selection
4. Best method with 5-fold cross-validation

Designed to run autonomously for ~2 hours.
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
from sklearn.feature_selection import mutual_info_classif
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


class ExperimentLogger:
    """Logger to track experiment progress."""
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = datetime.now()
        
    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        elapsed = (datetime.now() - self.start_time).total_seconds() / 60
        log_msg = f"[{timestamp} | {elapsed:.1f}min] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')


def normalize_features_per_subject(df, features, subject_col='sid', method='zscore'):
    """Normalize features separately for each subject."""
    df_normalized = df.copy()
    
    for sid in df[subject_col].unique():
        subject_mask = df[subject_col] == sid
        subject_data = df.loc[subject_mask, features].values
        
        if method == 'zscore':
            mean = subject_data.mean(axis=0)
            std = subject_data.std(axis=0)
            std[std == 0] = 1
            normalized_data = (subject_data - mean) / std
            
        elif method == 'robust':
            median = np.median(subject_data, axis=0)
            q75 = np.percentile(subject_data, 75, axis=0)
            q25 = np.percentile(subject_data, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1
            normalized_data = (subject_data - median) / iqr
            
        elif method == 'percentile':
            normalized_data = np.zeros_like(subject_data)
            for i in range(subject_data.shape[1]):
                feature_col = subject_data[:, i]
                ranks = np.argsort(np.argsort(feature_col))
                normalized_data[:, i] = ranks / (len(ranks) - 1) if len(ranks) > 1 else 0.5
                
        elif method == 'minmax':
            min_val = subject_data.min(axis=0)
            max_val = subject_data.max(axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            normalized_data = (subject_data - min_val) / range_val
        
        df_normalized.loc[subject_mask, features] = normalized_data
    
    return df_normalized


def identify_subject_specific_features(df, features, subject_col='sid', threshold=0.7):
    """
    Identify features that are highly subject-specific (high between-subject variance).
    These features don't generalize well across subjects.
    """
    subject_means = df.groupby(subject_col)[features].mean()
    
    # Calculate variance ratio: between-subject variance / total variance
    variance_ratios = []
    for feature in features:
        total_var = df[feature].var()
        between_subject_var = subject_means[feature].var()
        
        if total_var > 0:
            ratio = between_subject_var / total_var
        else:
            ratio = 0
        
        variance_ratios.append({
            'feature': feature,
            'variance_ratio': ratio,
            'is_subject_specific': ratio > threshold
        })
    
    variance_df = pd.DataFrame(variance_ratios)
    
    generalizable_features = variance_df[~variance_df['is_subject_specific']]['feature'].tolist()
    subject_specific_features = variance_df[variance_df['is_subject_specific']]['feature'].tolist()
    
    return generalizable_features, subject_specific_features, variance_df


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
    """Train LightGBM with more trials for better optimization."""
    
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
    
    best_f1 = -trials.best_trial['result']['loss']
    
    return final_model, best_params, best_f1


def evaluate_model(model, X, y):
    """Quick evaluation."""
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    f1_weighted = f1_score(y, y_pred, average='weighted')
    kappa = cohen_kappa_score(y, y_pred)
    f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
    
    return {
        'accuracy': acc,
        'f1_weighted': f1_weighted,
        'kappa': kappa,
        'f1_per_class': f1_per_class.tolist() if len(f1_per_class) > 0 else [0]*5,
    }


def run_experiment(name, df, features, train_sids, val_sids, test_sids, group_variable, logger, max_evals=150):
    """Run a single experiment."""
    logger.log(f"Starting experiment: {name}")
    logger.log(f"  Features: {len(features)}, Subjects: train={len(train_sids)}, val={len(val_sids)}, test={len(test_sids)}")
    
    # Prepare data
    X_train, y_train, _ = train_test_split(df, train_sids, features, group_variable)
    X_val, y_val, _ = train_test_split(df, val_sids, features, group_variable)
    X_test, y_test, _ = train_test_split(df, test_sids, features, group_variable)
    
    logger.log(f"  Training with {max_evals} hyperparameter trials...")
    model, best_params, best_val_f1 = train_model(X_train, y_train, X_val, y_val, max_evals=max_evals)
    
    logger.log(f"  Best validation F1: {best_val_f1:.4f}")
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)
    
    overfit_gap = train_metrics['accuracy'] - val_metrics['accuracy']
    
    logger.log(f"  Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")
    logger.log(f"  Overfit Gap: {overfit_gap:.4f}")
    
    results = {
        'name': name,
        'n_features': len(features),
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'overfit_gap': overfit_gap,
        'best_params': best_params,
        'best_val_f1': best_val_f1,
    }
    
    return results, model


def run_cross_validation_best(df, features, good_quality_sids, group_variable, logger, n_splits=5):
    """Run cross-validation on best approach."""
    logger.log(f"Running {n_splits}-fold cross-validation on best approach...")
    
    cv_results = []
    
    for split_idx in range(n_splits):
        logger.log(f"  CV Split {split_idx + 1}/{n_splits}")
        
        random.seed(split_idx)
        np.random.seed(split_idx)
        
        train_sids = random.sample(good_quality_sids, 56)
        remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
        val_sids = random.sample(remaining_sids, 8)
        test_sids = [subj for subj in remaining_sids if subj not in val_sids]
        
        X_train, y_train, _ = train_test_split(df, train_sids, features, group_variable)
        X_val, y_val, _ = train_test_split(df, val_sids, features, group_variable)
        X_test, y_test, _ = train_test_split(df, test_sids, features, group_variable)
        
        model, _, _ = train_model(X_train, y_train, X_val, y_val, max_evals=100)
        
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)
        
        cv_results.append({
            'split': split_idx,
            'train_acc': train_metrics['accuracy'],
            'val_acc': val_metrics['accuracy'],
            'test_acc': test_metrics['accuracy'],
            'overfit_gap': train_metrics['accuracy'] - val_metrics['accuracy'],
        })
        
        logger.log(f"    Val: {val_metrics['accuracy']:.4f}, Test: {test_metrics['accuracy']:.4f}, Gap: {cv_results[-1]['overfit_gap']:.4f}")
    
    # Aggregate
    val_accs = [r['val_acc'] for r in cv_results]
    test_accs = [r['test_acc'] for r in cv_results]
    gaps = [r['overfit_gap'] for r in cv_results]
    
    logger.log(f"  CV Results: Val={np.mean(val_accs):.4f}±{np.std(val_accs):.4f}, Test={np.mean(test_accs):.4f}±{np.std(test_accs):.4f}, Gap={np.mean(gaps):.4f}±{np.std(gaps):.4f}")
    
    return cv_results


def main():
    """Main comprehensive experiment suite."""
    
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    results_dir = f'./results/comprehensive_suite_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    log_file = os.path.join(results_dir, 'experiment_log.txt')
    logger = ExperimentLogger(log_file)
    
    logger.log("="*80)
    logger.log("COMPREHENSIVE 2-HOUR OPTIMIZATION SUITE")
    logger.log("="*80)
    
    # Data preparation
    logger.log("\n" + "="*80)
    logger.log("DATA PREPARATION")
    logger.log("="*80)
    
    quality_df_dir = './results/quality_scores_per_subject.csv'
    features_dir = "dataset_sample/features_df/"
    info_dir = "dataset_sample/participant_info.csv"
    
    clean_df, new_features, good_quality_sids = data_preparation(
        threshold=0.2,
        quality_df_dir=quality_df_dir,
        features_dir=features_dir,
        info_dir=info_dir
    )
    
    logger.log(f"Data shape: {clean_df.shape}")
    logger.log(f"Features: {len(new_features)}, Subjects: {len(good_quality_sids)}")
    
    SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)
    
    # Create splits
    random.seed(0)
    train_sids = random.sample(good_quality_sids, 56)
    remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
    val_sids = random.sample(remaining_sids, 8)
    test_sids = [subj for subj in remaining_sids if subj not in val_sids]
    
    group_variables = ['AHI_Severity', 'Obesity']
    group_variable = get_variable(group_variables, idx=0)
    
    all_results = []
    
    # EXPERIMENT 1: Baseline (no normalization)
    logger.log("\n" + "="*80)
    logger.log("EXPERIMENT 1/7: BASELINE (No Normalization)")
    logger.log("="*80)
    
    baseline_results, baseline_model = run_experiment(
        "baseline", SW_df, final_features, train_sids, val_sids, test_sids, 
        group_variable, logger, max_evals=150
    )
    all_results.append(baseline_results)
    
    # EXPERIMENT 2-5: Different normalization methods
    norm_methods = ['zscore', 'robust', 'percentile', 'minmax']
    
    for idx, method in enumerate(norm_methods, start=2):
        logger.log("\n" + "="*80)
        logger.log(f"EXPERIMENT {idx}/7: {method.upper()} NORMALIZATION")
        logger.log("="*80)
        
        df_normalized = normalize_features_per_subject(SW_df, final_features, 'sid', method)
        
        norm_results, norm_model = run_experiment(
            f"norm_{method}", df_normalized, final_features, train_sids, val_sids, test_sids,
            group_variable, logger, max_evals=150
        )
        all_results.append(norm_results)
    
    # EXPERIMENT 6: Feature selection (remove subject-specific features)
    logger.log("\n" + "="*80)
    logger.log("EXPERIMENT 6/7: FEATURE SELECTION (Remove Subject-Specific)")
    logger.log("="*80)
    
    generalizable_features, subject_specific_features, variance_df = identify_subject_specific_features(
        SW_df, final_features, 'sid', threshold=0.7
    )
    
    logger.log(f"  Total features: {len(final_features)}")
    logger.log(f"  Generalizable features: {len(generalizable_features)}")
    logger.log(f"  Subject-specific features (removed): {len(subject_specific_features)}")
    
    if len(generalizable_features) > 50:  # Only if we have enough features left
        fs_results, fs_model = run_experiment(
            "feature_selection", SW_df, generalizable_features, train_sids, val_sids, test_sids,
            group_variable, logger, max_evals=150
        )
        all_results.append(fs_results)
    else:
        logger.log("  WARNING: Too few generalizable features, skipping this experiment")
    
    # EXPERIMENT 7: Best normalization + feature selection
    logger.log("\n" + "="*80)
    logger.log("EXPERIMENT 7/7: COMBINED (Best Norm + Feature Selection)")
    logger.log("="*80)
    
    # Find best normalization method
    norm_results_only = [r for r in all_results if 'norm_' in r['name']]
    if norm_results_only:
        best_norm = max(norm_results_only, key=lambda x: x['val']['accuracy'])
        best_norm_method = best_norm['name'].replace('norm_', '')
        
        logger.log(f"  Best normalization method: {best_norm_method}")
        logger.log(f"  Using {len(generalizable_features)} generalizable features")
        
        if len(generalizable_features) > 50:
            df_norm_fs = normalize_features_per_subject(SW_df, generalizable_features, 'sid', best_norm_method)
            
            combined_results, combined_model = run_experiment(
                f"combined_{best_norm_method}_fs", df_norm_fs, generalizable_features,
                train_sids, val_sids, test_sids, group_variable, logger, max_evals=150
            )
            all_results.append(combined_results)
    
    # Summary
    logger.log("\n" + "="*80)
    logger.log("SUMMARY OF ALL EXPERIMENTS")
    logger.log("="*80)
    
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Experiment': r['name'],
            'Features': r['n_features'],
            'Train_Acc': r['train']['accuracy'],
            'Val_Acc': r['val']['accuracy'],
            'Test_Acc': r['test']['accuracy'],
            'Val_F1': r['val']['f1_weighted'],
            'Overfit_Gap': r['overfit_gap'],
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Val_Acc', ascending=False)
    
    logger.log("\n" + summary_df.to_string(index=False))
    
    # Save detailed results
    summary_df.to_csv(os.path.join(results_dir, 'experiment_summary.csv'), index=False)
    
    with open(os.path.join(results_dir, 'detailed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Find best approach
    best_experiment = all_results[summary_df.index[0]]
    
    logger.log("\n" + "="*80)
    logger.log("BEST APPROACH")
    logger.log("="*80)
    logger.log(f"Experiment: {best_experiment['name']}")
    logger.log(f"Features: {best_experiment['n_features']}")
    logger.log(f"Validation Accuracy: {best_experiment['val']['accuracy']:.4f}")
    logger.log(f"Test Accuracy: {best_experiment['test']['accuracy']:.4f}")
    logger.log(f"Overfit Gap: {best_experiment['overfit_gap']:.4f}")
    
    # Improvement analysis
    baseline = all_results[0]
    improvement_val = best_experiment['val']['accuracy'] - baseline['val']['accuracy']
    improvement_gap = baseline['overfit_gap'] - best_experiment['overfit_gap']
    
    logger.log(f"\nImprovement over baseline:")
    logger.log(f"  Validation Accuracy: {improvement_val:+.4f} ({improvement_val/baseline['val']['accuracy']*100:+.1f}%)")
    logger.log(f"  Overfit Gap Reduction: {improvement_gap:.4f} ({improvement_gap/baseline['overfit_gap']*100:.1f}% reduction)")
    
    # Cross-validation on best approach if time permits
    if best_experiment['name'] != 'baseline':
        logger.log("\n" + "="*80)
        logger.log("CROSS-VALIDATION ON BEST APPROACH")
        logger.log("="*80)
        
        # Reconstruct the best dataframe
        if 'norm_' in best_experiment['name']:
            method = best_experiment['name'].replace('norm_', '').replace('_fs', '').replace('combined_', '')
            if 'fs' in best_experiment['name'] or 'combined' in best_experiment['name']:
                df_best = normalize_features_per_subject(SW_df, generalizable_features, 'sid', method)
                features_best = generalizable_features
            else:
                df_best = normalize_features_per_subject(SW_df, final_features, 'sid', method)
                features_best = final_features
        elif 'feature_selection' in best_experiment['name']:
            df_best = SW_df
            features_best = generalizable_features
        else:
            df_best = SW_df
            features_best = final_features
        
        cv_results = run_cross_validation_best(df_best, features_best, good_quality_sids, group_variable, logger, n_splits=5)
        
        # Save CV results
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(os.path.join(results_dir, 'cross_validation_best.csv'), index=False)
    
    logger.log("\n" + "="*80)
    logger.log("COMPREHENSIVE SUITE COMPLETE")
    logger.log("="*80)
    logger.log(f"All results saved to: {results_dir}")
    
    total_time = (datetime.now() - logger.start_time).total_seconds() / 60
    logger.log(f"Total runtime: {total_time:.1f} minutes")


if __name__ == "__main__":
    main()

