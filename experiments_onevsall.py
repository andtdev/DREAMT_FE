"""
One-vs-All Binary Classification with LightGBM
Based on original experiments.ipynb method

5 binary classifiers: Wake vs All, REM vs All, N1 vs All, N2 vs All, N3 vs All
Saves confusion matrices and SHAP plots for each model
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
import shap

from utils import *
from datasets import *

warnings.filterwarnings("ignore")

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 50)

random.seed(0)
np.random.seed(1)


class OneVsAllClassifier:
    """One-vs-All multiclass classifier using 5 binary LightGBM models."""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.class_names = ['Wake', 'REM', 'N1', 'N2', 'N3']
        self.models = {}
        self.best_params = {}
        
    def train_binary_classifier(self, X_train, y_train, X_val, y_val, class_idx, class_name):
        """Train a binary classifier for one class vs all others."""
        
        print(f"\n{'='*80}")
        print(f"TRAINING: {class_name} (Class {class_idx}) vs All Others")
        print(f"{'='*80}")
        
        # Convert to binary labels
        y_train_binary = (y_train == class_idx).astype(int)
        y_val_binary = (y_val == class_idx).astype(int)
        
        pos_count = y_train_binary.sum()
        neg_count = len(y_train_binary) - pos_count
        
        print(f"Training set: {pos_count} positive, {neg_count} negative")
        print(f"Validation set: {y_val_binary.sum()} positive, {len(y_val_binary) - y_val_binary.sum()} negative")
        
        # Calculate class weights (focal-loss inspired from original paper)
        total = len(y_train_binary)
        pos_weight = (total / pos_count) ** 1.4  # Higher gamma for harder classes
        neg_weight = (total / neg_count) ** 1.2
        
        class_weights = {0: neg_weight, 1: pos_weight}
        
        # Hyperparameter space (from original LightGBM_engine)
        space = {
            "max_depth": hp.quniform("max_depth", 2, 4, 1),
            "reg_alpha": hp.quniform("reg_alpha", 50, 300, 10),
            "reg_lambda": hp.uniform("reg_lambda", 5, 15),
            "num_leaves": hp.quniform("num_leaves", 10, 40, 5),
            "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
            "min_child_samples": hp.quniform("min_child_samples", 50, 200, 10),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
            "subsample": hp.uniform("subsample", 0.6, 0.9),
            "min_data_in_leaf": hp.quniform("min_data_in_leaf", 20, 100, 10),
        }
        
        def objective(params):
            clf = lgb.LGBMClassifier(
                objective="binary",
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
                X_train, y_train_binary,
                eval_set=[(X_val, y_val_binary)],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val_binary, y_pred, average='binary')
            
            return {"loss": -f1, "status": STATUS_OK}
        
        print("Running hyperparameter optimization (100 trials)...")
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            verbose=False
        )
        
        # Convert params
        for key in ['max_depth', 'n_estimators', 'num_leaves', 'min_child_samples', 'min_data_in_leaf']:
            if key in best_params:
                best_params[key] = int(best_params[key])
        
        best_f1 = -trials.best_trial['result']['loss']
        print(f"Best validation F1: {best_f1:.4f}")
        print(f"Best params: {best_params}")
        
        # Train final model
        final_model = lgb.LGBMClassifier(
            objective="binary",
            max_depth=best_params["max_depth"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            min_child_samples=best_params["min_child_samples"],
            min_data_in_leaf=best_params["min_data_in_leaf"],
            colsample_bytree=best_params["colsample_bytree"],
            subsample=best_params["subsample"],
            class_weight=class_weights,
            random_state=1,
            num_iterations=50,
            verbose=-1,
        )
        
        final_model.fit(
            X_train, y_train_binary,
            eval_set=[(X_val, y_val_binary)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        self.models[class_idx] = final_model
        self.best_params[class_idx] = best_params
        
        return final_model
    
    def predict_proba(self, X):
        """Get probability predictions from all 5 binary classifiers."""
        probas = np.zeros((len(X), 5))
        
        for class_idx in range(5):
            # Get probability for positive class (this class vs all)
            probas[:, class_idx] = self.models[class_idx].predict_proba(X)[:, 1]
        
        # Normalize probabilities to sum to 1
        probas = probas / probas.sum(axis=1, keepdims=True)
        
        return probas
    
    def predict(self, X):
        """Predict class labels."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def plot_binary_confusion_matrix(self, y_true, y_pred, class_name, dataset_name):
        """Plot confusion matrix for binary classification."""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Percentages
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1%',
            cmap='Blues',
            xticklabels=['Other', class_name],
            yticklabels=['Other', class_name],
            cbar_kws={'label': 'Percentage'},
            ax=ax1,
            vmin=0,
            vmax=1
        )
        ax1.set_title(f'{class_name} vs All - {dataset_name} (Percentages)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=11)
        ax1.set_ylabel('True', fontsize=11)
        
        # Counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=['Other', class_name],
            yticklabels=['Other', class_name],
            cbar_kws={'label': 'Count'},
            ax=ax2
        )
        ax2.set_title(f'{class_name} vs All - {dataset_name} (Counts)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=11)
        ax2.set_ylabel('True', fontsize=11)
        
        plt.tight_layout()
        filename = f'{class_name.lower()}_vs_all_{dataset_name.lower()}_binary_cm.png'
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        recall = cm_percent[1, 1] if len(cm) > 1 else 0
        precision = cm[1, 1] / cm[:, 1].sum() if len(cm) > 1 and cm[:, 1].sum() > 0 else 0
        
        print(f"\n{class_name} Binary Classification - {dataset_name}:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Saved to: {filepath}")
    
    def plot_multiclass_confusion_matrix(self, y_true, y_pred, dataset_name):
        """Plot confusion matrix for final multiclass predictions."""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Percentages
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1%',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage'},
            ax=ax1,
            vmin=0,
            vmax=1
        )
        ax1.set_title(f'Multiclass - {dataset_name} (Percentages)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('True', fontsize=12)
        
        # Counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'},
            ax=ax2
        )
        ax2.set_title(f'Multiclass - {dataset_name} (Counts)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=12)
        ax2.set_ylabel('True', fontsize=12)
        
        plt.tight_layout()
        filename = f'multiclass_{dataset_name.lower()}_cm.png'
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print per-class metrics
        acc = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)
        
        print(f"\n{'='*80}")
        print(f"MULTICLASS RESULTS - {dataset_name}")
        print(f"{'='*80}")
        print(f"Overall Accuracy: {acc:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"\nPer-Class Recall:")
        for i, name in enumerate(self.class_names):
            if i < len(cm):
                recall = cm_percent[i, i]
                count = cm[i, i]
                total = cm[i, :].sum()
                print(f"  {name}: {recall:.1%} ({count}/{total} samples)")
        print(f"Saved to: {filepath}")
    
    def plot_shap_summary(self, X_train, class_idx, class_name, feature_names):
        """Plot SHAP summary for a binary classifier."""
        print(f"\nGenerating SHAP values for {class_name}...")
        
        model = self.models[class_idx]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        
        # SHAP returns [negative_class, positive_class] for binary
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class (this class vs all)
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train, plot_type="bar", 
                         feature_names=feature_names, show=False, max_display=20)
        plt.title(f'SHAP Feature Importance - {class_name} vs All', fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f'{class_name.lower()}_vs_all_shap_bar.png'
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Bar plot saved to: {filepath}")
        
        # Beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_train, 
                         feature_names=feature_names, show=False, max_display=20)
        plt.title(f'SHAP Feature Impact - {class_name} vs All', fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f'{class_name.lower()}_vs_all_shap_beeswarm.png'
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Beeswarm plot saved to: {filepath}")


def main():
    """Main function."""
    
    print("="*80)
    print("ONE-VS-ALL BINARY CLASSIFICATION (5 MODELS)")
    print("="*80)
    
    # Data preparation (from original experiments.ipynb)
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
    
    # Create splits (same as original)
    random.seed(0)
    train_sids = random.sample(good_quality_sids, 56)
    remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
    val_sids = random.sample(remaining_sids, 8)
    test_sids = [subj for subj in remaining_sids if subj not in val_sids]
    
    group_variables = ['AHI_Severity', 'Obesity']
    group_variable = get_variable(group_variables, idx=0)
    
    # Prepare datasets
    print("\nPreparing datasets...")
    X_train, y_train, _ = train_test_split(SW_df, train_sids, final_features, group_variable)
    X_val, y_val, _ = train_test_split(SW_df, val_sids, final_features, group_variable)
    X_test, y_test, _ = train_test_split(SW_df, test_sids, final_features, group_variable)
    
    print(f"Train: {X_train.shape[0]} samples, {np.bincount(y_train.astype(int))}")
    print(f"Val: {X_val.shape[0]} samples, {np.bincount(y_val.astype(int))}")
    print(f"Test: {X_test.shape[0]} samples, {np.bincount(y_test.astype(int))}")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    results_dir = f'./results/onevsall_{timestamp}'
    
    # Initialize classifier
    classifier = OneVsAllClassifier(results_dir)
    
    # Train 5 binary classifiers
    print(f"\n{'='*80}")
    print("TRAINING 5 BINARY CLASSIFIERS")
    print(f"{'='*80}")
    
    for class_idx, class_name in enumerate(classifier.class_names):
        classifier.train_binary_classifier(
            X_train, y_train, X_val, y_val, 
            class_idx, class_name
        )
        
        # Generate binary confusion matrices
        y_train_binary = (y_train == class_idx).astype(int)
        y_val_binary = (y_val == class_idx).astype(int)
        y_test_binary = (y_test == class_idx).astype(int)
        
        y_train_pred_binary = classifier.models[class_idx].predict(X_train)
        y_val_pred_binary = classifier.models[class_idx].predict(X_val)
        y_test_pred_binary = classifier.models[class_idx].predict(X_test)
        
        classifier.plot_binary_confusion_matrix(y_train_binary, y_train_pred_binary, class_name, 'Train')
        classifier.plot_binary_confusion_matrix(y_val_binary, y_val_pred_binary, class_name, 'Val')
        classifier.plot_binary_confusion_matrix(y_test_binary, y_test_pred_binary, class_name, 'Test')
        
        # Generate SHAP plots
        classifier.plot_shap_summary(X_train, class_idx, class_name, final_features)
    
    # Get multiclass predictions
    print(f"\n{'='*80}")
    print("MULTICLASS PREDICTIONS (COMBINING 5 BINARY MODELS)")
    print(f"{'='*80}")
    
    y_train_pred = classifier.predict(X_train)
    y_val_pred = classifier.predict(X_val)
    y_test_pred = classifier.predict(X_test)
    
    # Plot multiclass confusion matrices
    classifier.plot_multiclass_confusion_matrix(y_train, y_train_pred, 'Train')
    classifier.plot_multiclass_confusion_matrix(y_val, y_val_pred, 'Val')
    classifier.plot_multiclass_confusion_matrix(y_test, y_test_pred, 'Test')
    
    # Save summary
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    summary = {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'overfit_gap': train_acc - val_acc,
        'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
        'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted'),
        'train_kappa': cohen_kappa_score(y_train, y_train_pred),
        'val_kappa': cohen_kappa_score(y_val, y_val_pred),
        'test_kappa': cohen_kappa_score(y_test, y_test_pred),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(results_dir, 'summary_metrics.csv'), index=False)
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {results_dir}")
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

