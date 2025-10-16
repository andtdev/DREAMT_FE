"""
Direct Multiclass Sleep Stage Classification Script

This script implements a single multiclass LightGBM classifier for all 5 sleep stages:
- Wake (0)
- REM (1)
- N1 (2)
- N2 (3)
- N3 (4)

Advantages over hierarchical:
- No error cascading
- All classes learned simultaneously
- Better feature sharing
- Simpler and more reliable
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
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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


class DirectMulticlassClassifier:
    """
    Direct multiclass classifier for sleep stages using LightGBM.
    """
    
    def __init__(self, results_dir='./results/direct_multiclass', use_focal_weights=True, gamma=1.2):
        self.model = None
        self.best_params = None
        self.features = None
        self.use_focal_weights = use_focal_weights
        self.gamma = gamma
        
        # Store training data for SHAP
        self.X_train = None
        
        # Add timestamp to results directory
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.results_dir = f"{results_dir}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def calculate_focal_class_weights(self, y, gamma=None):
        """
        Calculate focal-loss-inspired class weights.
        
        Parameters
        ----------
        y : np.ndarray
            Labels
        gamma : float, optional
            Focusing parameter. If None, uses self.gamma
            
        Returns
        -------
        class_weights_dict : dict
            Dictionary mapping class labels to weights
        """
        if not self.use_focal_weights:
            return 'balanced'
        
        if gamma is None:
            gamma = self.gamma
        
        class_counts = np.bincount(y.astype(int))
        total_samples = len(y)
        
        # Calculate weights with focal emphasis
        class_weights = {}
        for class_idx, count in enumerate(class_counts):
            if count > 0:
                class_weights[class_idx] = (total_samples / count) ** gamma
        
        print(f"\n  Class distribution: {class_counts}")
        print(f"  Class names: W={class_counts[0]}, R={class_counts[1]}, N1={class_counts[2]}, N2={class_counts[3]}, N3={class_counts[4]}")
        print(f"  Focal weights (gamma={gamma}): {class_weights}")
        
        return class_weights
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, title, filename):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Percentage'}
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {filepath}")
        print(f"\nConfusion Matrix (counts):")
        print(cm)
        
    def plot_shap_summary(self, model, X_sample, feature_names, title, filename, max_display=20):
        """Generate and save SHAP summary plots."""
        print(f"Generating SHAP values...")
        
        # Use a sample for SHAP calculation (for speed)
        if X_sample.shape[0] > 1000:
            sample_indices = np.random.choice(X_sample.shape[0], 1000, replace=False)
            X_shap = X_sample[sample_indices]
        else:
            X_shap = X_sample
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # Create DataFrame for easier handling
        X_shap_df = pd.DataFrame(X_shap, columns=feature_names)
        
        # For multiclass, shap_values is a list of arrays (one per class)
        # We'll plot the mean absolute SHAP value across all classes
        if isinstance(shap_values, list):
            shap_values_mean = np.abs(shap_values).mean(axis=0)
        else:
            shap_values_mean = np.abs(shap_values)
        
        # Summary plot (bar)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_mean,
            X_shap_df,
            plot_type="bar",
            max_display=max_display,
            show=False
        )
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filepath = os.path.join(self.results_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP plot saved to: {filepath}")
        
        # Also save detailed SHAP plot (beeswarm) for each class
        if isinstance(shap_values, list):
            for class_idx, class_name in enumerate(['Wake', 'REM', 'N1', 'N2', 'N3']):
                beeswarm_filename = filename.replace('.png', f'_class_{class_name}.png')
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values[class_idx],
                    X_shap_df,
                    max_display=max_display,
                    show=False
                )
                plt.title(f"{title} - {class_name} Class", fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                
                beeswarm_filepath = os.path.join(self.results_dir, beeswarm_filename)
                plt.savefig(beeswarm_filepath, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"SHAP beeswarm plot for {class_name} saved to: {beeswarm_filepath}")
    
    def fit(self, X_train, y_train, X_val, y_val, feature_names):
        """
        Fit the multiclass classifier.
        
        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : validation data
        feature_names : list
            List of feature names
        """
        print("\n" + "="*80)
        print("DIRECT MULTICLASS SLEEP STAGE CLASSIFICATION")
        print("="*80)
        
        print(f"\nTraining samples: {len(y_train)}")
        print(f"Validation samples: {len(y_val)}")
        
        self.features = feature_names
        self.X_train = X_train
        
        # Calculate focal class weights
        class_weights = self.calculate_focal_class_weights(y_train)
        
        # Balanced hyperparameters - not too aggressive
        space = {
            "max_depth": hp.quniform("max_depth", 3, 6, 1),  # Allow more depth
            "reg_alpha": hp.quniform("reg_alpha", 10, 150, 10),  # Moderate regularization
            "reg_lambda": hp.uniform("reg_lambda", 1, 10),  # Moderate regularization
            "num_leaves": hp.quniform("num_leaves", 20, 60, 5),  # Allow more leaves
            "n_estimators": hp.quniform("n_estimators", 100, 300, 25),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
            "min_child_samples": hp.quniform("min_child_samples", 20, 100, 10),
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
        
        print(f"\nStarting hyperparameter optimization (150 trials)...")
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=150,  # More trials since we only have one model
            trials=trials
        )
        
        # Convert params
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["num_leaves"] = int(best_params["num_leaves"])
        best_params["min_child_samples"] = int(best_params["min_child_samples"])
        best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"])
        
        self.best_params = best_params
        
        print(f"\nBest hyperparameters: {best_params}")
        
        # Train final model
        final_n_estimators = best_params["n_estimators"] + 50
        
        self.model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=5,
            max_depth=best_params["max_depth"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            n_estimators=final_n_estimators,
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            min_child_samples=best_params["min_child_samples"],
            min_data_in_leaf=best_params["min_data_in_leaf"],
            colsample_bytree=best_params["colsample_bytree"],
            subsample=best_params["subsample"],
            class_weight=class_weights,
            random_state=1,
        )
        
        print("\nTraining final model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=True)]
        )
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        y_pred_val = self.model.predict(X_val)
        self.plot_confusion_matrix(
            y_val, y_pred_val,
            class_names=['Wake', 'REM', 'N1', 'N2', 'N3'],
            title='Direct Multiclass - Validation Set',
            filename='validation_confusion_matrix.png'
        )
        
        self.plot_shap_summary(
            self.model, X_train, feature_names,
            title='Feature Importance (All Classes)',
            filename='shap_summary.png'
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
    
    def predict(self, X):
        """Predict sleep stages."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities for all sleep stages."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """Evaluate the classifier."""
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name} Set")
        print(f"{'='*60}")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # Per-class metrics
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  Cohen's Kappa: {kappa:.4f}")
        
        print(f"\nPer-Class F1 Scores:")
        class_names = ['Wake', 'REM', 'N1', 'N2', 'N3']
        for i, name in enumerate(class_names):
            if i < len(f1_per_class):
                print(f"  {name}: {f1_per_class[i]:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Save confusion matrix visualization
        filename = f'confusion_matrix_{dataset_name.lower()}.png'
        self.plot_confusion_matrix(
            y_test, y_pred,
            class_names=['Wake', 'REM', 'N1', 'N2', 'N3'],
            title=f'Direct Multiclass - {dataset_name} Set',
            filename=filename
        )
        
        results_df = pd.DataFrame([{
            'Model': 'Direct_Multiclass_LightGBM',
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'F1_Weighted': f1_weighted,
            'F1_Macro': f1_macro,
            'Cohens_Kappa': kappa,
            'F1_Wake': f1_per_class[0] if len(f1_per_class) > 0 else 0,
            'F1_REM': f1_per_class[1] if len(f1_per_class) > 1 else 0,
            'F1_N1': f1_per_class[2] if len(f1_per_class) > 2 else 0,
            'F1_N2': f1_per_class[3] if len(f1_per_class) > 3 else 0,
            'F1_N3': f1_per_class[4] if len(f1_per_class) > 4 else 0,
        }])
        
        return results_df, cm, y_pred, y_pred_proba
    
    def save_method_info(self, filename='method_info.txt'):
        """Save information about the method and its parameters."""
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DIRECT MULTICLASS SLEEP STAGE CLASSIFICATION METHOD\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("METHOD DESCRIPTION:\n")
            f.write("-" * 80 + "\n")
            f.write("Single multiclass LightGBM classifier for all 5 sleep stages:\n")
            f.write("  - Wake (0)\n")
            f.write("  - REM (1)\n")
            f.write("  - N1 (2)\n")
            f.write("  - N2 (3)\n")
            f.write("  - N3 (4)\n\n")
            
            f.write("DATA PREPROCESSING:\n")
            f.write(f"  - Focal Class Weighting: {'Enabled (gamma=' + str(self.gamma) + ')' if self.use_focal_weights else 'Disabled'}\n")
            f.write("  - Class Weight Formula: (total_samples / class_count) ^ gamma\n")
            f.write("  - Hyperparameter Optimization: 150 trials\n")
            f.write("  - Early Stopping: 20 rounds patience\n")
            f.write("  - Memory Efficient: No data resampling required\n\n")
            
            f.write("ADVANTAGES OVER HIERARCHICAL:\n")
            f.write("  - No error cascading between levels\n")
            f.write("  - All classes learned simultaneously with shared features\n")
            f.write("  - Simpler architecture with fewer hyperparameters\n")
            f.write("  - More robust and reliable\n")
            f.write("  - Better generalization\n\n")
            
            f.write("="*80 + "\n")
            f.write("BEST HYPERPARAMETERS\n")
            f.write("="*80 + "\n")
            f.write(json.dumps(self.best_params, indent=2) + "\n\n")
            
            f.write("="*80 + "\n")
            f.write("FEATURE INFORMATION\n")
            f.write("="*80 + "\n")
            f.write(f"Number of features: {len(self.features)}\n")
        
        print(f"\nMethod information saved to: {filepath}")


def main():
    """Main execution function."""
    print("="*80)
    print("DIRECT MULTICLASS SLEEP STAGE CLASSIFICATION")
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
    
    # Split data
    print("\n" + "="*80)
    print("DATA SPLITTING")
    print("="*80)
    
    SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)
    
    # Create train/val/test splits
    random.seed(0)
    train_sids = random.sample(good_quality_sids, 56)
    remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
    val_sids = random.sample(remaining_sids, 8)
    test_sids = [subj for subj in remaining_sids if subj not in val_sids]
    
    group_variables = ['AHI_Severity', 'Obesity']
    group_variable = get_variable(group_variables, idx=0)
    
    # Prepare datasets
    X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
    X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
    X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"  Class distribution: {np.bincount(y_train.astype(int))}")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"  Class distribution: {np.bincount(y_val.astype(int))}")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"  Class distribution: {np.bincount(y_test.astype(int))}")
    
    # Train Classifier
    print("\n" + "="*80)
    print("TRAINING DIRECT MULTICLASS CLASSIFIER")
    print("="*80)
    
    classifier = DirectMulticlassClassifier(
        results_dir='./results/direct_multiclass',
        use_focal_weights=True,
        gamma=1.2  # Moderate focal weight
    )
    classifier.fit(X_train, y_train, X_val, y_val, final_features)
    
    # Save method information
    classifier.save_method_info()
    
    # Evaluate on all sets
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    train_results, train_cm, train_pred, train_proba = classifier.evaluate(X_train, y_train, "Train")
    val_results, val_cm, val_pred, val_proba = classifier.evaluate(X_val, y_val, "Validation")
    test_results, test_cm, test_pred, test_proba = classifier.evaluate(X_test, y_test, "Test")
    
    # Combine results
    all_results = pd.concat([train_results, val_results, test_results], ignore_index=True)
    
    # Save results
    results_path = os.path.join(classifier.results_dir, 'results.csv')
    all_results.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Display final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(all_results.to_string(index=False))
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

