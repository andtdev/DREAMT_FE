"""
Hierarchical Sleep Stage Classification Script

This script implements a hierarchical classification approach for sleep stages:
- Level 1: Wake vs All Others (binary classification)
- Level 2: REM vs Non-REM (within sleep stages)
- Level 3: Different Non-REM phases (N1, N2, N3 classification)

Each level can use different features optimized for that classification task.
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
import gpboost as gpb  # Keep library but not used in this script
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from utils import *
from datasets import *
from models import *

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


class HierarchicalSleepClassifier:
    """
    Hierarchical classifier for sleep stages using LightGBM.
    
    Three-level hierarchy:
    1. Wake vs Sleep
    2. REM vs Non-REM (within sleep)
    3. N1 vs N2 vs N3 (within non-REM sleep)
    """
    
    def __init__(self, results_dir='./results/hierarchical', use_focal_weights=True):
        self.level1_model = None  # Wake vs Sleep
        self.level2_model = None  # REM vs Non-REM
        self.level3_model = None  # N1 vs N2 vs N3
        
        self.level1_features = None
        self.level2_features = None
        self.level3_features = None
        
        self.level1_params = None
        self.level2_params = None
        self.level3_params = None
        
        # Store training data for SHAP
        self.X_train_l1 = None
        self.X_train_l2 = None
        self.X_train_l3 = None
        
        self.use_focal_weights = use_focal_weights
        
        # Add timestamp to results directory
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        self.results_dir = f"{results_dir}_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def select_features_for_level(self, all_features, level):
        """
        Select relevant features for each classification level.
        Different levels may benefit from different feature subsets.
        
        Parameters
        ----------
        all_features : list
            List of all available feature names
        level : int
            Classification level (1, 2, or 3)
            
        Returns
        -------
        selected_features : list
            Features selected for this level
        """
        # For now, use all features for all levels
        # Can be customized based on feature importance analysis
        
        if level == 1:
            # Wake vs Sleep: activity and heart rate features are most relevant
            priority_keywords = ['ACC', 'HRV', 'HR_', 'TEMP']
        elif level == 2:
            # REM vs Non-REM: heart rate variability and eye movement proxies
            priority_keywords = ['HRV', 'HR_', 'ACC_INDEX', 'derivative']
        elif level == 3:
            # N1 vs N2 vs N3: deep sleep features (heart rate, HRV, temperature)
            priority_keywords = ['HRV', 'HR_', 'TEMP', 'EDA', 'gaussian']
        else:
            priority_keywords = []
        
        # For now, return all features
        # TODO: Implement feature selection based on importance
        return all_features
    
    def calculate_focal_class_weights(self, y, gamma=1.5):
        """
        Calculate focal-loss-inspired class weights.
        
        This is much more memory-efficient than SMOTE and often works better.
        Uses the formula: weight = (total_samples / class_count) ^ gamma
        
        Parameters
        ----------
        y : np.ndarray
            Labels
        gamma : float
            Focusing parameter. Higher = more focus on minority classes.
            Default 1.5 provides balanced improvement without over-focusing.
            
        Returns
        -------
        class_weights_dict : dict
            Dictionary mapping class labels to weights
        """
        if not self.use_focal_weights:
            return 'balanced'
        
        class_counts = np.bincount(y.astype(int))
        total_samples = len(y)
        
        # Calculate weights with focal emphasis
        class_weights = {}
        for class_idx, count in enumerate(class_counts):
            if count > 0:
                # Use gamma to control focus on minority classes
                class_weights[class_idx] = (total_samples / count) ** gamma
        
        print(f"\n  Class distribution: {class_counts}")
        print(f"  Focal weights (gamma={gamma}): {class_weights}")
        
        return class_weights
    
    def prepare_level1_data(self, X, y):
        """
        Prepare data for Level 1: Wake (0) vs Sleep (1, 2, 3, 4)
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels (0=W, 1=R, 2=N1, 3=N2, 4=N3)
            
        Returns
        -------
        X_level1, y_level1 : tuple
            Data for level 1 classification
        """
        # Convert to binary: 0=Wake, 1=Sleep (any sleep stage)
        y_level1 = (y > 0).astype(int)
        return X, y_level1
    
    def prepare_level2_data(self, X, y):
        """
        Prepare data for Level 2: REM (1) vs Non-REM (2, 3, 4)
        Only includes samples that are already classified as sleep.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels (0=W, 1=R, 2=N1, 3=N2, 4=N3)
            
        Returns
        -------
        X_level2, y_level2 : tuple
            Data for level 2 classification (sleep stages only)
        """
        # Filter only sleep stages (exclude wake)
        sleep_mask = y > 0
        X_level2 = X[sleep_mask]
        y_sleep = y[sleep_mask]
        
        # Convert to binary: 0=REM (1), 1=Non-REM (2, 3, 4)
        y_level2 = (y_sleep > 1).astype(int)
        
        return X_level2, y_level2
    
    def prepare_level3_data(self, X, y):
        """
        Prepare data for Level 3: N1 (2) vs N2 (3) vs N3 (4)
        Only includes samples that are Non-REM.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels (0=W, 1=R, 2=N1, 3=N2, 4=N3)
            
        Returns
        -------
        X_level3, y_level3 : tuple
            Data for level 3 classification (non-REM stages only)
        """
        # Filter only non-REM stages
        nonrem_mask = y >= 2
        X_level3 = X[nonrem_mask]
        y_level3 = y[nonrem_mask]
        
        # Remap labels: N1=0, N2=1, N3=2
        y_level3 = y_level3 - 2
        
        return X_level3, y_level3
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, title, filename):
        """
        Plot and save confusion matrix.
        
        Parameters
        ----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        class_names : list
            Names of classes
        title : str
            Plot title
        filename : str
            Output filename
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
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
        
        # Also print counts
        print(f"\nConfusion Matrix (counts):")
        print(cm)
        
    def plot_shap_summary(self, model, X_sample, feature_names, title, filename, max_display=20):
        """
        Generate and save SHAP summary plot.
        
        Parameters
        ----------
        model : LightGBM model
            Trained model
        X_sample : np.ndarray
            Sample of training data for SHAP (use subset for speed)
        feature_names : list
            Names of features
        title : str
            Plot title
        filename : str
            Output filename
        max_display : int
            Maximum number of features to display
        """
        print(f"Generating SHAP values for {title}...")
        
        # Use a sample for SHAP calculation (for speed)
        if X_sample.shape[0] > 1000:
            sample_indices = np.random.choice(X_sample.shape[0], 1000, replace=False)
            X_shap = X_sample[sample_indices]
        else:
            X_shap = X_sample
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Create DataFrame for easier handling
        X_shap_df = pd.DataFrame(X_shap, columns=feature_names)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
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
        
        # Also save detailed SHAP plot (beeswarm)
        beeswarm_filename = filename.replace('.png', '_beeswarm.png')
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_shap_df,
            max_display=max_display,
            show=False
        )
        plt.title(f"{title} (Detailed)", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        beeswarm_filepath = os.path.join(self.results_dir, beeswarm_filename)
        plt.savefig(beeswarm_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP beeswarm plot saved to: {beeswarm_filepath}")
    
    def train_lgb_binary(self, X_train, y_train, X_val, y_val, level_name):
        """
        Train a binary LightGBM classifier.
        
        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : validation data
        level_name : str
            Name of the level for logging
            
        Returns
        -------
        model, best_params : tuple
            Trained model and best hyperparameters
        """
        print(f"\n{'='*60}")
        print(f"Training {level_name}")
        print(f"{'='*60}")
        print(f"Training samples: {len(y_train)}, Class distribution: {np.bincount(y_train.astype(int))}")
        print(f"Validation samples: {len(y_val)}, Class distribution: {np.bincount(y_val.astype(int))}")
        
        # Calculate focal class weights (memory-efficient alternative to SMOTE)
        class_weights = self.calculate_focal_class_weights(y_train, gamma=1.5)
        
        # More conservative hyperparameters to prevent overfitting
        space = {
            "max_depth": hp.quniform("max_depth", 2, 4, 1),  # Reduced from 6
            "reg_alpha": hp.quniform("reg_alpha", 50, 300, 10),  # Increased regularization
            "reg_lambda": hp.uniform("reg_lambda", 5, 20),  # Increased regularization
            "num_leaves": hp.quniform("num_leaves", 10, 40, 5),  # Reduced from 60
            "n_estimators": hp.quniform("n_estimators", 50, 200, 25),  # Reduced from 300
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),  # More conservative
            "min_child_samples": hp.quniform("min_child_samples", 50, 200, 10),  # Increased from 20
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 0.9),  # Reduced max from 1.0
            "subsample": hp.uniform("subsample", 0.6, 0.85),  # Reduced from 0.9
            "min_data_in_leaf": hp.quniform("min_data_in_leaf", 20, 100, 10),  # Added for regularization
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
                class_weight=class_weights,  # Use focal weights
                verbose=-1,
            )
            
            clf.fit(
                X_train, y_train,  # No resampling needed
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
            )
            
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='binary')
            
            return {"loss": -f1, "status": STATUS_OK}
        
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,  # Increased from 50 for better optimization
            trials=trials
        )
        
        # Convert params
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["num_leaves"] = int(best_params["num_leaves"])
        best_params["min_child_samples"] = int(best_params["min_child_samples"])
        best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"])
        
        print(f"\nBest hyperparameters for {level_name}: {best_params}")
        
        # Train final model with more iterations for better convergence
        # Add extra iterations to the best params
        final_n_estimators = best_params["n_estimators"] + 50
        
        final_model = lgb.LGBMClassifier(
            objective="binary",
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
            class_weight=class_weights,  # Use focal weights
            random_state=1,
        )
        
        final_model.fit(
            X_train, y_train,  # No resampling needed
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]  # More patience
        )
        
        return final_model, best_params
    
    def train_lgb_multiclass(self, X_train, y_train, X_val, y_val, level_name, num_classes):
        """
        Train a multiclass LightGBM classifier.
        
        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : validation data
        level_name : str
            Name of the level for logging
        num_classes : int
            Number of classes
            
        Returns
        -------
        model, best_params : tuple
            Trained model and best hyperparameters
        """
        print(f"\n{'='*60}")
        print(f"Training {level_name}")
        print(f"{'='*60}")
        print(f"Training samples: {len(y_train)}, Class distribution: {np.bincount(y_train.astype(int))}")
        print(f"Validation samples: {len(y_val)}, Class distribution: {np.bincount(y_val.astype(int))}")
        
        # Calculate focal class weights (memory-efficient alternative to SMOTE)
        class_weights = self.calculate_focal_class_weights(y_train, gamma=1.5)
        
        # More conservative hyperparameters to prevent overfitting
        space = {
            "max_depth": hp.quniform("max_depth", 2, 4, 1),  # Reduced from 6
            "reg_alpha": hp.quniform("reg_alpha", 50, 300, 10),  # Increased regularization
            "reg_lambda": hp.uniform("reg_lambda", 5, 20),  # Increased regularization
            "num_leaves": hp.quniform("num_leaves", 10, 40, 5),  # Reduced from 60
            "n_estimators": hp.quniform("n_estimators", 50, 200, 25),  # Reduced from 300
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),  # More conservative
            "min_child_samples": hp.quniform("min_child_samples", 50, 200, 10),  # Increased from 20
            "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 0.9),  # Reduced max from 1.0
            "subsample": hp.uniform("subsample", 0.6, 0.85),  # Reduced from 0.9
            "min_data_in_leaf": hp.quniform("min_data_in_leaf", 20, 100, 10),  # Added for regularization
        }
        
        def objective(params):
            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
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
                class_weight=class_weights,  # Use focal weights
                verbose=-1,
            )
            
            clf.fit(
                X_train, y_train,  # No resampling needed
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
            max_evals=100,  # Increased from 50 for better optimization
            trials=trials
        )
        
        # Convert params
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["n_estimators"] = int(best_params["n_estimators"])
        best_params["num_leaves"] = int(best_params["num_leaves"])
        best_params["min_child_samples"] = int(best_params["min_child_samples"])
        best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"])
        
        print(f"\nBest hyperparameters for {level_name}: {best_params}")
        
        # Train final model with more iterations for better convergence
        # Add extra iterations to the best params
        final_n_estimators = best_params["n_estimators"] + 50
        
        final_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
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
            class_weight=class_weights,  # Use focal weights
            random_state=1,
        )
        
        final_model.fit(
            X_train, y_train,  # No resampling needed
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]  # More patience
        )
        
        return final_model, best_params
    
    def fit(self, X_train, y_train, X_val, y_val, all_features):
        """
        Fit all three levels of the hierarchy.
        
        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val : validation data
        all_features : list
            List of all feature names
        """
        # Level 1: Wake vs Sleep
        print("\n" + "="*80)
        print("LEVEL 1: Wake vs All Sleep Stages")
        print("="*80)
        
        self.level1_features = self.select_features_for_level(all_features, 1)
        X_train_l1, y_train_l1 = self.prepare_level1_data(X_train, y_train)
        X_val_l1, y_val_l1 = self.prepare_level1_data(X_val, y_val)
        
        self.level1_model, self.level1_params = self.train_lgb_binary(
            X_train_l1, y_train_l1, X_val_l1, y_val_l1, "Level 1: Wake vs Sleep"
        )
        
        # Store training data for SHAP
        self.X_train_l1 = X_train_l1
        
        # Generate visualizations for Level 1
        print("\nGenerating Level 1 visualizations...")
        y_pred_l1 = self.level1_model.predict(X_val_l1)
        self.plot_confusion_matrix(
            y_val_l1, y_pred_l1,
            class_names=['Wake', 'Sleep'],
            title='Level 1: Wake vs Sleep - Validation Set',
            filename='level1_confusion_matrix.png'
        )
        self.plot_shap_summary(
            self.level1_model, X_train_l1, self.level1_features,
            title='Level 1: Feature Importance (Wake vs Sleep)',
            filename='level1_shap.png'
        )
        
        # Level 2: REM vs Non-REM
        print("\n" + "="*80)
        print("LEVEL 2: REM vs Non-REM (within sleep)")
        print("="*80)
        
        self.level2_features = self.select_features_for_level(all_features, 2)
        X_train_l2, y_train_l2 = self.prepare_level2_data(X_train, y_train)
        X_val_l2, y_val_l2 = self.prepare_level2_data(X_val, y_val)
        
        self.level2_model, self.level2_params = self.train_lgb_binary(
            X_train_l2, y_train_l2, X_val_l2, y_val_l2, "Level 2: REM vs Non-REM"
        )
        
        # Store training data for SHAP
        self.X_train_l2 = X_train_l2
        
        # Generate visualizations for Level 2
        print("\nGenerating Level 2 visualizations...")
        y_pred_l2 = self.level2_model.predict(X_val_l2)
        self.plot_confusion_matrix(
            y_val_l2, y_pred_l2,
            class_names=['REM', 'Non-REM'],
            title='Level 2: REM vs Non-REM - Validation Set',
            filename='level2_confusion_matrix.png'
        )
        self.plot_shap_summary(
            self.level2_model, X_train_l2, self.level2_features,
            title='Level 2: Feature Importance (REM vs Non-REM)',
            filename='level2_shap.png'
        )
        
        # Level 3: N1 vs N2 vs N3
        print("\n" + "="*80)
        print("LEVEL 3: N1 vs N2 vs N3 (within non-REM)")
        print("="*80)
        
        self.level3_features = self.select_features_for_level(all_features, 3)
        X_train_l3, y_train_l3 = self.prepare_level3_data(X_train, y_train)
        X_val_l3, y_val_l3 = self.prepare_level3_data(X_val, y_val)
        
        self.level3_model, self.level3_params = self.train_lgb_multiclass(
            X_train_l3, y_train_l3, X_val_l3, y_val_l3, "Level 3: N1 vs N2 vs N3", num_classes=3
        )
        
        # Store training data for SHAP
        self.X_train_l3 = X_train_l3
        
        # Generate visualizations for Level 3
        print("\nGenerating Level 3 visualizations...")
        y_pred_l3 = self.level3_model.predict(X_val_l3)
        self.plot_confusion_matrix(
            y_val_l3, y_pred_l3,
            class_names=['N1', 'N2', 'N3'],
            title='Level 3: N1 vs N2 vs N3 - Validation Set',
            filename='level3_confusion_matrix.png'
        )
        self.plot_shap_summary(
            self.level3_model, X_train_l3, self.level3_features,
            title='Level 3: Feature Importance (N1 vs N2 vs N3)',
            filename='level3_shap.png'
        )
        
        print("\n" + "="*80)
        print("HIERARCHICAL TRAINING COMPLETE")
        print("="*80)
    
    def predict(self, X):
        """
        Predict sleep stages using the hierarchical approach.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        predictions : np.ndarray
            Final predictions (0=W, 1=R, 2=N1, 3=N2, 4=N3)
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        # Level 1: Wake vs Sleep
        level1_pred = self.level1_model.predict(X)
        wake_mask = (level1_pred == 0)
        predictions[wake_mask] = 0  # Wake
        
        # For samples predicted as sleep, go to level 2
        sleep_mask = ~wake_mask
        X_sleep = X[sleep_mask]
        
        if X_sleep.shape[0] > 0:
            # Level 2: REM vs Non-REM
            level2_pred = self.level2_model.predict(X_sleep)
            rem_mask_local = (level2_pred == 0)
            
            # Convert local mask to global indices
            sleep_indices = np.where(sleep_mask)[0]
            rem_indices = sleep_indices[rem_mask_local]
            nonrem_indices = sleep_indices[~rem_mask_local]
            
            predictions[rem_indices] = 1  # REM
            
            # For Non-REM samples, go to level 3
            if len(nonrem_indices) > 0:
                X_nonrem = X[nonrem_indices]
                level3_pred = self.level3_model.predict(X_nonrem)
                
                # Map: 0->N1(2), 1->N2(3), 2->N3(4)
                predictions[nonrem_indices] = level3_pred + 2
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probabilities for all sleep stages using hierarchical approach.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        probabilities : np.ndarray
            Probability matrix (n_samples, 5) for classes [W, R, N1, N2, N3]
        """
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 5))
        
        # Level 1: P(Wake) and P(Sleep)
        level1_proba = self.level1_model.predict_proba(X)
        p_wake = level1_proba[:, 0]
        p_sleep = level1_proba[:, 1]
        
        probs[:, 0] = p_wake  # P(Wake)
        
        # Level 2: P(REM | Sleep) and P(Non-REM | Sleep)
        level2_proba = self.level2_model.predict_proba(X)
        p_rem_given_sleep = level2_proba[:, 0]
        p_nonrem_given_sleep = level2_proba[:, 1]
        
        probs[:, 1] = p_sleep * p_rem_given_sleep  # P(REM)
        
        # Level 3: P(N1, N2, N3 | Non-REM)
        level3_proba = self.level3_model.predict_proba(X)
        p_n1_given_nonrem = level3_proba[:, 0]
        p_n2_given_nonrem = level3_proba[:, 1]
        p_n3_given_nonrem = level3_proba[:, 2]
        
        probs[:, 2] = p_sleep * p_nonrem_given_sleep * p_n1_given_nonrem  # P(N1)
        probs[:, 3] = p_sleep * p_nonrem_given_sleep * p_n2_given_nonrem  # P(N2)
        probs[:, 4] = p_sleep * p_nonrem_given_sleep * p_n3_given_nonrem  # P(N3)
        
        # Normalize to ensure probabilities sum to 1
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        return probs
    
    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """
        Evaluate the hierarchical classifier.
        
        Parameters
        ----------
        X_test, y_test : test data
        dataset_name : str
            Name of the dataset for reporting
            
        Returns
        -------
        results_df : pd.DataFrame
            Evaluation metrics
        """
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
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  Cohen's Kappa: {kappa:.4f}")
        
        print(f"\nPer-Class F1 Scores:")
        class_names = ['Wake', 'REM', 'N1', 'N2', 'N3']
        for i, name in enumerate(class_names):
            print(f"  {name}: {f1_per_class[i]:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Save confusion matrix visualization
        filename = f'final_confusion_matrix_{dataset_name.lower()}.png'
        self.plot_confusion_matrix(
            y_test, y_pred,
            class_names=['Wake', 'REM', 'N1', 'N2', 'N3'],
            title=f'Final Hierarchical Classification - {dataset_name} Set',
            filename=filename
        )
        
        results_df = pd.DataFrame([{
            'Model': 'Hierarchical_LightGBM',
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'F1_Weighted': f1_weighted,
            'F1_Macro': f1_macro,
            'Cohens_Kappa': kappa,
            'F1_Wake': f1_per_class[0],
            'F1_REM': f1_per_class[1],
            'F1_N1': f1_per_class[2],
            'F1_N2': f1_per_class[3],
            'F1_N3': f1_per_class[4],
        }])
        
        return results_df, cm, y_pred, y_pred_proba
    
    def save_method_info(self, filename='method_info.txt'):
        """
        Save information about the method and its parameters to a file.
        
        Parameters
        ----------
        filename : str
            Output filename
        """
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HIERARCHICAL SLEEP STAGE CLASSIFICATION METHOD\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("METHOD DESCRIPTION:\n")
            f.write("-" * 80 + "\n")
            f.write("Three-level hierarchical classification using LightGBM:\n")
            f.write("  Level 1: Wake (0) vs All Sleep Stages (1,2,3,4)\n")
            f.write("  Level 2: REM (1) vs Non-REM (2,3,4) [within sleep]\n")
            f.write("  Level 3: N1 (2) vs N2 (3) vs N3 (4) [within non-REM]\n\n")
            
            f.write("DATA PREPROCESSING:\n")
            f.write(f"  - Focal Class Weighting: {'Enabled (gamma=1.5)' if self.use_focal_weights else 'Disabled'}\n")
            f.write("  - Class Weight Formula: (total_samples / class_count) ^ gamma\n")
            f.write("  - Hyperparameter Optimization: 100 trials per level\n")
            f.write("  - Early Stopping: 20 rounds patience\n")
            f.write("  - Memory Efficient: No data resampling required\n\n")
            
            f.write("ADVANTAGES:\n")
            f.write("  - Handles class imbalance at each level separately\n")
            f.write("  - Can use different features optimized for each classification task\n")
            f.write("  - More interpretable decision process\n")
            f.write("  - Each level focuses on distinguishing similar patterns\n")
            f.write("  - Strong regularization to prevent overfitting\n\n")
            
            f.write("="*80 + "\n")
            f.write("LEVEL 1 PARAMETERS: Wake vs Sleep\n")
            f.write("="*80 + "\n")
            f.write(json.dumps(self.level1_params, indent=2) + "\n\n")
            
            f.write("="*80 + "\n")
            f.write("LEVEL 2 PARAMETERS: REM vs Non-REM\n")
            f.write("="*80 + "\n")
            f.write(json.dumps(self.level2_params, indent=2) + "\n\n")
            
            f.write("="*80 + "\n")
            f.write("LEVEL 3 PARAMETERS: N1 vs N2 vs N3\n")
            f.write("="*80 + "\n")
            f.write(json.dumps(self.level3_params, indent=2) + "\n\n")
            
            f.write("="*80 + "\n")
            f.write("FEATURE SELECTION:\n")
            f.write("="*80 + "\n")
            f.write(f"Level 1 features: {len(self.level1_features)} features\n")
            f.write(f"Level 2 features: {len(self.level2_features)} features\n")
            f.write(f"Level 3 features: {len(self.level3_features)} features\n\n")
        
        print(f"\nMethod information saved to: {filepath}")


def main():
    """
    Main execution function for hierarchical sleep stage classification.
    """
    print("="*80)
    print("HIERARCHICAL SLEEP STAGE CLASSIFICATION")
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
    
    # Train Hierarchical Classifier
    print("\n" + "="*80)
    print("TRAINING HIERARCHICAL CLASSIFIER")
    print("="*80)
    
    classifier = HierarchicalSleepClassifier(results_dir='./results/hierarchical')
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
    
    # Save results - use the classifier's results directory
    results_path = os.path.join(classifier.results_dir, 'hierarchical_results.csv')
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

