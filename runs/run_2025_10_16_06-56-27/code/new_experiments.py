import pandas as pd
import numpy as np
import random
import warnings
from utils import *
from datasets import *
from models import *
from datetime import datetime
import os
import shutil
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import label_binarize
import sys
import torch

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set display options
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 50)

# Create timestamped run folder
timestamp = datetime.now().strftime("run_%Y_%m_%d_%H-%M-%S")
run_dir = f"runs/{timestamp}"
code_dir = f"{run_dir}/code"
output_dir = f"{run_dir}/output"

# Create directories
os.makedirs(code_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Set up logging to both console and file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(f"{run_dir}/log.txt")
print(f"Logging to: {run_dir}/log.txt")

def calculate_multiclass_metrics(y_true, y_pred, y_proba, class_names):
    """Calculate comprehensive metrics for multiclass classification."""
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 scores
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
    
    # Binarize labels for ROC-AUC and PR-AUC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    
    # ROC-AUC (one-vs-rest)
    try:
        metrics['auroc_macro'] = roc_auc_score(y_true_bin, y_proba, average='macro', multi_class='ovr')
        metrics['auroc_weighted'] = roc_auc_score(y_true_bin, y_proba, average='weighted', multi_class='ovr')
        metrics['auroc_per_class'] = roc_auc_score(y_true_bin, y_proba, average=None, multi_class='ovr')
    except:
        metrics['auroc_macro'] = np.nan
        metrics['auroc_weighted'] = np.nan
        metrics['auroc_per_class'] = [np.nan] * len(class_names)
    
    # PR-AUC (average precision)
    try:
        metrics['auprc_macro'] = average_precision_score(y_true_bin, y_proba, average='macro')
        metrics['auprc_weighted'] = average_precision_score(y_true_bin, y_proba, average='weighted')
        metrics['auprc_per_class'] = average_precision_score(y_true_bin, y_proba, average=None)
    except:
        metrics['auprc_macro'] = np.nan
        metrics['auprc_weighted'] = np.nan
        metrics['auprc_per_class'] = [np.nan] * len(class_names)
    
    return metrics

def print_metrics_report(metrics, class_names, dataset_name=""):
    """Print a formatted metrics report."""
    print(f"\n{'='*80}")
    print(f"{dataset_name} Metrics Report")
    print(f"{'='*80}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  F1 (Macro):      {metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted):   {metrics['f1_weighted']:.4f}")
    print(f"  AUROC (Macro):   {metrics['auroc_macro']:.4f}")
    print(f"  AUROC (Weighted):{metrics['auroc_weighted']:.4f}")
    print(f"  AUPRC (Macro):   {metrics['auprc_macro']:.4f}")
    print(f"  AUPRC (Weighted):{metrics['auprc_weighted']:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<10} {'F1':<10} {'AUROC':<10} {'AUPRC':<10}")
    print(f"{'-'*40}")
    for i, class_name in enumerate(class_names):
        f1 = metrics['f1_per_class'][i]
        auroc = metrics['auroc_per_class'][i]
        auprc = metrics['auprc_per_class'][i]
        print(f"{class_name:<10} {f1:<10.4f} {auroc:<10.4f} {auprc:<10.4f}")
    print(f"{'='*80}")

# Copy code files
code_files = ['utils.py', 'datasets.py', 'models.py', 'new_experiments.py']
for code_file in code_files:
    if os.path.exists(code_file):
        shutil.copy(code_file, os.path.join(code_dir, code_file))

print(f"Run folder created: {run_dir}")
print(f"Code files saved to: {code_dir}\n")

# Prepare the data (preserving original sleep stages, P mapped to W)
quality_df_dir = './results/quality_scores_per_subject.csv'
features_dir = "dataset_sample/features_df/"
info_dir = "dataset_sample/participant_info.csv"

print("Loading data for multiclass classification...")
clean_df, new_features, good_quality_sids = data_preparation_multiclass(
    threshold=0.2,
    quality_df_dir=quality_df_dir,
    features_dir=features_dir,
    info_dir=info_dir
)
print(f"Data loaded: {clean_df.shape}")
print(f"Number of features: {len(new_features)}")
print(f"Sleep stage distribution:\n{clean_df.Sleep_Stage.value_counts()}\n")

# Map sleep stages to numeric labels for multiclass classification
clean_df_numeric, stage_to_num, num_to_stage = map_stages_to_numeric(clean_df)
class_names = ['W', 'R', 'N1', 'N2', 'N3']  # In order of numeric labels
num_classes = len(class_names)

print(f"Class mapping: {stage_to_num}")
print(f"Training {num_classes}-class classifier: {class_names}\n")

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)

# Split data and remove highly correlated features
SW_df, final_features = split_data(clean_df_numeric, good_quality_sids, new_features)

# Define train/val/test splits
train_sids = random.sample(good_quality_sids, 56)
remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
val_sids = random.sample(remaining_sids, 8)
test_sids = [subj for subj in remaining_sids if subj not in val_sids]

group_variables = ['AHI_Severity', 'Obesity']
group_variable = get_variable(group_variables, idx=0)

print(f"{'='*80}")
print(f"Multiclass Sleep Stage Classification")
print(f"{'='*80}\n")

# Create train/val/test splits
X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
print(f"Train data class distribution:")
for class_idx in range(num_classes):
    count = (y_train == class_idx).sum()
    print(f"  {class_names[class_idx]}: {count}")

X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
print(f"\nValidation data class distribution:")
for class_idx in range(num_classes):
    count = (y_val == class_idx).sum()
    print(f"  {class_names[class_idx]}: {count}")

X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)
print(f"\nTest data class distribution:")
for class_idx in range(num_classes):
    count = (y_test == class_idx).sum()
    print(f"  {class_names[class_idx]}: {count}")

# Calculate class weights based on ORIGINAL training data (before SMOTE)
# Inverse of class frequency: weight = n_samples / (n_classes * class_count)
# Then apply aggressive scaling for minority classes
print("\nCalculating AGGRESSIVE class weights for cost-sensitive learning...")
class_counts = np.bincount(y_train.astype(int), minlength=num_classes)
total_samples = len(y_train)
class_weights = {}

# Base weights (inverse frequency)
for class_idx in range(num_classes):
    if class_counts[class_idx] > 0:
        class_weights[class_idx] = total_samples / (num_classes * class_counts[class_idx])
    else:
        class_weights[class_idx] = 1.0

# Apply aggressive scaling to minority classes
# R (class 1): 2x boost
# N1 (class 2): 2x boost  
# N3 (class 4): 5x boost (most aggressive)
scaling_factors = {
    0: 1.0,  # W - no scaling
    1: 2.0,  # R - 2x boost
    2: 2.0,  # N1 - 2x boost
    3: 1.0,  # N2 - no scaling
    4: 5.0,  # N3 - 5x AGGRESSIVE boost
}

for class_idx in range(num_classes):
    class_weights[class_idx] *= scaling_factors[class_idx]

print("Aggressive class weights (higher = more penalty for misclassification):")
for class_idx, class_name in enumerate(class_names):
    base_weight = total_samples / (num_classes * class_counts[class_idx]) if class_counts[class_idx] > 0 else 1.0
    print(f"  {class_name}: {class_weights[class_idx]:.3f} (base: {base_weight:.3f}, boost: {scaling_factors[class_idx]}x, n={class_counts[class_idx]})")

# Resample training data using SMOTE for multiclass
print("\nResampling training data with SMOTE...")
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled train data class distribution:")
for class_idx in range(num_classes):
    count = (y_train_resampled == class_idx).sum()
    print(f"  {class_names[class_idx]}: {count}")

# Train LightGBM multiclass model with aggressive class weights + reduced regularization
print("\nTraining LightGBM multiclass model with AGGRESSIVE class weights...")
print("Using class weights to heavily penalize minority class misclassifications...")
print("Combined with reduced regularization to allow better fitting of minority classes...")
final_lgb_model = LightGBM_engine_multiclass(
    X_train_resampled, 
    y_train_resampled, 
    X_val, 
    y_val, 
    num_classes=num_classes,
    class_weight=class_weights,
    use_focal_loss=False,  # Disabled due to LightGBM API compatibility
)

# Save the trained model
model_path = f"{output_dir}/LightGBM_multiclass_model.pkl"
joblib.dump(final_lgb_model, model_path)
print(f"✓ Model saved: {model_path}")

# Calculate training scores
print("\nEvaluating on training data...")
prob_ls_train, len_train, true_ls_train = compute_probabilities_multiclass(
    train_sids, SW_df, final_features, final_lgb_model, group_variable, num_classes=num_classes
)

# Concatenate probabilities and true labels
y_train_true = np.concatenate(true_ls_train)
y_train_proba = np.concatenate(prob_ls_train)
y_train_pred = np.argmax(y_train_proba, axis=1)

# Calculate metrics
train_metrics = calculate_multiclass_metrics(y_train_true, y_train_pred, y_train_proba, class_names)

# Plot confusion matrix
train_cm_path = f"{output_dir}/multiclass_train_CM.png"
plot_cm_multiclass(prob_ls_train, true_ls_train, "LightGBM Multiclass (Train)", class_names, train_cm_path)
print(f"✓ Training CM saved: {train_cm_path}")

# Print metrics report
print_metrics_report(train_metrics, class_names, "TRAINING SET")

# Calculate testing scores
print("\nEvaluating on test data...")
prob_ls_test, len_test, true_ls_test = compute_probabilities_multiclass(
    test_sids, SW_df, final_features, final_lgb_model, group_variable, num_classes=num_classes
)

# Concatenate probabilities and true labels
y_test_true = np.concatenate(true_ls_test)
y_test_proba = np.concatenate(prob_ls_test)
y_test_pred = np.argmax(y_test_proba, axis=1)

# Calculate metrics
test_metrics = calculate_multiclass_metrics(y_test_true, y_test_pred, y_test_proba, class_names)

# Plot confusion matrix
test_cm_path = f"{output_dir}/multiclass_test_CM.png"
plot_cm_multiclass(prob_ls_test, true_ls_test, "LightGBM Multiclass (Test)", class_names, test_cm_path)
print(f"✓ Testing CM saved: {test_cm_path}")

# Print metrics report
print_metrics_report(test_metrics, class_names, "TEST SET")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Dataset': ['Train', 'Test'],
    'Accuracy': [train_metrics['accuracy'], test_metrics['accuracy']],
    'F1_Macro': [train_metrics['f1_macro'], test_metrics['f1_macro']],
    'F1_Weighted': [train_metrics['f1_weighted'], test_metrics['f1_weighted']],
    'AUROC_Macro': [train_metrics['auroc_macro'], test_metrics['auroc_macro']],
    'AUROC_Weighted': [train_metrics['auroc_weighted'], test_metrics['auroc_weighted']],
    'AUPRC_Macro': [train_metrics['auprc_macro'], test_metrics['auprc_macro']],
    'AUPRC_Weighted': [train_metrics['auprc_weighted'], test_metrics['auprc_weighted']],
})
metrics_csv_path = f"{output_dir}/metrics_summary.csv"
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"✓ Metrics summary saved: {metrics_csv_path}")

# ==============================================================================
# LSTM POST-PROCESSING
# ==============================================================================
print(f"\n{'='*80}")
print(f"MULTICLASS LSTM POST-PROCESSING")
print(f"{'='*80}")
print(f"Adding temporal modeling to improve sequence predictions...")

# Define specific features to add to LSTM input (in addition to 5 class probabilities)
lstm_feature_names = [
    'rolling_var_ACC_Z_MAD_trimmed_max',
    'max_SCR_Rise_Time',
    'rolling_var_ACC_X_trimmed_IQR',
    'rolling_var_HRV_Prc80NN',
    'gaussian_ACC_X_trimmed_max'
]

# Verify features exist in the dataset
print(f"\nVerifying LSTM features...")
missing_features = [f for f in lstm_feature_names if f not in final_features]
if missing_features:
    print(f"WARNING: Missing features: {missing_features}")
    print(f"Using available features from the final_features list instead...")
    # Use first 5 available features as fallback
    lstm_feature_names = final_features[:5]
    
print(f"LSTM additional features: {lstm_feature_names}")

# Extract LSTM features for training
print(f"\nExtracting LSTM features for training...")
features_train = extract_lstm_features(train_sids, SW_df, final_features, lstm_feature_names)

# Create LSTM dataloader for training (5 probabilities + 5 features = 10 inputs)
print(f"Creating LSTM dataloader...")
lstm_dataloader_train = LSTM_dataloader_multiclass(
    prob_ls_train, features_train, len_train, true_ls_train, batch_size=32
)

# Train LSTM
print(f"\nTraining multiclass LSTM (5 classes)...")
print(f"Input: 5 LightGBM probabilities + 5 additional features = 10 features per timestep")
print(f"Using same aggressive class weights as LightGBM")
lstm_model = LSTM_engine_multiclass(
    lstm_dataloader_train, 
    num_epoch=300,  # Same as original paper, but now with class weights
    hidden_layer_size=64,  # Increased capacity for multiclass
    learning_rate=0.001,
    num_classes=num_classes,
    class_weight=class_weight,  # Critical: forces LSTM to learn minority classes (R, N1, N3)
    use_focal_loss=False
)

# Save LSTM model
lstm_model_path = f"{output_dir}/LSTM_multiclass_model.pth"
torch.save(lstm_model.state_dict(), lstm_model_path)
print(f"✓ LSTM model saved: {lstm_model_path}")

# Evaluate LSTM on test set
print(f"\nEvaluating LSTM on test data...")
features_test = extract_lstm_features(test_sids, SW_df, final_features, lstm_feature_names)

lstm_dataloader_test = LSTM_dataloader_multiclass(
    prob_ls_test, features_test, len_test, true_ls_test, batch_size=1
)

# Get LSTM predictions
prob_ls_test_lstm = LSTM_eval_multiclass(
    lstm_model, lstm_dataloader_test, true_ls_test, class_names, test_name="LSTM Test"
)

# Calculate LSTM metrics
y_test_lstm_proba = np.concatenate(prob_ls_test_lstm)
y_test_lstm_pred = np.argmax(y_test_lstm_proba, axis=1)

lstm_test_metrics = calculate_multiclass_metrics(y_test_true, y_test_lstm_pred, y_test_lstm_proba, class_names)

# Plot LSTM confusion matrix
lstm_test_cm_path = f"{output_dir}/multiclass_LSTM_test_CM.png"
plot_cm_multiclass(prob_ls_test_lstm, true_ls_test, "LSTM Multiclass (Test)", class_names, lstm_test_cm_path)
print(f"✓ LSTM Testing CM saved: {lstm_test_cm_path}")

# Print LSTM metrics report
print_metrics_report(lstm_test_metrics, class_names, "LSTM POST-PROCESSED TEST SET")

# Compare LightGBM vs LSTM
print(f"\n{'='*80}")
print(f"COMPARISON: LightGBM vs LightGBM+LSTM")
print(f"{'='*80}")
comparison_df = pd.DataFrame({
    'Model': ['LightGBM', 'LightGBM+LSTM', 'Improvement'],
    'Accuracy': [
        test_metrics['accuracy'], 
        lstm_test_metrics['accuracy'],
        lstm_test_metrics['accuracy'] - test_metrics['accuracy']
    ],
    'F1_Macro': [
        test_metrics['f1_macro'], 
        lstm_test_metrics['f1_macro'],
        lstm_test_metrics['f1_macro'] - test_metrics['f1_macro']
    ],
    'F1_Weighted': [
        test_metrics['f1_weighted'], 
        lstm_test_metrics['f1_weighted'],
        lstm_test_metrics['f1_weighted'] - test_metrics['f1_weighted']
    ],
    'AUROC_Macro': [
        test_metrics['auroc_macro'], 
        lstm_test_metrics['auroc_macro'],
        lstm_test_metrics['auroc_macro'] - test_metrics['auroc_macro']
    ],
})
print(comparison_df.to_string(index=False))

# Save comparison
comparison_csv_path = f"{output_dir}/lgb_vs_lstm_comparison.csv"
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"\n✓ Comparison saved: {comparison_csv_path}")

# Per-class comparison
print(f"\nPer-Class F1 Score Comparison:")
print(f"{'Class':<10} {'LightGBM':<12} {'LightGBM+LSTM':<15} {'Improvement':<12}")
print(f"{'-'*50}")
for i, class_name in enumerate(class_names):
    lgb_f1 = test_metrics['f1_per_class'][i]
    lstm_f1 = lstm_test_metrics['f1_per_class'][i]
    improvement = lstm_f1 - lgb_f1
    print(f"{class_name:<10} {lgb_f1:<12.4f} {lstm_f1:<15.4f} {improvement:+.4f}")

print(f"{'='*80}\n")

# SHAP analysis
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(final_lgb_model)
shap_values = explainer.shap_values(X_train_resampled)

# For multiclass, shap_values is a list of arrays, one for each class
# Save SHAP summary plot for each class (bar chart only, no beeswarm for multiclass)
for class_idx, class_name in enumerate(class_names):
    shap_bar_plot_path = f"{output_dir}/multiclass_shap_bar_{class_name}.png"
    shap.summary_plot(
        shap_values[class_idx], 
        X_train_resampled, 
        plot_type="bar", 
        feature_names=final_features, 
        show=False
    )
    plt.title(f"SHAP Feature Importance for {class_name}")
    plt.savefig(shap_bar_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ SHAP bar plot for {class_name} saved: {shap_bar_plot_path}")

print(f"\n{'='*80}")
print(f"MULTICLASS CLASSIFICATION COMPLETE!")
print(f"{'='*80}")
print(f"All outputs saved to: {run_dir}")
print(f"  - Code snapshots: {code_dir}/")
print(f"  - Results: {output_dir}/")
print(f"  - Complete log: {run_dir}/log.txt")
print(f"\nGenerated files:")
print(f"  - LightGBM_multiclass_model.pkl")
print(f"  - multiclass_train_CM.png")
print(f"  - multiclass_test_CM.png")
print(f"  - metrics_summary.csv")
print(f"  - multiclass_shap_bar_<class>.png (one per class: W, R, N1, N2, N3)")
print(f"  - log.txt (complete output with all metrics)")
print(f"\nModel Settings:")
print(f"  - Loss function: Cross-Entropy with aggressive class weights")
print(f"  - Class weights: W=0.47, R=4.94, N1=5.34, N2=0.52, N3=30.88")
print(f"  - Regularization: REDUCED (reg_alpha 0-20, reg_lambda 0.01-1.0)")
print(f"  - Data: SMOTE resampling for balanced training")
print(f"  - Focus: Minority classes (R, N1, N3) via heavy misclassification penalties")
print(f"{'='*80}")
