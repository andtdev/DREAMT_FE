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

# Resample training data using SMOTE for multiclass
print("\nResampling training data with SMOTE...")
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Resampled train data class distribution:")
for class_idx in range(num_classes):
    count = (y_train_resampled == class_idx).sum()
    print(f"  {class_names[class_idx]}: {count}")

# Train LightGBM multiclass model
print("\nTraining LightGBM multiclass model...")
final_lgb_model = LightGBM_engine_multiclass(
    X_train_resampled, 
    y_train_resampled, 
    X_val, 
    y_val, 
    num_classes=num_classes
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
train_cm_path = f"{output_dir}/multiclass_train_CM.png"
plot_cm_multiclass(prob_ls_train, true_ls_train, "LightGBM Multiclass", class_names, train_cm_path)
print(f"✓ Training CM saved: {train_cm_path}")

# Calculate testing scores
print("\nEvaluating on test data...")
prob_ls_test, len_test, true_ls_test = compute_probabilities_multiclass(
    test_sids, SW_df, final_features, final_lgb_model, group_variable, num_classes=num_classes
)
test_cm_path = f"{output_dir}/multiclass_test_CM.png"
plot_cm_multiclass(prob_ls_test, true_ls_test, "LightGBM Multiclass", class_names, test_cm_path)
print(f"✓ Testing CM saved: {test_cm_path}")

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
print(f"\nGenerated files:")
print(f"  - LightGBM_multiclass_model.pkl")
print(f"  - multiclass_train_CM.png")
print(f"  - multiclass_test_CM.png")
print(f"  - multiclass_shap_bar_<class>.png (one per class: W, R, N1, N2, N3)")
print(f"{'='*80}")
