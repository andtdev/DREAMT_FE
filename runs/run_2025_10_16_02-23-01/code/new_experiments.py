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

# Copy code files (only once)
code_files = ['utils.py', 'datasets.py', 'models.py', 'new_experiments.py']
for code_file in code_files:
    if os.path.exists(code_file):
        shutil.copy(code_file, os.path.join(code_dir, code_file))

print(f"Run folder created: {run_dir}")
print(f"Code files saved to: {code_dir}\n")

# Prepare the data (preserving original sleep stages)
quality_df_dir = './results/quality_scores_per_subject.csv'
features_dir = "dataset_sample/features_df/"
info_dir = "dataset_sample/participant_info.csv"

print("Loading data with original sleep stages preserved...")
clean_df, new_features, good_quality_sids = data_preparation_multiclass(
    threshold=0.2,
    quality_df_dir=quality_df_dir,
    features_dir=features_dir,
    info_dir=info_dir
)
print(f"Data loaded: {clean_df.shape}")
print(f"Number of features: {len(new_features)}")
print(f"Sleep stage distribution:\n{clean_df.Sleep_Stage.value_counts()}\n")

# Define the 5 scenarios: each sleep stage vs all others
sleep_stages = ['W', 'R', 'N1', 'N2', 'N3']
stage_names = {
    'W': 'Wake',
    'R': 'REM',
    'N1': 'N1',
    'N2': 'N2',
    'N3': 'N3'
}

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)

# Define train/val/test splits (same for all scenarios)
train_sids = random.sample(good_quality_sids, 56)
remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
val_sids = random.sample(remaining_sids, 8)
test_sids = [subj for subj in remaining_sids if subj not in val_sids]

group_variables = ['AHI_Severity', 'Obesity']
group_variable = get_variable(group_variables, idx=0)

print(f"{'='*80}")
print(f"Starting analysis for 5 sleep stages (one-vs-rest)")
print(f"{'='*80}\n")

# Loop through each sleep stage scenario
for stage_idx, target_stage in enumerate(sleep_stages, 1):
    stage_name = stage_names[target_stage]
    print(f"\n{'='*80}")
    print(f"SCENARIO {stage_idx}/5: {stage_name} ({target_stage}) vs Others")
    print(f"{'='*80}\n")
    
    # Relabel data for this scenario (target stage = 1, others = 0)
    scenario_df = relabel_for_stage_vs_rest(clean_df, target_stage)
    
    # Check class balance
    print(f"Class distribution for {stage_name}:")
    print(f"  {stage_name}: {scenario_df.Sleep_Stage.sum()}")
    print(f"  Others: {(scenario_df.Sleep_Stage == 0).sum()}\n")
    
    # Split data and remove highly correlated features
    SW_df, final_features = split_data(scenario_df, good_quality_sids, new_features)
    
    # Create train/val/test splits
    X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
    print(f"Train data balance: {np.unique(y_train, return_counts=True)[1]}")
    
    X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
    print(f"Validation data balance: {np.unique(y_val, return_counts=True)[1]}")
    
    X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)
    print(f"Test data balance: {np.unique(y_test, return_counts=True)[1]}\n")
    
    # Resample training data
    print("Resampling training data...")
    X_train_resampled, y_train_resampled, group_train_resampled = resample_data(
        X_train, y_train, group_train, group_variable
    )
    print(f"Resampled train data balance: {np.unique(y_train_resampled, return_counts=True)[1]}\n")
    
    # Train LightGBM model
    print("Training LightGBM model...")
    final_lgb_model = LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val)
    
    # Save the trained model
    model_path = f"{output_dir}/{stage_name}_LightGBM_model.pkl"
    joblib.dump(final_lgb_model, model_path)
    print(f"✓ Model saved: {model_path}")
    
    # Calculate training scores
    print("Evaluating on training data...")
    prob_ls_train, len_train, true_ls_train = compute_probabilities(
        train_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable
    )
    train_cm_path = f"{output_dir}/{stage_name}_train_CM.png"
    lgb_train_results_df = LightGBM_result(
        final_lgb_model, X_train, y_train, prob_ls_train, true_ls_train, train_cm_path
    )
    print(f"✓ Training CM saved: {train_cm_path}")
    
    # Calculate testing scores
    print("Evaluating on test data...")
    prob_ls_test, len_test, true_ls_test = compute_probabilities(
        test_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable
    )
    test_cm_path = f"{output_dir}/{stage_name}_test_CM.png"
    lgb_test_results_df = LightGBM_result(
        final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test, test_cm_path
    )
    print(f"✓ Testing CM saved: {test_cm_path}")
    
    # SHAP analysis
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(final_lgb_model)
    shap_values = explainer.shap_values(X_train)
    
    # Save SHAP summary plot (bar chart)
    shap_bar_plot_path = f"{output_dir}/{stage_name}_shap_bar.png"
    shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=final_features, show=False)
    plt.savefig(shap_bar_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ SHAP bar plot saved: {shap_bar_plot_path}")
    
    # Save SHAP summary plot (beeswarm plot)
    # For binary classification, use class 1 (the positive class - target stage)
    shap_beeswarm_plot_path = f"{output_dir}/{stage_name}_shap_beeswarm.png"
    shap.summary_plot(shap_values[1], X_train, plot_type="dot", feature_names=final_features, show=False)
    plt.savefig(shap_beeswarm_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ SHAP beeswarm plot saved: {shap_beeswarm_plot_path}")
    
    print(f"\n✓ Scenario {stage_idx}/5 complete: {stage_name}")

print(f"\n{'='*80}")
print(f"ALL SCENARIOS COMPLETE!")
print(f"{'='*80}")
print(f"All outputs saved to: {run_dir}")
print(f"  - Code snapshots: {code_dir}/")
print(f"  - Results: {output_dir}/")
print(f"\nGenerated files per stage:")
print(f"  - <Stage>_LightGBM_model.pkl")
print(f"  - <Stage>_train_CM.png")
print(f"  - <Stage>_test_CM.png")
print(f"  - <Stage>_shap_bar.png")
print(f"  - <Stage>_shap_beeswarm.png")
print(f"{'='*80}")
