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

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set display options
pd.set_option("display.max_rows", 500)  # Replace 500 with your desired number of rows
pd.set_option(
    "display.max_columns", 10
)  # Replace 10 with your desired number of columns
pd.set_option("display.width", 1000)  # Adjust the width as needed
pd.set_option("display.max_colwidth", 50)  # Adjust the column width as needed

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

# Prepare the data
# Adjust your path here
quality_df_dir = './results/quality_scores_per_subject.csv'
features_dir = "dataset_sample/features_df/"
info_dir = "dataset_sample/participant_info.csv"
clean_df, new_features, good_quality_sids = data_preparation(
    threshold = 0.2, 
    quality_df_dir = quality_df_dir,
    features_dir = features_dir,
    info_dir = info_dir)
print(clean_df.shape)
print(len(new_features))

SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)

# Save SW_df to CSV
sw_df_path = f"{output_dir}/SW_df.csv"
SW_df.to_csv(sw_df_path, index=False)
print(f"SW_df saved to: {sw_df_path}")

SW_df

import random
random.seed(0)
train_sids = random.sample(good_quality_sids, 56)
remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
val_sids = random.sample(remaining_sids, 8)
test_sids = [subj for subj in remaining_sids if subj not in val_sids]

group_variables = ['AHI_Severity', 'Obesity']
# when idx == 0, it returns ['AHI_Severity'], the first variable in the list
# when idx == 1, it returns ['Obesity'], the second variable in the list
group_variable = get_variable(group_variables, idx = 0) # set your variable

X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
print("Train data balance: ")
print(np.unique(y_train, return_counts=True))
print('')

X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
print("Validation data balancpredicted_probabilitiese: ")
print(np.unique(y_val, return_counts=True))
print('')

X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)
print("Test data balance: ")
print(np.unique(y_test, return_counts=True))
print(y_train.sum())

### Resample data
X_train_resampled, y_train_resampled, group_train_resampled = resample_data(X_train, y_train, group_train, group_variable)

### LightGBM
final_lgb_model = LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val)

# Save the trained model
model_path = f"{output_dir}/LightGBM_model.pkl"
joblib.dump(final_lgb_model, model_path)
print(f"Trained LightGBM model saved to: {model_path}")

# calculate training scores
prob_ls_train, len_train, true_ls_train = compute_probabilities(
    train_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable)
train_cm_path = f"{output_dir}/LightGBM_train_CM.png"
lgb_train_results_df = LightGBM_result(final_lgb_model, X_train, y_train, prob_ls_train, true_ls_train, train_cm_path)
print(f"Training confusion matrix saved to: {train_cm_path}")

# # calculate testing scores
prob_ls_test, len_test, true_ls_test = compute_probabilities(
    test_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable)
test_cm_path = f"{output_dir}/LightGBM_test_CM.png"
lgb_test_results_df = LightGBM_result(final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test, test_cm_path)
print(f"Testing confusion matrix saved to: {test_cm_path}")
lgb_test_results_df

import shap
import matplotlib.pyplot as plt
explainer = shap.TreeExplainer(final_lgb_model)
shap_values = explainer.shap_values(X_train)

# Save SHAP summary plot (bar chart)
shap_bar_plot_path = f"{output_dir}/LightGBM_shap_tree_explainer_bar.png"
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=final_features, show=False)
plt.savefig(shap_bar_plot_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"SHAP tree explainer bar plot saved to: {shap_bar_plot_path}")

# Save SHAP summary plot (violin/beeswarm plot)
shap_violin_plot_path = f"{output_dir}/LightGBM_shap_tree_explainer_violin.png"
shap.summary_plot(shap_values, X_train, feature_names=final_features, show=False)
plt.savefig(shap_violin_plot_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"SHAP tree explainer violin plot saved to: {shap_violin_plot_path}")

print(f"\n{'='*60}")
print(f"Run complete! All outputs saved to: {run_dir}")
print(f"{'='*60}")