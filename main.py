# load packages
import pandas as pd
import numpy as np
import random
import shap


from utils import *
from models import *
from datasets import *

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Prepare the data
# Adjust your path here
print("=== STEP 1: Starting data preparation ===")
quality_df_dir = './results/quality_scores_per_subject.csv'
features_dir = "dataset_sample/features_df/"
info_dir = "dataset_sample/participant_info.csv"
clean_df, new_features, good_quality_sids = data_preparation(
    threshold = 0.2, 
    quality_df_dir = quality_df_dir,
    features_dir = features_dir,
    info_dir = info_dir)
print("=== STEP 1 COMPLETED: Data preparation successful ===")
print(f"Found {len(good_quality_sids)} good quality subjects")

# Split data to train, validation, and test set
print("=== STEP 2: Starting data splitting ===")
SW_df, final_features = split_data(clean_df, good_quality_sids, new_features)
print("=== STEP 2 COMPLETED: Data splitting successful ===")
print(f"Final features count: {len(final_features)}")

print("=== STEP 3: Creating train/val/test splits ===")
random.seed(0)
train_sids = random.sample(good_quality_sids, 56)
remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
val_sids = random.sample(remaining_sids, 8)
test_sids = [subj for subj in remaining_sids if subj not in val_sids]
print("=== STEP 3 COMPLETED: Train/val/test splits created ===")
print(f"Train: {len(train_sids)}, Val: {len(val_sids)}, Test: {len(test_sids)}")

print("=== STEP 4: Setting up group variables ===")
group_variables = ["AHI_Severity", "Obesity"]
# when idx == 0, it returns ['AHI_Severity'], the first variable in the list
# when idx == 1, it returns ['Obesity'], the second variable in the list
group_variable = get_variable(group_variables, idx=0)
print(f"Selected group variable: {group_variable}")

print("=== STEP 5: Creating feature matrices ===")
X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable)
print("Train set created")
X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable)
print("Validation set created")
X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable)
print("Test set created")
print("=== STEP 5 COMPLETED: Feature matrices created ===")

# Resample all the data
print("=== STEP 6: Resampling training data ===")
X_train_resampled, y_train_resampled, group_train_resampled = resample_data(X_train, y_train, group_train, group_variable)
print("=== STEP 6 COMPLETED: Data resampling successful ===")
print(f"Original train size: {len(X_train)}, Resampled size: {len(X_train_resampled)}")

# Run LightGBM model
print("=== STEP 7: Training LightGBM model ===")
final_lgb_model = LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val)
print("=== STEP 7 COMPLETED: LightGBM training successful ===")

# calculate training scores
print("=== STEP 8: Computing training probabilities ===")
prob_ls_train, len_train, true_ls_train = compute_probabilities(
    train_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable)
print("Training probabilities computed")
lgb_train_results_df = LightGBM_result(final_lgb_model, X_train, y_train, prob_ls_train, true_ls_train)
print("=== STEP 8 COMPLETED: Training results calculated ===")

# calculate testing scores
print("=== STEP 9: Computing testing probabilities ===")
prob_ls_test, len_test, true_ls_test = compute_probabilities(
    test_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable)
print("Testing probabilities computed")
lgb_test_results_df = LightGBM_result(final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test)
print("=== STEP 9 COMPLETED: Testing results calculated ===")

# Identify best features
print("=== STEP 10: Starting SHAP analysis ===")
print("Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(final_lgb_model)
print("TreeExplainer created successfully")
print("Computing SHAP values...")
shap_values = explainer.shap_values(X_train)
print("SHAP values computed successfully")
print("Creating SHAP summary plot...")
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=final_features)
print("=== STEP 10 COMPLETED: SHAP analysis successful ===")

# Add LSTM for post processing
print("=== STEP 11: Starting LSTM processing ===")
# create train data
print("Creating LSTM training dataloader...")
dataloader_train = LSTM_dataloader(
    prob_ls_train, len_train, true_ls_train, batch_size=32
)
print("Training dataloader created")

# Run LSTM model
print("Training LSTM model...")
LSTM_model = LSTM_engine(dataloader_train, num_epoch=300, hidden_layer_size=32, learning_rate=0.001) # set your num_epoch
print("LSTM training completed")

# test LSMT model
print("Creating LSTM test dataloader...")
dataloader_test = LSTM_dataloader(
    prob_ls_test, len_test, true_ls_test, batch_size=1
)
print("Evaluating LSTM model...")
lgb_lstm_test_results_df = LSTM_eval(LSTM_model, dataloader_test, true_ls_test, 'LightGBM_LSTM')
print("=== STEP 11 COMPLETED: LSTM processing successful ===")


# Run GPBoost model
print("=== STEP 12: Training GPBoost model ===")
final_gpb_model = GPBoost_engine(X_train_resampled, group_train_resampled, y_train_resampled, X_val, y_val, group_val)
print("=== STEP 12 COMPLETED: GPBoost training successful ===")

# calculate training scores
print("=== STEP 13: Computing GPBoost training probabilities ===")
prob_ls_train, len_train, true_ls_train = compute_probabilities(
    train_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable)
print("GPBoost training probabilities computed")
gpb_train_results_df = GPBoost_result(final_gpb_model, X_train, y_train, group_train, prob_ls_train, true_ls_train)
print("=== STEP 13 COMPLETED: GPBoost training results calculated ===")

# calculate testing scores
print("=== STEP 14: Computing GPBoost testing probabilities ===")
prob_ls_test, len_test, true_ls_test = compute_probabilities(
    test_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable)
print("GPBoost testing probabilities computed")
gpb_test_results_df = GPBoost_result(final_gpb_model, X_test, y_test, group_test, prob_ls_test, true_ls_test)
print("=== STEP 14 COMPLETED: GPBoost testing results calculated ===")


# Get LSTM dataset
print("=== STEP 15: Creating LSTM datasets for GPBoost ===")
dataloader_train = LSTM_dataloader(
    prob_ls_train, len_train, true_ls_train, batch_size=32
)
print("GPBoost LSTM training dataloader created")
dataloader_test = LSTM_dataloader(
    prob_ls_test, len_test, true_ls_test, batch_size=1
)
print("GPBoost LSTM test dataloader created")

# Run LSTM model
print("Training GPBoost LSTM model...")
LSTM_model = LSTM_engine(dataloader_train, num_epoch=300, hidden_layer_size=32, learning_rate = 0.001) # set your num_epoch
print("Evaluating GPBoost LSTM model...")
gpb_lstm_test_results_df = LSTM_eval(LSTM_model, dataloader_test, true_ls_test, 'GPBoost_LSTM')
print("=== STEP 15 COMPLETED: GPBoost LSTM processing successful ===")

# overall result
print("=== STEP 16: Generating final results ===")
overall_result = pd.concat([lgb_test_results_df, lgb_lstm_test_results_df, 
                            gpb_test_results_df, gpb_lstm_test_results_df])
print("=== FINAL RESULTS ===")
print(group_variable)
print(overall_result)
print("=== SCRIPT COMPLETED SUCCESSFULLY ===")
