# load packages
import pandas as pd
import numpy as np
import random
import shap
import matplotlib.pyplot as plt


from utils import *
from models import *
from datasets import *

import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration: Set to True to save all plots and results
SAVE_OUTPUTS = True

# Prepare the data
# Adjust your path here
quality_df_dir = './results/quality_scores_per_subject.csv'
features_dir = "dataset_sample/features_df/"
info_dir = "dataset_sample/participant_info.csv"
clean_df, new_features, good_quality_sids = data_preparation(
    threshold = 0.2, 
    quality_df_dir = quality_df_dir,
    features_dir = features_dir,
    info_dir = info_dir,
    verbose = SAVE_OUTPUTS)

# Split data to train, validation, and test set
SW_df, final_features = split_data(clean_df, good_quality_sids, new_features, verbose=SAVE_OUTPUTS)

random.seed(0)
train_sids = random.sample(good_quality_sids, 56)
remaining_sids = [subj for subj in good_quality_sids if subj not in train_sids]
val_sids = random.sample(remaining_sids, 8)
test_sids = [subj for subj in remaining_sids if subj not in val_sids]

group_variables = ["AHI_Severity", "Obesity"]
# when idx == 0, it returns ['AHI_Severity'], the first variable in the list
# when idx == 1, it returns ['Obesity'], the second variable in the list
group_variable = get_variable(group_variables, idx=0)

X_train, y_train, group_train = train_test_split(SW_df, train_sids, final_features, group_variable, verbose=SAVE_OUTPUTS)
X_val, y_val, group_val = train_test_split(SW_df, val_sids, final_features, group_variable, verbose=SAVE_OUTPUTS)
X_test, y_test, group_test = train_test_split(SW_df, test_sids, final_features, group_variable, verbose=SAVE_OUTPUTS)

# Resample all the data
X_train_resampled, y_train_resampled, group_train_resampled = resample_data(X_train, y_train, group_train, group_variable, verbose=SAVE_OUTPUTS)

# Run LightGBM model
final_lgb_model = LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val, verbose=SAVE_OUTPUTS)
# calculate training scores
prob_ls_train, len_train, true_ls_train = compute_probabilities(
    train_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable, verbose=SAVE_OUTPUTS)
lgb_train_results_df = LightGBM_result(final_lgb_model, X_train, y_train, prob_ls_train, true_ls_train, save_plots=SAVE_OUTPUTS)

# calculate testing scores
prob_ls_test, len_test, true_ls_test = compute_probabilities(
    test_sids, SW_df, final_features, "lgb", final_lgb_model, group_variable, verbose=SAVE_OUTPUTS)
lgb_test_results_df = LightGBM_result(final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test, save_plots=SAVE_OUTPUTS)

# Identify best features
explainer = shap.TreeExplainer(final_lgb_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=final_features, show=not SAVE_OUTPUTS)

if SAVE_OUTPUTS:
    import os
    os.makedirs('./new_results', exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig('./new_results/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    print("SHAP feature importance plot saved to: ./new_results/shap_feature_importance.png")
    plt.close()
else:
    plt.show()

# Add LSTM for post processing
# create train data
dataloader_train = LSTM_dataloader(
    prob_ls_train, len_train, true_ls_train, batch_size=32
)

# Run LSTM model
LSTM_model = LSTM_engine(dataloader_train, num_epoch=300, hidden_layer_size=32, learning_rate=0.001, verbose=SAVE_OUTPUTS) # set your num_epoch

# test LSMT model
dataloader_test = LSTM_dataloader(
    prob_ls_test, len_test, true_ls_test, batch_size=1
)
lgb_lstm_test_results_df = LSTM_eval(LSTM_model, dataloader_test, true_ls_test, 'LightGBM_LSTM', save_plots=SAVE_OUTPUTS)


# Run GPBoost model
final_gpb_model = GPBoost_engine(X_train_resampled, group_train_resampled, y_train_resampled, X_val, y_val, group_val, verbose=SAVE_OUTPUTS)
# calculate training scores
prob_ls_train, len_train, true_ls_train = compute_probabilities(
    train_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable, verbose=SAVE_OUTPUTS)
gpb_train_results_df = GPBoost_result(final_gpb_model, X_train, y_train, group_train, prob_ls_train, true_ls_train, save_plots=SAVE_OUTPUTS)

# calculate testing scores
prob_ls_test, len_test, true_ls_test = compute_probabilities(
    test_sids, SW_df, final_features, 'gpb', final_gpb_model, group_variable, verbose=SAVE_OUTPUTS)
gpb_test_results_df = GPBoost_result(final_gpb_model, X_test, y_test, group_test, prob_ls_test, true_ls_test, save_plots=SAVE_OUTPUTS)


# Get LSTM dataset
dataloader_train = LSTM_dataloader(
    prob_ls_train, len_train, true_ls_train, batch_size=32
)
dataloader_test = LSTM_dataloader(
    prob_ls_test, len_test, true_ls_test, batch_size=1
)

# Run LSTM model
LSTM_model = LSTM_engine(dataloader_train, num_epoch=300, hidden_layer_size=32, learning_rate = 0.001, verbose=SAVE_OUTPUTS) # set your num_epoch
gpb_lstm_test_results_df = LSTM_eval(LSTM_model, dataloader_test, true_ls_test, 'GPBoost_LSTM', save_plots=SAVE_OUTPUTS)

# overall result
overall_result = pd.concat([lgb_test_results_df, lgb_lstm_test_results_df, 
                            gpb_test_results_df, gpb_lstm_test_results_df])

# Save results if specified
if SAVE_OUTPUTS:
    # Save comprehensive results
    overall_result.to_csv('./new_results/main_experiment_results.csv', index=False)
    print("Overall results saved to: ./new_results/main_experiment_results.csv")
    
    # Save individual model results for detailed analysis
    lgb_test_results_df.to_csv('./new_results/lightgbm_only.csv', index=False)
    lgb_lstm_test_results_df.to_csv('./new_results/lightgbm_lstm.csv', index=False)
    gpb_test_results_df.to_csv('./new_results/gpboost_only.csv', index=False)
    gpb_lstm_test_results_df.to_csv('./new_results/gpboost_lstm.csv', index=False)
    
    print("Individual model results saved to ./new_results/")
    print("=== ALL RESULTS AND PLOTS SAVED TO new_results/ ===")

print("=== FINAL RESULTS ===")
print("Group variable:", group_variable)
print(overall_result)
print("=== SCRIPT COMPLETED SUCCESSFULLY ===")
