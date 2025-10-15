"""
This module provides a set of functions for modeling training and evaluation.

Main Functions:
- transform_data: Converts input data into a specified format.
- validate_data: Checks data against a set of validation rules.
- format_output: Formats data for output based on a specified template.

Usage:
To use these functions, import this script and call the desired function with the appropriate parameters. 

For example:

from model import *

Author: 
License: 
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
import lightgbm as lgb
import gpboost as gpb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
from utils import *
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(1)


class BiLSTMPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=5, dropout=0.5, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,  # Multiple layers enable dropout to work
            bidirectional=True,
        )
        self.linear = nn.Linear(
            2 * hidden_layer_size, output_size
        )  # output_size units for multiclass (default 5 for sleep stages)

    def forward(self, input_seq, lengths):
        packed_input = pack_padded_sequence(
            input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.linear(output)  # Shape: [batch_size, seq_len, output_size]
        return output


class LSTMPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=5, dropout=0.5, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_layer_size, batch_first=True, dropout=dropout,
            num_layers=num_layers  # Multiple layers enable dropout to work
        )
        self.linear = nn.Linear(
            hidden_layer_size, output_size
        )  # output_size units for multiclass (default 5 for sleep stages)

    def forward(self, input_seq, lengths):
        packed_input = pack_padded_sequence(
            input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.linear(output)  # Shape: [batch_size, seq_len, output_size]
        return output


def LightGBM_engine(X_train_resampled, y_train_resampled, X_val, y_val):
    """Train a LightGBM model using hyperparameter optimization.
    
    Parameters
    ----------
    X_train_resampled : array-like
        Training data.
    y_train_resampled : array-like
        Training labels.
    X_val : array-like
        Validation data for early stopping.
    y_val : array-like
        Validation labels for early stopping.

    Returns
    -------
    final_lgb_model : LightGBM model
    """
    space = {
        "max_depth": hp.quniform("max_depth", 2, 4, 1),  # Reduced max depth to prevent overfitting
        "reg_alpha": hp.quniform("reg_alpha", 50, 300, 10),  # Increased L1 regularization
        "reg_lambda": hp.uniform("reg_lambda", 5, 15),  # Increased L2 regularization
        "num_leaves": hp.quniform("num_leaves", 10, 40, 5),  # Reduced to prevent overfitting
        "n_estimators": hp.quniform("n_estimators", 50, 200, 10),  # Slightly reduced
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),  # More conservative learning rate
        "min_child_samples": hp.quniform("min_child_samples", 50, 200, 10),  # Add minimum samples per leaf
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),  # Feature sampling to reduce overfitting
        "subsample": hp.uniform("subsample", 0.6, 0.9),  # Row sampling to reduce overfitting
        "min_data_in_leaf": hp.quniform("min_data_in_leaf", 20, 100, 10),  # Additional regularization
        "boosting_type": hp.choice("boosting_type", ["gbdt"]),  # DART has issues with early stopping
    }

    def objective(space):
        # Custom class weights that mimic focal loss behavior
        # Balanced weighting for both R and N3 to prevent R degradation
        # Distribution: W=10698, R=4666, N1=4320, N2=22209, N3=1868
        # Using ^1.2 for most, ^1.6 for R and N3 (balanced focus)
        total_samples = 43761
        class_weights_dict = {
            0: (total_samples / 10698) ** 1.2,  # W: ~4.9
            1: (total_samples / 4666) ** 1.6,   # R: ~29.4 (boosted to protect R)
            2: (total_samples / 4320) ** 1.2,   # N1: ~13.2
            3: (total_samples / 22209) ** 1.2,  # N2: ~2.3
            4: (total_samples / 1868) ** 1.6,   # N3: ~47.1 (balanced with R)
        }
        
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=5,
            max_depth=int(space["max_depth"]),
            reg_alpha=space["reg_alpha"],
            reg_lambda=space["reg_lambda"],
            n_estimators=int(space["n_estimators"]),
            learning_rate=space["learning_rate"],
            num_leaves=int(space["num_leaves"]),
            min_child_samples=int(space["min_child_samples"]),
            colsample_bytree=space["colsample_bytree"],
            subsample=space["subsample"],
            min_data_in_leaf=int(space["min_data_in_leaf"]),
            boosting_type=space["boosting_type"],  # hp.choice returns the actual value, not index
            class_weight=class_weights_dict,  # Use custom focal-loss-inspired weights
            verbose=-1,
        )

        clf.fit(
            X_train_resampled, 
            y_train_resampled,
            eval_set=[(X_val, y_val)],
            eval_metric='multi_logloss',
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )

        predicted_probabilities = clf.predict_proba(X_val)
        predicted_labels = np.argmax(predicted_probabilities, axis=1)

        # Use weighted F1 for better overall performance (balances classes by support)
        f1 = f1_score(y_val, predicted_labels, average='weighted')
        return {"loss": -f1, "status": STATUS_OK}

    # Run the hyperparameter search with more trials for better optimization
    trials = Trials()
    lgb_best_hyperparams = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials
    )
    print("Best hyperparameters:", lgb_best_hyperparams)

    # Adjust the data types of the best hyperparameters
    lgb_best_hyperparams["max_depth"] = int(lgb_best_hyperparams["max_depth"])
    lgb_best_hyperparams["n_estimators"] = int(lgb_best_hyperparams["n_estimators"])
    lgb_best_hyperparams["num_leaves"] = int(lgb_best_hyperparams["num_leaves"])
    lgb_best_hyperparams["min_child_samples"] = int(lgb_best_hyperparams["min_child_samples"])
    lgb_best_hyperparams["min_data_in_leaf"] = int(lgb_best_hyperparams["min_data_in_leaf"])
    lgb_best_hyperparams["boosting_type"] = ["gbdt", "dart"][lgb_best_hyperparams["boosting_type"]]

    # Use the same balanced class weights in final model (matching objective function)
    total_samples = 43761
    class_weights_dict = {
        0: (total_samples / 10698) ** 1.2,  # W: ~4.9
        1: (total_samples / 4666) ** 1.6,   # R: ~29.4 (boosted to protect R)
        2: (total_samples / 4320) ** 1.2,   # N1: ~13.2
        3: (total_samples / 22209) ** 1.2,  # N2: ~2.3
        4: (total_samples / 1868) ** 1.6,   # N3: ~47.1 (balanced with R)
    }
    
    final_lgb_model = lgb.LGBMClassifier(
        **lgb_best_hyperparams, 
        objective="multiclass",
        num_class=5,
        class_weight=class_weights_dict,  # Focal-loss-inspired weights
        random_state=1, 
        num_iterations=50
    )

    final_lgb_model.fit(
        X_train_resampled, 
        y_train_resampled,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Calculate and print N3 recall on validation set after training
    val_preds = final_lgb_model.predict(X_val)
    n3_mask = (y_val == 4)
    if n3_mask.sum() > 0:
        n3_recall = (val_preds[n3_mask] == 4).mean()
        print(f"Validation N3 Recall: {n3_recall:.4f} ({n3_mask.sum()} N3 samples)")
    else:
        print("No N3 samples in validation set")

    return final_lgb_model


def LightGBM_predict(final_lgb_model, X_test, y_test):
    """Predict using a trained LightGBM model and calculate evaluation metrics.
    
    Parameters
    ----------
    final_lgb_model : lgb.LGBMClassifier
        LightGBM model
    X_test : array-like
        Data to predict on.
    y_test : array-like
        True labels.

    Returns
    -------
    results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    pred_probabilities = final_lgb_model.predict_proba(X_test)
    results_df = calculate_metrics(y_test, pred_probabilities, "LightGBM")
    return results_df


def LightGBM_result(final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test):
    """Calculate evaluation metrics and plot confusion matrix for a trained LightGBM model.
    
    Parameters
    ----------
    final_lgb_model : lgb.LGBMClassifier
        LightGBM model
    X : array-like
        Data to predict on.
    y : array-like
        True labels.
    prob_ls : array-like
        Predicted probabilities from LightGBM model without post-processing.
    true_ls : array-like
        True labels.

    Returns
    -------
    results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    kappa = calculate_kappa(prob_ls_test, true_ls_test)
    results_df = LightGBM_predict(final_lgb_model, X_test, y_test)
    results_df["Cohen's Kappa"] = kappa
    plot_cm(prob_ls_test, true_ls_test, "LightGBM")

    return results_df


def GPBoost_engine(
    X_train_resampled, group_train_resampled, y_train_resampled, X_val, y_val, group_val
):

    """Train a GPBoost model using hyperparameter optimization.
    
    Parameters
    ----------
    X_train_resampled : array-like
        Training data.
    group_train_resampled : array-like
        Group data for training.
    y_train_resampled : array-like
        Training labels.
    X_val : array-like
        Validation data for early stopping.
    y_val : array-like
        Validation labels for early stopping.
    group_val : array-like
        Group data for validation.

    Returns
    -------
    final_gpb_model : GPBoost model
        the trained GPBoost model
    """
    space = {
        "max_depth": hp.quniform("max_depth", 3, 6, 1),
        "learning_rate": hp.uniform("learning_rate", 0.005, 0.01),
        "num_leaves": hp.quniform("num_leaves", 20, 200, 20),
        "feature_fraction": hp.uniform("feature_fraction", 0.5, 0.95),
        "lambda_l2": hp.uniform("lambda_l2", 1.0, 10.0),
        "lambda_l1": hp.quniform("lambda_l1", 10, 100, 10),
        "pos_bagging_fraction": hp.uniform("pos_bagging_fraction", 0.8, 0.95),
        "neg_bagging_fraction": hp.uniform("neg_bagging_fraction", 0.6, 0.8),
        "num_boost_round": hp.quniform("num_boost_round", 400, 1000, 100),
    }

    def objective(space):
        params = {
            "objective": "binary",
            "max_depth": int(space["max_depth"]),
            "learning_rate": space["learning_rate"],
            "num_leaves": int(space["num_leaves"]),
            "feature_fraction": space["feature_fraction"],
            "lambda_l2": space["lambda_l2"],
            "lambda_l1": space["lambda_l1"],
            "pos_bagging_fraction": space["pos_bagging_fraction"],
            "neg_bagging_fraction": space["neg_bagging_fraction"],
            "num_boost_round": int(space["num_boost_round"]),
            "verbose": -1,
        }
        num_boost_round = params.pop("num_boost_round")

        gp_model = gpb.GPModel(
            group_data=group_train_resampled, likelihood="bernoulli_probit"
        )

        data_train = gpb.Dataset(data=X_train_resampled, label=y_train_resampled)
        clf = gpb.train(
            params=params,
            train_set=data_train,
            gp_model=gp_model,
            num_boost_round=num_boost_round,
        )

        pred_resp = clf.predict(
            data=X_val, group_data_pred=group_val, predict_var=True, pred_latent=False
        )
        positive_probabilities = pred_resp["response_mean"]
        predicted_labels = (positive_probabilities > 0.5).astype(int)

        f1 = f1_score(y_val, predicted_labels)
        return {"loss": -f1, "status": STATUS_OK}

    # Run the hyperparameter search
    # for AHI and obesity together, it's okay to have number of max evaluations be 10 instead of 50
    # due to the much longer fitting time
    trials = Trials()
    gpb_best_hyperparams = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials
    )
    print("Best hyperparameters:", gpb_best_hyperparams)

    # Adjust the types of the best hyperparameters
    gpb_best_hyperparams["max_depth"] = int(gpb_best_hyperparams["max_depth"])
    gpb_best_hyperparams["num_leaves"] = int(gpb_best_hyperparams["num_leaves"])
    gpb_best_hyperparams["num_boost_round"] = int(
        gpb_best_hyperparams["num_boost_round"]
    )

    # Train the final model
    data_train = gpb.Dataset(X_train_resampled, y_train_resampled)
    data_eval = gpb.Dataset(X_val, y_val)
    gp_model = gpb.GPModel(
        group_data=group_train_resampled, likelihood="bernoulli_probit"
    )
    gp_model.set_prediction_data(group_data_pred=group_val)
    evals_result = {}  # record eval results for plotting
    final_gpb_model = gpb.train(
        params=gpb_best_hyperparams,
        train_set=data_train,
        gp_model=gp_model,
        valid_sets=data_eval,
        early_stopping_rounds=10,
        use_gp_model_for_validation=True,
        evals_result=evals_result,
    )

    return final_gpb_model


def GPBoost_predict(final_gpb_model, X_test, y_test, group_test):
    """ Predict using a trained GPBoost model and calculate evaluation metrics.
    
    Parameters
    ----------
    final_gpb_model : gpb.train
        Trained GPBoost model.
    X_test : array-like
        Data to predict on.
    y_test : array-like
        True labels.  
    group_test : array-like
        Group data for prediction.

    Returns
    -------
    gpb_train_results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    pred_resp = final_gpb_model.predict(
        data=X_test, group_data_pred=group_test, predict_var=True, pred_latent=False
    )
    positive_probabilities = pred_resp["response_mean"]
    negative_probabilities = 1 - positive_probabilities
    predicted_probabilities = np.stack(
        [negative_probabilities, positive_probabilities], axis=1
    )
    gpb_train_results_df = calculate_metrics(y_test, predicted_probabilities, "GPBoost")

    return gpb_train_results_df


def GPBoost_result(final_gpb_model, X_test, y_test, group, prob_ls_test, true_ls_test):
    """Calculate evaluation metrics and plot confusion matrix for a trained GPBoost model.
    
    Parameters
    ----------
    final_gpb_model : gpb.train
        Trained GPBoost model.
    X_test : array-like
        Data to predict on.
    y_test : array-like
        True labels.
    group_test : array-like
        Group data for prediction.
    prob_ls_test : array-like
        Predicted probabilities from GPBoost model without post-processing.
    true_ls_test : array-like
        True labels in time series.

    Returns
    -------
    results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    kappa = calculate_kappa(prob_ls_test, true_ls_test)
    results_df = GPBoost_predict(final_gpb_model, X_test, y_test, group)
    results_df["Cohen's Kappa"] = kappa
    plot_cm(prob_ls_test, true_ls_test, "GPBoost")

    return results_df


def LSTM_dataloader(list_probabilities_subject, lengths, list_true_stages, batch_size=1):
    """Create a DataLoader for a list of each subject's data.
    
    Parameters
    ----------
    list_probabilities_subject : list
        List of predicted probabilities for each subject.
    lengths : list
        List of lengths of each subject's data.
    list_true_stages : list
        List of true labels for each subject.

    Returns
    -------
    dataloader : DataLoader
        DataLoader for the LSTM model.
    """
    dataset = TimeSeriesDataset(list_probabilities_subject, lengths, list_true_stages)

    # DataLoader with the custom collate function for handling padding
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


def LSTM_engine(dataloader_train, num_epoch, hidden_layer_size=64, learning_rate=0.001, dataloader_val=None):
    """
    Train a LSTM model using a DataLoader with comprehensive regularization.
    
    Parameters
    ----------
    dataloader_train : DataLoader
        DataLoader for the training data.
    num_epoch : int
        Number of epochs to train the model.
    hidden_layer_size : int
        Size of hidden layer (default 64, increased from 32).
    learning_rate : float
        Initial learning rate.
    dataloader_val : DataLoader, optional
        DataLoader for validation data for early stopping.

    Returns
    -------
    model : BiLSTMPModel
        Trained LSTM model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    input_size = 7  # 5 probabilities from LightGBM + 2 features (ACC_INDEX, HRV_HFD)
    output_size = 5  # 5 sleep stage classes (W, R, N1, N2, N3)

    # Calculate class weights matching LightGBM's balanced approach
    # Distribution: W=10698, R=4666, N1=4320, N2=22209, N3=1868
    # Using ^1.2 for most, ^1.6 for R and N3 (balanced to protect both minority classes)
    total_samples = 43761
    class_counts = torch.tensor([10698, 4666, 4320, 22209, 1868], dtype=torch.float)
    # Calculate base weights
    base_weights = (total_samples / class_counts) ** 1.2
    # Override R and N3 weights with higher exponent for balanced focus
    class_weights = base_weights.clone()
    class_weights[1] = (total_samples / 4666) ** 1.6   # R: ~29.4 (protect R from degradation)
    class_weights[4] = (total_samples / 1868) ** 1.6   # N3: ~47.1 (balanced with R)
    class_weights = class_weights.to(device)
    
    # Add dropout (0.3) with 2 layers for proper regularization
    model = BiLSTMPModel(input_size, hidden_layer_size, output_size, dropout=0.3, num_layers=2).to(device)
    
    # Use Focal Loss with gamma=1.5 for balanced focusing on R and N3
    # gamma=1.5: Moderate focus on hard examples, prevents over-focusing on N3 at R's expense
    loss_function = FocalLoss(alpha=class_weights, gamma=1.5, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler: reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    # Training loop with early stopping
    epochs = num_epoch
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        model.train()  # Set the model to training mode

        for i, batch in enumerate(dataloader_train):
            sample = batch["sample"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)

            if sample.shape[1] == 0:
                print("Empty batch detected, skipping...")
                continue

            optimizer.zero_grad()
            y_pred = model(sample, length)

            # Reshape y_pred and label for FocalLoss
            # FocalLoss expects y_pred of shape [N, C], label of shape [N]
            y_pred = y_pred.view(-1, 5)  # Flatten output for FocalLoss (5 classes)
            label = label.view(-1)  # Flatten label tensor

            loss = loss_function(y_pred, label)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(y_pred, label).item()

        avg_loss = total_loss / len(dataloader_train)
        avg_accuracy = total_accuracy / len(dataloader_train)

        # Validation and early stopping
        if dataloader_val is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in dataloader_val:
                    sample = batch["sample"].to(device)
                    length = batch["length"]
                    label = batch["label"].to(device)
                    
                    y_pred = model(sample, length)
                    y_pred = y_pred.view(-1, 5)
                    label = label.view(-1)
                    
                    loss = loss_function(y_pred, label)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(dataloader_val)
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Train Acc: {avg_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}")
        else:
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return model


def LSTM_eval(lstm_model, dataloader_test, list_true_stages_test, test_name):
    """
    Evaluate a LSTM model using a DataLoader.
    
    Parameters
    ----------
    lstm_model : BiLSTMPModel
        Trained LSTM model.
    dataloader_test : DataLoader
        DataLoader for the test data.
    list_true_stages_test : list
        List of true labels for the test data.

    Returns
    -------
    lstm_test_results_df : DataFrame
        Dataframe with evaluation metrics.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.eval()  # Set the model to evaluation mode
    lstm_model.to(device)

    predicted_probabilities_test = []
    kappa = []

    with torch.no_grad():  # No need to track the gradients
        for batch in dataloader_test:
            sample = batch["sample"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)

            # Forward pass
            outputs = lstm_model(sample, length)

            predicted_probabilities_test.extend(outputs.cpu().numpy())

            # Calculating Cohen's Kappa Score, ensure labels and predictions are on CPU
            kappa.append(
                cohen_kappa_score(
                    label.cpu().numpy()[0], np.argmax(outputs.cpu().numpy()[0], axis=1)
                )
            )

    array_true = np.concatenate(list_true_stages_test)
    array_predict = np.concatenate(predicted_probabilities_test)

    lstm_test_results_df = calculate_metrics(array_true, array_predict, test_name)
    lstm_test_results_df["Cohen's Kappa"] = np.average(kappa)
    plot_cm(array_predict, list_true_stages_test, test_name)

    return lstm_test_results_df
