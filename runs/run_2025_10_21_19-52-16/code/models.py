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
    def __init__(self, input_size, hidden_layer_size, output_size=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_layer_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.linear = nn.Linear(
            2 * hidden_layer_size, output_size
        )  # 2 output units for 2 classes

    def forward(self, input_seq, lengths):
        packed_input = pack_padded_sequence(
            input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.linear(output)  # Shape: [batch_size, seq_len, 2]
        return output


class LSTMPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_layer_size, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(
            hidden_layer_size, output_size
        )  # 2 output units for 2 classes

    def forward(self, input_seq, lengths):
        packed_input = pack_padded_sequence(
            input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.linear(output)  # Shape: [batch_size, seq_len, 2]
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
        "max_depth": hp.quniform("max_depth", 2, 6, 1),
        "reg_alpha": hp.quniform("reg_alpha", 0, 180, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0.2, 5),
        "num_leaves": hp.quniform("num_leaves", 20, 100, 10),
        "n_estimators": hp.quniform("n_estimators", 50, 300, 10),
        "learning_rate": hp.uniform("learning_rate", 0.005, 0.5),
    }

    def objective(space):
        clf = lgb.LGBMClassifier(
            objective="binary",
            #is_unbalance=True,
            scale_pos_weight=1.5,
            max_depth=int(space["max_depth"]),
            reg_alpha=space["reg_alpha"],
            reg_lambda=space["reg_lambda"],
            n_estimators=int(space["n_estimators"]),
            learning_rate=space["learning_rate"],
            num_leaves=int(space["num_leaves"]),
            verbose=-1,
        )

        clf.fit(X_train_resampled, y_train_resampled)

        positive_probabilities = clf.predict_proba(X_val)[:, 1]
        predicted_labels = (positive_probabilities > 0.5).astype(int)

        f1 = f1_score(y_val, predicted_labels)
        return {"loss": -f1, "status": STATUS_OK}

    # Run the hyperparameter search
    trials = Trials()
    lgb_best_hyperparams = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials
    )
    print("Best hyperparameters:", lgb_best_hyperparams)

    # Adjust the data types of the best hyperparameters
    lgb_best_hyperparams["max_depth"] = int(lgb_best_hyperparams["max_depth"])
    lgb_best_hyperparams["n_estimators"] = int(lgb_best_hyperparams["n_estimators"])
    lgb_best_hyperparams["num_leaves"] = int(lgb_best_hyperparams["num_leaves"])

    final_lgb_model = lgb.LGBMClassifier(
        **lgb_best_hyperparams, random_state=1, num_iterations=50
    )

    final_lgb_model.fit(X_train_resampled, y_train_resampled)

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


def LightGBM_result(final_lgb_model, X_test, y_test, prob_ls_test, true_ls_test, image_output_file=None):
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
    image_output_file : str, optional
        If provided, save the confusion matrix plot to this file path.

    Returns
    -------
    results_df : DataFrame
        Dataframe with evaluation metrics.
    """
    kappa = calculate_kappa(prob_ls_test, true_ls_test)
    results_df = LightGBM_predict(final_lgb_model, X_test, y_test)
    results_df["Cohen's Kappa"] = kappa
    plot_cm(prob_ls_test, true_ls_test, "LightGBM", image_output_file)

    return results_df


def focal_loss_lgb_multiclass(y_true, y_pred, alpha=1.0, gamma=2.0, num_classes=5):
    """
    Focal loss for LightGBM multiclass classification.
    
    Parameters
    ----------
    y_true : array-like
        True labels (1D array of class indices)
    y_pred : array-like
        Predicted probabilities (2D array: n_samples x n_classes)
    alpha : float
        Weighting factor in [0, 1] to balance positive/negative examples
    gamma : float
        Focusing parameter for modulating loss (gamma=0 is equivalent to CE loss)
    num_classes : int
        Number of classes
    
    Returns
    -------
    grad : array
        Gradient
    hess : array
        Hessian
    """
    # Reshape predictions to (n_samples, n_classes)
    y_pred = y_pred.reshape(-1, num_classes, order='F')
    
    # Apply softmax to get probabilities
    y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
    
    # Clip probabilities to avoid log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    
    # One-hot encode true labels
    y_true_one_hot = np.zeros((len(y_true), num_classes))
    y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
    
    # Calculate focal loss components
    # pt = probability of true class
    pt = np.sum(y_pred * y_true_one_hot, axis=1, keepdims=True)
    
    # Focal loss modulating factor
    focal_weight = alpha * np.power(1 - pt, gamma)
    
    # Gradient: derivative of focal loss w.r.t. logits
    grad = focal_weight * (y_pred - y_true_one_hot)
    
    # Hessian: second derivative (approximation for LightGBM)
    hess = focal_weight * y_pred * (1 - y_pred)
    
    return grad.flatten('F'), hess.flatten('F')


def LightGBM_engine_multiclass(X_train_resampled, y_train_resampled, X_val, y_val, num_classes=5, class_weight=None, use_focal_loss=False, focal_alpha=1.0, focal_gamma=2.0):
    """Train a LightGBM model for multiclass classification using hyperparameter optimization.
    
    Parameters
    ----------
    X_train_resampled : array-like
        Training data.
    y_train_resampled : array-like
        Training labels (numeric 0 to num_classes-1).
    X_val : array-like
        Validation data for early stopping.
    y_val : array-like
        Validation labels for early stopping.
    num_classes : int
        Number of classes (default 5: W, R, N1, N2, N3)
    class_weight : dict or 'balanced', optional
        Weights associated with classes. If 'balanced', uses n_samples / (n_classes * np.bincount(y))
    use_focal_loss : bool
        If True, use focal loss instead of standard cross-entropy
    focal_alpha : float
        Focal loss alpha parameter (default 1.0)
    focal_gamma : float
        Focal loss gamma parameter (default 2.0, higher = more focus on hard examples)

    Returns
    -------
    final_lgb_model : LightGBM model
    """
    space = {
        "max_depth": hp.quniform("max_depth", 2, 6, 1),
        "reg_alpha": hp.quniform("reg_alpha", 0, 10, 1),  # Reduced L1 penalty (0-10) to allow more splits
        "reg_lambda": hp.uniform("reg_lambda", 0.1, 1.0),  # Reduced L2 penalty (0.1-1.0) to allow more splits
        "num_leaves": hp.quniform("num_leaves", 20, 100, 10),
        "n_estimators": hp.quniform("n_estimators", 50, 300, 10),
        "learning_rate": hp.uniform("learning_rate", 0.005, 0.5),
    }

    # Create focal loss objective if requested
    if use_focal_loss:
        def focal_obj(y_true, y_pred):
            return focal_loss_lgb_multiclass(y_true, y_pred, alpha=focal_alpha, gamma=focal_gamma, num_classes=num_classes)
        
        objective_func = focal_obj
        print(f"Using Focal Loss with alpha={focal_alpha}, gamma={focal_gamma}")
    else:
        objective_func = "multiclass"

    def objective(space):
        if use_focal_loss:
            # When using custom objective, must use lgb.train with fobj parameter
            train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
            
            params = {
                'num_class': num_classes,
                'max_depth': int(space["max_depth"]),
                'reg_alpha': space["reg_alpha"],
                'reg_lambda': space["reg_lambda"],
                'num_leaves': int(space["num_leaves"]),
                'learning_rate': space["learning_rate"],
                'verbose': -1,
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=int(space["n_estimators"]),
                fobj=focal_obj,
            )
            
            # Predict on validation
            y_pred_proba = model.predict(X_val)
            predicted_labels = np.argmax(y_pred_proba, axis=1)
        else:
            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                max_depth=int(space["max_depth"]),
                reg_alpha=space["reg_alpha"],
                reg_lambda=space["reg_lambda"],
                n_estimators=int(space["n_estimators"]),
                learning_rate=space["learning_rate"],
                num_leaves=int(space["num_leaves"]),
                class_weight=class_weight,
                verbose=-1,
            )

            clf.fit(X_train_resampled, y_train_resampled)
            predicted_labels = clf.predict(X_val)

        f1 = f1_score(y_val, predicted_labels, average='weighted')
        
        return {"loss": -f1, "status": STATUS_OK}

    # Run the hyperparameter search
    trials = Trials()
    lgb_best_hyperparams = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials
    )
    print("Best hyperparameters:", lgb_best_hyperparams)

    # Adjust the data types of the best hyperparameters
    lgb_best_hyperparams["max_depth"] = int(lgb_best_hyperparams["max_depth"])
    lgb_best_hyperparams["n_estimators"] = int(lgb_best_hyperparams["n_estimators"])
    lgb_best_hyperparams["num_leaves"] = int(lgb_best_hyperparams["num_leaves"])

    # Train final model
    if use_focal_loss:
        train_data = lgb.Dataset(X_train_resampled, label=y_train_resampled)
        
        params = {
            'objective': None,
            'num_class': num_classes,
            'max_depth': lgb_best_hyperparams["max_depth"],
            'reg_alpha': lgb_best_hyperparams["reg_alpha"],
            'reg_lambda': lgb_best_hyperparams["reg_lambda"],
            'num_leaves': lgb_best_hyperparams["num_leaves"],
            'learning_rate': lgb_best_hyperparams["learning_rate"],
            'verbose': -1,
        }
        
        final_lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=50,
            fobj=focal_obj,
        )
    else:
        final_lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            class_weight=class_weight,
            **lgb_best_hyperparams, 
            random_state=1, 
            num_iterations=50
        )

        final_lgb_model.fit(X_train_resampled, y_train_resampled)

    return final_lgb_model


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


def LSTM_engine(dataloader_train, num_epoch, hidden_layer_size=32, learning_rate = 0.001):
    """
    Train a LSTM model using a DataLoader.
    
    Parameters
    ----------
    dataloader_train : DataLoader
        DataLoader for the training data.
    num_epoch : int
        Number of epochs to train the model.

    Returns
    -------
    model : BiLSTMPModel
        Trained LSTM model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    input_size = 4  # Number of features
    output_size = 2

    # dropout must be 0 if using only one layer of LSTM
    model = BiLSTMPModel(input_size, hidden_layer_size, output_size).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # Training loop
    epochs = num_epoch

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

            # Reshape y_pred and label for CrossEntropyLoss
            # CrossEntropyLoss expects y_pred of shape [N, C], label of shape [N]
            y_pred = y_pred.view(-1, 2)  # Flatten output for CrossEntropyLoss
            label = label.view(-1)  # Flatten label tensor

            loss = loss_function(y_pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(y_pred, label).item()

        avg_loss = total_loss / len(dataloader_train)
        avg_accuracy = total_accuracy / len(dataloader_train)

        # Optionally, you can calculate loss and accuracy on a validation set he
        # re
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


def LSTM_dataloader_multiclass(list_probabilities_subject, list_features_subject, lengths, list_true_stages, batch_size=1):
    """Create a DataLoader for multiclass LSTM with probabilities + additional features.
    
    Parameters
    ----------
    list_probabilities_subject : list
        List of predicted probabilities for each subject (5 classes per time step).
    list_features_subject : list
        List of additional features for each subject (5 features per time step).
    lengths : list
        List of lengths of each subject's data.
    list_true_stages : list
        List of true labels for each subject.

    Returns
    -------
    dataloader : DataLoader
        DataLoader for the multiclass LSTM model.
    """
    # Concatenate probabilities and features
    combined_data = []
    for probs, feats in zip(list_probabilities_subject, list_features_subject):
        combined = np.concatenate([probs, feats], axis=1)  # Shape: (n_timesteps, 10)
        combined_data.append(combined)
    
    dataset = TimeSeriesDataset(combined_data, lengths, list_true_stages)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dataloader


def LSTM_engine_multiclass(dataloader_train, num_epoch, hidden_layer_size=32, learning_rate=0.001, num_classes=5, input_size=10, class_weight=None, use_focal_loss=False):
    """
    Train a multiclass LSTM model using a DataLoader.
    
    Parameters
    ----------
    dataloader_train : DataLoader
        DataLoader for the training data.
    num_epoch : int
        Number of epochs to train the model.
    hidden_layer_size : int
        Size of the hidden layer.
    learning_rate : float
        Learning rate for optimization.
    num_classes : int
        Number of output classes (default: 5 for W, R, N1, N2, N3).
    input_size : int
        Number of input features (default: 10 = 5 probabilities + 5 additional features).
    class_weight : dict or None
        Class weights for handling imbalance.
    use_focal_loss : bool
        Whether to use focal loss instead of cross-entropy.

    Returns
    -------
    model : BiLSTMPModel
        Trained LSTM model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training multiclass LSTM on {device}")

    output_size = num_classes

    model = BiLSTMPModel(input_size, hidden_layer_size, output_size).to(device)
    
    # Convert class weights to tensor if provided
    if class_weight is not None:
        weight_tensor = torch.tensor([class_weight[i] for i in range(num_classes)], dtype=torch.float32).to(device)
        print(f"Using class weights: {weight_tensor.cpu().numpy()}")
    else:
        weight_tensor = None
    
    if use_focal_loss:
        from utils import FocalLoss
        loss_function = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss for LSTM training")
    else:
        loss_function = nn.CrossEntropyLoss(weight=weight_tensor)
        if weight_tensor is not None:
            print("Using Weighted Cross-Entropy Loss for LSTM training")
        else:
            print("Using Cross-Entropy Loss for LSTM training")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    epochs = num_epoch

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        model.train()

        for i, batch in enumerate(dataloader_train):
            sample = batch["sample"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)

            if sample.shape[1] == 0:
                print("Empty batch detected, skipping...")
                continue

            optimizer.zero_grad()
            y_pred = model(sample, length)

            # Reshape for loss computation
            y_pred = y_pred.view(-1, output_size)
            label = label.view(-1)

            loss = loss_function(y_pred, label.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(y_pred, label).item()

        avg_loss = total_loss / len(dataloader_train)
        avg_accuracy = total_accuracy / len(dataloader_train)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return model


def LSTM_eval_multiclass(lstm_model, dataloader_test, list_true_stages_test, class_names, test_name=""):
    """
    Evaluate a multiclass LSTM model using a DataLoader.
    
    Parameters
    ----------
    lstm_model : BiLSTMPModel
        Trained LSTM model.
    dataloader_test : DataLoader
        DataLoader for the test data.
    list_true_stages_test : list
        List of true labels for the test data.
    class_names : list
        List of class names (e.g., ['W', 'R', 'N1', 'N2', 'N3']).
    test_name : str
        Name for the test set.

    Returns
    -------
    predicted_probabilities_test : list
        List of predicted probabilities for each subject.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model.eval()
    lstm_model.to(device)

    predicted_probabilities_test = []
    kappa = []

    with torch.no_grad():
        for batch in dataloader_test:
            sample = batch["sample"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)

            # Forward pass
            outputs = lstm_model(sample, length)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=-1)
            predicted_probabilities_test.extend(probs.cpu().numpy())

            # Calculating Cohen's Kappa Score
            kappa.append(
                cohen_kappa_score(
                    label.cpu().numpy()[0], np.argmax(probs.cpu().numpy()[0], axis=1)
                )
            )

    print(f"\n{test_name} - Average Cohen's Kappa: {np.average(kappa):.4f}")
    
    return predicted_probabilities_test
