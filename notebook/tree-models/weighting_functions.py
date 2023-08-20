# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import copy
import shap

from scipy import linalg
from scipy.special import expit
from scipy import stats
from tqdm import tqdm
from matplotlib import cm
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                             RocCurveDisplay, PrecisionRecallDisplay)
from pandas.api.types import CategoricalDtype


def PrepareWeightedData(res, outcome):
    """
    Prepare data set for ensemble model training using the output of MICE
    
    Parameters
    ----------
    res : dict
        Dictionary returned by `MICE()` which should contain the imputed data under `imp`
    outcome : str
        Outcome variable for the ensemble model
    
    Returns
    -------
    pandas.DataFrame
        Modified covariate matrices with all observed and imputed observations
    pandas.DataFrame
        Modified outcome vectors with all observed and imputed observations
    numpy.ndarray
        Weights for all observations
    """
    # Extract missing flags
    missingflag = res["missingflag"]
    any_missingflag = res["missingflag"].any(axis=1)
    
    # Call internal function
    return _weighted_data(res["imp"], any_missingflag, outcome)


def _weighted_data(imp, any_missingflag, outcome):
    # Internal function where the supersized data is constructed
    
    # Get number of multiply imputed data
    m = len(imp)
    
    # Calculate Nobs and Nmis
    Nobs = (~any_missingflag).sum()
    Nmis = any_missingflag.sum()
    
    # Placeholder for the dataset and assigned weights
    Xs, ys, ws = [], [], []
    
    # Iterate over multiply imputed data
    for i, df in enumerate(imp):
        # Extract covariate and outcome data frames
        tempX = df.drop(outcome, axis=1)
        tempy = df[outcome]
        
        if i == 0:
            # Use the first dataset to separate out observed and imputed
            Xs.append(tempX[~any_missingflag])
            ys.append(tempy[~any_missingflag])
            ws.append(np.ones((Nobs, 1)))
        
        # Append imputed data only with appropriate weight
        Xs.append(tempX[any_missingflag])
        ys.append(tempy[any_missingflag])
        ws.append(np.ones((Nmis, 1)) / m)
    
    # Merge data together
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    w = np.concatenate(ws).ravel()
    
    return X, y, w


def KFoldWeighted(n_splits, res, outcome, base_model, random_state=None):
    ## DOCUMENTATION TO BE ADDED
    # Wrapper to implement k-fold cross validation on weighted data
    # NOTE: Unlike KFoldEnsemble, this is to be used with the imputed
    # object directly (no need to call PrepareWeightedData)
    # Generate k-fold object
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    # Get number of multiply imputed data
    m = len(res["imp"])
    
    # Extract missing flags
    missingflag = res["missingflag"]
    any_missingflag = res["missingflag"].any(axis=1)
    
    # Placeholder for predictions and metrics
    preds = []
    
    # Iterate over the folds
    for i, (train_index, test_index) in enumerate(kf.split(res["imp"][0])):
        # Reconstruct missing flag
        misflag_train = any_missingflag.iloc[train_index]
        misflag_test = any_missingflag.iloc[test_index]
        
        # Separate out train and test data
        imp_train, imp_test = [], []
        for j in range(m):
            # Split the j-th dataset
            imp_train.append(res["imp"][j].iloc[train_index])
            # Filter observed data only for test data
            imp_test.append(
                res["imp"][j].iloc[test_index][misflag_test.values]
            )
        
        # Construct weighted data
        X_train, y_train, w_train = _weighted_data(
            imp_train, misflag_train, outcome
        )
        X_test, y_test, _ = _weighted_data(
            imp_test, misflag_test[misflag_test == 1], outcome
        )
        
        # Build model on training data
        curmodel = copy.deepcopy(base_model)
        curmodel.fit(X_train, y_train, sample_weight=w_train)
        
        # Predict on test data and store predictions
        pred_test = curmodel.predict_proba(X_test)[:, 1]
        preds.append(pd.DataFrame({"true": y_test, "pred": pred_test}))
    
    # Aggregate predictions and compute performance metrics
    # NOTE: For precision, recall, F1, a cutoff of 0.5 is assumed
    preds = pd.concat(preds)
    preds["pred_labels"] = preds["pred"] > 0.5
    all_metrics = {
        "AUROC": roc_auc_score(preds["true"], preds["pred"]),
        "Precision": precision_score(preds["true"], preds["pred_labels"]),
        "Recall": recall_score(preds["true"], preds["pred_labels"]),
        "F1": f1_score(preds["true"], preds["pred_labels"]),
    }
    
    # Construct ROC and precision-recall curves
    fig, axs = plt.subplots(figsize=(8, 4), ncols=2, nrows=1)
    RocCurveDisplay.from_predictions(
        preds["true"], preds["pred"], drop_intermediate=False, ax=axs[0]
    )
    PrecisionRecallDisplay.from_predictions(
        preds["true"], preds["pred"], ax=axs[1]
    )
    fig.tight_layout()
    
    return all_metrics, fig, preds

