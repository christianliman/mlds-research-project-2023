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
                             RocCurveDisplay, PrecisionRecallDisplay, mean_squared_error)
from pandas.api.types import CategoricalDtype


def PrepareWeightedData(res, outcome, covars=None):
    """
    Prepare data set for ensemble model training using the output of MICE
    
    Parameters
    ----------
    res : dict
        Dictionary returned by `MICE()` which should contain the imputed data under `imp`
    outcome : str
        Outcome variable for the ensemble model
    covars : list, optional
        List of covariates. If blank, assumed to be everything other than outcome
    
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
    return _weighted_data(res["imp"], any_missingflag, outcome, covars)


def _weighted_data(imp, any_missingflag, outcome, covars):
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
        if covars is None:
            tempX = df.drop(outcome, axis=1)
        else:
            tempX = df[covars]
        tempy = df[outcome]
        
        if i == 0:
            # Use the first dataset to separate out observed and imputed
            Xs.append(tempX[~any_missingflag])
            ys.append(tempy[~any_missingflag])
            ws.append(np.ones((Nobs, 1)))
        
        # Append imputed data only with appropriate weight
        Xs.append(tempX.iloc[any_missingflag])
        ys.append(tempy.iloc[any_missingflag])
        ws.append(np.ones((Nmis, 1)) / m)
    
    # Merge data together
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    w = np.concatenate(ws).ravel()
    
    return X, y, w


def KFoldWeighted(n_splits, res, outcome, base_model, classifier=True, random_state=None, covars=None):
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
        misflag_train = any_missingflag.iloc[train_index].values
        misflag_test = any_missingflag.iloc[test_index].values
        
        # Separate out train and test data
        imp_train, imp_test = [], []
        for j in range(m):
            # Split the j-th dataset
            imp_train.append(res["imp"][j].iloc[train_index])
            imp_test.append(res["imp"][j].iloc[test_index])
        
        # Construct weighted data
        X_train, y_train, w_train = _weighted_data(
            imp_train, misflag_train, outcome, covars=covars
        )
        X_test, y_test, w_test = _weighted_data(
            imp_test, misflag_test, outcome, covars=covars
        )

        # For test data, we are only interested in observed data
        X_test = X_test[w_test == 1]
        y_test = y_test[w_test == 1]
        
        # Build model on training data
        curmodel = copy.deepcopy(base_model)
        curmodel.fit(X_train, y_train, sample_weight=w_train)
        
        # Predict on test data and store predictions
        if classifier:
            # Compute probability if model is a classifier
            pred_test = curmodel.predict_proba(X_test)[:, 1]
        else:
            pred_test = curmodel.predict(X_test)
        preds.append(pd.DataFrame({"true": y_test, "pred": pred_test}))
    
    # Aggregate predictions and compute performance metrics
    # NOTE: For precision, recall, F1, a cutoff of 0.5 is assumed
    preds = pd.concat(preds)

    # Adapt metric depending on model type
    if classifier:
        preds["pred_labels"] = preds["pred"] > 0.5
        all_metrics = {
            "AUROC": roc_auc_score(preds["true"], preds["pred"]),
            #"Precision": precision_score(preds["true"], preds["pred_labels"]),
            #"Recall": recall_score(preds["true"], preds["pred_labels"]),
            "F1": f1_score(preds["true"], preds["pred_labels"]),
        }
        
        # Construct ROC and precision-recall curves
        #fig, axs = plt.subplots(figsize=(8, 4), ncols=2, nrows=1)
        #RocCurveDisplay.from_predictions(
        #    preds["true"], preds["pred"], drop_intermediate=False, ax=axs[0]
        #)
        #PrecisionRecallDisplay.from_predictions(
        #    preds["true"], preds["pred"], ax=axs[1]
        #)
        #fig.tight_layout()
    else:
        all_metrics = {
            "RMSE": mean_squared_error(preds["true"], preds["pred"], squared=False)
        }
    
    return all_metrics, preds#, fig, preds

