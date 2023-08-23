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
from sklearn.base import TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score, 
                             RocCurveDisplay, PrecisionRecallDisplay, mean_squared_error)
from pandas.api.types import CategoricalDtype


def PrepareEnsembleData(res, outcome, covars=None):
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
    list
        List of multiply imputed covariate matrices
    list
        List of multiply imputed outcome vectors
    """
    X, y = [], []
    for i, df in enumerate(res["imp"]):
        y.append(df[outcome])
        if covars is None:
            X.append(df.drop(outcome, axis=1))
        else:
            X.append(df[covars])
    return X, y


class EnsembleClassifier(ClassifierMixin):
    # DOCUMENTATION TO BE REFINED
    # Main class to implement ensemble model (can be RF or any sklearn classifiers basically)

    def __init__(self, component, **kwargs):
        super().__init__(**kwargs)
        self.comp_model = component
    
    def fit(self, X, y):
        self.m = len(X) # number of ensemble models to be constructed
        self.components = []
        for i in range(self.m):
            self.components.append(copy.deepcopy(self.comp_model).fit(X[i], y[i]))
        return self
    
    def predict(self, X):
        predicted = np.zeros((X.shape[0], self.m))
        for i in range(self.m):
            predicted[:, i] = self.components[i].predict(X)
        return stats.mode(predicted, axis=1).mode # most common
    
    def predict_proba(self, X):
        predicted = np.zeros((X.shape[0], self.m))
        for i in range(self.m):
            predicted[:, i] = self.components[i].predict_proba(X)[:, 1]
        pos_p = predicted.mean(axis=1).reshape(-1, 1)
        neg_p = 1 - pos_p
        return np.hstack((neg_p, pos_p))
    
    def predict_log_proba(self, X):
        probs = self.predict_proba(X)
        return np.log(probs)
    
    def predict_log_odds(self, X):
        probs = self.predict_proba(X)
        return np.log(probs[:, 1]) - np.log(probs[:, 0])


class EnsembleRegressor(RegressorMixin):
    # DOCUMENTATION TO BE REFINED
    # Main class to implement ensemble model (can be RF or any sklearn classifiers basically)

    def __init__(self, component, **kwargs):
        super().__init__(**kwargs)
        self.comp_model = component
    
    def fit(self, X, y):
        self.m = len(X) # number of ensemble models to be constructed
        self.components = []
        for i in range(self.m):
            self.components.append(copy.deepcopy(self.comp_model).fit(X[i], y[i]))
        return self
    
    def predict(self, X):
        predicted = np.zeros((X.shape[0], self.m))
        for i in range(self.m):
            predicted[:, i] = self.components[i].predict(X)
        return np.mean(predicted, axis=1)


def KFoldEnsemble(n_splits, X, y, misflag, base_model, classifier=True, random_state=None, resample=False, pred_cutoff=0.5):
    ## DOCUMENTATION TO BE ADDED
    # Wrapper to implement k-fold cross validation on ensemble data
    ## Note: Missing flag should be 1-dimensional
    # Generate k-fold object
    #kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    
    # Get number of multiply imputed data
    m = len(X)
    
    # Placeholder for predictions and metrics
    preds = []
    
    # Iterate over the folds
    # for i, (train_index, test_index) in enumerate(kf.split(X[0])):
    for i, (train_index, test_index) in enumerate(kf.split(X[0], misflag)):
        # If needed, we rebalance the training indices
        if resample:
            temp = y[0].iloc[train_index] # note that y is not imputed so this is fine
            train_index_pos = temp[temp == 1].index
            train_index_neg = temp[temp == 0].index
            train_index_pos_up = np.random.choice(
                train_index_pos, size=len(train_index_neg), replace=True
            )
            train_index_up = np.concatenate((train_index_pos_up, train_index_neg), axis=0)

        # Construct train and test data
        X_train, X_test, y_train, y_test = [], [], [], []
        for j in range(m):
            # Split the j-th dataset
            if resample:
                X_train.append(X[j].iloc[train_index_up])
                y_train.append(y[j].iloc[train_index_up])
            else:
                X_train.append(X[j].iloc[train_index])
                y_train.append(y[j].iloc[train_index])
            X_test.append(X[j].iloc[test_index])
            y_test.append(y[j].iloc[test_index])
        
        # Build model on training data
        if classifier:
            curmodel = EnsembleClassifier(base_model)
        else:
            curmodel = EnsembleRegressor(base_model)
        curmodel.fit(X_train, y_train)
        
        # Construct single dataframe for test data
        # NOTE: Focus on observed data for now
        misflag_test = misflag.iloc[test_index].values
        Xobs_test = X_test[0][~misflag_test]
        yobs_test = y_test[0][~misflag_test]
        
        # Predict on test data and store predictions
        if classifier:
            # Compute probability if model is a classifier
            pred_test = curmodel.predict_proba(Xobs_test)[:, 1]
        else:
            pred_test = curmodel.predict(Xobs_test)
        preds.append(pd.DataFrame({"true": yobs_test, "pred": pred_test}))
    
    # Aggregate predictions and compute performance metrics
    # NOTE: For precision, recall, F1, a cutoff of 0.5 is assumed
    preds = pd.concat(preds)

    # Adapt metric depending on model type
    if classifier:
        preds["pred_labels"] = preds["pred"] > pred_cutoff
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
