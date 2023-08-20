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


# Implement SHAP explainer wrapper for classifiers
def CreateSHAPExplainer(model, res, outcome):
    ## DOCUMENTATION TO BE ADDED
    # Get observed data only for the background distribution
    any_missingflag = res["missingflag"].any(axis=1)
    Xobs = res["imp"][0].drop(outcome, axis=1)[any_missingflag]
    background_data = shap.maskers.Independent(Xobs, max_samples=100)
    
    # Construct SHAP explainer
    # TO UPDATE WITH APPROPRIATE FUNCTION
    pos_prob_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(pos_prob_fn, background_data)
    
    # Calculate SHAP values on the entire observed data
    shap_values = explainer(Xobs)
    
    return explainer, shap_values


# Create wrapper for SHAP or partial dependence plot
def GenerateDependencePlot(model, res, feature, outcome, shap_values, 
                           shap_plot=False, ax=None):
    ## DOCUMENTATION TO BE ADDED
    # Create figure object if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)
    else:
        fig = None
        
    # Get observed data only
    any_missingflag = res["missingflag"].any(axis=1)
    Xobs = res["imp"][0].drop(outcome, axis=1)[any_missingflag]
    
    if shap_plot:
        # Construct SHAP dependence plot
        shap.plots.scatter(
            shap_values[:, feature], show=False, colors=shap_values, ax=ax
        )
    else:
        # Construct partial dependence plot
        # MIGHT NEED TO CHANGE THE PREDICT FUNCTION LATER
        pos_prob_fn = lambda x: model.predict_proba(x)[:, 1]
        shap.partial_dependence_plot(
            feature, pos_prob_fn, Xobs, model_expected_value=True, 
            feature_expected_value=True, show=False, ice=False, ax=ax
        )
    
    if fig is None:
        return ax
    else:
        return fig


# Create wrapper for beeswarm plot
def GenerateBeeswarmPlot(model, res, shap_values, ax=None):
    _ = shap.plots.beeswarm(shap_values, show=False)
    
    # Need to call plt.savefig in the same cell to save the plot
    
    return