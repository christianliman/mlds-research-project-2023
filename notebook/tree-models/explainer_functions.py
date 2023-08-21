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

# Helper function to create model explanation and store visualizations
def ExplainModel(model, res, outcome, selected_features, figpath, classifier=True):
    ## DOCUMENTATION TO BE ADDED
    # Construct explainer and (exact) SHAP values
    expl, shapvals = CreateSHAPExplainer(model, res, outcome, 
        classifier, covars=covars)

    # Create dependence plot for selected features
    for c in selected_features:
        f1 = GenerateDependencePlot(model, res, c, outcome, shapvals, 
            classifier=classifier, shap_plot=True, covars=covars)
        f1.savefig(figpath + "dependence_{}.pdf".format(c))

    # Create beeswarm plot
    plt.clf()
    _ = shap.plots.beeswarm(shapvals, show=False)
    plt.savefig(figpath + "beeswarm.pdf")

    return shapvals


# Implement SHAP explainer wrapper
def CreateSHAPExplainer(model, res, outcome, classifier=True, covars=None):
    ## DOCUMENTATION TO BE ADDED
    # Get observed data only for the background distribution
    any_missingflag = res["missingflag"].any(axis=1)
    if covars is None:
        Xobs = res["imp"][0].drop(outcome, axis=1)[any_missingflag]
    else:
        Xobs = res["imp"][0][covars][any_missingflag]
    background_data = shap.maskers.Independent(Xobs, max_samples=100)
    
    # Construct SHAP explainer
    if classifier:
        pred_fn = lambda x: model.predict_proba(x)[:, 1]
    else:
        pred_fn = lambda x: model.predict(x)
    explainer = shap.Explainer(pred_fn, background_data)
    
    # Calculate SHAP values on the entire observed data
    shap_values = explainer(Xobs)
    
    return explainer, shap_values


# Create wrapper for SHAP or partial dependence plot
def GenerateDependencePlot(model, res, feature, outcome, shap_values, classifier=True, shap_plot=False, ax=None, covars=None):
    ## DOCUMENTATION TO BE ADDED
    # Create figure object if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), ncols=1, nrows=1)
    else:
        fig = None
        
    # Get observed data only
    any_missingflag = res["missingflag"].any(axis=1)
    if covars is None:
        Xobs = res["imp"][0].drop(outcome, axis=1)[any_missingflag]
    else:
        Xobs = res["imp"][0][covars][any_missingflag]
    
    if shap_plot:
        # Construct SHAP dependence plot
        shap.plots.scatter(
            shap_values[:, feature], show=False, ax=ax
        )
    else:
        # Construct partial dependence plot
        # MIGHT NEED TO CHANGE THE PREDICT FUNCTION LATER
        if classifier:
            pred_fn = lambda x: model.predict_proba(x)[:, 1]
        else:
            pred_fn = lambda x: model.predict(x)
        shap.partial_dependence_plot(
            feature, pred_fn, Xobs, model_expected_value=True, 
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