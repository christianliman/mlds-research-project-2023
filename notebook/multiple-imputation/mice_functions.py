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


def ImputeRandomSample(data, seed=None):
    """
    Impute missing values using one of the observed values for a given variable 
    (univariate imputation).
    This function iterates over all columns in a data frame and imputes as necessary.
    
    Parameters
    ----------
    data : Pandas DataFrame
        Data frame to be imputed
    seed : None, optional
        Random seed for reproducibility
    
    Returns
    -------
    Pandas DataFrame
        Imputed data frame
    """
    # Copy data frame
    imp = data.copy()
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Iterate over all variables in the data frame
    for c in data.columns:
        # Find rows to be imputed and number of missing observations
        mr = data[c].isna()
        n = mr.sum()
        
        # Collect observed data for sampling
        obs = data[c].dropna().values
        
        # Impute using random samples
        # We assume that the data is not fully missing but only partially
        imp.loc[mr, c] = np.random.choice(obs, size=n, replace=True)
        
    return imp


def ImputePMM(data, missingflag, d=5, k=1e-5, seed=None, targets=None):
    """
    Perform imputation using the predictive mean matching (PMM) method. 
    If the target variables are not specified, it assumes that all variables 
    are continuous and can be modelled using the Bayesian linear model.
    
    Parameters
    ----------
    data : Pandas DataFrame
        Data frame to be imputed
    missingflag : Pandas DataFrame
        A boolean data frame with the exact same dimension as `data`, with an
        indicator that shows if a given observation and variable is missing in
        `data`
    d : int, optional
        Number of donors in the donor set (default = 5)
    k : float, optional
        Ridge parameter for numerical stability (default = 1e-5)
    seed : int, optional
        Random seed
    targets : list, optional
        List of target variables to be imputed (assumed to be continuous)
    
    Returns
    -------
    Pandas DataFrame
        Imputed data frame
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Iterate over all columns
    if targets is None:
        targets = data.columns
    for i, c in enumerate(targets):
        # Separate out variable to be imputed and predictors - with their respective
        # flags
        y = data[c]
        yflag = missingflag[c]
        X = data.drop(c, axis=1)
        Xflag = missingflag.drop(c, axis=1)
        
        # Skip if no missing value
        if yflag.sum() == 0:
            continue
        
        # Separate out observed and missing
        Xobs = X[~yflag].values
        Xmis = X[yflag].values
        yobs = y[~yflag].values
        ymis = y[yflag].values
        
        # Calculate regression weights (Algorithm 3.1)
        S = np.transpose(Xobs) @ Xobs
        V = np.linalg.inv(S + k * np.diag(np.diag(S)))
        bhat = V @ np.transpose(Xobs) @ yobs
        
        # Calculate noise variance
        df = len(yobs) - Xobs.shape[1]
        gdot = np.random.chisquare(df)
        res = yobs - Xobs @ bhat
        sigdot = np.sqrt(np.transpose(res) @ res / gdot)
        
        # Draw beta from the posterior distribution
        z1 = np.random.normal(size=Xobs.shape[1])
        bdot = bhat + sigdot * z1 @ np.linalg.cholesky(V)
        
        # Calculate the imputed values and overwrite data matrix
        # NOTE: This is the Bayesian linear model approach, not PMM
        #z2 = np.random.normal(size=len(ymis))
        #yimp = Xmis @ bdot + z2 * sigdot
        #data.loc[yflag, c] = yimp
        
        # Calculate distances
        eta = np.subtract.outer(np.dot(Xobs, bhat).ravel(), np.dot(Xmis, bdot).ravel())
        eta = np.abs(eta)
        
        # Identify donor sets for each missing value
        ind = np.argsort(eta, axis=0)
        donorind = ind[:d, :]
        
        # Draw random donor for each missing value
        selectedind = np.random.randint(0, d, size=len(ymis))
        selecteddonorind = np.diag(donorind[np.ix_(selectedind, np.arange(len(ymis)))])
        yimp = yobs[selecteddonorind]
        
        # Overwrite data matrix
        data.loc[yflag, c] = yimp

    return data


def ImputeLogRegBoot(data, missingflag, d=5, k=1e-5, seed=None, targets=None):
    """
    Perform imputation using the logistic regression method with bootstrapping. 
    Note that target columns should be specified if not all variables are binary.
    Note that this implementation also assumes that the binary variables have
    been encoded as 0s and 1s.
    
    Parameters
    ----------
    data : Pandas DataFrame
        Data frame to be imputed
    missingflag : Pandas DataFrame
        A boolean data frame with the exact same dimension as `data`, with an
        indicator that shows if a given observation and variable is missing in
        `data`
    d : int, optional
        Number of donors in the donor set (default = 5)
    k : float, optional
        Ridge parameter for numerical stability (default = 1e-5)
    seed : int, optional
        Random seed
    targets : list, optional
        Target columns to be imputed
    
    Returns
    -------
    Pandas DataFrame
        Imputed data frame
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Iterate over all columns
    if targets is None:
        targets = data.columns
    for i, c in enumerate(targets):
        # Separate out variable to be imputed and predictors - with their respective
        # flags
        y = data[c]
        yflag = missingflag[c]
        X = data.drop(c, axis=1)
        Xflag = missingflag.drop(c, axis=1)
        
        # Skip if no missing value
        if yflag.sum() == 0:
            continue
        
        # Separate out observed and missing
        Xobs = X[~yflag].values
        Xmis = X[yflag].values
        yobs = y[~yflag].values
        ymis = y[yflag].values
        
        # Stop process if y is not binary
        if len(np.unique(yobs)) > 2:
            raise ValueError("Column {} has {} unique values".format(c, len(np.unique(yobs))))
        
        # Resample observed data using bootstrap
        resampled_idx = np.random.choice(np.arange(yobs.shape[0]), size=yobs.shape[0], 
                                         replace=True)
        Xobs1 = Xobs[resampled_idx, :]
        yobs1 = yobs[resampled_idx]
        
        # Train logistic regression model
        Xobs1 = sm.add_constant(Xobs1)
        lr_model = sm.Logit(yobs1, Xobs1).fit(disp=0) # suppress optim message
        
        # Predict probabilities using the fitted model
        pmis = lr_model.predict(sm.add_constant(Xmis))
        
        # Generate imputed binary values based on probabilities
        yimp = (np.random.uniform(size=ymis.shape[0]) < pmis).astype(float)
        
        # Overwrite data matrix
        data.loc[yflag, c] = yimp

    return data


def ImputeLogRegAugment(data, missingflag, d=5, k=1e-5, seed=None, targets=None):
    """
    Perform imputation using the logistic regression method with data augmentation
    according to White, Daniel, and Royston (2010).
    Note that target columns should be specified if not all variables are binary.
    Note that this implementation also assumes that the binary variables have
    been encoded as 0s and 1s.
    A key difference against R implementation is the use of standard binomial
    instead of quasibinomial.
    
    Parameters
    ----------
    data : Pandas DataFrame
        Data frame to be imputed
    missingflag : Pandas DataFrame
        A boolean data frame with the exact same dimension as `data`, with an
        indicator that shows if a given observation and variable is missing in
        `data`
    d : int, optional
        Number of donors in the donor set (default = 5)
    k : float, optional
        Ridge parameter for numerical stability (default = 1e-5)
    seed : None, optional
        Random seed
    targets : None, optional
        Target columns to be imputed
    
    Returns
    -------
    Pandas DataFrame
        Imputed data frame
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    
    # Iterate over all columns
    if targets is None:
        targets = data.columns
    for i, c in enumerate(targets):
        # Separate out variable to be imputed and predictors - with their respective
        # flags
        y = data[c]
        yflag = missingflag[c]
        X = data.drop(c, axis=1)
        Xflag = missingflag.drop(c, axis=1)
        
        # Skip if no missing value
        if yflag.sum() == 0:
            continue
        
        # Perform data augmentation
        Xaug, yaug, Xflagaug, yflagaug = _augment_data(X, y, Xflag, yflag)
        
        # Separate out observed and missing
        Xobs = Xaug[~yflagaug].values
        Xmis = Xaug[yflagaug].values
        yobs = yaug[~yflagaug].values
        ymis = yaug[yflagaug].values
        
        # Stop process if y is not binary
        if len(np.unique(yobs)) > 2:
            raise ValueError("Column {} has {} unique values".format(c, len(np.unique(yobs))))
        
        # Train logistic regression model
        Xobs = sm.add_constant(Xobs)
        lr_model = sm.Logit(yobs, Xobs).fit(disp=0) # suppress optim message
        
        # Sample betas from (estimated) posterior distribution
        cov_unscaled = lr_model.cov_params(scale=False)
        cov_sqrt = linalg.cholesky(cov_unscaled, lower=True)
        betas = lr_model.params.values
        beta_star = betas + np.dot(cov_sqrt, np.random.normal(len(betas)))
        
        # Predict probabilities using betas
        pmis = expit(np.dot(sm.add_constant(Xmis), beta_star))
        
        # Generate imputed binary values based on probabilities
        yimp = (np.random.uniform(size=ymis.shape[0]) < pmis).astype(float)
        
        # Overwrite data matrix
        data.loc[yflag, c] = yimp

    return data


def _augment_data(X, y, Xflag, yflag):
    # Helper function for data augmentation based on White, Daniel, Royston (2010)
    
    # Get number of categories and number of predictors
    p = X.shape[1]
    k = y.dropna().nunique()
    nr = 2 * p * k
    
    # Calculate column wise mean and SD, then construct a matrix
    mu = X.mean()
    sig = X.std(ddof=1) # R implements sample variance by default, but not Python
    mu_mtrx = np.tile(mu.values.reshape(1, -1), (nr, 1))
    sig_mtrx = np.tile(sig.values.reshape(1, -1), (nr, 1))
    
    # Create shift matrix and outcome label vector
    shift_mtrx = linalg.block_diag(*tuple([[[.5], [-.5]]] * p))
    shift_mtrx = np.tile(shift_mtrx, (k, 1))
    ynew = np.repeat(np.arange(k), 2 * p)
    
    # Create augmented data
    aug = mu_mtrx + shift_mtrx * sig_mtrx
    aug = pd.DataFrame(aug, columns=X.columns)
    ynew = pd.Series(ynew)
    augflag = pd.DataFrame(np.zeros(nr, ).astype(bool), columns=X.columns)
    
    # Augment to the original data
    Xaug = pd.concat([X, aug], ignore_index=True)
    yaug = pd.concat([y, ynew], ignore_index=True)
    Xflagaug = pd.concat([Xflag, augflag], ignore_index=True)
    yflagaug = pd.concat([yflag, pd.Series(np.zeros(nr).astype(bool))], ignore_index=True)
    
    return Xaug, yaug, Xflagaug, yflagaug


def getImputedStats(data, missingflag):
    # Helper function to calculate statistics of imputed data
    # Initialize arrays for mean and SD
    mu = np.zeros(data.shape[1])
    sigma = np.zeros(data.shape[1])
    
    for i, c in enumerate(data.columns):
        # Extract missing data
        miss = data.loc[missingflag[c], c]
        
        # Get statistics
        mu[i] = miss.mean()
        sigma[i] = miss.std(ddof=1)
    
    return mu, sigma


def MICEPMM(data, m=10, maxit=5, d=5, k=1e-5, seed=123):
    """
    Implement multivariate imputation by chained equations (MICE) using 
    predictive mean matching (PMM) method. This function assumes that all
    variables are continuous and can be modelled using a Bayesian linear model.
    Furthermore, it assumes a fully conditional specification (FCS), which makes
    the original MICE framework.
    
    Parameters
    ----------
    data : Pandas DataFrame
        Data frame to be imputed
    m : int, optional
        Number of multiply imputed data to be generated (default = 10)
    maxit : int, optional
        Maximum number of iterations for the MICE algorithm (default = 5)
    d : int, optional
        Number of donors in the donor set (default = 5)
    k : float, optional
        Ridge parameter for numerical stability (default = 1e-5)
    seed : int, optional
        Random seed
    
    Returns
    -------
    dict
        Python dictionary with the imputed data (`imp`), missing data flag (`missingflag`),
        and the chain statistics (`chainmean` and `chainstd`)
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Create flags for missing value
    missingflag = data.isna()
    
    # Make m copies of the data
    imp = []
    for _ in range(m):
        # Initialize using random sample
        imp.append(ImputeRandomSample(data))
    
    # Initialize chain statistics
    chainmean = np.empty((data.shape[1], m, maxit+1))
    chainstd = np.empty((data.shape[1], m, maxit+1))
    for i in range(m):
        chainmean[:, i, 0], chainstd[:, i, 0] = getImputedStats(imp[i], missingflag)
    
    # Iterate over maxit
    for j in tqdm(range(maxit)):
        #print("Iteration {}".format(j))
        
        for i in range(m):
            # Impute using PMM
            imp[i] = ImputePMM(imp[i], missingflag, d=d, k=k)
        
            # Calculate updated chain statistics
            chainmean[:, i, j+1], chainstd[:, i, j+1] = getImputedStats(imp[i], missingflag)
    
    # Return multiply imputed data and chain statistics
    res = {
        "imp": imp,
        "missingflag": missingflag,
        "chainmean": chainmean,
        "chainstd": chainstd
    }
    return res


def MICELogReg(data, m=10, maxit=5, d=5, k=1e-5, seed=123, method="boot"):
    """
    Implement multivariate imputation by chained equations (MICE) using 
    logistic regression method. This function assumes that all
    variables are binary and are encoded as 0s and 1s. Two methods of logistic
    regression imputation are supported, one based on bootstrap approach and the
    other based on Bayesian approach (with data augmentation).
    Furthermore, it assumes a fully conditional specification (FCS), which makes
    the original MICE framework.
    
    Parameters
    ----------
    data : Pandas DataFrame
        Data frame to be imputed
    m : int, optional
        Number of multiply imputed data to be generated (default = 10)
    maxit : int, optional
        Maximum number of iterations for the MICE algorithm (default = 5)
    d : int, optional
        Number of donors in the donor set (default = 5)
    k : float, optional
        Ridge parameter for numerical stability (default = 1e-5)
    seed : int, optional
        Random seed
    method : str, optional
        Logistic regression method, "boot" for bootstrap approach (default) and
        "bayes" for Bayesian approach
    
    Returns
    -------
    dict
        Python dictionary with the imputed data (`imp`), missing data flag (`missingflag`),
        and the chain statistics (`chainmean` and `chainstd`)
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Create flags for missing value
    missingflag = data.isna()
    
    # Make m copies of the data
    imp = []
    for _ in range(m):
        # Initialize using random sample
        imp.append(ImputeRandomSample(data))
    
    # Initialize chain statistics
    chainmean = np.empty((data.shape[1], m, maxit+1))
    chainstd = np.empty((data.shape[1], m, maxit+1))
    for i in range(m):
        chainmean[:, i, 0], chainstd[:, i, 0] = getImputedStats(imp[i], missingflag)
    
    # Iterate over maxit
    for j in tqdm(range(maxit)):
        #print("Iteration {}".format(j))
        
        for i in range(m):
            # Impute using appropriate method
            if method == "boot":
                imp[i] = ImputeLogRegBoot(imp[i], missingflag, d=d, k=k)
            elif method == "bayes":
                imp[i] = ImputeLogRegBayes(imp[i], missingflag, d=d, k=k)
            else:
                raise ValueError("Invalid method: {}".format(method))
        
            # Calculate updated chain statistics
            chainmean[:, i, j+1], chainstd[:, i, j+1] = getImputedStats(imp[i], missingflag)
    
    # Return multiply imputed data and chain statistics
    res = {
        "imp": imp,
        "missingflag": missingflag,
        "chainmean": chainmean,
        "chainstd": chainstd
    }
    return res


def MICE(data, targets_cat, targets_num, m=10, maxit=5, d=5, k=1e-5, seed=123, method_cat="boot", method_num="pmm"):
    """
    Implement multivariate imputation by chained equations (MICE). This function
    requires continuous and categorical target variables to be explicitly specified.
    Appropriate imputation method will then be applied to each target variable type.
    
    Parameters
    ----------
    data : Pandas DataFrame
        Data frame to be imputed
    targets_cat : list
        List of categorical target variables to be imputed. Pass an empty list if
        no categorical variable is to be imputed
    targets_num : list
        List of numeric target variables to be imputed. Pass an empty list if
        no numeric variable is to be imputed
    m : int, optional
        Number of multiply imputed data to be generated (default = 10)
    maxit : int, optional
        Maximum number of iterations for the MICE algorithm (default = 5)
    d : int, optional
        Number of donors in the donor set (default = 5)
    k : float, optional
        Ridge parameter for numerical stability (default = 1e-5)
    seed : int, optional
        Random seed
    method_cat : str, optional
        Imputation method for categorical target variables. Supports "boot" (default),
        "bayes", and "pmm"
    method_num : str, optional
        Imputation method for numeric target variables. Only supports "pmm" (default)
    
    Returns
    -------
    dict
        Python dictionary with the imputed data (`imp`), missing data flag (`missingflag`),
        and the chain statistics (`chainmean` and `chainstd`)
    """
    # Check validity of imputation methods
    assert method_cat in ["pmm", "boot", "bayes"], "Invalid method : {}".format(method_cat)
    assert method_num in ["pmm"], "Invalid method: {}".format(method_num)
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Create flags for missing value
    missingflag = data.isna()
    
    # Make m copies of the data
    imp = []
    for _ in range(m):
        # Initialize using random sample
        imp.append(ImputeRandomSample(data))
    
    # Initialize chain statistics
    chainmean = np.empty((data.shape[1], m, maxit+1))
    chainstd = np.empty((data.shape[1], m, maxit+1))
    for i in range(m):
        chainmean[:, i, 0], chainstd[:, i, 0] = getImputedStats(imp[i], missingflag)
    
    # Define dictionary to map appropriate functions
    imputefunc = {
        "pmm" : ImputePMM,
        "boot": ImputeLogRegBoot,
        "bayes": ImputeLogRegAugment,
    }
    
    # Iterate over maxit
    for j in tqdm(range(maxit)):
        #print("Iteration {}".format(j))
        
        for i in range(m):
            # Impute using appropriate method for each data type
            if len(targets_cat) > 0:
                imp[i] = imputefunc[method_cat](imp[i], missingflag, d=d, k=k, 
                                                targets=targets_cat)
            if len(targets_num) > 0:
                imp[i] = imputefunc[method_num](imp[i], missingflag, d=d, k=k, 
                                                targets=targets_num)
        
            # Calculate updated chain statistics
            chainmean[:, i, j+1], chainstd[:, i, j+1] = getImputedStats(imp[i], missingflag)
    
    # Return multiply imputed data and chain statistics
    res = {
        "imp": imp,
        "missingflag": missingflag,
        "chainmean": chainmean,
        "chainstd": chainstd
    }
    return res


def getImputedData(res, colname):
    """
    Helper function to get a data frame of imputed values for a given variable
    
    Parameters
    ----------
    res : dict
        Python dictionary returned by `MICEPMM`
    colname : str
        Variable for which the imputed data is to be shown
    
    Returns
    -------
    Pandas DataFrame
        Data frame showing observations with missing data only. Each column represents
        the output from each imputation model
    """
    # Retrieve relevant data
    implist, missingflag = res["imp"], res["missingflag"]
    
    # Extract relevant missing flag
    yflag = missingflag[colname]
    
    # Iterate over all imputed data
    impcombined = []
    for i, data in enumerate(implist):
        # Extract imputed values
        data = data.loc[yflag]
        data = data.rename(columns={colname: colname + str(i)})
        impcombined.append(data[colname + str(i)])
    
    impcombined = pd.concat(impcombined, axis=1)
    
    return impcombined


def plotImputedData(res, colname):
    """
    Alternative helper function to construct a strip plot of imputed values for a given variable
    
    Parameters
    ----------
    res : dict
        Python dictionary returned by `MICEPMM`
    colname : str
        Variable for which the imputed data is to be shown
    
    Returns
    -------
    matplotlib.pyplot.figure
        Figure showing the strip plots
    """

    # Retrieve relevant data
    implist, missingflag = res["imp"], res["missingflag"]
    
    # Extract relevant missing flag
    yflag = missingflag[colname]
    
    # Iterate over all imputed data
    impdata, impnums, impflag = [], [], []
    for i, data in enumerate(implist):
        # Extract relevant column
        impdata.append(data[colname].values)
        impnums.append(np.ones(data.shape[0], dtype=np.int32) * (i+1))
        impflag.append(yflag.values)
    
    # Construct combined data for plotting
    impdata = np.concatenate(impdata)
    impnums = np.concatenate(impnums)
    impflag = np.concatenate(impflag)
    impdf = pd.DataFrame({
        colname: impdata,
        "Imputation number": impnums,
        "Imputed": impflag,
        })

    # Placeholder for plotting
    fig, ax = plt.subplots()

    # Generate strip plot
    sns.stripplot(data=impdf, x="Imputation number", y=colname, hue="Imputed", jitter=True, 
        ax=ax)

    return fig


def ChainStatsViz(res, maxvar=3):
    """
    Constructs a trace plot of chain statistics based on the `MICEPMM` output
    
    Parameters
    ----------
    res : dict
        Python dictionary returned by `MICEPMM`
    maxvar : int, optional
        Maximum number of variables to be visualized (default = 3). Note that only
        variables with missing data will be shown
    
    Returns
    -------
    matplotlib.pyplot.figure
        Figure showing the trace plots
    """
    # Retrieve chain mean and standard deviation
    chainmean, chainstd = res["chainmean"], res["chainstd"]
    
    # Pick first maxvar variables with missing data
    allvars = pd.Series(res["missingflag"].columns.values)
    missingvars = res["missingflag"].columns[res["missingflag"].sum() > 0]
    if len(missingvars) < maxvar:
        maxvar = len(missingvars)
    else:
        missingvars = missingvars[:maxvar]
    missingvarsind = allvars[allvars.isin(missingvars)].index
    
    # Get number of imputed data
    m = len(res["imp"])
    
    # Placeholder for plotting
    fig, axs = plt.subplots(figsize=(8, maxvar*3), ncols=2, nrows=maxvar, 
                            sharex=True)
    cmap = cm.get_cmap("jet", 10) # If m > 10, it will rotate back to start
    
    # Generate plot for each variable and imputation
    for i, idx in enumerate(missingvarsind):
        # Plot chain mean
        for j in range(m):
            axs[i, 0].plot(chainmean[i, j, :], color=cmap(j % 10), alpha=0.7)
        axs[i, 0].set_title("{}: mean".format(allvars[idx]))
        
        # Plot chain SD
        for j in range(m):
            axs[i, 1].plot(chainstd[i, j, :], color=cmap(j % 10), alpha=0.7)
        axs[i, 1].set_title("{}: SD".format(allvars[idx]))
    
    return fig
