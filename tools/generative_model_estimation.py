# This module implements estimation of generative models used in the study

import numpy as np
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
from sdv.tabular import GaussianCopula, CTGAN, CopulaGAN, TVAE
import torch
from sklearn.model_selection import KFold
import math
from sdv.evaluation import evaluate
import awkde
from statsmodels.nonparametric.kernel_density import EstimatorSettings, KDEMultivariate
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

n_components_list = list(range(1, 5))
n_components_list.extend(list(range(5, 101, 5)))


def get_gmm_model(data, cv_rs):
    """Estimates GMM model

    :param data: train dataset
    :param cv_rs: random state for CV
    :return gmm: generative model
    """
    params = {'n_components': n_components_list,
              'covariance_type': ['full', 'tied', 'diag', 'spherical']}
    kf = KFold(n_splits=3, random_state=cv_rs, shuffle=True)
    grid = GridSearchCV(GaussianMixture(random_state=42), params, cv=kf)
    grid.fit(data)
    gmm = grid.best_estimator_
    return gmm


def get_bayesian_gmm_model(data, cv_rs):
    """Estimates Bayesian GMM model

    :param data: train dataset
    :param cv_rs: random state for CV
    :return gmm: generative model
    """
    params = {'n_components':  list(range(1, 11, 1)),
              'covariance_type': ['full', 'tied', 'diag', 'spherical'],
              'weight_concentration_prior_type': ['dirichlet_process', 'dirichlet_distribution'],
              'weight_concentration_prior': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 5000, 10000]}
    kf = KFold(n_splits=3, random_state=cv_rs, shuffle=True)
    grid = GridSearchCV(BayesianGaussianMixture(random_state=42), params, cv=kf)
    grid.fit(data)
    gmm = grid.best_estimator_
    return gmm


def get_copula_model(data):
    """Estimates Copula model

    :param data: train dataset
    :return model: generative model
    """
    np.random.seed(42)
    model = GaussianCopula(default_distribution="univariate")
    model.fit(data)
    return model


def get_ctgan_model(data, cv_rs):
    """Estimates CTGAN model

    :param data: train dataset
    :param cv_rs: random state for CV
    :return model: generative model
    """
    np.random.seed(42)
    torch.manual_seed(42)
    batch_size = math.ceil(len(data)/100)*100
    skf = KFold(n_splits=3, shuffle=True, random_state=cv_rs)
    res_metric = []
    epoch_list = [500, 1000, 1500, 2000]
    for epoch_v in epoch_list:
        metric_list = []
        for index in skf.split(data):
            x_train, x_test = data.iloc[index[0]], data.iloc[index[1]]
            model = CTGAN(epochs=epoch_v, batch_size=batch_size)
            model.fit(x_train)
            sub_metric_list = []
            for i in range(10):
                sampled_data = model.sample(len(x_test))
                sub_metric_list.append(evaluate(sampled_data, x_test))
            metric_list.append(np.mean(sub_metric_list))
        res_metric.append(np.mean(metric_list))
    opt_epoch = epoch_list[np.argmax(res_metric)]
    model = CTGAN(epochs=opt_epoch, batch_size=batch_size)
    model.fit(data)
    return model


def get_copulagan_model(data, cv_rs):
    """Estimates CopulaGAN model

    :param data: train dataset
    :param cv_rs: random state for CV
    :return model: generative model
    """
    np.random.seed(42)
    torch.manual_seed(42)
    batch_size = math.ceil(len(data)/100)*100
    skf = KFold(n_splits=3, shuffle=True, random_state=cv_rs)
    res_metric = []
    epoch_list = [500, 1000, 1500, 2000]
    for epoch_v in epoch_list:
        metric_list = []
        for index in skf.split(data):
            x_train, x_test = data.iloc[index[0]], data.iloc[index[1]]
            model = CopulaGAN(epochs=epoch_v, batch_size=batch_size)
            model.fit(x_train)
            sub_metric_list = []
            for i in range(10):
                sampled_data = model.sample(len(x_test))
                sub_metric_list.append(evaluate(sampled_data, x_test))
            metric_list.append(np.mean(sub_metric_list))
        res_metric.append(np.mean(metric_list))
    opt_epoch = epoch_list[np.argmax(res_metric)]
    model = CopulaGAN(epochs=opt_epoch, batch_size=batch_size)
    model.fit(data)
    return model


def get_tvae_model(data, cv_rs):
    """Estimates TVAE model

    :param data: train dataset
    :param cv_rs: random state for CV
    :return model: generative model
    """
    np.random.seed(42)
    torch.manual_seed(42)
    batch_size = math.ceil(len(data)/100)*100
    skf = KFold(n_splits=3, shuffle=True, random_state=cv_rs)
    res_metric = []
    epoch_list = [500, 1000, 1500, 2000]
    for epoch_v in epoch_list:
        metric_list = []
        for index in skf.split(data):
            x_train, x_test = data.iloc[index[0]], data.iloc[index[1]]
            model = TVAE(epochs=epoch_v, batch_size=batch_size)
            model.fit(x_train)
            sub_metric_list = []
            for i in range(10):
                sampled_data = model.sample(len(x_test))
                sub_metric_list.append(evaluate(sampled_data, x_test))
            metric_list.append(np.mean(sub_metric_list))
        res_metric.append(np.mean(metric_list))
    opt_epoch = epoch_list[np.argmax(res_metric)]
    model = TVAE(epochs=opt_epoch, batch_size=batch_size)
    model.fit(data)
    return model


def sklearn_kde(data, cv_rs):
    """Estimates KDE (sklearn implementation)

    :param data: train dataset
    :param cv_rs: random state for CV
    :return kde: generative model
    """
    kf = KFold(n_splits=3, random_state=cv_rs, shuffle=True)
    bw_range = np.arange(0.01, 1.01, 0.01)
    params = {'bandwidth': bw_range}
    grid = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=kf)
    grid.fit(data)
    if grid.best_estimator_.bandwidth == bw_range[0]:
        bw_range = np.arange(0.0001, 0.01, 0.0001)
        params = {'bandwidth': bw_range}
        grid = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=kf)
        grid.fit(data)
    elif grid.best_estimator_.bandwidth == bw_range[-1]:
        bw_range = np.arange(1.01, 2.01, 0.01)
        params = {'bandwidth': bw_range}
        grid = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=kf)
        grid.fit(data)
    kde = grid.best_estimator_
    return kde


def awkde_kde(data, cv_rs):
    """Estimates AWKDE model

    :param data: train dataset
    :param cv_rs: random state for CV
    :return kde: generative model
    """
    kf = KFold(n_splits=3, random_state=cv_rs, shuffle=True)
    params = {'alpha': [0.1, 0.3, 0.5, 0.7, 0.9]}
    grid = GridSearchCV(awkde.GaussianKDE(), params, cv=kf)
    grid.fit(data)
    kde = grid.best_estimator_
    return kde


def stats_kde(data, method, var_type, efficient=False):
    """Estimates KDE (Statsmodels)

    :param data: train dataset
    :param method: bandwidth selection method
    :param var_type: type of variables
    :param efficient: settings for bandwidth estimation
    :return kde: generative model
    """
    if efficient:
        settings = EstimatorSettings(efficient=True, randomize=True)
        kde = KDEMultivariate(data, var_type, bw=method, defaults=settings)
    else:
        kde = KDEMultivariate(data, var_type, bw=method)
    return kde


def fit_model(gen_algorithm, data, cv_rs):
    """Calls a model estimation function

    :param gen_algorithm: generative algorithm name
    :param data: train dataset
    :param cv_rs: random state for CV
    :return model: generative model
    """
    if gen_algorithm == "sklearn_kde":
        model = sklearn_kde(data, cv_rs)
    elif gen_algorithm == "awkde":
        model = awkde_kde(data, cv_rs)
    elif gen_algorithm in ["kde_cv_ml", "kde_cv_ls"]:
        method_list = gen_algorithm.split("_")
        method_name = method_list[1]+"_"+method_list[2]
        model = stats_kde(data, method_name, "c"*data.shape[1])
    elif gen_algorithm == "gmm":
        model = get_gmm_model(data, cv_rs)
    elif gen_algorithm == "bayesian_gmm":
        model = get_bayesian_gmm_model(data, cv_rs)
    elif gen_algorithm == "ctgan":
        model = get_ctgan_model(data, cv_rs)
    elif gen_algorithm == "copula":
        model = get_copula_model(data)
    elif gen_algorithm == "copulagan":
        model = get_copulagan_model(data, cv_rs)
    elif gen_algorithm == "tvae":
        model = get_tvae_model(data, cv_rs)
    return model



