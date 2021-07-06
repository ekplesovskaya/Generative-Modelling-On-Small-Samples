# This module implements sampling from generative models and post-processing of datasets

import pandas as pd
import numpy as np
import torch
import random
from tools.generative_model_estimation import fit_model


def round_func(val, data_info):
    """Rounds generated value to the required number of decimal points

    :param val: value
    :param data_info: number of decimal points to round
    :return val: rounded value
    """
    if data_info != None:
        val = round(val, data_info)
    return val


def lim_func(val, data_info_high, data_info_low):
    """Limits generated value

    :param val: value
    :param data_info_high: upper limit
    :return data_info_low: lower limit
    """
    if data_info_high != None:
        if val > data_info_high:
            val = data_info_high
    if data_info_low != None:
        if val < data_info_low:
            val = data_info_low
    return val


def post_processing(sampled_data, data):
    """Post-processing function

    :param sampled_data: synthetic dataset
    :param data: sampling parameters
    :return sampled_data: post-processed synthetic dataset
    """
    for col in sampled_data.columns:
        sampled_data[col] = sampled_data[col].apply(lim_func, args=(data["lim"][col]["high"], data["lim"][col]["low"]))
        sampled_data[col] = sampled_data[col].apply(round_func, args=[data["round"][col]])
    return sampled_data


def simple_sample_procedure(model, sample_len, seed_list, cols, scaling, data):
    """Simple sampling function

    :param model: generative model
    :param sample_len: number of samples in synthetic dataset
    :param seed_list: list of seeds for sampling
    :param cols: dataset columns
    :param scaling: scaler for synthetic dataset
    :param data: sampling parameters
    :return sampled_data_list: list of synthetic datasets
    """
    sampled_data_list = []
    for seed in seed_list:
        sampled_data = model.sample(n_samples=sample_len, random_state=seed)
        if scaling:
            sampled_data = scaling.inverse_transform(sampled_data)
        sampled_data = pd.DataFrame(sampled_data, columns=cols)
        sampled_data = post_processing(sampled_data, data)
        sampled_data_list.append(sampled_data)
    return sampled_data_list


def sample_stats(kde, size, seed):
    """Sampling from Statsmodel's KDE

    :param kde: KDE
    :param size: number of samples in synthetic dataset
    :param seed: seed
    :return sampled_data: generated data
    """
    rng = np.random.RandomState(seed)
    n, d = kde.data.shape
    indices = rng.randint(0, n, size)
    cov = np.diag(kde.bw)**2
    means = kde.data[indices, :]
    norm = rng.multivariate_normal(np.zeros(d), cov, size)
    sampled_data = np.transpose(means + norm).T
    return sampled_data


def simple_sample_stats_procedure(model, sample_len, seed_list, cols, scaling, data):
    """Sampling synthetic datasets from Statsmodel's KDE

    :param model: generative model
    :param sample_len: number of samples in synthetic dataset
    :param seed_list: list of seeds for sampling
    :param cols: dataset columns
    :param scaling: scaler for synthetic dataset
    :param data: sampling parameters
    :return sampled_data_list: list of synthetic datasets
    """
    sampled_data_list = []
    for seed in seed_list:
        sampled_data = sample_stats(model, sample_len, seed)
        if scaling:
            sampled_data = scaling.inverse_transform(sampled_data)
        sampled_data = pd.DataFrame(sampled_data, columns=cols)
        if data:
            sampled_data = post_processing(sampled_data, data)
        sampled_data_list.append(sampled_data)
    return sampled_data_list


def gmm_sample_procedure(model, sample_len, cols, scaling, num_samples, data):
    """Sampling from GMM

    :param model: generative model
    :param sample_len: number of samples in synthetic dataset
    :param cols: dataset columns
    :param scaling: scaler for synthetic dataset
    :param num_samples: number of synthetic datasets
    :param data: sampling parameters
    :return sampled_data_list: list of synthetic datasets
    """
    sampled_data_list = []
    n_samples = model.sample(sample_len*num_samples)[0]
    for i in range(num_samples):
        sampled_data = n_samples[(i*sample_len):((i+1)*sample_len)]
        if scaling:
            sampled_data = scaling.inverse_transform(sampled_data)
        sampled_data = pd.DataFrame(sampled_data, columns=cols)
        sampled_data = post_processing(sampled_data, data)
        sampled_data_list.append(sampled_data)
    return sampled_data_list


def sample_sdv_procedure(model, sample_len, seed_list, cols, scaling, data):
    """Sampling from SDV library model

    :param model: generative model
    :param sample_len: number of samples in synthetic dataset
    :param seed_list: list of seeds for sampling
    :param cols: dataset columns
    :param scaling: scaler for synthetic dataset
    :param data: sampling parameters
    :return sampled_data_list: list of synthetic datasets
    """
    sampled_data_list = []
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)
        sampled_data = model.sample(sample_len)
        if scaling:
            sampled_data = scaling.inverse_transform(sampled_data)
        sampled_data = pd.DataFrame(sampled_data, columns=cols)
        sampled_data = post_processing(sampled_data, data)
        sampled_data_list.append(sampled_data)
    return sampled_data_list


def get_sampled_data(model, sample_len, seed_list, method, cols, scaling, data):
    """Calls a sampling function

    :param model: generative model
    :param sample_len: number of samples in synthetic dataset
    :param seed_list: list of seeds for sampling
    :param method: generative model name
    :param cols: dataset columns
    :param scaling: scaler for synthetic dataset
    :param data: sampling parameters
    :return sampled_data_list: list of synthetic datasets
    """
    if method in ["sklearn_kde", "awkde"]:
        sampled_data_list = simple_sample_procedure(model, sample_len, seed_list, cols, scaling, data)
    elif method in ["kde_cv_ml", "kde_cv_ls"]:
        sampled_data_list = simple_sample_stats_procedure(model, sample_len, seed_list, cols, scaling, data)
    elif method in ["gmm", "bayesian_gmm"]:
        sampled_data_list = gmm_sample_procedure(model, sample_len, cols, scaling, len(seed_list), data)
    elif method in ["ctgan", "copula", "copulagan", "tvae"]:
        sampled_data_list = sample_sdv_procedure(model, sample_len, seed_list, cols, scaling, data)
    return sampled_data_list


def get_sample_for_pairwise_plot(gen_model, ind_data, ds):
    """Chooses generative model with the lowest ROC AUC and samples a dataset

    :param gen_model: generative algorithm name
    :param ind_data: dataset similarity indicators for the generative algorithm
    :param ds: dataset parameters
    :return sampled_data: synthetic dataset
    """
    iter = 0
    min_roc_auc = 1
    for iter_num in ind_data:
        min_roc_auc_iter = min(ind_data[iter_num]["c2st_roc_auc"])
        if min_roc_auc_iter < min_roc_auc:
            min_roc_auc = min_roc_auc_iter
            iter = iter_num
    sample_num = np.argmin(ind_data[iter]["c2st_roc_auc"])
    random.seed(42)
    seed_val = random.sample(list(range(100000)), 100)
    seed_val_cv = seed_val[:50][iter]
    seed_val_sampling = seed_val[sample_num]
    fitted_model = fit_model(gen_model, ds["data_scaled"], seed_val_cv)
    sampled_data = get_sampled_data(fitted_model, ds["len"], [seed_val_sampling], gen_model, ds["cols"], ds["scaler"], ds)[0]
    return sampled_data