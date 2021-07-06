# This module generates plots for dataset comparison

from tools.plot_functions import make_pairwise_plot, make_qq_and_pp_plots
import random
import numpy as np
from tools.gen_param_for_norm_distr import get_list_of_random_generators
from dataset_quality_estimation import get_data
from tools.generative_model_estimation import stats_kde
from tools.generative_model_sampling import simple_sample_stats_procedure


def get_data_for_plot(dist_type, vol, n_var):
    """Generates real and synthetic dataset

    :param dist_type: type of distribution
    :param vol: sample size
    :param n_var: dimensionality
    :return train_data: train dataset
    :return sampled_data: sampled dataset
    """
    random.seed(42)
    rng = np.random.default_rng(42)
    np.random.seed(45)
    random_generator = get_list_of_random_generators(dist_type, n_var, 1, rng)[0]
    train_data = get_data(random_generator, vol, 42)
    kde = stats_kde(train_data, "cv_ml", "c"*train_data.shape[1], efficient=True)
    sampled_data = simple_sample_stats_procedure(kde, vol, [42], train_data.columns, None, None)[0]
    return train_data, sampled_data


def get_axis_lim(df1, df2):
    """Outputs axis view limits

    :param df1: train dataset
    :param df2: sampled dataset
    :return var_range: axis view limits
    """
    min_series = np.hstack((df1.min(), df2.min()))
    max_series = np.hstack((df1.max(), df2.max()))
    if min(min_series) < 0:
        var_range = (min(min_series)*1.2, max(max_series)*1.2)
    else:
        var_range = (min(min_series)*0.8, max(max_series)*1.2)
    return var_range


if __name__ == '__main__':
    # plots for normal distribution
    train_data, sampled_data = get_data_for_plot("normal", 500, 4)
    var_range = get_axis_lim(train_data, sampled_data)
    make_pairwise_plot(train_data, var_range, "Train sample, 500 obs.", "Train_sample_normal")
    make_pairwise_plot(sampled_data, var_range, "Sample from KDE, 500 obs.", "KDE_sample_normal")
    # plots for normal mixture distribution
    train_data, sampled_data = get_data_for_plot("normal_mixture", 500, 4)
    var_range = get_axis_lim(train_data, sampled_data)
    make_pairwise_plot(train_data, var_range, "Train sample, 500 obs.", "Train_sample_normal_mixture")
    make_pairwise_plot(sampled_data, var_range, "Sample from KDE, 500 obs.", "KDE_sample_normal_mixture")
    # qq and pp plots for normal mixture distribution
    train_data, sampled_data = get_data_for_plot("normal_mixture", 500, 6)
    make_qq_and_pp_plots(train_data, sampled_data, "Normal mixture distribution, 500 obs.", "PP_and_QQ_plots_normal_mixture")


