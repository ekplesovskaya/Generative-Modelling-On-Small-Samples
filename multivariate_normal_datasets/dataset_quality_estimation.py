# This module estimates the quality of synthetic datasets generated for samples
# from multivariate normal and normal mixture distributions

import random
import pickle
import pandas as pd
from tools.gen_param_for_norm_distr import get_list_of_random_generators
from tools.generative_model_estimation import stats_kde
from tools.generative_model_sampling import simple_sample_stats_procedure
from tools.dataset_similarity_metrics import calc_metrics, zu_overfitting_statistic
import numpy as np


def get_data(rg, vol, rs):
    """Generates random sample for a given distribution

    :param rg: multivariate random variable
    :param vol: sample size
    :param rs: random state
    :return data: random sample
    """
    if isinstance(rg, list):
        sample_list = []
        for rv in rg:
            sample_list.append(rv.rvs(int(vol/2), random_state=rs))
        data = np.vstack((sample_list))
    else:
        data = rg.rvs(vol, random_state=rs)
    data = pd.DataFrame(data)
    return data


def generate_dataset_and_calc_metrics(vol_list, n_var_list, dist_list):
    """Generates synthetic datasets from KDE and estimates the quality of samples

    :param vol_list: list of train data volumes
    :param n_var_list: list of train data dimensionality
    :param dist_list: list of distribution types
    :return result: dataset similarity indicators
    """
    result = {}
    for dist_type in dist_list:
        result[dist_type] = {}
        random.seed(42)
        for vol in vol_list:
            print("Sample volume:", vol)
            result[dist_type][vol] = {}
            rng = np.random.default_rng(42)
            np.random.seed(45)
            for n_var in n_var_list:
                print("Sample dimensionality:", n_var)
                result[dist_type][vol][n_var] = {}
                random_generators = get_list_of_random_generators(dist_type, n_var, 10, rng)
                for j, rg in enumerate(random_generators):
                    print("Distribution:", j)
                    data = get_data(rg, vol, 42)
                    kde = stats_kde(data, "cv_ml", "c"*data.shape[1], efficient=True)
                    seed_val = random.sample(list(range(100000)), 10)
                    sampled_data = simple_sample_stats_procedure(kde, vol, seed_val, data.columns, None, None)
                    result[dist_type][vol][n_var][j] = calc_metrics(data, sampled_data, "norm_dataset")
                    result[dist_type][vol][n_var][j]["Zu"] = []
                    for i, seed in enumerate(seed_val):
                        test_data = get_data(rg, vol, seed)
                        Zu_stat = zu_overfitting_statistic(test_data, sampled_data[i], data)
                        result[dist_type][vol][n_var][j]["Zu"].append(Zu_stat)
    return result


if __name__ == '__main__':
    vol_list = [500, 1000, 2000, 3000, 4000, 5000]
    n_var_list = [2, 4, 6]
    dist_list = ["normal", "normal_mixture"]
    result = generate_dataset_and_calc_metrics(vol_list, n_var_list, dist_list)
    with open('generation_results_for_multivariate_normal_datasets.pickle', "wb") as pickle_file:
        pickle.dump(result,  pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_file.close()