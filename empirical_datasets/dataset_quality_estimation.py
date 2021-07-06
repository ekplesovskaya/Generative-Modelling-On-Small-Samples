# This module estimates the quality of synthetic samples from a range of generative models
# trained on datasets from the OpenML database and Scikit-learn library

from tools.get_empirical_dataset import get_dataset
import pickle
import random
from tools.generative_model_estimation import fit_model
from tools.generative_model_sampling import get_sampled_data
from tools.dataset_similarity_metrics import calc_metrics


def generate_dataset_and_calc_metrics(dataset_names, method_list):
    """Generates synthetic datasets from generative models and estimates the quality of samples

    :param dataset_names: list of dataset names
    :param method_list: list of generative algorithm names
    :return result: dataset similarity indicators
    """
    random.seed(42)
    seed_val = random.sample(list(range(100000)), 100)
    seed_val_cv = seed_val[:50]
    seed_val_sample = seed_val
    result = {}
    for ds_name in dataset_names:
        print("Dataset:", ds_name)
        ds = get_dataset(ds_name)
        result[ds_name] = {}
        for method_name in method_list:
            print("Algorithm:", method_name)
            result[ds_name][method_name] = {}
            if method_name in ["copula", "kde_cv_ml", "kde_cv_ls"]:
                seed_val_cv_method = [None]
            else:
                seed_val_cv_method = seed_val_cv
            for i, seed_cv in enumerate(seed_val_cv_method):
                gen_model = fit_model(method_name, ds["data_scaled"], seed_cv)
                sampled_data_list = get_sampled_data(gen_model, ds["len"], seed_val_sample,
                                                     method_name, ds["cols"], ds["scaler"], ds)
                result[ds_name][method_name][i] = calc_metrics(ds["data"], sampled_data_list, "emp_dataset")
    return result


if __name__ == '__main__':
    dataset_names = ["iris", "visualizing_galaxy", "visualizing_environmental"]
    method_list = ["sklearn_kde", "awkde", "kde_cv_ml", "kde_cv_ls", "gmm",
                  "bayesian_gmm", "ctgan", "copula", "copulagan", "tvae"]
    result = generate_dataset_and_calc_metrics(dataset_names, method_list)
    with open('generation_results_for_the_empirical_datasets.pickle', "wb") as pickle_file:
        pickle.dump(result,  pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_file.close()
