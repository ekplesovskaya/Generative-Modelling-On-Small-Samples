# This module estimates the degree of overfitting of generative models
# trained on datasets from the OpenML database and Scikit-learn library

from tools.get_empirical_dataset import get_dataset, data_scaling
import pickle
import random
from tools.generative_model_estimation import fit_model
from tools.generative_model_sampling import get_sampled_data
from tools.dataset_similarity_metrics import zu_overfitting_statistic
from sklearn.model_selection import ShuffleSplit


def estimate_overfitting(dataset_names, method_list):
    """Estimates the degree of overfitting of generative models

    :param dataset_names: list of dataset names
    :param method_list: list of generative algorithm names
    :return result: Zu overfitting statistics
    """
    random.seed(42)
    seed_val = random.sample(list(range(100000)), 100)
    seed_val_cv = seed_val[:10]
    seed_val_sample = seed_val
    result = {}
    for ds_name in dataset_names:
        print("Dataset:", ds_name)
        ds = get_dataset(ds_name)
        result[ds_name] = {}
        rs = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.7, random_state=42)
        for j,indexes in enumerate(rs.split(ds["data"])):
            print("Split:", j)
            data_train =ds["data"].iloc[indexes[0], :]
            data_train.reset_index(drop=True, inplace=True)
            data_test = ds["data"].iloc[indexes[1], :]
            data_test.reset_index(drop=True, inplace=True)
            data_train_scaled, scaler = data_scaling(data_train, data_train.columns)
            for method_name in method_list:
                print("Algorithm:", method_name)
                if method_name not in result[ds_name]:
                    result[ds_name][method_name] = {}
                if method_name in ["copula", "kde_cv_ml", "kde_cv_ls"]:
                    seed_val_cv_method = [None]
                else:
                    seed_val_cv_method = seed_val_cv
                for i, seed_cv in enumerate(seed_val_cv_method):
                    gen_model = fit_model(method_name, data_train_scaled, seed_cv)
                    sampled_data_list = get_sampled_data(gen_model, len(data_train), seed_val_sample,
                                                         method_name, ds["cols"], scaler, ds)
                    if "Zu" not in result[ds_name][method_name]:
                        result[ds_name][method_name]["Zu"] = []
                    for sampled_data in sampled_data_list:
                        result[ds_name][method_name]["Zu"].append(zu_overfitting_statistic(data_test, sampled_data, data_train))
    return result


if __name__ == '__main__':
    dataset_names = ["iris", "visualizing_galaxy", "visualizing_environmental"]
    method_list = ["sklearn_kde", "awkde", "kde_cv_ml", "kde_cv_ls", "gmm",
                   "bayesian_gmm", "ctgan", "copula", "copulagan", "tvae"]
    result = estimate_overfitting(dataset_names, method_list)
    with open('overfitting_estimation_for_the_empirical_datasets.pickle', "wb") as pickle_file:
        pickle.dump(result,  pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_file.close()
