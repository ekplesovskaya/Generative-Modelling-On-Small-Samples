# This module generates plots for dataset comparison

from tools.plot_functions import make_pairwise_plot_for_emp_dataset, get_axis_lim
import pickle
from tools.generative_model_sampling import get_sample_for_pairwise_plot
from tools.get_empirical_dataset import get_dataset


def compare_pairwise_plots(dataset_name, gen_models_list, data):
    """Outputs pairwise plots for samples from generative models

    :param dataset_name: dataset name
    :param gen_models_list: list of generative algorithms to compare
    :param data: dataset similarity indicators
    """
    ds = get_dataset(dataset_name)
    var_range = get_axis_lim(ds["data"])
    ds_title = ds_name_for_title[dataset_name]
    make_pairwise_plot_for_emp_dataset(ds["data"], var_range, ds_title+": " + "train sample",
                                       dataset_name+"_train_sample")
    for gen_model in gen_models_list:
        print(gen_model)
        gen_model_title = method_for_title[gen_model]
        sampled_data = get_sample_for_pairwise_plot(gen_model, data[dataset_name][gen_model], ds)
        make_pairwise_plot_for_emp_dataset(sampled_data, var_range, ds_title+": " + gen_model_title,
                                           dataset_name + "_" + gen_model_title + "_sample")


ds_name_for_title = dict(zip(["iris", "visualizing_galaxy", "visualizing_environmental"],
                             ["Iris Dataset", "Visualizing Galaxy Dataset", "Visualizing Environmental Dataset"]))
method_for_title = {"kde_cv_ml": "KDE1", "sklearn_kde": "KDE2", "kde_cv_ls": "KDE3", "awkde": "KDE4", "gmm": "GMM1",
                    "bayesian_gmm": "GMM2", "ctgan": "CTGAN", "copula": "Copula", "copulagan": "CopulaGAN", "tvae":"TVAE"}


if __name__ == '__main__':
    with open('generation_results_for_the_empirical_datasets.pickle', 'rb') as f:
        data = pickle.load(f)
    ds_list = ["iris", "visualizing_galaxy", "visualizing_environmental"]
    gen_models_dict = dict(zip(ds_list, [["sklearn_kde", "ctgan"], ["sklearn_kde", "awkde"], ["awkde", "copulagan"]]))
    for i, dataset_name in enumerate(ds_list):
        print(dataset_name)
        compare_pairwise_plots(dataset_name, gen_models_dict[dataset_name], data)


