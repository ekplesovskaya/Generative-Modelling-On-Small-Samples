# This module generates result tables based on generation result

import pandas as pd
import pickle
import numpy as np


def prediction_interval(data):
    lv = np.percentile(data, 2.5)
    upv = np.percentile(data, 97.5)
    return lv, upv


def get_stat_test_result(data, round_param):
    lv, upv = prediction_interval(data["stat"])
    series = np.array(data["pval"])
    share = round(len(series[series < 0.05])/len(series)*100)
    res_val = "["+str(round(lv, round_param))+", "+str(round(upv, round_param))+"]\n("+str(share)+"%)"
    return res_val


def get_metric_val(data, ind):
    if ind == "MMD test":
        metric_dict = {"stat": [], "pval": []}
        for iter in data:
            metric_dict["pval"].extend(data[iter]["mmd_p_val"])
            metric_dict["stat"].extend(data[iter]["mmd_stat"])
    elif ind == "KS test":
        metric_dict = {"stat": [], "pval": []}
        for iter in data:
            metric_dict["pval"].extend(data[iter]["ks_p_val"])
            metric_dict["stat"].extend(data[iter]["ks_stat"])
    elif ind == "ROC AUC":
        metric_dict = {"val":[]}
        for iter in data:
            metric_dict["val"].extend(data[iter]["c2st_roc_auc"])
    elif ind == "Accuracy for real and generated data":
        metric_dict = {"c2st_acc_r": [], "c2st_acc_g": []}
        for iter in data:
            metric_dict["c2st_acc_r"].extend(data[iter]["c2st_acc_r"])
            metric_dict["c2st_acc_g"].extend(data[iter]["c2st_acc_g"])
    return metric_dict


method_dict = {"kde_cv_ml":"KDE1", "sklearn_kde":"KDE2","kde_cv_ls":"KDE3", "awkde":"KDE4", "gmm":"GMM1",
 "bayesian_gmm":"GMM2", "ctgan":"CTGAN", "copula":"Copula", "copulagan":"CopulaGAN", "tvae":"TVAE"}
ind_list = ["MMD test",  "KS test", "ROC AUC", "Accuracy for real and generated data", "Overfitting statistic Zu"]


def make_result_dataframes(data, data_overfitting):
    result = {}
    for ds_name in data:
        res_df = {}
        for method in method_dict:
            res_df[method_dict[method]] = []
            for ind in ind_list:
                if ind != "Overfitting statistic Zu":
                    metric_val = get_metric_val(data[ds_name][method], ind)
                else:
                    metric_val = {"val": data_overfitting[ds_name][method]["Zu"]}
                if ind == "MMD test":
                    res_val = get_stat_test_result(metric_val, 1)
                elif ind == "KS test":
                    res_val = get_stat_test_result(metric_val, 2)
                elif ind == "ROC AUC":
                    lv, upv = prediction_interval(metric_val["val"])
                    res_val = "["+str(round(lv, 2))+", "+str(round(upv, 2))+"]"
                elif ind == "Accuracy for real and generated data":
                    lv, upv = prediction_interval(metric_val["c2st_acc_r"])
                    lv2, upv2 = prediction_interval(metric_val["c2st_acc_g"])
                    res_val = "["+str(round(lv, 2))+", "+str(round(upv, 2))+"],\n["\
                              +str(round(lv2, 2))+", "+str(round(upv2, 2))+"]"
                else:
                    lv, upv = prediction_interval(metric_val["val"])
                    res_val = "["+str(round(lv, 2))+", "+str(round(upv, 2))+"]"
                res_df[method_dict[method]].append(res_val)
        result[ds_name] = pd.DataFrame.from_dict(res_df, orient="index", columns=ind_list)
    with open('result_dataframes.pickle', "wb") as pickle_file:
        pickle.dump(result,  pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_file.close()


if __name__ == '__main__':
    with open('generation_results_for_the_empirical_datasets.pickle', 'rb') as f:
        data = pickle.load(f)
    with open('overfitting_estimation_for_the_empirical_datasets.pickle', 'rb') as f:
        data_overfitting = pickle.load(f)
    make_result_dataframes(data, data_overfitting)