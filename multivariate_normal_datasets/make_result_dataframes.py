# This module generates result tables based on generation result

import pandas as pd
import pickle
import numpy as np


def prediction_interval(data):
    lv = np.percentile(data, 2.5)
    upv = np.percentile(data, 97.5)
    return lv, upv


def get_test_result(low_dict, up_dict, round_param, vol_list_name):
    df_low = pd.DataFrame.from_dict(low_dict, orient="index", columns=vol_list_name)
    df_up = pd.DataFrame.from_dict(up_dict, orient="index", columns=vol_list_name)
    for col in df_low:
        low_val = df_low[col]
        up_val = df_up[col]
        res_val = ["["+str(round(element1, round_param))+", "+str(round(element2, round_param))+"]"
                   for element1, element2 in zip(low_val, up_val)]
        df_low[col] = res_val
    return df_low


def get_2test_result(low_dict, up_dict, low_dict2, up_dict2, round_param, vol_list_name):
    df_low = pd.DataFrame.from_dict(low_dict, orient="index", columns=vol_list_name)
    df_up = pd.DataFrame.from_dict(up_dict, orient="index", columns=vol_list_name)
    df_low2 = pd.DataFrame.from_dict(low_dict2, orient="index", columns=vol_list_name)
    df_up2 = pd.DataFrame.from_dict(up_dict2, orient="index", columns=vol_list_name)
    for col in df_low:
        low_val = df_low[col]
        up_val = df_up[col]
        low_val_g = df_low2[col]
        up_val_g = df_up2[col]
        res_val = ["["+str(round(element1, round_param))+", "+str(round(element2, round_param))+"],\n[" +
                   str(round(element3, round_param))+", "+str(round(element4, round_param))+"]"
                   for element1, element2, element3, element4 in zip(low_val, up_val, low_val_g, up_val_g)]
        df_low[col] = res_val
    return df_low


def get_stat_test_result(low_dict, up_dict, share_dict, round_param, vol_list_name):
    df_low = pd.DataFrame.from_dict(low_dict, orient="index", columns=vol_list_name)
    df_up = pd.DataFrame.from_dict(up_dict, orient="index", columns=vol_list_name)
    p_df = pd.DataFrame.from_dict(share_dict, orient="index", columns=vol_list_name)
    for col in df_low:
        val_low = df_low[col]
        val_up = df_up[col]
        p_val = p_df[col]
        res_val = ["["+str(round(element1, round_param))+", "+str(round(element2, round_param))+"]\n(" +
                   str(element3)+"%)" for element1, element2, element3 in zip(val_low, val_up, p_val)]
        df_low[col] = res_val
    return df_low


def make_tables(low, up, share, vol_list):
    vol_list_name = [str(vol)+" obs." for vol in vol_list]
    df1 = get_test_result(low["c2st_roc_auc"], up["c2st_roc_auc"], 2, vol_list_name).copy()
    df2 = get_2test_result(low["c2st_acc_r"], up["c2st_acc_r"], low["c2st_acc_g"], up["c2st_acc_g"], 2, vol_list_name).copy()
    df3 = get_stat_test_result(low["ks_stat"], up["ks_stat"], share["ks_p_val"], 2, vol_list_name).copy()
    df4 = get_stat_test_result(low["mmd_stat"], up["mmd_stat"], share["mmd_p_val"], 1, vol_list_name).copy()
    df5 = get_test_result(low["Zu"], up["Zu"], 1, vol_list_name).copy()
    result = {"ROC AUC":  df1, "Accuracy for real and generated data": df2, "KS test": df3,
              "MMD test": df4, "Overfitting statistic Zu": df5}
    return result


def make_result_dataframes(data):
    result = {}
    for dist_type in data:
        low, up, share = {}, {}, {}
        for vol in data[dist_type]:
            for n_var in data[dist_type][vol]:
                n_var_name = str(n_var)+" variables"
                metric_val = {}
                for dist_num in data[dist_type][vol][n_var]:
                    for metric in data[dist_type][vol][n_var][dist_num]:
                        if metric not in metric_val:
                            metric_val[metric] = data[dist_type][vol][n_var][dist_num][metric]
                        else:
                            metric_val[metric].extend(data[dist_type][vol][n_var][dist_num][metric])
                for metric in metric_val:
                    if metric in ["mmd_p_val", "ks_p_val"]:
                        series = np.array(metric_val[metric])
                        series_val = round(len(series[series < 0.05])/len(series)*100)
                        if metric not in share:
                            share[metric] = {n_var_name: [series_val]}
                        elif n_var_name not in share[metric]:
                            share[metric][n_var_name] = [series_val]
                        else:
                            share[metric][n_var_name].append(series_val)
                    else:
                        lv, upv = prediction_interval(metric_val[metric])
                        if metric not in low:
                            low[metric] = {n_var_name: [lv]}
                            up[metric] = {n_var_name: [upv]}
                        elif n_var_name not in low[metric]:
                            low[metric][n_var_name] = [lv]
                            up[metric][n_var_name] = [upv]
                        else:
                            low[metric][n_var_name].append(lv)
                            up[metric][n_var_name].append(upv)
        result[dist_type] = make_tables(low, up, share, list(data[dist_type].keys()))
    with open('result_dataframes.pickle', "wb") as pickle_file:
        pickle.dump(result,  pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_file.close()


if __name__ == '__main__':
    with open('generation_results_for_multivariate_normal_datasets.pickle', 'rb') as f:
        data = pickle.load(f)
    make_result_dataframes(data)