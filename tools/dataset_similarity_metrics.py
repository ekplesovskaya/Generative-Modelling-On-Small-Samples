# This module implements dataset similarity and overfitting indicators used in paper

from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors as NN
from scipy.stats import mannwhitneyu
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import shogun as sg


def ks_test(df1, df2):
    """Kolmogorov-Smirnov test applied to each variable of two dataframes

    :param df1: train dataset
    :param df2: sampled dataset
    :return p_val: p-value for the variable with the maximum KS statistic
    :return stat: maximum KS statistic value
    :return n: variable with the maximum KS statistic
    :return p_val_list: p-value for each variable
    :return stat_list: KS statistic for each variable
    """
    p_val_list = []
    stat_list = []
    for element in df1.columns:
        res = stats.ks_2samp(df1[element], df2[element])
        p_val_list.append(res[1])
        stat_list.append(res[0])
    n = np.argmax(stat_list)
    p_val = p_val_list[n]
    stat = stat_list[n]
    return p_val, stat, n, p_val_list, stat_list


def c2st_roc_auc(df1, df2):
    """Classifier Two-Sample Test: ROC AUC for gradient boosting classifier

    :param df1: train dataset
    :param df2: sampled dataset
    :return roc_auc: ROC AUC

    References:
    Friedman, J. H. (2003) “On Multivariate Goodness–of–Fit and Two–Sample Testing” Statistical Problems in Particle Physics, Astrophysics and Cosmology, PHYSTAT2003: 311-313.
    """
    df1 = df1.copy()
    df2 = df2.copy()
    df1["Class"] = 1
    df2["Class"] = 0
    data = pd.concat([df1, df2], axis=0)
    y = data['Class']
    data.drop('Class', axis=1, inplace=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
    clf = xgb.XGBClassifier(seed=10)
    score = []
    for train_index, test_index in skf.split(data, y):
        x0, x1 = data.iloc[train_index], data.iloc[test_index]
        y0, y1 = y.iloc[train_index], y.iloc[test_index]
        clf.fit(x0, y0, eval_set=[(x1, y1)], eval_metric='logloss', verbose=False, early_stopping_rounds=10)
        prval = clf.predict_proba(x1)[:, 1]
        ras = roc_auc_score(y1, prval)
        score.append(ras)
    roc_auc = np.mean(score)
    return roc_auc


def c2st_accuracy(data_orig, sampled):
    """Classifier Two-Sample Test: LOO Accuracy for 1-NN classifier

    :param df1: train dataset
    :param df2: sampled dataset
    :return acc_r: accuracy for real samples
    :return acc_g: accuracy for generated samples

    References:
    Xu, Q. et al. (2018) “An empirical study on evaluation metrics of generative adversarial networks” arXiv preprint arXiv:1806.07755.
    """
    data_orig = data_orig.copy()
    sampled = sampled.copy()
    data_orig["class"] = 1
    sampled["class"] = 0
    data_res = pd.concat([data_orig, sampled], ignore_index=True, sort=False)
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    n_var = data_res.shape[1]-1
    for train_index, test_index in loo.split(data_res):
        X_train, X_test = data_res.iloc[train_index, :n_var-1], data_res.iloc[test_index, :n_var-1]
        y_train, y_test = data_res.iloc[train_index, n_var], data_res.iloc[test_index, n_var]
        NN_clf = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
        preds = NN_clf.predict(X_test)
        y_true.extend(y_test)
        y_pred.extend(preds)
    res = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
    res_1 = res[res["y_true"] == 1]
    acc_r = res_1["y_pred"].sum()/len(res_1)
    res_0 = res[res["y_true"] == 0]
    acc_g = 1-res_0["y_pred"].sum()/len(res_0)
    return acc_r, acc_g


def ks_permutation(stat, df1, df2):
    """Kolmogorov-Smirnov permutation test applied to each variable of two dataframes

    :param stat: list of KS test statistics for each variable
    :param df1: train dataset
    :param df2: sampled dataset
    :return p_val: resulting p-value for two datasets
    """
    p_val = None
    i = 0
    p_val_list = []
    while p_val == None:
        element = list(df1.columns)[i]
        x1 = df1[element].values
        x2 = df2[element].values
        p_val_stat = ks_permutation_var(stat[i], x1, x2)
        if p_val_stat < 0.05:
            p_val = p_val_stat
        else:
            i += 1
            p_val_list.append(p_val_stat)
        if element == list(df1.columns)[-1]:
            p_val = p_val_list[0]
    return p_val


def ks_permutation_var(stat, series1, series2):
    """Kolmogorov-Smirnov permutation test for a single variable

    :param stat: KS test statistic
    :param series1: train series
    :param series2: sampled series
    :return p_val: p-value
    """
    x1 = series1
    x2 = series2
    lx1 = len(x1)
    lx2 = len(x2)
    data_x = np.concatenate([x1, x2], axis=0)
    rng = np.random.default_rng(seed=42)
    ks_res = []
    n_samp = 1000
    for j in range(n_samp):
        x_con = rng.permutation(data_x)
        x1_perm = x_con[:lx1]
        x2_perm = x_con[lx2:]
        ks_res.append(stats.ks_2samp(x1_perm, x2_perm)[0])
    ks_list = np.sort(ks_res)
    ks_arg = np.arange(start=1, stop=n_samp+1)/n_samp
    p_val = 1-np.interp(stat, ks_list, ks_arg)
    return p_val


def rbf_mmd_test(X1, X2):
    """Maximum Mean Discrepancy Test

    :param X1: real feature array
    :param X2: generated feature array
    :return p_val: p-value
    :return stat: MMD test statistic

    References:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012) “A kernel two-sample test” The Journal of Machine Learning Research, 13(1): 723-773.
    Sutherland, D. J., Tung, H. Y., Strathmann, H., De, S., Ramdas, A., Smola, A. J., & Gretton, A. (2017) “Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy” International Conference on Learning Representations.
    """
    sub = lambda feats, n: feats[np.random.choice(feats.shape[0], min(feats.shape[0], n), replace=False)]
    Z = np.r_[sub(X1, 1000 // 2), sub(X2, 1000 // 2)]
    D2 = euclidean_distances(Z, squared=True)
    upper = D2[np.triu_indices_from(D2, k=1)]
    kernel_width = np.median(upper, overwrite_input=True)
    del Z, D2, upper
    mmd = sg.QuadraticTimeMMD()
    mmd.set_p(sg.RealFeatures(X1.T.astype(np.float64)))
    mmd.set_q(sg.RealFeatures(X2.T.astype(np.float64)))
    mmd.set_kernel(sg.GaussianKernel(32, kernel_width))
    mmd.set_null_approximation_method(sg.NAM_PERMUTATION)
    mmd.set_num_null_samples(500)
    stat = mmd.compute_statistic()
    p_val = mmd.compute_p_value(stat)
    return p_val, stat


def zu_overfitting_statistic(df1, df2, df3):
    """Zu overfitting statistic calculation

    :param df1: test dataset
    :param df2: synthetic dataset
    :param df3: train dataset
    :return zu_stat: Zu overfitting statistic

    References:
    Meehan C., Chaudhuri K., Dasgupta S. (2020) “A non-parametric test to detect data-copying in generative models” International Conference on Artificial Intelligence and Statistics.
    """
    m = df2.shape[0]
    n = df1.shape[0]
    T_NN = NN(n_neighbors=1).fit(df3)
    LQm, _ = T_NN.kneighbors(X=df2, n_neighbors=1)
    LPn, _ = T_NN.kneighbors(X=df1, n_neighbors=1)
    u, p = mannwhitneyu(LQm, LPn, alternative='less')
    mean = (n * m / 2) - 0.5
    std = np.sqrt(n*m*(n + m + 1) / 12)
    zu_stat = (u - mean) / std
    return zu_stat


def calc_metrics(data, sampled_data_list, dataset_type):
    """Calculates dataset similarity metrics

    :param data: real dataset
    :param sampled_data_list: list of synthetic datasets
    :param dataset_type: dataset type
    :return result: synthetic dataset quality metrics
    """
    result={}
    for sampled_data in sampled_data_list:
        c2st_roc_auc_metric = c2st_roc_auc(data, sampled_data)
        if "c2st_roc_auc" in result:
            result["c2st_roc_auc"].append(c2st_roc_auc_metric)
        else:
            result["c2st_roc_auc"] = [c2st_roc_auc_metric]
        mmd_p_val, mmd_stat = rbf_mmd_test(data.values, sampled_data.values)
        if "mmd_p_val" in result:
            result["mmd_p_val"].append(mmd_p_val)
            result["mmd_stat"].append(mmd_stat)
        else:
            result["mmd_p_val"] = [mmd_p_val]
            result["mmd_stat"] = [mmd_stat]
        ks_p_val, ks_stat, ks_n, ks_p_val_list, ks_stat_list = ks_test(data, sampled_data)
        if dataset_type != "norm_dataset":
            ks_p_val = ks_permutation(ks_stat_list, data, sampled_data)
        if "ks_p_val" in result:
            result["ks_p_val"].append(ks_p_val)
            result["ks_stat"].append(ks_stat)
        else:
            result["ks_p_val"] = [ks_p_val]
            result["ks_stat"] = [ks_stat]
        acc_r, acc_g = c2st_accuracy(data, sampled_data)
        if "c2st_acc_r" in result:
            result["c2st_acc_r"].append(acc_r)
            result["c2st_acc_g"].append(acc_g)
        else:
            result["c2st_acc_r"] = [acc_r]
            result["c2st_acc_g"] = [acc_g]
    return result




