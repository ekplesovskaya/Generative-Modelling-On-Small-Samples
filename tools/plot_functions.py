# This module makes plots for graphical comparison of real and generated datasets

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tools.dataset_similarity_metrics import ks_test


def make_pairwise_plot(df, var_range, title, name):
    """Outputs pairwise plot

    :param df: dataframe
    :param var_range: axis limit parameters
    :param title: plot title
    :param name: file name
    """
    sns.set(font_scale=3)
    g = sns.pairplot(df, diag_kind='kde', corner=True, height=4)
    g.set(xlim=var_range)
    g.set(ylim=var_range)
    g.fig.suptitle(title)
    plt.tight_layout()
    g.savefig(name+'.png', dpi=300)
    plt.close(g.fig)


def corr_axis_lim(coord, coord_type):
    """Outputs axis limit based on the maximum and minimum data coordinates

    :param coord: data coordinate
    :param coord_type: data coordinate type
    :return coord_corr: axis limit
    """
    if coord_type == "max":
        if coord>0:
            coord_corr = coord*1.1
        else:
            coord_corr = coord*0.8
    else:
        if coord>0:
            coord_corr = coord*0.8
        else:
            coord_corr = coord*1.1
    return coord_corr


def get_axis_lim(df):
    """Outputs axis view limits

    :param df: dataset
    :return var_range_dict: dictionary with axis view limits
    """
    x_max = list(df.max())
    x_min = list(df.min())
    var_range_dict = {}
    for i in range(1, len(x_max)):
        for j in range(0, i):
            x_max_coord = corr_axis_lim(x_max[j], "max")
            x_min_coord = corr_axis_lim(x_min[j], "min")
            y_max_coord = corr_axis_lim(x_max[i], "max")
            y_min_coord = corr_axis_lim(x_min[i], "min")
            var_range_dict[(i, j)] = {"x": (x_min_coord, x_max_coord), "y": (y_min_coord, y_max_coord)}
    return var_range_dict


def make_pairwise_plot_for_emp_dataset(df, var_range, title, name):
    """Outputs pairwise plot for the empirical datasets

    :param df: dataframe
    :param var_range: axis view limits
    :param title: plot title
    :param name: file name
    """
    sns.set(font_scale=3)
    g = sns.pairplot(df, diag_kind='kde', corner=True, height=7, plot_kws={"s": 150})
    for num in var_range:
        g.axes[num[0], num[1]].set_xlim(var_range[num]["x"])
        g.axes[num[0], num[1]].set_ylim(var_range[num]["y"])
    g.fig.suptitle(title)
    plt.tight_layout()
    g.savefig(name, dpi=300)
    plt.close(g.fig)


def make_qq_and_pp_plots(data_orig, sampled_data, title, name):
    """Outputs P-P and Q-Q plots for each variable

    :param data_orig: train dataset
    :param sampled_data: KDE sample
    :param title: plot title
    :param name: file name
    """
    plt.rcParams.update({'font.size': 15})
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    p_val, stat, n, p_list, st_list = ks_test(data_orig, sampled_data)
    m, n = 0, 0
    text_level_dict = {"row": {0: 0.94, 1: 0.645, 2: 0.355}, "col": {0: 0.305, 2: 0.715}}
    for i, element in enumerate(data_orig.columns):
        series_orig = data_orig[element].sort_values()
        series = sampled_data[element].sort_values()
        axes[m, n].scatter(series_orig, series, s=20)
        axes[m, n].plot(series_orig, series_orig, "--", c='r')
        axes[m, n].set_title("Q-Q plot")
        axes[m, n].set_xlabel("Train sample")
        axes[m, n].set_ylabel("KDE sample")
        series_unique = []
        series_unique.extend(series_orig)
        series_unique.extend(series)
        series_unique = np.unique(series_unique)
        quant_orig = [en/len(series_orig) for en in range(len(series_orig))]
        quant = [en/len(series) for en in range(len(series))]
        quant_est = np.interp(series_unique, series, quant)
        quant_orig_est = np.interp(series_unique, series_orig, quant_orig)
        axes[m, n+1].scatter(quant_orig_est, quant_est, s=20)
        axes[m, n+1].plot(quant_orig_est, quant_orig_est, "--", c='r')
        axes[m, n+1].set_title("P-P plot")
        axes[m, n+1].set_xlabel("Train dataset")
        axes[m, n+1].set_ylabel("KDE sample")
        plt.figtext(text_level_dict["col"][n], text_level_dict["row"][m], "Variable "+str(i+1)+"\nKS test statistic = " +
                    str(round(st_list[i], 3))+", p-value = "+str(round(p_list[i], 3)), ha="center", va="top", fontsize=15, color="k")
        if n == 0:
            n += 2
        else:
            n = 0
            m += 1
    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.suptitle(title)
    plt.savefig(name+".png")
    plt.clf()