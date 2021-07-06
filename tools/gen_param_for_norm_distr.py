# This module generates random parameters for theoretical distributions

from scipy import stats
import numpy as np
from scipy.stats import random_correlation


def norm_dist_generator(ndim, n_gen, rng):
    """Generates random parameters for normal distribution

    :param ndim: number of dimensions
    :param n_gen: number of distributions to generate
    :param rng: random generator with a fixed seed
    :return result: list of multivariate random variables
    """
    result = []
    for i in range(n_gen):
        rints_mean = rng.integers(low=0, high=100, size=ndim)
        eig = rng.uniform(0.1, 100, size=ndim)
        eig = eig/np.sum(eig)*ndim
        rv = False
        while not rv:
            try:
                x = random_correlation.rvs((eig))
                fac = rng.uniform(0.1, 100, size=ndim)
                x = x*fac
                rv = stats.multivariate_normal(mean=rints_mean, cov=x)
            except:
                rv = False
        result.append(rv)
    return result


def norm_mixture_dist_generator(ndim, n_gen, rng):
    """Generates random parameters for normal mixture distribution

    :param ndim: number of dimensions
    :param n_gen: number of distributions to generate
    :param rng: random generator with a fixed seed
    :return result: list of multivariate random variables
    """
    result = []
    for i in range(n_gen):
        rints_mean_1 = rng.integers(low=0, high=60, size=ndim)
        rints_mean_2 = rng.integers(low=40, high=100, size=ndim)
        mean_list = [rints_mean_1, rints_mean_2]
        rv_list = []
        for mean in mean_list:
            eig = rng.uniform(0.1, 100, size=ndim)
            eig = eig/np.sum(eig)*ndim
            rv = False
            while not rv:
                try:
                    x = random_correlation.rvs((eig))
                    fac = rng.uniform(0.1, 100, size=ndim)
                    x = x*fac
                    rv = stats.multivariate_normal(mean=mean, cov=x)
                except:
                    rv = False
            rv_list.append(rv)
        result.append(rv_list)
    return result


def get_list_of_random_generators(dist_type, n_var, n_gen, rng):
    """Calls a random sample generation function

    :param dist_type: type of distribution
    :param n_var: dimensionality
    :param n_gen: number of distributions to generate
    :param rng: random generator with a fixed seed
    :return rv_list: list of multivariate random variables
    """
    if dist_type == "normal":
        rv_list = norm_dist_generator(n_var, n_gen, rng)
    else:
        rv_list = norm_mixture_dist_generator(n_var, n_gen, rng)
    return rv_list




