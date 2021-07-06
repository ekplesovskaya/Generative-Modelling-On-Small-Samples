Code for the paper "An Empirical Analysis of KDE-based Generative Models on Small Datasets" // Procedia Computer Science Journal - 2021 by Plesovskaya E., Ivanov S. (In press).

This paper presents a framework for synthetic dataset similarity estimation based on two-sample tests. It also specifically accounts for the model overfitting detection. The framework is applied to samples, obtained from a range of KDE-based generative models trained on samples from [multivariate_normal_datasets](multivariate_normal_datasets) and [empirical_datasets](empirical_datasets). In addition to that, we compare the generative capability of KDE with other algorithms used to generate tabular data: gaussian mixture models, copulas, and deep learning models.

- [`tools/generative_model_estimation.py`](tools/generative_model_estimation.py) and [`tools/generative_model_sampling.py`](tools/generative_model_sampling.py) implement generative model estimation and sampling.
- Code for synthetic dataset similarity evaluation is in [`tools/dataset_similarity_metrics.py`](tools/dataset_similarity_metrics.py).
- We also use graphical analysis methods to analyse the similarity of samples which include pairwise plots, Quantile-Quantile and Probability-Probability plots [`tools/plot_functions.py`](tools/plot_functions.py).
