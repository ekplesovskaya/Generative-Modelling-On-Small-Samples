# This module loads datasets from the OpenML database and Scikit-learn library

from sklearn.datasets import fetch_openml, load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd


def get_post_process_param(dataset_name, dataset_cols):
    """Defines post-processing parameters

    :param dataset_name: dataset name
    :param dataset_cols: dataset columns
    :return parameters: post-process parameters
    """
    parameters = {"lim": {}, "round": {}}
    if dataset_name == "iris":
        for col in dataset_cols:
            parameters["lim"][col] = {"low": 0., "high": None}
            parameters["round"][col] = 1
    elif dataset_name == "visualizing_galaxy":
        for col in dataset_cols:
            if col == "velocity":
                parameters["lim"][col] = {"low": 0., "high": None}
                parameters["round"][col] = 0
            elif col == "radialposition":
                parameters["lim"][col] = {"low": None, "high": None}
                parameters["round"][col] = 1
            else:
                parameters["lim"][col] = {"low": None, "high": None}
                parameters["round"][col] = None
    elif dataset_name == "visualizing_environmental":
        for col in dataset_cols:
            if col == "wind":
                parameters["lim"][col] = {"low": 0., "high": None}
                parameters["round"][col] = 1
            else:
                parameters["lim"][col] = {"low": 0., "high": None}
                parameters["round"][col] = 0
    return parameters


def data_scaling(data, col_names):
    """Performes the standardization of a dataset

    :param data: dataset
    :param col_names: dataset columns
    :return data_scaled: standardized dataset
    :return scaler: scaler
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=col_names)
    return data_scaled, scaler


def get_dataset(dataset_name):
    """Loads dataset and its parameters

    :param dataset_name: dataset name
    :return result: dataset and its parameters
    """
    if dataset_name == "iris":
        data = load_iris(as_frame=True)["frame"]
    else:
        data = fetch_openml(name=dataset_name)["frame"]
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    if dataset_name == "iris":
        data.drop("target", inplace=True, axis=1)
    elif dataset_name == "visualizing_galaxy":
        data.drop("angle", inplace=True, axis=1)
    parameters = get_post_process_param(dataset_name, data.columns)
    data_scaled, scaler = data_scaling(data, data.columns)
    result = {"data": data, "data_scaled": data_scaled, "scaler": scaler,
              "lim": parameters["lim"].copy(), "round": parameters["round"].copy(),
              "len": len(data), "cols": data.columns}
    return result