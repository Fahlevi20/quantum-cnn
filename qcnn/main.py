#%%
import imp
import os
import time
import argparse

import pandas as pd

import json

import numpy as np

from data_utility import DataUtility
from experiment import run_experiment
from data_handler import (
    save_json,
    load_json,
    get_2d_modelling_data,
    get_image_data,
    create_train_test_samples,
)


#%%
# data_path = "../data/archive/Data/features_30_sec.csv"
# target = "label"
# raw = pd.read_csv(data_path)
# data_utility = DataUtility(raw, target=target, default_subset="modelling")
# columns_to_remove = ["filename", "length"]
# data_utility.update(columns_to_remove, "included", {"value": False, "reason": "manual"})
# Levels to consider
# target_levels = raw[data_utility.target].unique()
# # Here we get all possible pairwise comparisons, this is used for ovo classification
# target_pairs = [target_pair for target_pair in itertools.combinations(target_levels, 2)]


# Here we get all possible pairwise comparisons, this is used for ovo classification
# target_pairs = [target_pair for target_pair in itertools.combinations(target_levels, 2)]

#%%
# Configuration
# EXPERIMENT_PATH = "../experiments"
# # Ensure experiment doesn't get overwritten
# EXPERIMENT_ID = max([int(exp_str) for exp_str in os.listdir(EXPERIMENT_PATH)]) + 1
# # EXPERIMENT_ID = 106


# Setup for ova classifcation, each class should be in the "1" index the 0 index is arbitrary
# target_pairs = [(target_level, target_level) for target_level in target_levels]
# Setup expermiment config
# config = {
#     "scaler": {
#         "method": ["standard"],
#         "standard_params": {},
#         "minmax_params": {"feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]},
#     },
#     "feature_selection": {
#         "method": ["pca"],
#         "pca_params": {"n_components": [8]},
#         "tree_params": {"max_features": [8], "n_estimators": [50]},
#     },
# }
# quantum_experiment_config = {
#     "ID": EXPERIMENT_ID,
#     "path": EXPERIMENT_PATH,
#     "data": {
#         "target_pairs": [
#             (0, 1),
#             (1, 7)
#             # ("disco", "rock"),
#             # ("hiphop", "pop"),
#             # ("country", "reggae"),
#             # ("jazz", "metal"),
#         ],
#         "type": "image",
#         "path": "../data/archive/Data/features_30_sec.csv",
#         "sampling": {  # This sampling creates a test set which is never seen during any model training for all experiments.. this can be
#             # can be done differently, like based on time the dataset naturally is split into a train / test
#             "test_size": 0.3,
#             "random_state": 42,
#         },
#     },
#     "preprocessing": {
#         "quantum": {
#             "Angle": {
#                 "scaler": {
#                     "method": [None],
#                     "standard_params": {},
#                     "minmax_params": {
#                         "feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]
#                     },
#                 },
#                 "feature_selection": {
#                     "method": ["pca", "tree"],
#                     "pca_params": {"n_components": [8]},
#                     "tree_params": {"max_features": [8], "n_estimators": [50]},
#                 },
#                 "ignore": False,
#             },
#             "IQP": {
#                 "scaler": {
#                     "method": [None],
#                     "standard_params": {},
#                     "minmax_params": {
#                         "feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]
#                     },
#                 },
#                 "feature_selection": {
#                     "method": ["pca", "tree"],
#                     "pca_params": {"n_components": [8]},
#                     "tree_params": {"max_features": [8], "n_estimators": [50]},
#                 },
#                 "kwargs": {"depth": 10},
#                 "ignore": False,
#             },
#             "Amplitude": {
#                 "scaler": {
#                     "method": [None],
#                     "standard_params": {},
#                     "minmax_params": {
#                         "feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]
#                     },
#                 },
#                 "feature_selection": {
#                     "method": ["pca", "tree"],
#                     "pca_params": {"n_components": [8]},
#                     "tree_params": {"max_features": [8], "n_estimators": [50]},
#                 },
#                 "ignore": False,
#             },
#         },
#         "classical": {
#             "normal": {
#                 "scaler": {
#                     "method": ["standard"],
#                     "standard_params": {},
#                     "minmax_params": {
#                         "feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]
#                     },
#                 },
#                 "feature_selection": {
#                     "method": ["pca", "tree"],
#                     "pca_params": {"n_components": [8]},
#                     "tree_params": {"max_features": [8], "n_estimators": [50]},
#                 },
#                 "ignore": False,
#             },
#         },
#     },
#     "model": {
#         "quantum": {
#             "qcnn": {
#                 "circuit_list": ["U_5", "U_SU4"],
#                 "pooling_list": ["psatz1"],
#                 "custom_structures": {
#                     "custom_0": {  # order, convolution function name, circuite name, circuit paramaters, layer_name i.e. the keys should start with c for convolutional or p for pooling
#                         "c_1": [0, "c_1", "U_5"],
#                         "p_1": [1, "p_1", "psatz1"],
#                         "c_2": [2, "c_2", "U_SU4"],
#                         "p_2": [3, "p_2", "psatz1"],
#                         "c_3": [4, "c_3", "U_6"],
#                         "p_3": [5, "p_3", "psatz1"],
#                     },
#                     "custom_1": {
#                         "c_1": [0, "c_1", "U_5"],
#                         "p_1": [1, "p_1", "psatz1"],
#                         "c_2": [2, "c_2", "U_SU4"],
#                         "p_2": [3, "p_2", "psatz1"],
#                         "c_3": [4, "c_3", "U_5"],
#                         "p_3": [5, "p_3", "psatz1"],
#                     },
#                 },
#                 "ignore": False,
#             }
#         },
#         "classical": {
#             "logistic_regression": {
#                 "param_grid": {
#                     "C": np.logspace(-3, 3, 7).tolist(),
#                     "penalty": ["l1", "l2"],
#                 },
#                 "ignore": True,
#             },
#             "svm": {
#                 "param_grid": {
#                     "kernel": ("linear", "rbf", "poly", "sigmoid"),
#                     "C": [1, 10, 100],
#                 },
#                 "ignore": True,
#             },
#             "cnn": {"param_grid": {}, "ignore": True},
#         },
#         "classification_type": "binary",
#     },
#     "train": {
#         "iterations": 40,
#     },
#     "extra_info": "main",
# }
# Start experiment

# %%
# Get experiment config
# config = quantum_experiment_config


def main(args):
    # Load experiment config
    config = load_json(args.config_path)
    # Load data
    if config["data"].get("type", None) == "image":
        # With image data raw is a list consisting of X_train, y_train X_test, y_test
        samples = get_image_data(config["data"].get("path"))
    else:

        target = config["data"].get("target_column")
        path = config["data"].get("path")
        # TODO rename function to something more generic like read data
        raw = get_2d_modelling_data(path, target)
        # ==== Data Utility for data specific manipulations ====#
        """
        Datautility should be used only here to transform the data into a desirable train test set, then when the experiment is
        ran it is assumed that all "columns" and rows is as needs to be. This is specific data interaction from the user and should somehow
        be abstracted out TODO
        """
        columns_to_remove = ["filename", "length"]
        data_utility = DataUtility(raw, target=target, default_subset="modelling")
        data_utility.update(
            columns_to_remove, "included", {"value": False, "reason": "manual"}
        )
        X, y, _ = data_utility.get_samples(raw)
        # ==== End Data Utility ====#
        test_size = config["data"]["sampling"].get("test_size", 0.3)
        random_state = config["data"]["sampling"].get("random_state", 42)
        # Create test set
        samples = create_train_test_samples(
            X, y, test_size=test_size, random_state=random_state
        )
        # Move to function TODO
        samples.y_test.to_csv(f"{config.get('path')}/{config.get('ID')}/y_test.csv")
    model_execution_times = run_experiment(config, samples)
    # save_json(model_execution_times) TODO
    print("Experiment Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a machine learning experiment")

    # Add the arguments
    parser.add_argument(
        "--config_path",
        metavar="config_path",
        type=str,
        help="the path to the experiment config (json)",
    )

    args = parser.parse_args()
    main(args)
