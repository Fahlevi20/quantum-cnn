# Preprocessing should contain embedding + feature reduction logic
import numpy as np

import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# TODO compact + pure amplitude
EMBEDDING_OPTIONS = {
    8: ["Angle", "ZZMap", "IQP", "displacement", "Squeeze"],
    12: [f"Angular-Hybrid2-{i}" for i in range(1, 5)],
    16: [f"Amplitude-Hybrid2-{i}" for i in range(1, 5)] + ["Angle-Compact"],
    30: [f"Angular-Hybrid4-{i}" for i in range(1, 5)],
    32: [f"Amplitude-Hybrid4-{i}" for i in range(1, 5)] + ["Amplitude"],
}

from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1


def filter_embedding_options(embedding_list):
    """Method to filter out the embedding options dictionary. Removes all embeddings
    not specified in the provided list

    Args:
        embedding_list (list(str)): list containing embedding names such as Angle or Amplitude-Hybrid-4

    Returns:
        dictionary: a subset of all possible embedding options based on the names sent through.
    """
    embeddings = {
        red_size: set(embedding_list) & set(embedding_option)
        for red_size, embedding_option in EMBEDDING_OPTIONS.items()
        if len((set(embedding_list) & set(embedding_option))) > 0
    }

    return embeddings


def get_preprocessing_pipeline(config):
    """Returns a pipeline that handles the pre-processing part of the model (this step is quantum/classical agnostic).
    Currently the preprocessing pipeline consists of two steps, a scaling and feature selection step. Each of these have
    different configurable properties. This function takes in a slice of a more general config (1 permutation in a sense...example below):
    A general config will contain all the paramaters to try out
        general_config = {
                "scaler": {
                    "method": ["standard", "minmax"],
                    "standard_params": {},
                    "minmax_params": {"feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]},
                },
                "feature_selection": {
                    "method": ["pca"],
                    "pca_params": {"n_components": [8]},
                    "tree_params": {"max_features": [8], "n_estimators": [50]},
                },
            }
    The config that is sent through will be one specific iteration/slice of that general config
        config = {
                "scaler": {
                    "method": "minmax",
                    "minmax_params": {"feature_range": (0, np.pi / 2)},
                },
                "feature_selection": {
                    "method": "pca",
                    "pca_params": {"n_components": 8},
                },
            }

    Args:
        config (dict): This is a dictionary which contains the specific pipeline configuration
    """
    scaler_method = config["scaler"].get("method", None)
    scaler_params = config["scaler"].get(f"{scaler_method}_params", [])
    selection_method = config["feature_selection"].get("method", "pca")
    selection_params = config["feature_selection"].get(f"{selection_method}_params", [])

    # Define Scaler
    if scaler_method == None:
        scaler = (
            "scaler",
            IdentityTransformer(),
        )
    elif scaler_method == "minmax":
        scaler = (
            "scaler",
            preprocessing.MinMaxScaler(**scaler_params),
        )
    elif scaler_method == "standard":
        scaler = (
            "scaler",
            preprocessing.StandardScaler(**scaler_params),
        )

    # Define feature selector
    if selection_method == "tree":
        selection = (
            selection_method,
            SelectFromModel(
                ExtraTreesClassifier(
                    n_estimators=selection_params.get("n_estimators", 50)
                ),
                max_features=selection_params.get("max_features", 8),
            ),
        )
    elif selection_method == "pca":
        selection = (selection_method, PCA(**selection_params))
    pipeline = Pipeline(
        [
            scaler,
            selection,
        ]
    )

    return pipeline


def filter_levels(data, feature, levels):
    """Returns all rows belonging to the list of levels for a specific feature

    Args:
        data (pd.DataFrame): data to filter rows
        feature (str): name of the feature for which the levels are concerned
        levels (list[str or int]): distinct values of the feature to filter
    """
    filter_pat = "|".join(level for level in levels)
    indices = data[feature].str.contains(filter_pat)
    return data.loc[indices, :].copy()


def apply_preprocessing(
    raw, pipeline, data_utility, classification_type, target_levels, data_type
):

    if data_type == "image":
        X_train = raw[0]
        y_train = raw[1]
        X_test = raw[2]
        y_test = raw[3]

        X_test_all = X_test.copy()
        y_test_all = y_test.copy()
        y_test_all = np.where(y_test_all == target_levels[1], 1, 0)

        # TODO improve
        x_train_filter_01 = np.where(
            (y_train == target_levels[0]) | (y_train == target_levels[1])
        )
        x_test_filter_01 = np.where(
            (y_test == target_levels[0]) | (y_test == target_levels[1])
        )

        X_train, X_test = X_train[x_train_filter_01], X_test[x_test_filter_01]
        y_train, y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]

        y_train = np.where(y_train == target_levels[1], 1, 0)
        y_test = np.where(y_test == target_levels[1], 1, 0)

        X_train = tf.image.resize(X_train[:], (784, 1)).numpy()
        X_test = tf.image.resize(X_test[:], (784, 1)).numpy()
        X_test_all = tf.image.resize(X_test_all[:], (784, 1)).numpy()
        X_train, X_test = tf.squeeze(X_train), tf.squeeze(X_test)

        pipeline.fit(X_train, y_train)

        X_train_tfd = pipeline.transform(X_train)
        X_test_tfd = pipeline.transform(X_test)
        # TODO fix
        X_test_all_tfd = pd.DataFrame(X_test_tfd)

        # TODO test out this step
        X_train_tfd, X_test_tfd = (X_train_tfd - X_train_tfd.min()) * (
            np.pi / (X_train_tfd.max() - X_train_tfd.min())
        ), (X_test_tfd - X_test_tfd.min()) * (
            np.pi / (X_test_tfd.max() - X_test_tfd.min())
        )

        # X_test_all_tfd = (X_test_all_tfd - X_test_all_tfd.min()) * (
        #     np.pi / (X_test_all_tfd.max() - X_test_all_tfd.min())
        # )

        return (
            X_train_tfd,
            y_train,
            X_test_tfd,
            y_test,
            X_test_all_tfd,
            y_test_all,
            pd.DataFrame({"a": [1, 2, 3]}),
        )

    else:
        # Preprocessing
        if classification_type in ["ova"]:
            ## Make target binary 1 for target 0 rest
            raw[data_utility.target] = np.where(
                raw[data_utility.target] == target_levels[1], 1, 0
            )

            (
                X_train,
                y_train,
                Xy_test,
                X_test,
                y_test,
                Xy_test,
            ) = data_utility.get_samples(raw, row_samples=["train", "test"])
        else:
            # Get test set first
            X_test_all, y_test_all, Xy_test_all = data_utility.get_samples(
                raw, row_samples=["test"]
            )
            y_test_all = np.where(y_test_all == target_levels[1], 1, 0)
            ## Filter data
            raw = filter_levels(raw, data_utility.target, levels=target_levels)

            ## Make target binary TODO generalize more classes
            raw[data_utility.target] = np.where(
                raw[data_utility.target] == target_levels[1], 1, 0
            )
            ## Get train test splits, X_test here will be only for the subset of data, so used to evaluate the single model
            # but not the OvO combinded one
            (
                X_train,
                y_train,
                Xy_test,
                X_test,
                y_test,
                Xy_test,
            ) = data_utility.get_samples(raw, row_samples=["train", "test"])

        pipeline.fit(X_train, y_train)

        # Transform data
        X_train_tfd = pipeline.transform(X_train)
        X_test_tfd = pipeline.transform(X_test)
        if classification_type == "ovo":
            X_test_all_tfd = pipeline.transform(X_test_all)
        elif classification_type == "ova":
            X_test_all = X_test

    return (
        X_train_tfd,
        y_train,
        X_test_tfd,
        y_test,
        X_test_all_tfd,
        y_test_all,
        X_test_all,
    )
