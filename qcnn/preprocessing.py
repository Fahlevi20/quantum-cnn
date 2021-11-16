# Preprocessing should contain embedding + feature reduction logic
import numpy as np

import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Custom
from data_handler import Samples

# TODO compact + pure amplitude
EMBEDDING_OPTIONS = {
    8: ["Angle", "ZZMap", "IQP", "displacement", "Squeeze"],
    12: [f"Angular-Hybrid2-{i}" for i in range(1, 5)],
    16: [f"Amplitude-Hybrid2-{i}" for i in range(1, 5)] + ["Angle-Compact"],
    30: [f"Angular-Hybrid4-{i}" for i in range(1, 5)],
    32: [f"Amplitude-Hybrid4-{i}" for i in range(1, 5)] + ["Amplitude"],
}


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
    samples, pipeline, classification_type, data_type, target_pair=None
):

    if data_type == "image":

        train_filter = np.where(
            (samples.y_train == target_pair[0]) | (samples.y_train == target_pair[1])
        )

        test_filter = np.where(
            (samples.y_train == target_pair[0]) | (samples.y_train == target_pair[1])
        )
        X_train_filtered, X_test_filtered = (
            samples.X_train[train_filter],
            samples.X_test[test_filter],
        )
        y_train_filtered, y_test_filtered = (
            samples.y_train[train_filter],
            samples.y_test[test_filter],
        )
        # TODO this still very hardcoded
        y_train_filtered = np.where(y_train_filtered == target_pair[1], 1, 0)
        y_test_filtered = np.where(y_test_filtered == target_pair[1], 1, 0)

        X_train_filtered = tf.image.resize(X_train_filtered[:], (784, 1)).numpy()
        X_test_filtered = tf.image.resize(X_test_filtered[:], (784, 1)).numpy()
        X_train_filtered, X_teX_test_filteredst = tf.squeeze(
            X_train_filtered
        ), tf.squeeze(X_test_filtered)

        samples_filtered = Samples(
            X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered
        )
        pipeline.fit(samples_filtered.X_train, samples_filtered.y_train)

        X_train_tfd = pipeline.transform(samples_filtered.X_train)
        X_test_tfd = pipeline.transform(samples_filtered.y_train)

        # TODO test out this step
        X_train_tfd, X_test_tfd = (X_train_tfd - X_train_tfd.min()) * (
            np.pi / (X_train_tfd.max() - X_train_tfd.min())
        ), (X_test_tfd - X_test_tfd.min()) * (
            np.pi / (X_test_tfd.max() - X_test_tfd.min())
        )

        samples_tfd = Samples(
            X_train_tfd, y_train_filtered, X_test_tfd, y_test_filtered
        )

        return samples_tfd

    else:
        # Preprocessing
        if classification_type == "ova":
            samples_filtered = samples
        elif classification_type == "ova":
            samples_filtered = samples
        elif classification_type == "binary":

            train_filter = np.where(
                (samples.y_train == target_pair[0])
                | (samples.y_train == target_pair[1])
            )
            test_filter = np.where(
                (samples.y_train == target_pair[0])
                | (samples.y_train == target_pair[1])
            )
            X_train_filtered, X_test_filtered = (
                samples.X_train[train_filter],
                samples.X_test[test_filter],
            )
            y_train_filtered, y_test_filtered = (
                samples.y_train[train_filter],
                samples.y_test[test_filter],
            )

            y_train_filtered = np.where(y_train_filtered == target_pair[1], 1, 0)
            y_test_filtered = np.where(y_test_filtered == target_pair[1], 1, 0)

            samples_filtered = Samples(
                X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered
            )
        else:
            raise NotImplementedError(
                f"There is no implementation for classification type: {classification_type}"
            )
        pipeline.fit(samples_filtered.X_train, samples_filtered.y_train)

        # Transform data
        X_train_tfd = pipeline.transform(samples_filtered.X_train)
        X_test_tfd = pipeline.transform(samples_filtered.y_train)
        samples_tfd = Samples(
            X_train_tfd, samples_filtered.y_train, X_test_tfd, samples_filtered.y_test
        )

    return samples_tfd
