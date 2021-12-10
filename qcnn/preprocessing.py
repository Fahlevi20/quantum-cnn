# Preprocessing should contain embedding + feature reduction logic
import numpy as np
import os
from joblib import dump
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


class ImageResize(BaseEstimator, TransformerMixin):
    """
    Resizes an image
    """

    def __init__(self, size=None):
        self.size = size

    def fit(self, X, y=None):
        """returns itself"""
        if self.size == None:
            # assume image is n * width * height np array
            self.size = X.shape[1] * X.shape[2]
        return self

    def transform(self, X, y=None):
        """TODO automatically does squeezing.. this might not be wanted check"""
        X_resize = tf.image.resize(X[..., np.newaxis][:], (self.size, 1)).numpy()
        X_squeezed = tf.squeeze(X_resize).numpy()
        return X_squeezed


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


def get_preprocessing_pipeline(
    scaler_method, scaler_params, selection_method, selection_params, custom_steps=None
):
    """
    TODO update docstring it's out of date
    Returns a pipeline that handles the pre-processing part of the model (this step is quantum/classical agnostic).
    Currently the preprocessing pipeline consists of two steps, a scaling and feature selection step. Each of these have
    different configurable properties. This function takes in a slice of a more general config (1 permutation in a sense...example below):
    A general config will contain all the paramaters to try out


    """
    if custom_steps == None:
        all_custom_steps = []
        custom_step = (
            "identity_step",
            IdentityTransformer(),
        )
        all_custom_steps = all_custom_steps + [custom_step]
    else:
        # If there's custom steps to apply, for now it gets applied before
        
        all_custom_steps = []
        for step, step_info in custom_steps.items():
            if step_info["name"] == "image_resize":
                custom_step = (step_info["name"], ImageResize(**step_info["params"]))
            all_custom_steps = all_custom_steps + [custom_step]
    # Define Scaler
    if scaler_method == None or scaler_method == "identity":
        scaler = (
            "identity_scaler",
            IdentityTransformer(),
        )
    elif scaler_method == "minmax":
        scaler = (
            scaler_method,
            preprocessing.MinMaxScaler(**scaler_params),
        )
    elif scaler_method == "standard":
        scaler = (
            scaler_method,
            preprocessing.StandardScaler(**scaler_params),
        )

    # Define feature selector
    if selection_method == None or selection_method == "identity":
        selection = (
            "identity_selection",
            IdentityTransformer(),
        )
    elif selection_method == "tree":
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
    elif selection_method == "image_resize":
        selection = (selection_method, ImageResize(**selection_params))

    steps_list = all_custom_steps + [scaler, selection]
    pipeline = Pipeline(steps_list)

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
    samples,
    pipeline,
    classification_type,
    data_type,
    target_pair=None,
    model_name="dummy",
    result_path=None,
):

    if data_type == "image":
        if classification_type == "ovo":
            samples_filtered = samples
        elif classification_type == "ova":
            samples_filtered = samples
        elif classification_type == "binary":
            train_filter = np.where(
                (samples.y_train == target_pair[0]) | (samples.y_train == target_pair[1])
            )

            test_filter = np.where(
                (samples.y_test == target_pair[0]) | (samples.y_test == target_pair[1])
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

            samples_filtered = Samples(
                X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered
            )
        pipeline.fit(samples_filtered.X_train, samples_filtered.y_train)

        X_train_tfd = pipeline.transform(samples_filtered.X_train)
        X_test_tfd = pipeline.transform(samples_filtered.X_test)

        samples_tfd = Samples(
            X_train_tfd, y_train_filtered, X_test_tfd, y_test_filtered
        )

        return samples_tfd

    else:
        # Preprocessing
        if classification_type == "ovo":
            samples_filtered = samples
        elif classification_type == "ova":
            samples_filtered = samples
        elif classification_type == "binary":

            train_filter = np.where(
                (samples.y_train == target_pair[0])
                | (samples.y_train == target_pair[1])
            )
            test_filter = np.where(
                (samples.y_test == target_pair[0]) | (samples.y_test == target_pair[1])
            )
            X_train_filtered, X_test_filtered = (
                samples.X_train.iloc[train_filter],
                samples.X_test.iloc[test_filter],
            )
            y_train_filtered, y_test_filtered = (
                samples.y_train.iloc[train_filter],
                samples.y_test.iloc[test_filter],
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
        X_test_tfd = pipeline.transform(samples_filtered.X_test)
        samples_tfd = Samples(
            X_train_tfd, samples_filtered.y_train, X_test_tfd, samples_filtered.y_test
        )
    # Save pipeline TODO function
    if result_path:
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        dump(pipeline, f"{result_path}/{model_name}-pipeline.joblib")
    return samples_tfd
