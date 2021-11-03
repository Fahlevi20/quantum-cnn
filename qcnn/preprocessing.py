# Preprocessing should contain embedding + feature reduction logic
import numpy as np

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


def get_preprocessing_pipeline(encoding_option, reduction_size, config):
    """Returns the pipeline associated with provided encoding option.

    Args:
        embedding_option (str): Embedding option name, ex. Angle, Amplitude-Hybrid4-2
        reduction_size (int): Size of reduction, corresponds to how many features should be left over
        scale_range (list(float, float)): range that features should be scaled between if applicable.
    """
    reduction_method = config["preprocessing"].get("reduction_method")
    if reduction_method == "tree":
        reduction_step = (
            reduction_method,
            SelectFromModel(
                ExtraTreesClassifier(n_estimators=50), max_features=reduction_size
            ),
        )
    else:
        reduction_step = (reduction_method, PCA(reduction_size))
    if "Ang" in encoding_option:
        scale_range = config["preprocessing"]["scaler"].get(
            encoding_option, [0, np.pi / 2]
        )
        pipeline = Pipeline(
            [
                (
                    "scaler",
                    preprocessing.MinMaxScaler(scale_range),
                ),
                reduction_step,
            ]
        )
    elif ("ZZMap" in encoding_option) or ("IQP" in encoding_option):
        scale_range = config["preprocessing"]["scaler"].get(encoding_option, [-1, 1])
        pipeline = Pipeline(
            [
                (
                    "scaler",
                    preprocessing.MinMaxScaler(scale_range),
                ),
                reduction_step,
            ]
        )
    elif "Amplitude" in encoding_option:
        pipeline = Pipeline(
            [
                reduction_step,
            ]
        )
    else:
        scale_range = [-1, 1]
        pipeline = Pipeline(
            [
                (
                    "scaler",
                    preprocessing.MinMaxScaler(scale_range),
                ),
                reduction_step,
            ]
        )
    return pipeline
