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
    scaler_method = config["scaler"].get("method", "minmax")
    scaler_params = config["scaler"].get(f"{scaler_method}_params", [])
    selection_method = config["feature_selection"].get("method", "pca")
    selection_params = config["feature_selection"].get(f"{selection_method}_params", [])

    # Define Scaler
    if scaler_method == "minmax":
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
