#%%
import itertools
import os
from operator import mod
import pandas as pd
import pennylane as qml
from scipy.sparse import data
import circuit_presets
import json
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# Custom
from data_utility import DataUtility
from qcnn_structure import (
    QcnnStructure,
    Layer,
    train_qcnn,
)
from circuit_presets import c_1, c_2, c_3, p_1, p_2, p_3
import embedding

#%%
data_path = "../data/archive/Data/features_30_sec.csv"
target = "label"
raw = pd.read_csv(data_path)
data_utility = DataUtility(raw, target=target, default_subset="modelling")
columns_to_remove = ["filename", "length"]
data_utility.update(columns_to_remove, "included", {"value": False, "reason": "manual"})

#%%
# Specify options
# TODO compact + pure amplitude

EMBEDDING_OPTIONS = {
    8: ["Angle"],
    12: [f"Angular-Hybrid2-{i}" for i in range(1, 5)],
    16: [f"Amplitude-Hybrid2-{i}" for i in range(1, 5)] + ["Angle-Compact"],
    30: [f"Angular-Hybrid4-{i}" for i in range(1, 5)],
    32: [f"Amplitude-Hybrid4-{i}" for i in range(1, 5)],
}

# Name of circuit function from unitary.py along with param count
CIRCUIT_OPTIONS = {
    "U_TTN": 2,
    "U_5": 10,
    "U_6": 10,
    "U_9": 2,
    "U_13": 6,
    "U_14": 6,
    "U_15": 4,
    "U_SO4": 6,
    "U_SU4": 15,
}

POOLING_OPTIONS = {"psatz1": 2, "psatz2": 0, "psatz3": 3}

#%%
# Configuration
EXPERIMENT_PATH = "../experiments"
# Ensure experiment doesn't get overwritten
EXPERIMENT_ID = max([int(exp_str) for exp_str in os.listdir(EXPERIMENT_PATH)]) + 1
# EXPERIMENT_ID = 11

# levels to consider
target_levels = raw[data_utility.target].unique()
# here we get all possible pairwise comparisons
target_pairs = [target_pair for target_pair in itertools.combinations(target_levels, 2)]
# setup expermiment config
quantum_experiment_config = {
    "ID": EXPERIMENT_ID,
    "path": EXPERIMENT_PATH,
    "data": {"target_pairs": target_pairs},
    "type": "quantum",
    "preprocessing": {
        "reduction_method": "pca",
        "embedding_list": [
            "Angle",
        ],
    },
    "model": {
        "circuit_list": [
            "U_5",
        ]
    },
    "train": {
        "iterations": 100,
    },
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
        red_size: set(config["preprocessing"]["embedding_list"]) & set(embedding_option)
        for red_size, embedding_option in EMBEDDING_OPTIONS.items()
        if len((set(config["preprocessing"]["embedding_list"]) & set(embedding_option)))
        > 0
    }

    return embeddings
# Start experiment

# Get experiment config
config = quantum_experiment_config
# Set embeddings to run for the experiment
experiment_embeddings = filter_embedding_options(
    config["preprocessing"]["embedding_list"]
)
# Set circuits to run for the experiement
experiment_circuits = {
    circ_name: CIRCUIT_OPTIONS[circ_name]
    for circ_name in config["model"]["circuit_list"]
}

# Define embedding # TODO experiment function, log time taken
print(f"Running expirement: {config['ID']}")
for reduction_size, embedding_set in experiment_embeddings.items():
    for embedding_option in embedding_set:
        for circ_name, circ_param_count in experiment_circuits.items():
            for target_pair in config["data"]["target_pairs"]:
                # Only minmax scale if angle
                if "Ang" in embedding_option:
                    pipeline = Pipeline(
                        [
                            (
                                "scaler",
                                preprocessing.MinMaxScaler([0, np.pi]),
                            ),
                            ("pca", PCA(reduction_size)),
                        ]
                    )
                else:
                    pipeline = Pipeline(
                        [
                            ("pca", PCA(reduction_size)),
                        ]
                    )

                # Define QCNN structure
                layer_dict = {
                    "c_1": Layer(
                        c_1,
                        getattr(circuit_presets, circ_name),
                        "convolutional",
                        circ_param_count,
                        0,
                    ),
                    "p_1": Layer(
                        p_1,
                        getattr(circuit_presets, "psatz1"),
                        "pooling",
                        POOLING_OPTIONS["psatz1"],
                        1,
                    ),
                    "c_2": Layer(
                        c_2,
                        getattr(circuit_presets, circ_name),
                        "convolutional",
                        circ_param_count,
                        2,
                    ),
                    "p_2": Layer(
                        p_2,
                        getattr(circuit_presets, "psatz1"),
                        "pooling",
                        POOLING_OPTIONS["psatz1"],
                        3,
                    ),
                    "c_3": Layer(
                        c_3,
                        getattr(circuit_presets, circ_name),
                        "convolutional",
                        circ_param_count,
                        4,
                    ),
                    "p_3": Layer(
                        p_3,
                        getattr(circuit_presets, "psatz1"),
                        "pooling",
                        POOLING_OPTIONS["psatz1"],
                        5,
                    ),
                }

                # Create QCNN structure
                qcnn_structure = QcnnStructure(layer_dict)
                # TODO improve
                model_name = f"{config['preprocessing'].get('reduction_method', 'pca')}-{reduction_size}-{config.get('type', 'quantum')}-{embedding_option}-{circ_name}-{'-'.join(target_pair)}"
                # Train and store results
                (
                    y_hat,
                    y_hat_class,
                    loss_train_history,
                    loss_test_history,
                    params_history,
                    cf_matrix,
                ) = train_qcnn(
                    qcnn_structure,
                    embedding_option,
                    pipeline,
                    target_pair,
                    raw,
                    data_utility,
                    config,
                    model_name=model_name,
                )
print("Experiment Done")

# %%
