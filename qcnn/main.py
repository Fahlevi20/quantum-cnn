#%%
import itertools
import os
import time
import pandas as pd
import circuit_presets
import json
from pennylane import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# Custom
from data_utility import DataUtility
from qcnn_structure import (
    QcnnStructure,
    Layer,
    train_qcnn,
)
from circuit_presets import (
    filter_embedding_options,
    c_1,
    c_2,
    c_3,
    p_1,
    p_2,
    p_3,
    EMBEDDING_OPTIONS,
    CIRCUIT_OPTIONS,
    POOLING_OPTIONS,
)


#%%
data_path = "../data/archive/Data/features_30_sec.csv"
target = "label"
raw = pd.read_csv(data_path)
data_utility = DataUtility(raw, target=target, default_subset="modelling")
columns_to_remove = ["filename", "length"]
data_utility.update(columns_to_remove, "included", {"value": False, "reason": "manual"})


#%%
# Configuration
EXPERIMENT_PATH = "../experiments"
# Ensure experiment doesn't get overwritten
EXPERIMENT_ID = max([int(exp_str) for exp_str in os.listdir(EXPERIMENT_PATH)]) + 1
# EXPERIMENT_ID = 20

# levels to consider
target_levels = raw[data_utility.target].unique()
# here we get all possible pairwise comparisons
target_pairs = [target_pair for target_pair in itertools.combinations(target_levels, 2)]
# setup expermiment config
quantum_experiment_config = {
    "ID": EXPERIMENT_ID,
    "path": EXPERIMENT_PATH,
    "data": {"target_pairs": [("pop", "classical")]},
    "type": "quantum",
    "preprocessing": {
        "reduction_method": "pca",
        "scaler": {"angle": None, "Havlicek": "StandardScalar"},
        "embedding_list": ["Havlicek"],
    },
    "model": {"circuit_list": ["U_5", "U_TTN", "U_6"]},
    "train": {
        "iterations": 100,
    },
    "extra": "double:\nn_col = X.shape[0]"
    "\nfor i in range(n_col):"
    "\nqml.Hadamard(wires=[i])"
    "\nfor i in range(n_col):"
    "\nqml.RZ(X[i], wires=[i])"
    "\nfor i in range(n_col - 1):"
    "\nj = i + 1"
    "\nqml.CNOT(wire=[i, j])"
    "\nqml.RZ((np.pi - X[i]) * (np.pi - X[j]), wire=[i, j])"
    "\nqml.CNOT(wire=[i, j])",
}
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
model_time = {}
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
                                preprocessing.MinMaxScaler([0, np.pi / 2]),
                            ),
                            ("pca", PCA(reduction_size)),
                        ]
                    )
                elif "Havlicek" in embedding_option:
                    pipeline = Pipeline(
                        [
                            (
                                "scaler",
                                preprocessing.StandardScaler(),
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
                t1 = time.time()
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
                t2 = time.time()
                model_time[f"{model_name}"] = t2 - t1

result_path = (
    f"{quantum_experiment_config.get('path')}/{quantum_experiment_config.get('ID')}"
)
# Give expirment context
with open(f"{result_path}/experiment_time.json", "w+") as f:
    json.dump(model_time, f, indent=4)

print("Experiment Done")

# %%
