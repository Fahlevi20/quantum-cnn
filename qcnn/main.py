#%%
import itertools
import os
import time
import pandas as pd
import circuit_presets
import json

# TODO use numpy normally
from pennylane import numpy as np

from sklearn.model_selection import train_test_split

# Custom
from data_utility import DataUtility
from preprocessing import (
    get_preprocessing_pipeline,
    filter_embedding_options,
    EMBEDDING_OPTIONS,
)
from postprocessing import get_ovo_classication
from qcnn_structure import (
    QcnnStructure,
    Layer,
    train_qcnn,
)
from circuit_presets import (
    c_1,
    c_2,
    c_3,
    p_1,
    p_2,
    p_3,
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

# Levels to consider
target_levels = raw[data_utility.target].unique()
# Here we get all possible pairwise comparisons
target_pairs = [target_pair for target_pair in itertools.combinations(target_levels, 2)]
# Setup expermiment config
quantum_experiment_config = {
    "ID": EXPERIMENT_ID,
    "path": EXPERIMENT_PATH,
    "data": {
        "target_pairs": target_pairs,
    },
    "type": "quantum",
    "preprocessing": {
        "reduction_method": "pca",
        "scaler": {
            "Angle": [0, np.pi / 2],
            "Havlicek": [-1, 1],
        },
        "embedding_list": ["Angle"],
    },
    "model": {"circuit_list": ["U_5"], "multi_class": "ovo"},
    "train": {
        "iterations": 1,
        "test_size": 0.3,
        "random_state": 40,
    },
    "extra_info": "debug",
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

result_path = (
    f"{quantum_experiment_config.get('path')}/{quantum_experiment_config.get('ID')}"
)

# Define embedding # TODO experiment function, log time taken
print(f"Running expirement: {config['ID']}")
model_time = {}
for reduction_size, embedding_set in experiment_embeddings.items():
    for embedding_option in embedding_set:
        for circ_name, circ_param_count in experiment_circuits.items():
            # Set prefix to define model
            prefix = f"{config['preprocessing'].get('reduction_method', 'pca')}-{reduction_size}-{config.get('type', 'quantum')}-{embedding_option}-{circ_name}"
            test_size = config["train"].get("test_size", 0.3)
            random_state = config["train"].get("random_state", 42)
            X, y, Xy = data_utility.get_samples(raw)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
            data_utility.row_sample["train"] = X_train.index
            data_utility.row_sample["test"] = X_test.index
            y_hat_history = {
                "model_name": [],
                "target_pair": [],
                "y_hat": [],
                "X_test_ind": [],
                "best_params": [],
            }
            for target_pair in config["data"]["target_pairs"]:
                # Get preprocessing pipeline for configuration
                pipeline = get_preprocessing_pipeline(
                    embedding_option, reduction_size, config
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
                model_name = f"{prefix}-{'-'.join(target_pair)}"
                t1 = time.time()
                # Train and store results
                (y_hat, X_test_ind, best_params, cf_matrix,) = train_qcnn(
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
                y_hat_history["model_name"].append(model_name)
                y_hat_history["target_pair"].append(target_pair)
                y_hat_history["y_hat"].append(y_hat)
                y_hat_history["X_test_ind"].append(X_test_ind)
                y_hat_history["best_params"].append(best_params)
                model_time[f"{model_name}"] = t2 - t1

            # Store test set
            y_test.to_csv(f"{result_path}/{prefix}-ytest.csv")

            if config["model"]["multi_class"] == "ovo":
                # If model should apply ovo strategy, TODO this can be done better if I can represent the model as a SKLearn classifier
                y_class_multi, row_prediction_history = get_ovo_classication(
                    y_hat_history, y_test, config, store_results=True, prefix=prefix
                )


# Give expirment context
with open(f"{result_path}/experiment_time.json", "w+") as f:
    json.dump(model_time, f, indent=4)

print("Experiment Done")

# %%
