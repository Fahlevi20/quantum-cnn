#%%
from ast import Str
import itertools
import os
import time
from numpy.lib.function_base import append
import pandas as pd
import circuit_presets
import json

from collections import Counter
import pickle

from pennylane import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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
    "data": {
        "target_pairs": target_pairs,
    },
    "type": "quantum",
    "preprocessing": {
        "reduction_method": "pca",
        "scaler": {
            "angle": "MinMaxScaler([0, np.pi / 2])",
            "Havlicek": "MinMaxScaler([0, np.pi])",
        },
        "embedding_list": ["Angle"],
    },
    "model": {"circuit_list": ["U_5"]},
    "train": {
        "iterations": 100,
        "test_size": 0.3,
        "random_state":40,
    },
    "extra_info": "with qml.RZ(np.pi - X[i], wires=[i]) # i.e with pi minus",
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
            test_size = config["train"].get("test_size", 0.3)
            random_state = config["train"].get("random_state", 42)
            # if config["multi_class"] == True:
            #     # Multiclass currently means we are going to do a a pairwise comparison
            #     # for all possible (can be extended to specified) level pairs. If there are 10 levels
            #     # to predict then there are 45 possible comparisons 10c2
            #     # TODO make this configurable in a better way
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
                # TODO move to function
                y_hat_history["model_name"].append(model_name)
                y_hat_history["target_pair"].append(target_pair)
                y_hat_history["y_hat"].append(y_hat)
                y_hat_history["X_test_ind"].append(X_test_ind)
                y_hat_history["best_params"].append(best_params)
                model_time[f"{model_name}"] = t2 - t1

            y_hat_history = pd.DataFrame(y_hat_history)
            y_class_multi = pd.Series(dtype=str)
            row_prediction_history = {}
            # Calculate overall performance on test said OneVsOne style
            for test_idx, test_row in y_test.iteritems():
                # Which models predicted this row
                model_ind = y_hat_history["X_test_ind"].isin(
                    [item for item in y_hat_history["X_test_ind"] if test_idx in item]
                )
                if model_ind.any():
                    # Which of the models predictions corresponds to the specific rows
                    row_predictions = {"label": [], "y_hat": []}
                    for model_index, model_row in y_hat_history[model_ind].iterrows():
                        prediction_idx = list(model_row["X_test_ind"]).index(test_idx)
                        y_hat_tmp = model_row["y_hat"][prediction_idx]
                        mx_idx = list(y_hat_tmp).index(max(y_hat_tmp))
                        label = model_row["target_pair"][mx_idx]
                        # .numpy to convert from tensor to value
                        mx_yhat = max(y_hat_tmp).numpy()
                        row_predictions["label"].append(label)
                        row_predictions["y_hat"].append(mx_yhat)
                    # Index of row predictions having highest prediction

                    value_counts = Counter(row_predictions["label"])
                    # Get most common label from comparisons
                    final_label = value_counts.most_common()[0][0]
                    # if value_counts.most_common()[0][1] == 1:
                    #     # If all occur the same amount of times, choose most confident occurance (still not really a good way to do it though)
                    #     best_idx = list(row_predictions["y_hat"]).index(
                    #         max(row_predictions["y_hat"])
                    #     )
                    #     final_label = row_predictions["label"][best_idx]

                    y_class_multi.loc[test_idx] = final_label
                    row_prediction_history[test_idx] = row_predictions
            # Store prefix
            prefix = f"{result_path}/{config['preprocessing'].get('reduction_method', 'pca')}-{reduction_size}-{config.get('type', 'quantum')}-{embedding_option}-{circ_name}"
            with open(f"{prefix}-row-prediction-history.pkl", "wb+") as f:
                pickle.dump(row_prediction_history, f, pickle.HIGHEST_PROTOCOL)
            # how to load again
            # with open(f"{prefix}-row-prediction-history.pkl", 'rb') as f:
            #     pickle.load(f)
            y_class_multi.to_csv(f"{prefix}-yclass-multi.csv")
            y_test.to_csv(f"{prefix}-ytest.csv")


# Give expirment context
with open(f"{result_path}/experiment_time.json", "w+") as f:
    json.dump(model_time, f, indent=4)

print("Experiment Done")

# %%
