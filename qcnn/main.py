#%%
import os
import time
import argparse

import pandas as pd

import json

# TODO use numpy normally
import numpy as np

from data_handler import save_json, load_json, get_2d_modelling_data, get_image_data, create_train_test_samples



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
EXPERIMENT_PATH = "../experiments"
# Ensure experiment doesn't get overwritten
EXPERIMENT_ID = max([int(exp_str) for exp_str in os.listdir(EXPERIMENT_PATH)]) + 1
# EXPERIMENT_ID = 106



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
quantum_experiment_config = {
    "ID": EXPERIMENT_ID,
    "path": EXPERIMENT_PATH,
    "data": {
        "target_pairs": [
            (0, 1),
            (1,7)
            # ("disco", "rock"),
            # ("hiphop", "pop"),
            # ("country", "reggae"),
            # ("jazz", "metal"),
        ],
        "type":"image",
        "path":"../data/archive/Data/features_30_sec.csv",
        "sampling":{ # This sampling creates a test set which is never seen during any model training for all experiments.. this can be
                     # can be done differently, like based on time the dataset naturally is split into a train / test
            "test_size": 0.3,
            "random_state": 42,
            }
        
    },
    "preprocessing": {
        "quantum": {
            "Angle": {
                "scaler": {
                    "method": [None],
                    "standard_params": {},
                    "minmax_params": {
                        "feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]
                    },
                },
                "feature_selection": {
                    "method": ["pca", "tree"],
                    "pca_params": {"n_components": [8]},
                    "tree_params": {"max_features": [8], "n_estimators": [50]},
                },
                "ignore": False,
            },
            "IQP": {
                "scaler": {
                    "method": [None],
                    "standard_params": {},
                    "minmax_params": {
                        "feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]
                    },
                },
                "feature_selection": {
                    "method": ["pca", "tree"],
                    "pca_params": {"n_components": [8]},
                    "tree_params": {"max_features": [8], "n_estimators": [50]},
                },
                "kwargs": {"depth": 10},
                "ignore": False,
            },
            "Amplitude": {
                "scaler": {
                    "method": [None],
                    "standard_params": {},
                    "minmax_params": {
                        "feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]
                    },
                },
                "feature_selection": {
                    "method": ["pca", "tree"],
                    "pca_params": {"n_components": [8]},
                    "tree_params": {"max_features": [8], "n_estimators": [50]},
                },
                "ignore": False,
            },
        },
        "classical": {
            "normal": {
                "scaler": {
                    "method": ["standard"],
                    "standard_params": {},
                    "minmax_params": {
                        "feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]
                    },
                },
                "feature_selection": {
                    "method": ["pca", "tree"],
                    "pca_params": {"n_components": [8]},
                    "tree_params": {"max_features": [8], "n_estimators": [50]},
                },
                "ignore": False,
            },
        },
    },
    "model": {
        "quantum": {
            "qcnn": 
                {"circuit_list": ["U_5", "U_SU4"],
                 "pooling_list":["psatz1"],
                 "custom_structures":{"custom_0":{    # order, convolution function name, circuite name, circuit paramaters, layer_name i.e. the keys should start with c for convolutional or p for pooling                          
                                        "c_1": [0, "c_1", "U_5"],
                                        "p_1": [1, "p_1", "psatz1"],
                                        "c_2": [2, "c_2", "U_SU4"],
                                        "p_2": [3, "p_2", "psatz1"],
                                        "c_3": [4, "c_3", "U_6"],
                                        "p_3": [5, "p_3", "psatz1"],
                                    },
                                    "custom_1":{                                 
                                        "c_1": [0, "c_1", "U_5"],
                                        "p_1": [1, "p_1", "psatz1"],
                                        "c_2": [2, "c_2", "U_SU4"],
                                        "p_2": [3, "p_2", "psatz1"],
                                        "c_3": [4, "c_3", "U_5"],
                                        "p_3": [5, "p_3", "psatz1"],
                                    }
                                }, "ignore": False
            }
        },
        "classical": {
            "logistic_regression": {"param_grid": {"C":np.logspace(-3,3,7).tolist(), "penalty":["l1","l2"]}, "ignore": True},
            "svn": {"param_grid": {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [1, 10, 100]}, "ignore": True},
            "cnn": {"param_grid": {}, "ignore": True},
        },
        "classification_type": "binary",
    },
    "train": {
        "iterations": 40,
        
    },
    "extra_info": "main",
}
# Start experiment

# %%
# Get experiment config
config = quantum_experiment_config

# == Rerun previous experiment ==#
# EXPERIMENT_PATH = "../experiments"
# EXPERIMENT_ID = 104
# with open(f"{EXPERIMENT_PATH}/{EXPERIMENT_ID}/experiment_config.json", "r") as f:
#     config = json.load(f)
# == Rerun previous experiment ==#

# Split data by creating test(unseen by any model)


# test_size = config["train"].get("test_size", 0.3)
# random_state = config["train"].get("random_state", 42)

# if not image_data:
#     X, y, Xy = data_utility.get_samples(raw)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=test_size,
#         random_state=random_state,
#     )
#     data_utility.row_sample["train"] = X_train.index
#     data_utility.row_sample["test"] = X_test.index
# else:
#     data_utility = DataUtility(pd.DataFrame({"a":[1,2,3]}))

# def run_experiment(config):
#     y_hat_history = {
#         "model_name": [],
#         "target_pair": [],
#         "y_hat": [],
#         "X_test_ind": [],
#         "best_params": [],
#     }
#     result_path = f"{config.get('path')}/{config.get('ID')}"
#     model_time = {}
#     for model_type in ("quantum", "classical"):
#         for embedding_type in config["preprocessing"].get(model_type):
#             if config["preprocessing"][model_type][embedding_type].get("ignore") is False:
#                 for scaler_method in config["preprocessing"][model_type][embedding_type]["scaler"].get(
#                     "method", "standard"
#                 ):
#                     for selection_method in config["preprocessing"][model_type][embedding_type]["feature_selection"].get("method", "pca"):
#                         scaler_param = config["preprocessing"][model_type][embedding_type]["scaler"].get(f"{scaler_method}_params", {})
#                         selection_param = config["preprocessing"][model_type][embedding_type]["feature_selection"].get(
#                             f"{selection_method}_params", {}
#                         )
#                         # create dictionary of every possible paramater permutation
#                         scaler_keys, scaler_values = (
#                             zip(*scaler_param.items())
#                             if len(scaler_param.keys()) > 0
#                             else zip([[], []])
#                         )
#                         scaler_permutations_dicts = [
#                             dict(zip(scaler_keys, v)) for v in it.product(*scaler_values)
#                         ]
#                         # Ensure there is atleast one empty dictionary
#                         scaler_permutations_dicts = (
#                             [{}]
#                             if len(scaler_permutations_dicts) == 0
#                             else scaler_permutations_dicts
#                         )
#                         selection_keys, selection_values = (
#                             zip(*selection_param.items())
#                             if len(selection_param.keys()) > 0
#                             else zip([[], []])
#                         )
#                         selection_permutations_dicts = [
#                             dict(zip(selection_keys, v)) for v in it.product(*selection_values)
#                         ]
#                         # Ensure there is atleast one empty dictionary
#                         selection_permutations_dicts = (
#                             [{}]
#                             if len(selection_permutations_dicts) == 0
#                             else selection_permutations_dicts
#                         )
#                         for selection_param_permutation in selection_permutations_dicts:
#                             for scaler_param_permutation in scaler_permutations_dicts:
#                                 preprocessing_config = {
#                                     "scaler": {
#                                         "method": scaler_method,
#                                         f"{scaler_method}_params": scaler_param_permutation,
#                                     },
#                                     "feature_selection": {
#                                         "method": selection_method,
#                                         f"{selection_method}_params": selection_param_permutation,
#                                     },
#                                 }
#                                 pipeline = get_preprocessing_pipeline(preprocessing_config)
#                                 selection_param_str = "-".join(
#                                     [f"{k}={v}" for k, v in selection_param_permutation.items()]
#                                 )
#                                 scaler_param_str = "-".join(
#                                     [f"{k}={v}" for k, v in scaler_param_permutation.items()]
#                                 )
#                                 for algorithm in config["model"].get(model_type):
#                                     # quantum algo like qcnn or some other quantum model
#                                     if (
#                                         config["model"][model_type][algorithm].get("ignore", True)
#                                         == False
#                                     ):
#                                         for target_pair in config["data"]["target_pairs"]:
#                                             prefix = (
#                                                 f"{model_type}-{embedding_type}-{scaler_method}"
#                                                 f"-{scaler_param_str}-{selection_method}-{selection_param_str}-{algorithm}"
#                                             )
#                                             if model_type == "quantum":
#                                                 experiment_circuits = {
#                                                     circ_name: CIRCUIT_OPTIONS[circ_name]
#                                                     for circ_name in config["model"][
#                                                         model_type
#                                                     ][algorithm]["circuit_list"]
#                                                 }
#                                                 for (
#                                                     circ_name,
#                                                     circ_param_count,
#                                                 ) in experiment_circuits.items():
#                                                     model_name = f"{prefix}-{circ_name}-{target_pair}"
#                                                     if not (
#                                                         os.path.exists(
#                                                             f"{result_path}/{model_name}-confusion-matrix.csv"
#                                                         )
#                                                     ):
#                                                         # Define QCNN structure
#                                                         # fmt: off
#                                                         layer_dict = {
#                                                             "c_1": Layer(c_1, getattr(circuit_presets, circ_name), "convolutional", circ_param_count, 0,),
#                                                             "p_1": Layer(p_1, getattr(circuit_presets, "psatz1"),"pooling",POOLING_OPTIONS["psatz1"],1,),
#                                                             "c_2": Layer(c_2, getattr(circuit_presets, circ_name),"convolutional",circ_param_count,2,),
#                                                             "p_2": Layer(p_2, getattr(circuit_presets, "psatz1"),"pooling",POOLING_OPTIONS["psatz1"],3,),
#                                                             "c_3": Layer(c_3, getattr(circuit_presets, circ_name),"convolutional",circ_param_count,4,),
#                                                             "p_3": Layer(p_3, getattr(circuit_presets, "psatz1"),"pooling",POOLING_OPTIONS["psatz1"],5,),
#                                                         }

#                                                         # Create QCNN structure
#                                                         # fmt: off
#                                                         qcnn_structure = QcnnStructure(layer_dict)
#                                                         # fmt: off
#                                                         y_hat_history = {"model_name": [],"target_pair": [],"y_hat": [],"X_test_ind": [],"best_params": [],}
#                                                         t1 = time.time()
#                                                         # Train and store results
#                                                         (
#                                                             y_hat,
#                                                             X_test_ind,
#                                                             best_params,
#                                                             cf_matrix,
#                                                         ) = train_qcnn(
#                                                             qcnn_structure,
#                                                             embedding_type,
#                                                             pipeline,
#                                                             target_pair,
#                                                             raw.copy(),
#                                                             data_utility,
#                                                             config,
#                                                             model_name=model_name,
#                                                         )
#                                                         t2 = time.time()
#                                                         y_hat_history["model_name"].append(model_name)
#                                                         y_hat_history["target_pair"].append(target_pair)
#                                                         y_hat_history["y_hat"].append(y_hat)
#                                                         y_hat_history["X_test_ind"].append(X_test_ind)
#                                                         y_hat_history["best_params"].append(best_params)
#                                                         model_time[f"{model_name}"] = t2 - t1
#                                                 if config["model"]["classification_type"] == "ovo":
#                                                 # If model should apply ovo strategy, TODO this can be done better if I can represent the model as a SKLearn classifier
#                                                     y_class_multi, row_prediction_history = get_ovo_classication(
#                                                         y_hat_history, y_test, config, store_results=True, prefix=prefix
#                                                     )
#                                                 elif config["model"]["classification_type"] == "ova":
#                                                     # If model should apply ovo strategy, TODO this can be done better if I can represent the model as a SKLearn classifier
#                                                     y_class_multi, row_prediction_history = get_ova_classication(
#                                                         y_hat_history, y_test, config, store_results=True, prefix=prefix
#                                                     )
#                                             elif model_type == "classical":
#                                                 model_name = f"{prefix}-{algorithm}-{target_pair}"
#                                                 # train classical modelmodel_classical
#                                                 t1 = time.time()
#                                                 # Train and store results                                            
#                                                 train_classical(
#                                                     algorithm,
#                                                     pipeline,
#                                                     target_pair,
#                                                     raw.copy(),
#                                                     data_utility,
#                                                     config,
#                                                     model_name=model_name,
#                                                 )
#                                                 t2 = time.time()
#                                                 model_time[f"{model_name}"] = t2 - t1
                                            
                                        

# Give expirment context
# with open(f"{result_path}/experiment_time.json", "w+") as f:
#     json.dump(model_time, f, indent=4)

# print("Experiment Done")


def main(args):
    # Load experiment config
    config = load_json(args["config_path"])
    # Load data
    
    if config["data"].get("type", None)=="image":
        # With image data raw is a list consisting of X_train, y_train X_test, y_test
        samples = get_image_data(config["data_path"]) 
    else:
        raw, data_utility = get_2d_modelling_data(config["data"].get("path"))
        # Datautility should be used only here to transform the data into a desirable train test set
        # Then when the xeperiment is ran it is assumed that all "columns" and rows as is needs to be used
        # TODO check if data_utility updates
        samples = create_train_test_samples(raw, data_utility, test\_size=conig)
    model_execution_times = run_experiment(config, samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a machine learning experiment')

    # Add the arguments
    parser.add_argument('config_path',
                        metavar='config_path',
                        type=str,
                        help='the path to the experiment config (json)')
    
    args = parser.parse_args()
    main(args)

    
# %%
from collections import namedtuple
Samples = namedtuple("Samples", ["X_train", "y_train", "X_test", "y_test"])
samples = Samples(1,2,3,4)
# %%
