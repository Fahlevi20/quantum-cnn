# %%
import os
import time
import itertools as it
import numpy as np
from sklearn import feature_selection

import tensorflow as tf

import circuit_presets
from qcnn_structure import (
    QcnnStructure,
    Layer,
    train_qcnn,
)

from classical_models import train_classical
from preprocessing import (
    get_preprocessing_pipeline,
)
from postprocessing import get_ovo_classication, get_ova_classication
from sklearn.model_selection import ParameterGrid

# %%
# config = {
#         "a": [1,2,3],
#         "b": [4,5,6]
#         }
# general_config = {
#                 "scaler": {
#                     "method": ["standard", "minmax"],
#                     "standard_params": {},
#                     "minmax_params": {"feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]},
#                 },
#                 "feature_selection": {
#                     "method": ["pca"],
#                     "pca_params": {"n_components": [8]},
#                     "tree_params": {"max_features": [8], "n_estimators": [50]},
#                 },
#             }

# general_config = {
#     "scaler": {
#         "method": {
#             "standard": {},
#             "minmax": {"feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]},
#         },
#         "ignore": False,
#     },
#     "feature_selection": {
#         "method": {
#             "pca": {"n_components": [8]},
#             "tree": {"max_features": [8], "n_estimators": [50]},
#         },
#     },
# }
# for scaler, feature_selection in it.product(general_config["scaler"].get("method"), general_config["feature_selection"].get("method")):
#     scaler_params = list(ParameterGrid(general_config["scaler"]["method"].get(scaler)))
#     feature_selection_params = list(ParameterGrid(general_config["feature_selection"]["method"].get(feature_selection)))
#     ordered_combinations = list(
#         it.product(scaler_params, feature_selection_params)
#     )  # Cartesian product is fine here since a(first element) from S1 is different from a(first element) from S2, i.e the combination (a,a) is unique and (a,b)<>(b,a)
#     for preprocess_combination in ordered_combinations:
#         print(f"{scaler}-{feature_selection}\n{preprocess_combination}")

# %%
# def get_dict_permutation(dict_of_list):
#     """[summary]
#     """
# %%
import itertools as it
print(list(it.product([],[1])))

# %%
def run_quantum_model(config, embedding_type, scaler_method, scaler_param_str, selection_method, selection_param_str, algorithm, result_path, pipeline, samples):
    model_type ="quantum"
    prefix = (
                f"{model_type}-{embedding_type}-{scaler_method}"
                f"-{scaler_param_str}-{selection_method}-{selection_param_str}-{algorithm}"
            )
    circuit_list = config["model"][model_type][algorithm].get("circuit_list", [])
    pooling_list = config["model"][model_type][algorithm].get("pooling_list", [])
    classification_type = config["model"].get("classification_type", None)
    model_time = {}
    for circ_pool_combination in it.product(circuit_list, pooling_list):
        if classification_type == "binary":
            for target_pair in config["data"].get("target_pairs", []):
                model_name = f"{prefix}-{circ_pool_combination[0]}-{target_pair}"
                if not (
                    os.path.exists(
                        # Check to see if model was already built
                        f"{result_path}/{model_name}-confusion-matrix.csv"
                    )
                ):
                    qcnn_structure = QcnnStructure(circ_pool_combination)
                    t1 = time.time()
                    train_qcnn(
                        qcnn_structure,
                        embedding_type,
                        pipeline,
                        target_pair,
                        samples,
                        config,
                        model_name=model_name,
                    )
                    t2 = time.time()
                    model_time[f"{model_name}"] = t2 - t1
                for custom_structure in config["model"][model_type][algorithm].get("custom_structures", []):
                    model_name = f"{prefix}-{custom_structure}-{target_pair}"
                    if not (
                        os.path.exists(
                            # Check to see if model was already built
                            f"{result_path}/{model_name}-confusion-matrix.csv"
                        )
                    ):
                        qcnn_structure = QcnnStructure(custom_structure)
                        t1 = time.time()
                        train_qcnn(
                            qcnn_structure,
                            embedding_type,
                            pipeline,
                            target_pair,
                            samples,
                            config,
                            model_name=model_name,
                        )
                        t2 = time.time()
                        model_time[f"{model_name}"] = t2 - t1
    return model_time
    
    # if config["model"]["classification_type"] == "ovo":
    #     # TODO this should work by loading saved files and then doing ovo or ova
    #     # If model should apply ovo strategy, TODO this can be done better if I can represent the model as a SKLearn classifier
    #     y_class_multi, row_prediction_history = get_ovo_classication(
    #         y_hat_history, y_test, config, store_results=True, prefix=prefix
    #     )
    # elif config["model"]["classification_type"] == "ova":
    #     # If model should apply ovo strategy, TODO this can be done better if I can represent the model as a SKLearn classifier
    #     y_class_multi, row_prediction_history = get_ova_classication(
    #         y_hat_history, y_test, config, store_results=True, prefix=prefix
    #     )
# I can take the preprocessing one step further by searching for the built in methods in SKLearn

def run_classical_model(config, samples, pipeline, prefix, algorithm):
    classification_type = config["model"].get("classification_type", None)
    model_time = {}
    # TODO below is a bit redundant, the train_classical function can probably handle the target_pair + classification
    # type logic, but can be improved in future
    if classification_type == "binary":
        for target_pair in config["data"].get("target_pairs", []):
            model_name = f"{prefix}-{algorithm}-{target_pair}"
            t1 = time.time()
            # Train and store results
            train_classical(
                config,
                algorithm,
                pipeline,
                samples,
                target_pair=target_pair,
                model_name=model_name,
            )
            t2 = time.time()
            model_time[f"{model_name}"] = t2 - t1
    elif classification_type in ["ovo", "ova"]:
            model_name = f"{prefix}-{algorithm}-{classification_type}"
            t1 = time.time()
            # Train and store results
            train_classical(
                config,
                algorithm,
                pipeline,
                samples,
                model_name=model_name,
            )
            t2 = time.time()
            model_time[f"{model_name}"] = t2 - t1

    return (t2 - t1)

def run_experiment(config, samples):
    result_path = f"{config.get('path')}/{config.get('ID')}"
    all_model_time = {}
    for model_type in ("quantum", "classical"):
        for embedding_type in config["preprocessing"].get(model_type):
            if config["preprocessing"][model_type][embedding_type].get("ignore") is False:
                for scaler_method, selection_method in it.product(config["preprocessing"][model_type][embedding_type]["scaler"].get("method"), config["preprocessing"][model_type][embedding_type]["feature_selection"].get("method")):
                    scaler_params = list(ParameterGrid(config["preprocessing"][model_type][embedding_type]["scaler"]["method"].get(scaler_method)))
                    selection_params = list(ParameterGrid(config["preprocessing"][model_type][embedding_type]["feature_selection"]["method"].get(selection_method)))
                    ordered_combinations = list(
                        it.product(scaler_params, selection_params)
                    )  # Cartesian product is fine here since a(first element) from S1 is different from a(first element) from S2, i.e the combination (a,a) is unique and (a,b)<>(b,a)
                    for scaler_param, selection_parm in ordered_combinations:
                        preprocessing_config=dict(zip([scaler_method, selection_method], [scaler_param, selection_parm]))
                        pipeline = get_preprocessing_pipeline(preprocessing_config)                
                        selection_param_str = "-".join(
                            [f"{k}={v}" for k, v in scaler_param.items()]
                        )
                        scaler_param_str = "-".join(
                            [f"{k}={v}" for k, v in selection_method.items()]
                        )
                        for algorithm in config["model"].get(model_type):
                            # quantum algo like qcnn or some other quantum model
                            if (
                                config["model"][model_type][algorithm].get("ignore", True)
                                == False
                            ):
                                
                                all_model_time[f"{algorithm}"] = {}
                                if model_type=="quantum":
                                    model_time= run_quantum_model(config, embedding_type, scaler_method, scaler_param_str, selection_method, selection_param_str, algorithm, result_path, pipeline, samples)
                                elif model_type == "classical":
                                    model_time = run_classical_model(config, embedding_type, scaler_method, scaler_param_str, selection_method, selection_param_str, algorithm, result_path, pipeline, samples)  
                                all_model_time[f"{algorithm}"] = model_time
    return all_model_time
