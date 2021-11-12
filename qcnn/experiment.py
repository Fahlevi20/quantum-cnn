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

general_config = {
    "scaler": {
        "method": {
            "standard": {},
            "minmax": {"feature_range": [(0, 1), (-1, 1), (0, np.pi / 2)]},
        },
        "ignore": False,
    },
    "feature_selection": {
        "method": {
            "pca": {"n_components": [8]},
            "tree": {"max_features": [8], "n_estimators": [50]},
        },
    },
}
for scaler, feature_selection in it.product(general_config["scaler"].get("method"), general_config["feature_selection"].get("method")):
    scaler_params = list(ParameterGrid(general_config["scaler"]["method"].get(scaler)))
    feature_selection_params = list(ParameterGrid(general_config["feature_selection"]["method"].get(feature_selection)))
    ordered_combinations = list(
        it.product(scaler_params, feature_selection_params)
    )  # Cartesian product is fine here since a(first element) from S1 is different from a(first element) from S2, i.e the combination (a,a) is unique and (a,b)<>(b,a)
    for preprocess_combination in ordered_combinations:
        print(f"{scaler}-{feature_selection}\n{preprocess_combination}")

# %%
# def get_dict_permutation(dict_of_list):
#     """[summary]
#     """
# I can take the preprocessing one step further by searching for the built in methods in SKLearn
def run_experiment(config, raw):
    y_hat_history = {
        "model_name": [],
        "target_pair": [],
        "y_hat": [],
        "X_test_ind": [],
        "best_params": [],
    }
    result_path = f"{config.get('path')}/{config.get('ID')}"
    model_time = {}
    for model_type in ("quantum", "classical"):
        for embedding_type in config["preprocessing"].get(model_type):
            if config["preprocessing"][model_type][embedding_type].get("ignore") is False:
                for scaler_method, selection_method in it.product(general_config["scaler"].get("method"), general_config["feature_selection"].get("method")):
                    scaler_params = list(ParameterGrid(general_config["scaler"]["method"].get(scaler_method)))
                    selection_params = list(ParameterGrid(general_config["feature_selection"]["method"].get(selection_method)))
                    ordered_combinations = list(
                        it.product(scaler_params, selection_params)
                    )  # Cartesian product is fine here since a(first element) from S1 is different from a(first element) from S2, i.e the combination (a,a) is unique and (a,b)<>(b,a)
                    for scaler_param, selection_parm in ordered_combinations:
                        print(f"{scaler_method}-{selection_method}\n{preprocess_combination}")
                        preprocessing_config=dict(zip([scaler, feature_selection], [scaler_param, selection_parm]))
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
                                for target_pair in config["data"]["target_pairs"]:
                                    prefix = (
                                        f"{model_type}-{embedding_type}-{scaler_method}"
                                        f"-{scaler_param_str}-{selection_method}-{selection_param_str}-{algorithm}"
                                    )
                                    if model_type == "quantum":
                                        experiment_circuits = {
                                            circ_name: CIRCUIT_OPTIONS[circ_name]
                                            for circ_name in config["model"][
                                                model_type
                                            ][algorithm]["circuit_list"]
                                        }
                                        for (
                                            circ_name,
                                            circ_param_count,
                                        ) in experiment_circuits.items():
                                            model_name = f"{prefix}-{circ_name}-{target_pair}"
                                            if not (
                                                os.path.exists(
                                                    f"{result_path}/{model_name}-confusion-matrix.csv"
                                                )
                                            ):
                                                # Define QCNN structure
                                                # fmt: off
                                                layer_dict = {
                                                    "c_1": Layer(c_1, getattr(circuit_presets, circ_name), "convolutional", circ_param_count, 0,),
                                                    "p_1": Layer(p_1, getattr(circuit_presets, "psatz1"),"pooling",POOLING_OPTIONS["psatz1"],1,),
                                                    "c_2": Layer(c_2, getattr(circuit_presets, circ_name),"convolutional",circ_param_count,2,),
                                                    "p_2": Layer(p_2, getattr(circuit_presets, "psatz1"),"pooling",POOLING_OPTIONS["psatz1"],3,),
                                                    "c_3": Layer(c_3, getattr(circuit_presets, circ_name),"convolutional",circ_param_count,4,),
                                                    "p_3": Layer(p_3, getattr(circuit_presets, "psatz1"),"pooling",POOLING_OPTIONS["psatz1"],5,),
                                                }

                                                # Create QCNN structure
                                                # fmt: off
                                                qcnn_structure = QcnnStructure(layer_dict)
                                                # fmt: off
                                                y_hat_history = {"model_name": [],"target_pair": [],"y_hat": [],"X_test_ind": [],"best_params": [],}
                                                t1 = time.time()
                                                # Train and store results
                                                (
                                                    y_hat,
                                                    X_test_ind,
                                                    best_params,
                                                    cf_matrix,
                                                ) = train_qcnn(
                                                    qcnn_structure,
                                                    embedding_type,
                                                    pipeline,
                                                    target_pair,
                                                    raw.copy(),
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
                                        if config["model"]["classification_type"] == "ovo":
                                        # If model should apply ovo strategy, TODO this can be done better if I can represent the model as a SKLearn classifier
                                            y_class_multi, row_prediction_history = get_ovo_classication(
                                                y_hat_history, y_test, config, store_results=True, prefix=prefix
                                            )
                                        elif config["model"]["classification_type"] == "ova":
                                            # If model should apply ovo strategy, TODO this can be done better if I can represent the model as a SKLearn classifier
                                            y_class_multi, row_prediction_history = get_ova_classication(
                                                y_hat_history, y_test, config, store_results=True, prefix=prefix
                                            )
                                    elif model_type == "classical":
                                        model_name = f"{prefix}-{algorithm}-{target_pair}"
                                        # train classical modelmodel_classical
                                        t1 = time.time()
                                        # Train and store results
                                        train_classical(
                                            algorithm,
                                            pipeline,
                                            target_pair,
                                            raw.copy(),
                                            data_utility,
                                            config,
                                            model_name=model_name,
                                        )
                                        t2 = time.time()
                                        model_time[f"{model_name}"] = t2 - t1
