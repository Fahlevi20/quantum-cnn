# %%
from collections import namedtuple
import os
import time
import itertools as it
from typing import NamedTuple
import numpy as np
from sklearn import feature_selection

import tensorflow as tf

import circuit_presets

from quantum_models import train_quantum
from classical_models import train_classical
from preprocessing import (
    get_preprocessing_pipeline,
)
from postprocessing import get_ovo_classication, get_ova_classication
from sklearn.model_selection import ParameterGrid


Model_Configurations = namedtuple(
    "Model_Configurations",
    [
        "model_type",
        "algorithm",
        "classification_type",
        "embedding_type",
        "scaler_method",
        "scaler_param_str",
        "selection_method",
        "selection_param_str",
        "target_pair",
        "additional_structure",
    ],
)


def run_quantum_model(
    config,
    samples,
    pipeline,
    model_type,
    algorithm,
    classification_type,
    embedding_type,
    scaler_method,
    scaler_param_str,
    selection_method,
    selection_param_str,
    target_pair=None,
):
    # TODO Probably don't want these functions to rely so heavily on the config
    result_path = f"{config.get('path')}/{config.get('ID')}"
    circuit_list = config["model"][model_type][algorithm].get("circuit_list", [])
    pooling_list = config["model"][model_type][algorithm].get("pooling_list", [])
    wire_pattern_list = config["model"][model_type][algorithm].get("wire_pattern_list", [])
    classification_type = config["model"].get("classification_type", None)
    model_time = {}

    for circ_pool_combo in it.product(circuit_list, pooling_list, wire_pattern_list):
        # model name depends on circ_pool_combo which causes some redundancy in the code, i.e. it can be improved by deriving the name earlier
        model_configuration = Model_Configurations(
            model_type=model_type,
            algorithm=algorithm,
            classification_type=classification_type,
            embedding_type=embedding_type,
            scaler_method=scaler_method,
            scaler_param_str=scaler_param_str,
            selection_method=selection_method,
            selection_param_str=selection_param_str,
            target_pair=target_pair,
            additional_structure=circ_pool_combo,
        )
        model_name = "-".join([str(item) for item in model_configuration])
        if not (
            os.path.exists(
                # Check to see if model was already built
                f"{result_path}/{model_name}-confusion-matrix.csv"
            )
        ):
            t1 = time.time()
            train_quantum(
                config,
                circ_pool_combo,
                embedding_type,
                algorithm,
                pipeline,
                samples,
                target_pair,
                model_name=model_name,
                model_configuration=model_configuration,
            )
            t2 = time.time()
            model_time[f"{model_name}"] = t2 - t1
    for custom_structure in config["model"][model_type][algorithm].get(
        "custom_structures", []
    ):
        model_configuration = Model_Configurations(
            model_type=model_type,
            algorithm=algorithm,
            classification_type=classification_type,
            embedding_type=embedding_type,
            scaler_method=scaler_method,
            scaler_param_str=scaler_param_str,
            selection_method=selection_method,
            selection_param_str=selection_param_str,
            target_pair=target_pair,
            additional_structure=custom_structure,
        )
        model_name = "-".join([str(item) for item in model_configuration])
        if not (
            os.path.exists(
                # Check to see if model was already built
                f"{result_path}/{model_name}-confusion-matrix.csv"
            )
        ):
            qcnn_structure = config["model"][model_type][algorithm][
                "custom_structures"
            ][custom_structure]
            t1 = time.time()
            train_quantum(
                config,
                qcnn_structure,
                embedding_type,
                algorithm,
                pipeline,
                samples,
                target_pair,
                model_name=model_name,
                model_configuration=model_configuration,
            )
            t2 = time.time()
            model_time[f"{model_name}"] = t2 - t1
    return model_time


# TODO I can take the preprocessing one step further by searching for the built in methods in SKLearn


def run_classical_model(
    config,
    samples,
    pipeline,
    model_type,
    algorithm,
    classification_type,
    embedding_type,
    scaler_method,
    scaler_param_str,
    selection_method,
    selection_param_str,
    target_pair=None,
):
    classification_type = config["model"].get("classification_type", None)
    model_time = {}
    if target_pair:
        model_configuration = Model_Configurations(
            model_type=model_type,
            algorithm=algorithm,
            classification_type=classification_type,
            embedding_type=embedding_type,
            scaler_method=scaler_method,
            scaler_param_str=scaler_param_str,
            selection_method=selection_method,
            selection_param_str=selection_param_str,
            target_pair=target_pair,
            additional_structure=None,
        )
    else:
        model_configuration = Model_Configurations(
            model_type=model_type,
            algorithm=algorithm,
            classification_type=classification_type,
            embedding_type=embedding_type,
            scaler_method=scaler_method,
            scaler_param_str=scaler_param_str,
            selection_method=selection_method,
            selection_param_str=selection_param_str,
            target_pair=None,
            additional_structure=None,
        )
    model_name = "-".join([str(item) for item in model_configuration])
    t1 = time.time()
    # Train and store results
    train_classical(
        config,
        algorithm,
        pipeline,
        samples,
        target_pair=target_pair,
        model_name=model_name,
        model_configuration=model_configuration,
    )
    t2 = time.time()
    model_time[f"{model_name}"] = t2 - t1

    return model_time


def run_experiment(config, samples):

    all_model_time = {}
    for model_type in ("quantum", "classical"):
        for embedding_type in config["preprocessing"].get(model_type):
            if (
                config["preprocessing"][model_type][embedding_type].get("ignore")
                is False
            ):
                scaler_methods = config["preprocessing"][model_type][embedding_type][
                    "scaler"
                ].get("method")
                selection_methods = config["preprocessing"][model_type][embedding_type][
                    "feature_selection"
                ].get("method")
                custom_steps = config["preprocessing"][model_type][embedding_type].get(
                    "custom", None
                )
                for scaler_method, selection_method in it.product(
                    scaler_methods, selection_methods
                ):
                    scaler_params = list(
                        ParameterGrid(
                            config["preprocessing"][model_type][embedding_type][
                                "scaler"
                            ]["method"].get(scaler_method)
                        )
                    )
                    selection_params = list(
                        ParameterGrid(
                            config["preprocessing"][model_type][embedding_type][
                                "feature_selection"
                            ]["method"].get(selection_method)
                        )
                    )
                    ordered_combinations = list(
                        it.product(scaler_params, selection_params)
                    )  # Cartesian product is fine here since a(first element) from S1 is different from a(first element) from S2, i.e the combination (a,a) is unique and (a,b)<>(b,a)
                    for scaler_param, selection_param in ordered_combinations:
                        pipeline = get_preprocessing_pipeline(
                            scaler_method,
                            scaler_param,
                            selection_method,
                            selection_param,
                            custom_steps=custom_steps,
                        )
                        selection_param_str = "_".join(
                            [f"{k}={v}" for k, v in scaler_param.items()]
                        )
                        scaler_param_str = "_".join(
                            [f"{k}={v}" for k, v in selection_param.items()]
                        )
                        for algorithm in config["model"].get(model_type):
                            # quantum algo like qcnn or some other quantum model
                            if (
                                config["model"][model_type][algorithm].get(
                                    "ignore", True
                                )
                                == False
                            ):
                                classification_type = config["model"].get(
                                    "classification_type", None
                                )

                                prefix = (
                                    f"{model_type}-{algorithm}-{classification_type}-{embedding_type}-{scaler_method}"
                                    f"-{scaler_param_str}-{selection_method}-{selection_param_str}"
                                )

                                if classification_type == "binary":
                                    for target_pair in config["data"].get(
                                        "target_pairs", []
                                    ):
                                        if model_type == "quantum":
                                            model_time = run_quantum_model(
                                                config,
                                                samples,
                                                pipeline,
                                                model_type,
                                                algorithm,
                                                classification_type,
                                                embedding_type,
                                                scaler_method,
                                                scaler_param_str,
                                                selection_method,
                                                selection_param_str,
                                                target_pair=target_pair,
                                            )
                                        elif model_type == "classical":
                                            model_time = run_classical_model(
                                                config,
                                                samples,
                                                pipeline,
                                                model_type,
                                                algorithm,
                                                classification_type,
                                                embedding_type,
                                                scaler_method,
                                                scaler_param_str,
                                                selection_method,
                                                selection_param_str,
                                                target_pair=target_pair,
                                            )
                                    all_model_time[f"{algorithm}"] = model_time
                                elif classification_type in ["ovo", "ova"]:
                                    if model_type == "quantum":
                                        model_time = run_quantum_model(
                                            config,
                                            samples,
                                            pipeline,
                                            model_type,
                                            algorithm,
                                            classification_type,
                                            embedding_type,
                                            scaler_method,
                                            scaler_param_str,
                                            selection_method,
                                            selection_param_str,
                                            target_pair=None,
                                        )
                                    elif model_type == "classical":
                                        model_time = run_classical_model(
                                            config,
                                            samples,
                                            pipeline,
                                            model_type,
                                            algorithm,
                                            classification_type,
                                            embedding_type,
                                            scaler_method,
                                            scaler_param_str,
                                            selection_method,
                                            selection_param_str,
                                            target_pair=None,
                                        )
                                    all_model_time[f"{algorithm}"] = model_time
                                else:
                                    raise NotImplementedError(
                                        f"No implementation for classification type: {classification_type}"
                                    )
    return all_model_time
