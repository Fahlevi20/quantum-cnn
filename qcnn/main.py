#%%
import imp
import os
import time
import argparse

import pandas as pd

import json

import numpy as np

from data_utility import DataUtility
from experiment import run_experiment
from data_handler import (
    save_json,
    load_json,
    get_2d_modelling_data,
    get_image_data,
    create_train_test_samples,
)


def main(args):
    # Load experiment config
    config = load_json(args.config_path)
    # Load data
    if config["data"].get("type", None) == "image":
        # With image data raw is a list consisting of X_train, y_train X_test, y_test
        path = config["data"].get("path", None)
        set_name = config["data"].get("set_name", None)

        test_size = config["data"]["sampling"].get("test_size", 0.3)
        random_state = config["data"]["sampling"].get("random_state", 42)
        kwargs = config["data"].get("kwargs", {})

        samples = get_image_data(
            path,
            set_name=set_name,
            test_size=test_size,
            random_state=random_state,
            **kwargs,
        )

        np.save(
            f"{config.get('path')}/{config.get('ID')}/y_test",
            samples.y_test,
        )
        # np.savetxt(f"{config.get('path')}/{config.get('ID')}/X_test.csv", samples.X_test, delimiter=",")
    else:
        # assume data is 2d
        target = config["data"].get("target_column")
        path = config["data"].get("path")
        colnames = config["data"].get("column_names", None)
        set_name = config["data"].get("set_name", None)
        kwargs = config["data"].get("kwargs", {})
        # TODO rename function to something more generic like read data
        raw = get_2d_modelling_data(path, colnames, set_name, **kwargs)
        # ==== Data Utility for data specific manipulations ====#
        """
        Datautility should be used only here to transform the data into a desirable train test set, then when the experiment is
        ran it is assumed that all "columns" and rows is as needs to be. This is specific data interaction from the user and should somehow
        be abstracted out TODO
        """
        # This was for music data
        columns_to_remove = ["filename", "length"]
        data_utility = DataUtility(raw, target=target, default_subset="modelling")
        data_utility.update(
            columns_to_remove, "included", {"value": False, "reason": "manual"}
        )
        # data_utility = DataUtility(raw, target=target, default_subset="modelling")
        X, y, _ = data_utility.get_samples(raw)
        # ==== End Data Utility ====#
        test_size = config["data"]["sampling"].get("test_size", 0.3)
        random_state = config["data"]["sampling"].get("random_state", 42)
        # Create test set
        samples = create_train_test_samples(
            X, y, test_size=test_size, random_state=random_state
        )
        # Move to function TODO
        samples.y_test.to_csv(f"{config.get('path')}/{config.get('ID')}/y_test.csv")
        samples.X_test.to_csv(f"{config.get('path')}/{config.get('ID')}/X_test.csv")
    model_execution_times = run_experiment(config, samples)
    # save_json(model_execution_times) TODO
    print("Experiment Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a machine learning experiment")

    # Add the arguments
    parser.add_argument(
        "--config_path",
        metavar="config_path",
        type=str,
        help="the path to the experiment config (json)",
    )

    args = parser.parse_args()
    main(args)
