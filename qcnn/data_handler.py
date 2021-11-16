import os
import json
import pandas as pd
import numpy as np
from collections import namedtuple
import itertools as it
from data_utility import DataUtility

from sklearn.model_selection import train_test_split


Samples = namedtuple("Samples", ["X_train", "y_train", "X_test", "y_test"])

def save_json(path, dict_obj):
    """Save json file

    Args:
        path (str): path containing file name
    """
    # Give expirment context
    with open(path, "w+") as f:
        json.dump(dict_obj, f, indent=4)


def load_json(path):
    """Load json file

    Args:
        path (str: path to json file
    """
    with open(path, "r") as f:
        dict_obj = json.load(f)
    return dict_obj


def get_2d_modelling_data(path, target, columns_to_remove=[]):

    # Get variables from config
    
    filename, file_extension = os.path.splitext(path)

    if file_extension == ".csv":
        raw = pd.read_csv(path)
    elif file_extension in [".parq", ".parquet"]:
        raw = pd.read_parquet(path, engine="auto")

    data_utility = DataUtility(raw, target=target, default_subset="modelling")
    data_utility.update(
        columns_to_remove, "included", {"value": False, "reason": "manual"}
    )

    return raw, data_utility

def create_train_test_samples(data, data_utility, test_size=.3, random_state=42):
    X, y, Xy = data_utility.get_samples(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    data_utility.row_sample["train"] = X_train.index
    data_utility.row_sample["test"] = X_test.index

    return Samples(X_train, X_test, y_train, y_test)


def get_image_data(path):
    # TODO this tensforflow import slows things down and isn't needed
    import tensorflow as tf
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_test = (
        X_train[..., np.newaxis] / 255.0,
        X_test[..., np.newaxis] / 255.0,
    )
    ind = np.random.choice(range(X_test.shape[0]), 1000, replace=False)
    X_test = X_test[ind]
    y_test = y_test[ind]
    
    # # Levels to consider
    # target_levels = range(10)
    # target_pairs = [target_pair for target_pair in it.combinations(target_levels, 2)]

    return Samples(X_train, X_test, y_train, y_test)
