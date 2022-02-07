import os
import json
import pandas as pd
import numpy as np
from collections import namedtuple
import itertools as it


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

    filename, file_extension = os.path.splitext(path)

    if file_extension == ".csv":
        raw = pd.read_csv(path)
    elif file_extension in [".parq", ".parquet"]:
        raw = pd.read_parquet(path, engine="auto")    

    return raw

def create_train_test_samples(X, y, test_size=.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return Samples(X_train, y_train, X_test, y_test)


def get_image_data(path, set_name="mnist", **kwargs):
    if path:
        pass
    else:
        if set_name=="mnist":
            # TODO this tensforflow import slows things down and isn't needed
            # %%
            import tensorflow as tf
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        elif set_name=="f_mnist":
            import tensorflow as tf
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        elif set_name=="gtzan":
            pass

    return Samples(X_train, y_train, X_test, y_test)
