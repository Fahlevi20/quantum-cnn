import os
import numpy as np
import pandas as pd


import sklearn
import librosa

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf


import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display


def load_image_data(path, classes, target):
    Xy = pd.read_csv(path)

    # returns "classical|pop" etc depending on class selection
    filter_pat = "|".join(genre for genre in classes)
    indices = Xy["filename"].str.contains(filter_pat)

    # Select only data for relevant classes
    Xy = Xy.loc[indices, :].copy()

    Xy.drop(["filename", "length"], axis="columns", inplace=True)
    Xy.head()
    # %%
    # Split data
    y = Xy[target]
    X = Xy.loc[:, Xy.columns != target]
    # %%
    # Normalize
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)

    # new data frame with the new scaled data.
    X = pd.DataFrame(np_scaled, columns=cols)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    #%%
    # X_train = tf.image.resize(X_train[:], (784, 1)).numpy()
    # X_test = tf.image.resize(X_test[:], (784, 1)).numpy()
    # X_train, X_test = tf.squeeze(X_train), tf.squeeze(X_test)
    n_features = 32
    pca = PCA(n_features)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, y_train, X_test, y_test
