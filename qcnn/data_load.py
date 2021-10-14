# %%

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
import simpleaudio as sa
import wave

# %%
def get_rnd_audio_sample(data, class_type, target="label"):
    idx = data.loc[class_type[class_type == True].sample(1).index].index
    filename, genre = data.loc[idx, "filename"].iloc[0], data.loc[idx, target].iloc[0]
    return filename, genre


def plot_spectogram(audio_path, filename, genre):
    audio_ts, sample_rate = librosa.load(f"{audio_path}/{genre}/{filename}")
    X = librosa.stft(audio_ts)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(20, 10))
    librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
    plt.colorbar()
# %%
model_name = "pca-8-Angle-U_5"
data_path = "../data/archive/Data/features_30_sec.csv"
audio_path = "../data/archive/Data/genres_original"
target = "label"
classes = ["classical", "hiphop"]

genre = "classical"
filename = f"{genre}.00005.wav"
# %%
audio_ts, sample_rate = librosa.load(f"{audio_path}/{genre}/{filename}", sr=44100)

plt.figure(figsize=(14, 5))
print(sample_rate)
librosa.display.waveplot(audio_ts, sr=sample_rate)


# %%
X = librosa.stft(audio_ts)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='log')
plt.colorbar()

# %%
librosa.output.write_wav(f"{audio_path}/{genre}/{filename}", audio_ts, sample_rate)
# %%
# play
wave_obj = sa.WaveObject.from_wave_file(f"{audio_path}/{genre}/{filename}")
play_obj = wave_obj.play()
# %%
play_obj.stop()

















#%%
def load_audio_data(path, classes, target):
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

