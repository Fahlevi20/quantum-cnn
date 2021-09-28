# %%
# Import libraries
import os
import numpy as np
import pandas as pd

import sklearn
import librosa

import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display

# %%
# TODO remove relative import
AUDIO_PATH = "../data/Data/genres_original"
CLASSICAL_AUDIO_PATH = f"{AUDIO_PATH}/classical"
POP_AUDIO_PATH = f"{AUDIO_PATH}/pop"


# %%
def get_genre_list_n_path(genre):
    genre_audio_path = f"{AUDIO_PATH}/{genre}"
    return genre_audio_path, os.listdir(genre_audio_path)


classical_path, classical_song_list = get_genre_list_n_path("classical")
random_classical_song = np.random.choice(classical_song_list, 1)[0]

song_path = f"{classical_path}/{random_classical_song}"

# %%
# Play audio
audio_ts, sample_rate = librosa.load(song_path)

print(song_path)
print(type(audio_ts), type(sample_rate))
print(audio_ts.shape, sample_rate)
# ipd.Audio(song_path)
# %%
# Visualize

plt.figure(figsize=(14, 5))
librosa.display.waveplot(audio_ts, sr=sample_rate)
# %%
# Spectogram (visual representation of the spectrum of frequencies)
# Fourier transform
audio_ts_fft = librosa.stft(audio_ts)
Xdb = librosa.amplitude_to_db(abs(audio_ts_fft))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="hz")
plt.colorbar()
# %%
# Convert to log scale
librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
plt.colorbar()

# %%
# Plot the signal again and get zero crossing rate:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(audio_ts, sr=sample_rate)
# %%
# Zoom in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))

plt.plot(audio_ts[n0:n1])
plt.grid()
# %%
# Get zero crossings
"""
Zero crossing is the rate of sign changes (amplitudes) of a signal
"""
zero_crossings = librosa.zero_crossings(audio_ts[n0:n1], pad=False)
print(sum(zero_crossings))
# %%
# Spectral Centroid
"""
Calculates where the 'center of mass' fpr a sound is located and is calcualted as the weighted mean of the frequencies in the sound. Wonder how it's weighted?
"""

spectral_centroids = librosa.feature.spectral_centroid(audio_ts, sr=sample_rate)[0]
print(f"spectral_centroid shape: {spectral_centroids.shape}")

# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


# Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(audio_ts, sr=sample_rate, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color="r")
# %%
# Spectral Rolloff
"""
It is a measure of the shape of the signal. It represents the frequency below which a specified percentage of the total spectral energy, e.g., 85%, lies.
"""
spectral_rolloff = librosa.feature.spectral_rolloff(audio_ts + 0.01, sr=sample_rate)[0]
librosa.display.waveplot(audio_ts, sr=sample_rate, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color="r")
# %%
librosa.display.waveplot(audio_ts, sr=sample_rate)

mfccs = librosa.feature.mfcc(audio_ts, sr=sample_rate)
print(mfccs.shape)
# Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sample_rate, x_axis="time")
# %%
# scale
import sklearn

mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))
librosa.display.specshow(mfccs, sr=sample_rate, x_axis="time")
# %%
# Chroma frequencies
# Loadign the file
# x, sr = librosa.load('../simple_piano.wav')
hop_length = 512
chromagram = librosa.feature.chroma_stft(
    audio_ts, sr=sample_rate, hop_length=hop_length
)
plt.figure(figsize=(15, 5))
librosa.display.specshow(
    chromagram, x_axis="time", y_axis="chroma", hop_length=hop_length, cmap="coolwarm"
)
# %%
# Data load + initial cleaning

Xy = pd.read_csv(f"../data/Data/features_3_sec.csv")
classes = ["pop", "classical"]
# returns "classical|pop" etc depending on class selection
filter_pat = "|".join(genre for genre in classes)
indices = Xy["filename"].str.contains(filter_pat)

# Select only data for relevant classes
Xy = Xy.loc[indices, :].copy()

Xy.drop(["filename", "length"], axis="columns", inplace=True)
Xy.head()
# %%
# Split data
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf

TARGET = "label"

y = Xy[TARGET]
X = Xy.loc[:, Xy.columns!=TARGET]
# %%
# Normalize
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)

# new data frame with the new scaled data. 
X = pd.DataFrame(np_scaled, columns = cols)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#%%
# X_train = tf.image.resize(X_train[:], (784, 1)).numpy()
# X_test = tf.image.resize(X_test[:], (784, 1)).numpy()
# X_train, X_test = tf.squeeze(X_train), tf.squeeze(X_test)
n_features = 32
pca = PCA(n_features)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# %%
