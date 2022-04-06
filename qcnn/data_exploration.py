# %%
# %matplotlib widget
import os
import numpy as np
import pandas as pd
import math
import itertools as it

import sklearn
import librosa

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# import tensorflow as tf


import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display

# import simpleaudio as sa
import wave
from data_utility import DataUtility
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import ipympl
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# %%

# with plt.style.context("seaborn-whitegrid"):
#     # plt.figure(figsize=(30, 50))
#     for feature in data_utility.get(filter_dict={"subset": "modelling"}):
#         for target_level in target_levels:
#             plt.hist(
#                 X[y == target_level][feature],
#                 label=target_level,
#                 bins=100,
#                 alpha=0.3,
#             )
#         plt.xlabel(feature)
#         plt_id += 1
#         plt.legend(loc="upper right", fancybox=True, fontsize=8)
#         plt.tight_layout()
#         plt.show()

# %%
def get_pipeline(config):
    """config should be a subset of the big config, i.e. a specific slice/iteration on the lowest level

    Args:
        config ([type]): [description]
    """
    scaler_method = config["scaler"].get("method", "minmax")
    scaler_params = config["scaler"].get(
        f"{scaler_method}_params", {"feature_range": (0, 1)}
    )
    selection_method = config["feature_selection"].get("method", "pca")
    selection_params = config["feature_selection"].get(
        f"{selection_method}_params", {"n_components": 8}
    )

    # Define Scaler
    if scaler_method == "minmax":
        scaler = (
            "scaler",
            preprocessing.MinMaxScaler(**scaler_params),
        )
    elif scaler_method == "standard":
        scaler = (
            "scaler",
            preprocessing.StandardScaler(**scaler_params),
        )
    # Define feature selector
    if selection_method == "tree":
        selection = (
            selection_method,
            SelectFromModel(
                ExtraTreesClassifier(
                    n_estimators=selection_params.get("n_estimators", 50)
                ),
                max_features=selection_params.get("max_features", 8),
            ),
        )
    elif selection_method == "pca":
        selection = (selection_method, PCA(**selection_params))
    pipeline = Pipeline(
        [
            scaler,
            selection,
        ]
    )
    return pipeline


def plot_top3d(pipe_Xy_df, config, data_utility):
    # TODO code for None if incorrect config
    scaler_method = config["scaler"].get("method")
    scaler_params = config["scaler"].get(f"{scaler_method}_params")
    selection_method = config["feature_selection"].get("method")
    selection_params = config["feature_selection"].get(f"{selection_method}_params")
    # To include combination in title
    selection_param_str = "-".join([f"{k}={v}" for k, v in selection_params.items()])
    scaler_param_str = "-".join([f"{k}={v}" for k, v in scaler_params.items()])

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.set_title(
        f"{selection_method}-{scaler_method}-{selection_param_str}-{scaler_param_str}"
    )
    target_levels = pipe_Xy_df[data_utility.target].unique()
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    for target_level, color in zip(target_levels, colors):
        ind = pipe_Xy_df[data_utility.target] == target_level
        ax.scatter3D(
            pipe_Xy_df.loc[ind, f"{selection_method}-0"],
            pipe_Xy_df.loc[ind, f"{selection_method}-1"],
            pipe_Xy_df.loc[ind, f"{selection_method}-2"],
            c=color,
            s=50,
        )
    fig.legend(target_levels, loc="upper right", fancybox=True)
    fig.show()


def plot_var_exp(fig, ax, pipeline, selection_method):
    with plt.style.context("seaborn-whitegrid"):
        var_exp = pipeline.named_steps[selection_method].explained_variance_ratio_
        cum_var_exp = var_exp.cumsum()

        tmp_ax = ax.bar(
            range(len(var_exp)),
            var_exp,
            alpha=0.5,
            align="center",
            label="individual explained variance",
        )
        tmp_ax = ax.step(
            range(len(var_exp)),
            cum_var_exp,
            where="mid",
            label="cumulative explained variance",
        )
        a = ax.set_ylabel("Explained variance ratio")
        b = ax.set_xlabel("Principal components")
        # f#ig.legend(loc="best")


def plot_top2d(fig, ax, pipe_Xy_df, config, feature_names, data_utility):
    scaler_method = config["scaler"].get("method")
    scaler_params = config["scaler"].get(f"{scaler_method}_params")
    selection_method = config["feature_selection"].get("method")
    selection_params = config["feature_selection"].get(f"{selection_method}_params")
    # To include combination in title
    # selection_param_str = "-".join([f"{k}={v}" for k, v in selection_params.items()])
    # scaler_param_str = "-".join([f"{k}={v}" for k, v in scaler_params.items()])
    # # fig = plt.figure(figsize=figsize)
    # # ax = plt.axes()
    # ax.set_title(
    #     f"{selection_method}-{scaler_method}-{selection_param_str}-{scaler_param_str}"
    # )

    target_levels = pipe_Xy_df[data_utility.target].unique()
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    for target_level, color in zip(target_levels, colors):
        ind = pipe_Xy_df[data_utility.target] == target_level
        tmp_ax = ax.scatter(
            pipe_Xy_df.loc[ind, feature_names[0]],
            pipe_Xy_df.loc[ind, feature_names[1]],
            c=color,
            s=50,
        )
    a = ax.set_xlabel(feature_names[0])
    b = ax.set_ylabel(feature_names[1])
    fig.legend(target_levels, loc="upper right", fancybox=True)


# %%
# Setup
import itertools as it


def run_preprocess_experiment(config, X, y, data_utility, figsize=(8, 8)):
    for scaler_method in config["scaler"].get("method", "standard"):
        for selection_method in config["feature_selection"].get("method", "pca"):
            scaler_param = config["scaler"].get(f"{scaler_method}_params", {})
            selection_param = config["feature_selection"].get(
                f"{selection_method}_params", {}
            )
            # create dictionary of every possible paramater permutation
            scaler_keys, scaler_values = (
                zip(*scaler_param.items())
                if len(scaler_param.keys()) > 0
                else zip([[], []])
            )
            scaler_permutations_dicts = [
                dict(zip(scaler_keys, v)) for v in it.product(*scaler_values)
            ]
            # Ensure there is atleast one empty dictionary
            scaler_permutations_dicts = (
                [{}]
                if len(scaler_permutations_dicts) == 0
                else scaler_permutations_dicts
            )
            selection_keys, selection_values = (
                zip(*selection_param.items())
                if len(selection_param.keys()) > 0
                else zip([[], []])
            )
            selection_permutations_dicts = [
                dict(zip(selection_keys, v)) for v in it.product(*selection_values)
            ]
            # Ensure there is atleast one empty dictionary
            selection_permutations_dicts = (
                [{}]
                if len(selection_permutations_dicts) == 0
                else selection_permutations_dicts
            )
            for selection_param_permutation in selection_permutations_dicts:
                for scaler_param_permutation in scaler_permutations_dicts:
                    tmp_config = {
                        "scaler": {
                            "method": scaler_method,
                            f"{scaler_method}_params": scaler_param_permutation,
                        },
                        "feature_selection": {
                            "method": selection_method,
                            f"{selection_method}_params": selection_param_permutation,
                        },
                    }
                    pipeline = get_pipeline(tmp_config)
                    pipe_X = pipeline.fit_transform(X, y)
                    selection_param_str = "-".join(
                        [f"{k}={v}" for k, v in selection_param_permutation.items()]
                    )
                    scaler_param_str = "-".join(
                        [f"{k}={v}" for k, v in scaler_param_permutation.items()]
                    )
                    if selection_method == "pca":
                        fig, axs = plt.subplots(
                            1,
                            2,
                            figsize=figsize,
                            num=f"{selection_method}-{scaler_method}-{selection_param_str}-{scaler_param_str}",
                        )
                        plot_var_exp(fig, axs[1], pipeline, selection_method)
                        feature_names = [
                            f"{selection_method}-{i}" for i in range(pipe_X.shape[1])
                        ]
                        pipe_X_df = pd.DataFrame(
                            pipe_X,
                            columns=feature_names,
                            index=X.index,
                        )
                        pipe_Xy_df = pipe_X_df.join(y)
                        plot_top2d(
                            fig,
                            axs[0],
                            pipe_Xy_df,
                            tmp_config,
                            feature_names,
                            data_utility,
                        )
                        fig.tight_layout()
                        # fig.show()
                    else:
                        feature_names = X.columns[
                            pipeline.named_steps[selection_method].get_support()
                        ]

                        pipe_X_df = pd.DataFrame(
                            pipe_X,
                            columns=feature_names,
                            index=X.index,
                        )
                        pipe_Xy_df = pipe_X_df.join(y)
                        feature_combos = [
                            combo for combo in it.combinations(feature_names, 2)
                        ]
                        n_plot_row = (
                            1
                            if len(feature_combos) <= 2
                            else int(np.around(len(feature_combos) / 2))
                        )
                        fig, axs = plt.subplots(
                            n_plot_row,
                            2,
                            figsize=figsize,
                            num=f"{selection_method}-{scaler_method}-{selection_param_str}-{scaler_param_str}",
                        )
                        perm_it = 0
                        for feature_combo in feature_combos:
                            # int(it/2), it % 2 breaks 1 number into 2 counting 0,0 -> 0,1 -> 1,0 -> 1,1 -> 2,0 -> 2,1
                            plot_top2d(
                                fig,
                                axs[int(perm_it / 2), perm_it % 2],
                                pipe_Xy_df,
                                tmp_config,
                                feature_combo,
                                data_utility,
                            )
                            perm_it = perm_it + 1
                        fig.tight_layout()


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


model_name = "pca-8-Angle-U_5"
data_path = "../data/features_30_sec.csv"
classes = ["classical", "pop"]
# %%
target = "label"
audio_path = "/home/matt/dev/projects/quantum-cnn/data/genres_original"
image_path = "/home/matt/dev/projects/quantum-cnn/data/genres_image_1"
genre = "classical"
filename = f"{genre}.00005"
ext = "wav"
sample_rate = 21500
audio_ts, sample_rate = librosa.load(
    f"{audio_path}/{genre}/{filename}.{ext}", sr=sample_rate
)
# %%
# trim
n_fft = 2048
hop_length = 2048
n_mels = 8
audio_ts, _ = librosa.effects.trim(audio_ts)
# First 3 seconds
audio_ts = audio_ts[0 : sample_rate * 3]
mel_spec = librosa.feature.melspectrogram(
    y=audio_ts,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    fmax=8000,
)
mel_sdb = librosa.power_to_db(mel_spec, ref=np.max)

# %% display
# For plotting headlessly

fig, ax = plt.subplots()
img = librosa.display.specshow(
    mel_sdb, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel"
)
ax.set_title("Mel spetogram")
#ax.set_xlabel("Time")
#plt.axis('off')
fig.colorbar(img, ax=ax)

# plt.colorbar(format='%+2.0f dB')
print(mel_sdb.shape)
# fig.savefig(f"{image_path}/{filename.replace('.','_')}.png")

# %% ml data
# %%
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


img = scale_minmax(mel_sdb, 0, 255).astype(np.uint8)
img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
img = 255 - img  # invert. make black==more energy
plt.imshow(img)
# %%
import torch
import torchvision.transforms as T


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(size=(256, 256)),
        ]
    )
    batch = transforms(batch)
    return batch


a = preprocess(torch.as_tensor(mel_sdb))
# %%
S = librosa.feature.melspectrogram(y=audio_ts, sr=sample_rate, n_mels=128, fmax=8000)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(
    S_dB, x_axis="time", y_axis="mel", sr=sample_rate, fmax=8000, ax=ax
)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set(title="Mel-frequency spectrogram")

# Display
# %%
params = {'legend.fontsize': 18,
        #'figure.figsize': (9, 6),
         'axes.labelsize': 16,
         'axes.titlesize':22,
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
plt.rcParams.update(params)
audio_ts, sample_rate = librosa.load(
    f"{audio_path}/{genre}/{filename}.{ext}", sr=sample_rate
)
audio_ts, _ = librosa.effects.trim(audio_ts)
fig, ax = plt.subplots(figsize=(14,4))
img = librosa.display.waveshow(audio_ts, sr=sample_rate, ax=ax, color="#28708a")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Audio Signal")
fig.savefig(f"/home/matt/dev/projects/quantum-cnn/reports/20220202/audio_signal.svg")
# mel_feat = librosa.feature.melspectrogram(y=audio_ts, sr=sample_rate)

# %%
params = {'legend.fontsize': 18,
        'figure.figsize': (9, 6),
         'axes.labelsize': 18,
         'axes.titlesize':24,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

# %%
n_fft = 2048
hop_length = 2048
n_mels = 8
audio_ts, _ = librosa.effects.trim(audio_ts)
# First 3 seconds
audio_ts = audio_ts[0 : sample_rate * 3]
mel_spec = librosa.feature.melspectrogram(
    y=audio_ts,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    fmax=8000,
)
mel_sdb = librosa.power_to_db(mel_spec, ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(
    mel_sdb, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel"
)
ax.set_title("Mel spectogram")

fig.colorbar(img, ax=ax)
fig.savefig(f"/home/matt/dev/projects/quantum-cnn/reports/20220202/mel_3s.svg")

# %%
params = {'legend.fontsize': 18,
        'figure.figsize': (9, 6),
         'axes.labelsize': 18,
         'axes.titlesize':24,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

audio_ts, sample_rate = librosa.load(
    f"{audio_path}/{genre}/{filename}.{ext}"
)
n_fft = 2048
hop_length = 512
audio_ts, _ = librosa.effects.trim(audio_ts)
# First 3 seconds
#audio_ts = audio_ts[0 : sample_rate * 3]
mel_spec = librosa.feature.melspectrogram(
    y=audio_ts,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    fmax=8000,
)
mel_sdb = librosa.power_to_db(mel_spec, ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(
    mel_sdb, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel"
)
ax.set_title("Mel spectogram")

fig.colorbar(img, ax=ax)
fig.savefig(f"/home/matt/dev/projects/quantum-cnn/reports/20220202/mel_full.svg")

# %%
params = {'legend.fontsize': 18,
        'figure.figsize': (9, 6),
         'axes.labelsize': 18,
         'axes.titlesize':24,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

audio_ts, sample_rate = librosa.load(
    f"{audio_path}/{genre}/{filename}.{ext}"
)
n_fft = 2048
hop_length = 512
audio_ts, _ = librosa.effects.trim(audio_ts)
# First 3 seconds
audio_ts = audio_ts[0 : sample_rate * 3]
mel_spec = librosa.feature.melspectrogram(
    y=audio_ts,
    sr=sample_rate,
    n_fft=n_fft,
    hop_length=hop_length,
    fmax=8000,
)
mel_sdb = librosa.power_to_db(mel_spec, ref=np.max)
fig, ax = plt.subplots()
img = librosa.display.specshow(
    mel_sdb, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="mel"
)
ax.set_title("Mel spectogram")

fig.colorbar(img, ax=ax)
fig.savefig(f"/home/matt/dev/projects/quantum-cnn/reports/20220202/mel_full_3s.svg")
# %%
# Audio signal waveplot amplitude over time
X = librosa.stft(audio_ts)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
plt.colorbar()

# Audio signal waveplot amplitude over time
X = librosa.stft(audio_ts)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sample_rate, x_axis="time", y_axis="log")
plt.colorbar()


# %%
mel_feat = librosa.feature.melspectrogram(y=audio_ts, sr=sample_rate)
# all_wave.append(np.expand_dims(mel_feat, axis=2))
# all_label.append(label)
# # %%
# librosa.output.write_wav(f"{audio_path}/{genre}/{filename}", audio_ts, sample_rate)
# # %%
# # play
# wave_obj = sa.WaveObject.from_wave_file(f"{audio_path}/{genre}/{filename}")
# play_obj = wave_obj.play()
# # %%
# play_obj.stop()
# %%
import sklearn

mel_feat = sklearn.preprocessing.scale(mel_feat, axis=1)
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_feat, sr=sample_rate, x_axis="time")
plt.colorbar()
# %%
# Default FFT window size
n_fft = 2048  # FFT window size
hop_length = (
    512  # number audio of frames between STFT columns (looks like a good default
)
y, _ = librosa.effects.trim(audio_ts)
mel_s = librosa.feature.melspectrogram(y=y, sr=sample_rate)
mel_sdb = librosa.amplitude_to_db(mel_s, ref=np.max)
plt.figure(figsize=(16, 6))
librosa.display.specshow(
    mel_sdb,
    sr=sample_rate,
    hop_length=hop_length,
    x_axis="time",
    y_axis="log",
    cmap="cool",
)
plt.colorbar()
plt.title("Metal Mel Spectrogram", fontsize=23)

# %% Tutorial
# Convert mp3 to wav
from os import path
from pydub import AudioSegment

path = "/home/matt/dev/projects/quantum-cnn/data"
filename = "whale_song.mp3"

sound = AudioSegment.from_mp3(f"{path}/{filename}")
sound.export(f"{path}/whale.wav", format="wav")
# %%
import librosa
import librosa.display

path = "/home/matt/dev/projects/quantum-cnn/data"
filename = "whale.wav"
y, sr = librosa.load(f"{path}/{filename}")
whale_song, _ = librosa.effects.trim(y)
librosa.display.waveshow(whale_song, sr=sr)
# %% Fourier transform
import numpy as np
import matplotlib.pyplot as plt

n_fft = 2048
D = np.abs(librosa.stft(whale_song[:n_fft], n_fft=n_fft, hop_length=n_fft + 1))
plt.plot(D)
# %%
# use full song
hop_length = 512
D = np.abs(librosa.stft(whale_song, n_fft=n_fft, hop_length=hop_length))
librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="linear")
plt.colorbar()
# %%
DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log")
plt.colorbar(format="%+2.0f dB")

# %%
n_mels = 128
mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
# %%
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis="linear")
plt.ylabel("Mel filter")
plt.colorbar()
plt.title("1. Our filter bank for converting from Hz to mels.")

plt.subplot(1, 3, 2)
mel_10 = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=10)
librosa.display.specshow(mel_10, sr=sr, hop_length=hop_length, x_axis="linear")
plt.ylabel("Mel filter")
plt.colorbar()
plt.title("2. Easier to see what is happening with only 10 mels.")

plt.subplot(1, 3, 3)
idxs_to_plot = [0, 9, 49, 99, 127]
for i in idxs_to_plot:
    plt.plot(mel[i])
plt.legend(labels=[f"{i+1}" for i in idxs_to_plot])
plt.title("3. Plotting some triangular filters separately.")

plt.tight_layout()
# %%
S = librosa.feature.melspectrogram(
    whale_song, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(
    S_DB, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel"
)
plt.colorbar(format="%+2.0f dB")
# %%
