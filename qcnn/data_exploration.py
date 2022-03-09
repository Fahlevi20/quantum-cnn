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
import tensorflow as tf


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
        b =ax.set_xlabel("Principal components")
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
audio_path = "../data/genres_original"
target = "label"
classes = ["classical", "pop"]

genre = "classical"
filename = f"{genre}.00005.wav"

audio_ts, sample_rate = librosa.load(f"{audio_path}/{genre}/{filename}", sr=44100)

# First second
print(audio_ts[0:sample_rate])
# %%
librosa.display.waveshow(audio_ts, sr=sample_rate)


# %%
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
import sklearn
mel_feat = sklearn.preprocessing.scale(mel_feat, axis=1)
plt.figure(figsize=(14, 5))
librosa.display.specshow(mel_feat, sr=sample_rate, x_axis='time')
plt.colorbar()
# %%
