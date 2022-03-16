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


def get_2d_modelling_data(path, colnames=None):

    filename, file_extension = os.path.splitext(path)

    if colnames:
        if file_extension == ".csv":
            raw = pd.read_csv(path, names=colnames, header=None)
        elif file_extension in [".parq", ".parquet"]:
            raw = pd.read_parquet(path, engine="auto", names=colnames, header=None)
    else:
        if file_extension == ".csv":
            raw = pd.read_csv(path)
        elif file_extension in [".parq", ".parquet"]:
            raw = pd.read_parquet(path, engine="auto")

    return raw


def create_train_test_samples(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return Samples(X_train, y_train, X_test, y_test)


def get_image_data(path, set_name="mnist", **kwargs):
    if path:
        if set_name == "GTZAN":
            if kwargs.get("from_file", False):
                X = np.load(f"{path}/X.npy")
                y = np.load(f"{path}/y.npy")
            else:
                import librosa

                X = []
                y = []
                sr = kwargs.get("sr", 21500)
                n_fft = kwargs.get("n_fft", 2048)
                hop_length = kwargs.get("hop_length", 2048)
                n_mels = kwargs.get("n_mels", 8)
                fmax = kwargs.get("fmax", 8000)
                # Number of seconds to sample
                duration = kwargs.get("duration", 3)
                for genre in os.listdir(path):
                    for filename in os.listdir(f"{path}/{genre}"):
                        if not (filename in ["jazz.00054.wav"]):
                            audio_ts, sr = librosa.load(
                                f"{path}/{genre}/{filename}", sr=sr
                            )
                            audio_ts, _ = librosa.effects.trim(audio_ts)
                            audio_ts = audio_ts[0 : sr * duration]
                            # Get melspec
                            mel_spec = librosa.feature.melspectrogram(
                                y=audio_ts,
                                sr=sr,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                n_mels=n_mels,
                                fmax=fmax,
                            )
                            # Convert to dB
                            mel_sdb = librosa.power_to_db(mel_spec, ref=np.max)
                            X.append(mel_sdb)
                            y.append(genre)
                X = np.array(X)
                y = np.array(y)
                if kwargs.get("save", False):
                    np.save(f"{path}/X.npy", X)
                    np.save(f"{path}/y.npy", y)

            samples = create_train_test_samples(
                X,
                y,
                test_size=kwargs.get("test_size", 0.3),
                random_state=kwargs.get("random_state", 42),
            )
        return samples

    else:
        if set_name == "mnist":
            # TODO this tensforflow import slows things down and isn't needed
            # %%
            import tensorflow as tf

            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        elif set_name == "f_mnist":
            import tensorflow as tf

            (X_train, y_train), (
                X_test,
                y_test,
            ) = tf.keras.datasets.fashion_mnist.load_data()
        elif set_name == "gtzan":
            pass

    return Samples(X_train, y_train, X_test, y_test)
