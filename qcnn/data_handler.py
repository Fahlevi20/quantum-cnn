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


def get_2d_modelling_data(path, colnames=None, set_name=None, **kwargs):

    if kwargs.get("from_file", False):
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
    else:
        if set_name == "GTZAN":

            import librosa
            genres = ['hiphop', 'classical', 'blues', 'metal', 'jazz', 'country', 'pop', 'rock', 'disco', 'reggae']
            sr = kwargs.get("sr", 22050)
            n_fft = kwargs.get("n_fft", 2048)
            hop_length = kwargs.get("hop_length", 512)
            n_mels = kwargs.get("n_mels", 128)
            fmax = kwargs.get("fmax", 8000)
            # Number of seconds to sample
            duration = kwargs.get("duration", None)
            raw = pd.DataFrame(
                {
                    "filename": [],
                    "length": [],
                    "chroma_stft_mean": [],
                    "chroma_stft_var": [],
                    "rms_mean": [],
                    "rms_var": [],
                    "spectral_centroid_mean": [],
                    "spectral_centroid_var": [],
                    "spectral_bandwidth_mean": [],
                    "spectral_bandwidth_var": [],
                    "rolloff_mean": [],
                    "rolloff_var": [],
                    "zero_crossing_rate_mean": [],
                    "zero_crossing_rate_var": [],
                    "harmony_mean": [],
                    "harmony_var": [],
                    "perceptr_mean": [],
                    "perceptr_var": [],
                    "tempo": [],
                    "mfcc1_mean": [],
                    "mfcc1_var": [],
                    "mfcc2_mean": [],
                    "mfcc2_var": [],
                    "mfcc3_mean": [],
                    "mfcc3_var": [],
                    "mfcc4_mean": [],
                    "mfcc4_var": [],
                    "mfcc5_mean": [],
                    "mfcc5_var": [],
                    "mfcc6_mean": [],
                    "mfcc6_var": [],
                    "mfcc7_mean": [],
                    "mfcc7_var": [],
                    "mfcc8_mean": [],
                    "mfcc8_var": [],
                    "mfcc9_mean": [],
                    "mfcc9_var": [],
                    "mfcc10_mean": [],
                    "mfcc10_var": [],
                    "mfcc11_mean": [],
                    "mfcc11_var": [],
                    "mfcc12_mean": [],
                    "mfcc12_var": [],
                    "mfcc13_mean": [],
                    "mfcc13_var": [],
                    "mfcc14_mean": [],
                    "mfcc14_var": [],
                    "mfcc15_mean": [],
                    "mfcc15_var": [],
                    "mfcc16_mean": [],
                    "mfcc16_var": [],
                    "mfcc17_mean": [],
                    "mfcc17_var": [],
                    "mfcc18_mean": [],
                    "mfcc18_var": [],
                    "mfcc19_mean": [],
                    "mfcc19_var": [],
                    "mfcc20_mean": [],
                    "mfcc20_var": [],
                    "label": [],
                }
            )
            for genre in genres:
                for filename in os.listdir(f"{path}/{genre}"):
                    if not (filename in ["jazz.00054.wav"]):
                        audio_ts, sr = librosa.load(f"{path}/{genre}/{filename}", sr=sr)
                        audio_ts, _ = librosa.effects.trim(y=audio_ts)
                        if duration:
                            audio_ts = audio_ts[0 : sr * duration]
                        # Get mfcc's
                        mfccs = librosa.feature.mfcc(y=audio_ts, sr=sr)
                        # Chromogram
                        chromagram = librosa.feature.chroma_stft(y=audio_ts, sr=sr)
                        chroma_stft_mean = chromagram.mean()
                        chroma_stft_var = chromagram.var()
                        # rms
                        rms = librosa.feature.rms(y=audio_ts)
                        rms_mean = rms.mean()
                        rms_var = rms.var()
                        # Spectral centroids
                        spectral_centroids = librosa.feature.spectral_centroid(
                            y=audio_ts, sr=sr
                        )
                        sc_mean = spectral_centroids.mean()
                        sc_var = spectral_centroids.var()
                        # Spectral centroids
                        spectral_bandwidths = librosa.feature.spectral_bandwidth(
                            y=audio_ts, sr=sr
                        )
                        sb_mean = spectral_bandwidths.mean()
                        sb_var = spectral_bandwidths.var()
                        # Roll off
                        spectral_rolloffs = librosa.feature.spectral_rolloff(
                            y=audio_ts, sr=sr
                        )
                        ro_mean = spectral_rolloffs.mean()
                        ro_var = spectral_rolloffs.var()
                        # zero crossing rate
                        zero_crossing_rates = librosa.feature.zero_crossing_rate(
                            y=audio_ts
                        )
                        zc_mean = zero_crossing_rates.mean()
                        zc_var = zero_crossing_rates.var()
                        # Median-filtering harmonic percussive source separation
                        harm, perc = librosa.effects.hpss(y=audio_ts)
                        harm_mean = harm.mean()
                        harm_var = harm.var()
                        perc_mean = perc.mean()
                        perc_var = perc.var()
                        # tempo
                        tempo, _ = librosa.beat.beat_track(y=audio_ts, sr=sr)
                        # Store to data frame
                        audio_track_stats = {
                            "filename": [filename],
                            "length": [audio_ts.shape[0]],
                            "chroma_stft_mean": [chroma_stft_mean],
                            "chroma_stft_var": [chroma_stft_var],
                            "rms_mean": [rms_mean],
                            "rms_var": [rms_var],
                            "spectral_centroid_mean": [sc_mean],
                            "spectral_centroid_var": [sc_var],
                            "spectral_bandwidth_mean": [sb_mean],
                            "spectral_bandwidth_var": [sb_var],
                            "rolloff_mean": [ro_mean],
                            "rolloff_var": [ro_var],
                            "zero_crossing_rate_mean": [zc_mean],
                            "zero_crossing_rate_var": [zc_var],
                            "harmony_mean": [harm_mean],
                            "harmony_var": [harm_var],
                            "perceptr_mean": [perc_mean],
                            "perceptr_var": [perc_var],
                            "tempo": [tempo],
                            "mfcc1_mean": [mfccs[0].mean()],
                            "mfcc1_var": [mfccs[0].var()],
                            "mfcc2_mean": [mfccs[1].mean()],
                            "mfcc2_var": [mfccs[1].var()],
                            "mfcc3_mean": [mfccs[2].mean()],
                            "mfcc3_var": [mfccs[2].var()],
                            "mfcc4_mean": [mfccs[3].mean()],
                            "mfcc4_var": [mfccs[3].var()],
                            "mfcc5_mean": [mfccs[4].mean()],
                            "mfcc5_var": [mfccs[4].var()],
                            "mfcc6_mean": [mfccs[5].mean()],
                            "mfcc6_var": [mfccs[5].var()],
                            "mfcc7_mean": [mfccs[6].mean()],
                            "mfcc7_var": [mfccs[6].var()],
                            "mfcc8_mean": [mfccs[7].mean()],
                            "mfcc8_var": [mfccs[7].var()],
                            "mfcc9_mean": [mfccs[8].mean()],
                            "mfcc9_var": [mfccs[8].var()],
                            "mfcc10_mean": [mfccs[9].mean()],
                            "mfcc10_var": [mfccs[9].var()],
                            "mfcc11_mean": [mfccs[10].mean()],
                            "mfcc11_var": [mfccs[10].var()],
                            "mfcc12_mean": [mfccs[11].mean()],
                            "mfcc12_var": [mfccs[11].var()],
                            "mfcc13_mean": [mfccs[12].mean()],
                            "mfcc13_var": [mfccs[12].var()],
                            "mfcc14_mean": [mfccs[13].mean()],
                            "mfcc14_var": [mfccs[13].var()],
                            "mfcc15_mean": [mfccs[14].mean()],
                            "mfcc15_var": [mfccs[14].var()],
                            "mfcc16_mean": [mfccs[15].mean()],
                            "mfcc16_var": [mfccs[15].var()],
                            "mfcc17_mean": [mfccs[16].mean()],
                            "mfcc17_var": [mfccs[16].var()],
                            "mfcc18_mean": [mfccs[17].mean()],
                            "mfcc18_var": [mfccs[17].var()],
                            "mfcc19_mean": [mfccs[18].mean()],
                            "mfcc19_var": [mfccs[18].var()],
                            "mfcc20_mean": [mfccs[19].mean()],
                            "mfcc20_var": [mfccs[19].var()],
                            "label": [genre],
                        }
                        raw = pd.concat(
                            [raw, pd.DataFrame.from_dict(audio_track_stats)],
                            ignore_index=True,
                        )

            if kwargs.get("save", False):
                raw.to_csv(f"{path}/raw.csv")

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
