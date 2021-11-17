import os
import json
from joblib import dump, load
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D

from preprocessing import apply_preprocessing

# TODO move to seperate file since this is duplicated in qcnn_structure
def filter_levels(data, feature, levels):
    """Returns all rows belonging to the list of levels for a specific feature

    Args:
        data (pd.DataFrame): data to filter rows
        feature (str): name of the feature for which the levels are concerned
        levels (list[str or int]): distinct values of the feature to filter
    """
    filter_pat = "|".join(level for level in levels)
    indices = data[feature].str.contains(filter_pat)
    return data.loc[indices, :].copy()

def store_results(
    config,
    model_name,
    best_estimator,
    clf_grid_results,
    y_hat,
    cf_matrix,
):
    """
    Method to store results to a desired path
    """
    result_path = f"{config.get('path')}/{config.get('ID')}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Give expirment context
    # with open(f"{result_path}/experiment_config.json", "w+") as f:
    #     json.dump(config, f, indent=4)

    # print(f"Storing resuts to:\n {result_path}")
    pd.DataFrame(y_hat).to_csv(f"{result_path}/{model_name}-yhat.csv")
    pd.DataFrame(cf_matrix).to_csv(f"{result_path}/{model_name}-confusion-matrix.csv")
    dump(best_estimator, f"{result_path}/{model_name}-estimator.joblib")
    dump(clf_grid_results, f"{result_path}/{model_name}-clf-grid-results.joblib")    
    


def train_classical(config, algorithm, pipeline, samples, target_pair=None, model_name="dummy"):
    """[summary]

    Args:
        pipeline ([type]): [description]
        target_pair ([type]): [description]
        raw ([type]): [description]
        data_utility ([type]): [description]
        config ([type]): [description]
        model_name (str, optional): [description]. Defaults to "dummy".
    """
    model_type = "classical"
    save_results = False if config.get("path", None) is None else True

    # Get training job information TODO fix
    # iterations = config["train"].get("iterations", 200)
    # learning_rate = config["train"].get("learning_rate", 0.01)
    # batch_size = config["train"].get("batch_size", 25)
    # cost_fn = config["train"].get("cost_fn", "cross_entropy")
    # test_size = config["train"].get("test_size", 0.3)
    # random_state = config["train"].get("random_state", 42)
    data_type = config["data"].get("type", None)
    # Get model information
    classification_type = config["model"].get("classification_type", "binary")

    # Get algorithm information
    param_grid = config["model"][model_type][algorithm].get("param_grid", {})

    # Preprocessing
    samples_tfd = apply_preprocessing(samples, pipeline, classification_type, data_type, target_pair)

    if algorithm == "svm":
        model = svm.SVC()
    elif algorithm == "logistic_regression":
        model= LogisticRegression()
    elif algorithm == "cnn":
        pass
        # n_cols = X_train_tfd.shape[1]
        # n_rows = X_train_tfd.shape[0]
        # def cnn(optimizer='rmsprop', init='glorot_uniform'):
        #     # create model
        #     model = Sequential()
        #     model.add(Conv1D(1, kernel_size=2, activation='relu', input_shape=(n_rows,n_cols)))
        #     model.add(MaxPool1D(1, kernel_size=2, activation='relu', input_shape=(n_rows,n_cols)))
        #     model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
        #     model.add(Dense(8, kernel_initializer=init, activation='relu'))
        #     model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
        #     # Compile model
        #     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        #     return model    
        #     nn.Conv1d(in_channels=1, out_channels=n_feature, kernel_size=2, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=2, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(n_feature * final_layer_size, 2),
        # model = KerasClassifier(build_fn=cnn, verbose=0)


    # TODO ovo already implemented https://scikit-learn.org/stable/modules/svm.html in svm
    clf = GridSearchCV(model, param_grid)
    if classification_type == "ovo":
        clf = OneVsOneClassifier(clf)
    elif classification_type == "ova":
        clf = OneVsRestClassifier(clf)

    clf.fit(samples_tfd.X_train, samples_tfd.y_train)

    best_estimator = clf.best_estimator_
    # Get predictions
    y_hat = best_estimator.predict(samples_tfd.X_test)
    cf_matrix = confusion_matrix(samples_tfd.y_test, y_hat)

    if save_results:
        store_results(
            config,
            model_name,
            best_estimator,
            clf,
            y_hat,
            cf_matrix,
        )