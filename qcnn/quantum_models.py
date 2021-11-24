import os
import json
from joblib import dump
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from qcnn_estimator import Qcnn_Classifier
from preprocessing import apply_preprocessing

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


def train_quantum(config, qcnn_structure_permutation, embedding_type, algorithm, pipeline, samples, target_pair=None, model_name="dummy"):
    """

    """
    model_type = "quantum"
    save_results = False if config.get("path", None) is None else True

    
    data_type = config["data"].get("type", None)
    # Get model information
    classification_type = config["model"].get("classification_type", "binary")
    cv_folds = config["model"].get("cv_folds", None)
    n_jobs = config["model"].get("cv_folds", -1)

    # Get algorithm information
    param_grid = config["model"][model_type][algorithm].get("param_grid", {})

    # Preprocessing
    samples_tfd = apply_preprocessing(samples, pipeline, classification_type, data_type, target_pair)

    # TODO naming consistency with encoding_type, qcnn_structure_permutation etc
    encoding_kwargs = config["preprocessing"][model_type][embedding_type].get("kwargs", {})
    if algorithm == "qcnn":
        model = Qcnn_Classifier(layer_defintion=qcnn_structure_permutation, encoding_type=embedding_type, encoding_kwargs=encoding_kwargs)

    
    if not(classification_type=="binary"):
        # This means param grid should contain estimator__ for each of the estimators paramaters
        if not(all("estimator__" in key for key in param_grid.keys())):
            param_grid = {f"estimator__{key}": value for key, value in param_grid.items()}
        if classification_type == "ovo":
            model = OneVsOneClassifier(model)
        elif classification_type == "ova":
            model = OneVsRestClassifier(model)
    clf = GridSearchCV(model, param_grid, n_jobs=n_jobs, cv=cv_folds) # error_score="raise" <- for debugging
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