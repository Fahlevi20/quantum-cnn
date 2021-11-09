import os
import json
from joblib import dump, load
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


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
    with open(f"{result_path}/experiment_config.json", "w+") as f:
        json.dump(config, f, indent=4)

    # print(f"Storing resuts to:\n {result_path}")
    pd.DataFrame(y_hat).to_csv(f"{result_path}/{model_name}-yhat.csv")
    pd.DataFrame(cf_matrix).to_csv(f"{result_path}/{model_name}-confusion-matrix.csv")
    dump(best_estimator, f"{result_path}/{model_name}-estimator.joblib")
    dump(clf_grid_results, f"{result_path}/{model_name}-clf-grid-results.joblib")    
    
    

def train_classical(algorithm, pipeline, target_levels, raw, data_utility, config, model_name="dummy"):
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

    # Get training job information
    iterations = config["train"].get("iterations", 200)
    learning_rate = config["train"].get("learning_rate", 0.01)
    batch_size = config["train"].get("batch_size", 25)
    cost_fn = config["train"].get("cost_fn", "cross_entropy")
    test_size = config["train"].get("test_size", 0.3)
    random_state = config["train"].get("random_state", 42)

    # Get model information
    classification_type = config["model"].get("classification_type", "binary")

    # Get algorithm information
    param_grid = config["model"][model_type][algorithm].get("param_grid", {})

    # Preprocessing
    if classification_type in ["ova"]:
        ## Make target binary 1 for target 0 rest
        raw[data_utility.target] = np.where(
            raw[data_utility.target] == target_levels[1], 1, 0
        )

        X_train, y_train, Xy_test, X_test, y_test, Xy_test = data_utility.get_samples(
            raw, row_samples=["train", "test"]
        )
    else:
        # Get test set first
        X_test_all, y_test_all, Xy_test_all = data_utility.get_samples(
            raw, row_samples=["test"]
        )
        y_test_all = np.where(y_test_all == target_levels[1], 1, 0)
        ## Filter data
        raw = filter_levels(raw, data_utility.target, levels=target_levels)

        ## Make target binary TODO generalize more classes
        raw[data_utility.target] = np.where(
            raw[data_utility.target] == target_levels[1], 1, 0
        )
        ## Get train test splits, X_test here will be only for the subset of data, so used to evaluate the single model
        # but not the OvO combinded one
        X_train, y_train, Xy_test, X_test, y_test, Xy_test = data_utility.get_samples(
            raw, row_samples=["train", "test"]
        )

    pipeline.fit(X_train, y_train)

    # Transform data
    X_train_tfd = pipeline.transform(X_train)
    X_test_tfd = pipeline.transform(X_test)
    if classification_type == "ovo":
        X_test_all_tfd = pipeline.transform(X_test_all)
    elif classification_type == "ova":
        X_test_all = X_test

    if algorithm == "SVM":
        model = svm.SVC()
    elif algorithm == "logistic_regression":
        model= LogisticRegression()
    
    # TODO ovo already implemented https://scikit-learn.org/stable/modules/svm.html in svm
    clf = GridSearchCV(model, param_grid)
    clf.fit(X_train, y_train)

    best_estimator = clf.best_estimator_
    # Get predictions
    y_hat = best_estimator.predict(X_test)
    cf_matrix = confusion_matrix(y_test, y_hat)

    if save_results:
        store_results(
            config,
            model_name,
            best_estimator,
            clf,
            y_hat,
            cf_matrix,
        )