# %%

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import os
from joblib import dump, load
from collections import namedtuple
from ast import literal_eval
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
import networkx as nx


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    accuracy_score,
)

from circuit_presets import (
    CIRCUIT_OPTIONS,
    POOLING_OPTIONS,
)

from preprocessing import filter_embedding_options, EMBEDDING_OPTIONS


def get_model_result_list(experiment_config):
    path = f"{experiment_config.get('path')}/{experiment_config.get('ID')}"
    # X_test = pd.read_csv(f"{path}/X_test.csv", index_col=0)
    Results = namedtuple(
        "Results",
        [
            "model_name",
            "y_test_hat",
            "clf",
            "model_configuration",
            "samples_tfd",
            "pipeline",
        ],
    )
    model_names = [
        filename.split("-model_configuration.joblib")[0]
        for filename in os.listdir(path)
        if "-model_configuration.joblib" in filename
    ]
    result_list = []
    for model_name in model_names:
        y_test_hat = pd.read_csv(f"{path}/{model_name}-yhat.csv", index_col=0)
        clf = load(f"{path}/{model_name}-clf_results.joblib")
        model_configuration = load(f"{path}/{model_name}-model_configuration.joblib")
        samples_tfd = load(f"{path}/{model_name}-samples_tfd.joblib")
        if os.path.exists(f"{path}/{model_name}-pipeline.joblib"):
            pipeline = load(f"{path}/{model_name}-pipeline.joblib")
        else:
            pipeline = {}
        result_list = result_list + [
            Results(
                model_name,
                y_test_hat=y_test_hat,
                clf=clf,
                model_configuration=model_configuration,
                samples_tfd=samples_tfd,
                pipeline=pipeline,
            )
        ]
    return result_list


def confusion_matrix_stats(cf_matrix):
    cf = np.array(cf_matrix)
    accuracy = np.trace(cf) / float(np.sum(cf))
    precision = cf[1, 1] / sum(cf[:, 1])
    recall = cf[1, 1] / sum(cf[1, :])
    f1_score = 2 * precision * recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        accuracy, precision, recall, f1_score
    )
    return accuracy, precision, recall, f1_score, stats_text


def plot_confusion_matrix_nn(cf_matrix, classes, stats_text):
    sns.set_theme()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    fig.suptitle("Confusion Matrix")

    sns.heatmap(
        cf_matrix,
        ax=axes[0],
        annot=True,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        annot_kws={"size": 16},
    )

    sns.heatmap(
        cf_matrix / np.sum(cf_matrix),
        ax=axes[1],
        annot=True,
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        annot_kws={"size": 16},
    )
    axes[0].set_xlabel(stats_text)
    axes[1].set_xlabel(stats_text)


def get_tp_tn_lists(estimate_vs_class, y_test):
    TP = pd.Series(
        np.where(
            (estimate_vs_class["y_test"] == 1) & (estimate_vs_class["yhat_class"] == 1),
            True,
            False,
        )
    )
    TN = pd.Series(
        np.where(
            (estimate_vs_class["y_test"] == 0) & (estimate_vs_class["yhat_class"] == 0),
            True,
            False,
        )
    )
    FP = pd.Series(
        np.where(
            (estimate_vs_class["y_test"] == 0) & (estimate_vs_class["yhat_class"] == 1),
            True,
            False,
        )
    )
    FN = pd.Series(
        np.where(
            (estimate_vs_class["y_test"] == 1) & (estimate_vs_class["yhat_class"] == 0),
            True,
            False,
        )
    )

    TP.index = y_test.index
    TN.index = y_test.index
    FP.index = y_test.index
    FN.index = y_test.index

    return TP, TN, FP, FN


def get_result_table(data, groupby, metric):
    # groupby =["circuit", "embedding_option"]
    grouped_data = data.groupby(groupby)[metric].mean()
    display_table = grouped_data.copy()
    display_table = display_table.unstack(level=-1)
    display_table.loc[f"{groupby[1]} Average"] = display_table.mean(axis=0)
    display_table[f"{groupby[0]} Average"] = display_table.mean(axis=1)
    return display_table


def get_result_table_target_pairs(data, each_var, group_var, metric, group_filter=[]):
    """TODO move to generic function

    Args:
        data ([type]): [description]
        groupby ([type]): [description]
        metric ([type]): [description]

    Returns:
        [type]: [description]
    """
    # groupby =["circuit", "embedding_option"]
    grouped_data = data.groupby([each_var, group_var])[metric].max()
    grouped_data = grouped_data.unstack(level=0).copy()
    all_combos = [pair.split("_") for pair in grouped_data.index]
    distinct_levels = sorted({item for combo in all_combos for item in combo})
    display_table = pd.DataFrame(columns=distinct_levels, index=distinct_levels)
    for index, row in grouped_data.iterrows():
        target_pair = index.split("_")
        display_table.loc[target_pair[0], target_pair[1]] = row[0]
        display_table.loc[target_pair[1], target_pair[0]] = row[0]

    display_table.loc[f"Average"] = display_table.mean(axis=0)
    display_table[f"Average"] = display_table.mean(axis=1)
    return display_table


def plot_triangle_accuracies(
    plot_data, figsize=(10, 10), title="Accuracy for pairs of genre's"
):
    mask = np.zeros_like(plot_data)
    mask[np.triu_indices_from(mask)] = True
    mask[mask.shape[0] - 1, mask.shape[1] - 1] = False
    with sns.axes_style("whitegrid"):
        f, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        sns.heatmap(
            plot_data,
            annot=True,
            fmt=".0%",
            ax=ax,
            vmin=0.3,
            cmap=sns.dark_palette("#28708a", reverse=False, as_cmap=True),
            mask=mask,
        )
    return plt


def plot_loss(data, groupby, group_filter=[], figsize=(30, 5), save_plot=False):
    grouped_data_train = data.groupby(groupby)["loss_train_history"].max()
    loss_history_train = grouped_data_train.copy()
    loss_history_train = loss_history_train.unstack(level=0)

    grouped_data_test = data.groupby(groupby)["loss_test_history"].max()
    loss_history_test = grouped_data_test.copy()
    loss_history_test = loss_history_test.unstack(level=0)

    sns.set(font_scale=1.2)
    for col in loss_history_train.columns:
        plot_data_train = pd.DataFrame()
        plot_data_test = pd.DataFrame()
        for index, row in loss_history_train.iterrows():
            if type(row[col]) == np.ndarray and check_filter_on_list(
                group_filter, index[len(index) - 1].split("-")
            ):
                plot_data_train[index] = row[col]
        plot_data_train["Iteration"] = plot_data_train.index

        plot_data_test = pd.DataFrame()
        for index, row in loss_history_test.iterrows():
            if type(row[col]) == np.ndarray and check_filter_on_list(
                group_filter, index[len(index) - 1].split("-")
            ):
                plot_data_test[index] = row[col]
        plot_data_test["Iteration"] = plot_data_test.index

        if not (plot_data_train.empty):
            with sns.axes_style("whitegrid"):
                fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
                sns.lineplot(
                    ax=axes[0],
                    data=pd.melt(
                        plot_data_train,
                        "Iteration",
                        value_name="MSE Cost",
                        var_name=groupby[0],
                    ),
                    x="Iteration",
                    y="MSE Cost",
                    hue=groupby[0],
                    markers=True,
                    dashes=False,
                )
                axes[0].set_title(
                    f"{col}-{'-'.join(group_filter)} Train Cost Per Iteration"
                )
                axes[0].savefig(
                    f"{col}-{'-'.join(group_filter)} Train Cost Per Iteration.svg"
                )
                sns.lineplot(
                    ax=axes[1],
                    data=pd.melt(
                        plot_data_test,
                        "Iteration",
                        value_name="MSE Cost",
                        var_name=groupby[0],
                    ),
                    x="Iteration",
                    y="MSE Cost",
                    hue=groupby[0],
                    markers=True,
                    dashes=False,
                )
                axes[1].set_title(
                    f"{col}-{'-'.join(group_filter)} Test Cost Per Iteration"
                )
            if save_plot == True:
                fig.savefig(f"{col}-{'-'.join(group_filter)}.svg")


def get_line_plot_data(data, groupby, metric):
    grouped_data = data.groupby(groupby)[metric].max()
    grouped_data_unstack = grouped_data.copy().unstack(level=-1)
    grouped_data_unstack[grouped_data_unstack.index.name] = grouped_data_unstack.index
    return grouped_data_unstack.copy()


def plot_119_accuracy_per_structure(result_data, figsize=(14, 7)):
    groupby = ["additional_structure", "selection_method"]
    metric = "accuracy"
    data_0 = result_data[result_data["target_pair_str"] == "rock_reggae"].copy()
    data_1 = result_data[result_data["target_pair_str"] == "classical_pop"].copy()
    plot_data_0 = get_line_plot_data(data_0, groupby, metric)
    plot_data_1 = get_line_plot_data(data_1, groupby, metric)
    # sns.set(font_scale=1.2)
    import matplotlib.ticker as ticker

    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        # axes[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # axes[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # axes[1].set_aspect('equal', adjustable='box')
        axes[0].set_title("Rock vs Reggae")
        axes[0].set_xlabel("Additional Structure")
        sns.lineplot(
            ax=axes[0],
            data=pd.melt(
                plot_data_0,
                "additional_structure",
                value_name="Accuracy",
                var_name="selection_method",
            ),
            x="additional_structure",
            y="Accuracy",
            hue="selection_method",
            markers=True,
            dashes=False,
            marker="o",
        )
        axes[1].set_title("Classical vs Pop")
        axes[1].set_xlabel("Additional Structure")
        sns.lineplot(
            ax=axes[1],
            data=pd.melt(
                plot_data_1,
                "additional_structure",
                value_name="Accuracy",
                var_name="selection_method",
            ),
            x="additional_structure",
            y="Accuracy",
            hue="selection_method",
            markers=True,
            dashes=False,
            marker="o",
        )
    return fig


def check_filter_on_list(filter_list, check_list):
    """
    Filter list contains items to check whether it's any is in the check_list.
    Returns true if filter list is empty which has the meaning of a wild card, i.e.
    all items should match
    """
    if len(filter_list) == 0:
        return True
    else:
        return len(set(filter_list) & set(check_list)) > 0


def gather_results_118_135_deprecated(
    exp_id, path_experiments=f"/home/matt/dev/projects/quantum-cnn/experiments"
):
    path_single_experiment = f"{path_experiments}/{exp_id}"
    model_names = get_model_names(path_single_experiment)

    Results = namedtuple(
        "Results",
        [
            "model_name",
            "y_test_hat",
            "clf",
            "model_configuration",
            "samples_tfd",
            "pipeline",
        ],
    )
    result_list = []
    for model_name in model_names:
        y_test_hat = pd.read_csv(
            f"{path_single_experiment}/{model_name}-yhat.csv", index_col=0
        )
        clf = load(f"{path_single_experiment}/{model_name}-clf_results.joblib")
        model_configuration = load(
            f"{path_single_experiment}/{model_name}-model_configuration.joblib"
        )
        samples_tfd = load(f"{path_single_experiment}/{model_name}-samples_tfd.joblib")
        pipeline = load(f"{path_single_experiment}/{model_name}-pipeline.joblib")
        result_list = result_list + [
            Results(
                model_name,
                y_test_hat=y_test_hat,
                clf=clf,
                model_configuration=model_configuration,
                samples_tfd=samples_tfd,
                pipeline=pipeline,
            )
        ]

    result_data = pd.DataFrame(
        {
            "model_name": [],
            "model_type": [],
            "algorithm": [],
            "classification_type": [],
            "embedding_type": [],
            "scaler_method": [],
            "scaler_param_str": [],
            "selection_method": [],
            "selection_param_str": [],
            "target_pair": [],
            "additional_structure": [],
            "target_pair_str": [],
            "mean_test_score": [],
            "std_test_score": [],
            "params": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "loss_train_history": [],
        }
    )
    for result in result_list:
        y_test_hat = result.y_test_hat
        clf = result.clf
        model_configuration = result.model_configuration
        samples_tfd = result.samples_tfd
        model_name = result.model_name

        precision, recall, fscore, support = precision_recall_fscore_support(
            samples_tfd.y_test, y_test_hat, average="binary"  # TODO multiclass
        )
        accuracy = accuracy_score(samples_tfd.y_test, y_test_hat)
        tmp_result = model_configuration._asdict()
        tmp_result["model_name"] = model_name

        if not (type(model_configuration.target_pair[0]) == str):
            # if target pair are ints
            tmp_result[
                "target_pair_str"
            ] = f"{model_configuration.target_pair[0]}_{model_configuration.target_pair[1]}"
        else:
            tmp_result["target_pair_str"] = "_".join(model_configuration.target_pair)

        tmp_result["mean_test_score"] = clf.cv_results_["mean_test_score"][
            clf.best_index_
        ]
        tmp_result["std_test_score"] = clf.cv_results_["std_test_score"][
            clf.best_index_
        ]
        tmp_result["params"] = clf.cv_results_["params"][clf.best_index_]

        tmp_result["accuracy"] = accuracy
        tmp_result["precision"] = precision
        tmp_result["recall"] = recall
        tmp_result["f1"] = fscore
        tmp_result["loss_train_history"] = None  # set for quantum
        result_data = result_data.append(tmp_result, ignore_index=True)

    return result_data


def gather_results_deprecated(
    result_path, circuit_options, embedding_options, reduction_method="pca"
):
    # Setup structure for results
    result_data = pd.DataFrame(
        {
            "model": [],
            "circuit": [],
            "circuit_param_count": [],
            "reduction_method": [],
            "reduction_size": [],
            "embeded_full_name": [],
            "embedding_class": [],
            "embedding_permutation": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "loss_train_history": [],
        }
    )
    for reduction_size, embed_list in embedding_options.items():
        for embeded_full_name in embed_list:
            for circ_name, circ_param_count in circuit_options.items():
                if "Hybrid" in embeded_full_name:
                    name_parts = embeded_full_name.split("-")
                    embedding_class = f"{name_parts[0]}-{name_parts[1]}"
                    embedding_permutation = name_parts[2]
                else:
                    # Case when no hybrid encoding was done
                    embedding_class = embeded_full_name
                    embedding_permutation = 1
                model_name = f"{reduction_method}-{reduction_size}-{embeded_full_name}-{circ_name}"
                result_files = os.listdir(result_path)
                if f"{model_name}-param-history.csv" in result_files:
                    # print(model_name)
                    # TODO implement loss history data combination
                    cf_matrix = pd.read_csv(
                        f"{result_path}/{model_name}-confusion-matrix.csv", index_col=0
                    )
                    # TODO improve this is just to make it backwards compatible with expirement 0
                    # print(model_name)
                    if f"{model_name}-loss-history.csv" in result_files:
                        loss_train_history = pd.read_csv(
                            f"{result_path}/{model_name}-loss-history.csv"
                        )
                        loss_test_history = pd.read_csv(
                            f"{result_path}/{model_name}-loss-test-history.csv"
                        )
                        loss_train_history.columns = ["Iteration", "Train_Cost"]
                        loss_test_history.columns = ["Iteration", "Test_Cost"]
                    else:
                        loss_train_history = pd.read_csv(
                            f"{result_path}/{model_name}-loss-train-history.csv"
                        )
                        loss_test_history = pd.read_csv(
                            f"{result_path}/{model_name}-loss-test-history.csv"
                        )
                        if loss_train_history.shape[1] > 2:
                            loss_train_history.drop(
                                loss_train_history.columns[0], inplace=True, axis=1
                            )
                            loss_test_history.drop(
                                loss_test_history.columns[0], inplace=True, axis=1
                            )

                        loss_train_history.columns = ["Iteration", "Train_Cost"]
                        loss_test_history.columns = ["Iteration", "Test_Cost"]

                    # estimate_vs_class = pd.read_csv(
                    #     f"{result_path}/{model_name}-yhat-class-vs-y-test.csv",
                    #     index_col=0,
                    # )
                    # y_hat = pd.read_csv(
                    #     f"{result_path}/{model_name}-yhat.csv", index_col=0
                    # )
                    (
                        accuracy,
                        precision,
                        recall,
                        f1,
                        stats_text,
                    ) = confusion_matrix_stats(cf_matrix)
                    result = {
                        "model": model_name,
                        "circuit": circ_name,
                        "circuit_param_count": circ_param_count,
                        "reduction_method": reduction_method,
                        "reduction_size": reduction_size,
                        "embeded_full_name": embeded_full_name,
                        "embedding_class": embedding_class,
                        "embedding_permutation": embedding_permutation,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "loss_train_history": loss_train_history["Train_Cost"],
                        "loss_test_history": loss_test_history["Test_Cost"],
                    }
                    result_data = result_data.append(result, ignore_index=True)

    return result_data


def gather_experiment_results(result_path):
    # Setup structure for results
    result_data = pd.DataFrame(
        {
            "model": [],
            "circuit": [],
            "circuit_param_count": [],
            "reduction_method": [],
            "reduction_size": [],
            "embedding_option": [],
            "embedding_class": [],
            "embedding_permutation": [],
            "target_levels_list": [],
            "target_levels": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "loss_train_history": [],
        }
    )
    # Load experiment config
    config_path = f"{result_path}/experiment_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    experiment_embeddings = filter_embedding_options(
        config["preprocessing"]["embedding_list"]
    )
    # Set circuits to run for the experiement
    experiment_circuits = {
        circ_name: CIRCUIT_OPTIONS[circ_name]
        for circ_name in config["model"]["circuit_list"]
    }
    for reduction_size, embedding_set in experiment_embeddings.items():
        for embedding_option in embedding_set:
            for circ_name, circ_param_count in experiment_circuits.items():
                for target_pair in config["data"]["target_pairs"]:
                    if "Hybrid" in embedding_option:
                        name_parts = embedding_option.split("-")
                        embedding_class = f"{name_parts[0]}-{name_parts[1]}"
                        embedding_permutation = name_parts[2]
                    else:
                        # Case when no hybrid encoding was done
                        embedding_class = embedding_option
                        embedding_permutation = 1
                    if type(target_pair[0] == int):
                        target_pair_str = f"{target_pair[0]}-{target_pair[1]}"
                    else:
                        target_pair_str = ("-".join(target_pair),)

                    model_name = f"{config['preprocessing'].get('reduction_method', 'pca')}-{reduction_size}-{config.get('type', 'quantum')}-{embedding_option}-{circ_name}-{target_pair_str}"

                    result_files = os.listdir(result_path)
                    # Check if file exist, if not experiment might still be running TODO warning
                    if f"{model_name}-param-history.csv" in result_files:
                        # TODO implement loss history data combination
                        cf_matrix = pd.read_csv(
                            f"{result_path}/{model_name}-confusion-matrix.csv",
                            index_col=0,
                        )

                        loss_train_history = pd.read_csv(
                            f"{result_path}/{model_name}-loss-train-history.csv"
                        )
                        loss_test_history = pd.read_csv(
                            f"{result_path}/{model_name}-loss-test-history.csv"
                        )
                        yhat_ytest = pd.read_csv(
                            f"{result_path}/{model_name}-yhat-class-vs-y-test.csv"
                        )
                        if loss_train_history.shape[1] > 2:
                            loss_train_history.drop(
                                loss_train_history.columns[0], inplace=True, axis=1
                            )
                            loss_test_history.drop(
                                loss_test_history.columns[0], inplace=True, axis=1
                            )

                        loss_train_history.columns = ["Iteration", "Train_Cost"]
                        loss_test_history.columns = ["Iteration", "Test_Cost"]
                        (
                            accuracy,
                            precision,
                            recall,
                            f1,
                            stats_text,
                        ) = confusion_matrix_stats(cf_matrix)
                        result = {
                            "model": model_name,
                            "circuit": circ_name,
                            "circuit_param_count": circ_param_count,
                            "reduction_method": config["preprocessing"].get(
                                "reduction_method", "pca"
                            ),
                            "reduction_size": reduction_size,
                            "embedding_option": embedding_option,
                            "embedding_class": embedding_class,
                            "embedding_permutation": embedding_permutation,
                            "target_levels_list": target_pair,
                            "target_levels": target_pair_str,
                            "y_hat": yhat_ytest[
                                "y_test"
                            ],  # TODO super misleading column names, wrong order
                            "y_test": yhat_ytest["yhat_class"],
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "loss_train_history": loss_train_history["Train_Cost"],
                            "loss_test_history": loss_test_history["Test_Cost"],
                        }
                        result_data = result_data.append(result, ignore_index=True)
    return result_data


# %% Testing
# from ast import literal_eval

# experiments_path = "../experiments"
# experiment_filename = "experiment_config.json"  # "experiment.txt"

# experiment_id = 98
# result_data = gather_experiment_results(f"{experiments_path}/{experiment_id}")

# config_path = f"{experiments_path}/{experiment_id}/experiment_config.json"
# with open(config_path, "r") as f:
#     config = json.load(f)

# # %%
# plot_loss(result_data, ["embedding_option", "circuit"], figsize=(30, 5))
# distinct_levels = list(
#     {item for combo in config["data"]["target_pairs"] for item in combo}
# )
# for level in distinct_levels:
#     plot_loss(result_data, ["circuit", "embedding_option", "target_levels"], [f"{level}"], figsize=(28, 5))

# plot_loss(result_data, "circuit", "target_levels", [0], figsize=(28, 5))
# %%
def get_multiclass_results_deprecated(path, config, prefix):

    y_class_multi = pd.read_csv(
        f"{path}/{config['ID']}/{prefix}-yclass-multi.csv", index_col=0
    ).squeeze()
    y_test = pd.read_csv(
        f"{path}/{config['ID']}/{prefix}-ytest.csv", index_col=0
    ).squeeze()
    distinct_levels = list(
        {item for combo in config["data"]["target_pairs"] for item in combo}
    )
    # print('\nClassification Report\n')
    display_report = classification_report(y_test, y_class_multi)
    confusion = confusion_matrix(y_test, y_class_multi)
    display_table = pd.DataFrame(
        confusion, columns=distinct_levels, index=distinct_levels
    )
    fig, axes = plt.subplots(1, 1, figsize=(30, 10))
    axes.grid(False)
    dsp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_class_multi, ax=axes, cmap=plt.cm.Blues
    )

    # display_table.loc[f"Truth Average"] = display_table.mean(axis=0)
    # display_table[f"{groupby[0]} Average"] = display_table.mean(axis=1)
    return dsp, display_report


def get_multiclass_results(y_test, y_class_multi, target_levels=None, model_name=None):

    display_report = classification_report(y_test, y_class_multi)
    confusion = confusion_matrix(y_test, y_class_multi, labels=target_levels)

    fig, axes = plt.subplots(1, 1, figsize=(30, 10))
    axes.grid(False)
    dsp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_class_multi, ax=axes, cmap=plt.cm.Blues
    )
    axes.set_title(model_name)
    axes.set_xlabel(display_report)
    # display_table.loc[f"Truth Average"] = display_table.mean(axis=0)
    # display_table[f"{groupby[0]} Average"] = display_table.mean(axis=1)
    return dsp, display_report


def get_experiment_config(path_experiment, exp_id):

    config_filename = "experiment.json" if exp_id >= 108 else "experiment_config.json"
    config_filename = config_filename if exp_id >= 12 else "experiment.txt"
    experiment_info = get_file_content(f"{path_experiment}/{exp_id}/{config_filename}")
    return experiment_info


def get_model_names(
    path_single_experiment, reference_filename="model_configuration.joblib"
):
    model_names = (
        filename.split(f"-{reference_filename}")[0]
        for filename in os.listdir(path_single_experiment)
        if f"-{reference_filename}" in filename
    )
    return model_names


def gather_results_0_12(
    exp_id, path_experiments=f"/home/matt/dev/projects/quantum-cnn/experiments"
):
    path_single_experiment = f"{path_experiments}/{exp_id}"
    model_names = get_model_names(path_single_experiment, "confusion-matrix.csv")

    result_data = pd.DataFrame(
        {
            "model": [],
            "circuit": [],
            "embedding_type": [],
            "selection_method": [],
            "selection_param_str": [],
            "target_levels": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "loss_train_history": [],
            "loss_test_history": [],
        }
    )

    for model_name in model_names:
        cf_matrix = pd.read_csv(
            f"{path_single_experiment}/{model_name}-confusion-matrix.csv", index_col=0
        )
        if f"{model_name}-loss-history.csv" in os.listdir(path_single_experiment):
            loss_train_history = pd.read_csv(
                f"{path_single_experiment}/{model_name}-loss-history.csv"
            )
            loss_test_history = pd.read_csv(
                f"{path_single_experiment}/{model_name}-loss-test-history.csv"
            )
            loss_train_history.columns = ["Iteration", "Train_Cost"]
            loss_test_history.columns = ["Iteration", "Test_Cost"]
        else:
            loss_train_history = pd.read_csv(
                f"{path_single_experiment}/{model_name}-loss-train-history.csv"
            )
            loss_test_history = pd.read_csv(
                f"{path_single_experiment}/{model_name}-loss-test-history.csv"
            )
            if loss_train_history.shape[1] > 2:
                loss_train_history.drop(
                    loss_train_history.columns[0], inplace=True, axis=1
                )
                loss_test_history.drop(
                    loss_test_history.columns[0], inplace=True, axis=1
                )

            loss_train_history.columns = ["Iteration", "Train_Cost"]
            loss_test_history.columns = ["Iteration", "Test_Cost"]
        (
            accuracy,
            precision,
            recall,
            f1,
            stats_text,
        ) = confusion_matrix_stats(cf_matrix)

        scaler_method = ""
        selection_method = model_name.split("-")[0]
        selection_param_str = model_name.split("-")[1]
        if int(selection_param_str) > 8:
            if model_name.split("-")[3] == "Compact":
                embedding_type = (
                    f"{model_name.split('-')[2]}_{model_name.split('-')[3]}"
                )
                circ_name = model_name.split("-")[4]
            else:
                embedding_type = f"{model_name.split('-')[2]}_{model_name.split('-')[3]}_{model_name.split('-')[4]}"
                circ_name = model_name.split("-")[5]
        else:
            embedding_type = model_name.split("-")[2]
            circ_name = model_name.split("-")[3]

        result = {
            "model": model_name,
            "circuit": circ_name,
            "embedding_type": embedding_type,
            "selection_method": selection_method,
            "selection_param_str": selection_param_str,
            "target_levels": "pop_classical",
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss_train_history": loss_train_history["Train_Cost"],
            "loss_test_history": loss_test_history["Test_Cost"],
        }
        result_data = result_data.append(result, ignore_index=True)
    return result_data.copy()


def gather_results_118_135(
    exp_id, path_experiments=f"/home/matt/dev/projects/quantum-cnn/experiments"
):
    path_single_experiment = f"{path_experiments}/{exp_id}"
    model_names = get_model_names(path_single_experiment)

    Results = namedtuple(
        "Results",
        [
            "model_name",
            "y_test_hat",
            "clf",
            "model_configuration",
            "samples_tfd",
            "pipeline",
        ],
    )
    result_list = []
    for model_name in model_names:
        y_test_hat = pd.read_csv(
            f"{path_single_experiment}/{model_name}-yhat.csv", index_col=0
        )
        clf = load(f"{path_single_experiment}/{model_name}-clf_results.joblib")
        model_configuration = load(
            f"{path_single_experiment}/{model_name}-model_configuration.joblib"
        )
        samples_tfd = load(f"{path_single_experiment}/{model_name}-samples_tfd.joblib")
        pipeline = load(f"{path_single_experiment}/{model_name}-pipeline.joblib")
        result_list = result_list + [
            Results(
                model_name,
                y_test_hat=y_test_hat,
                clf=clf,
                model_configuration=model_configuration,
                samples_tfd=samples_tfd,
                pipeline=pipeline,
            )
        ]

    result_data = pd.DataFrame(
        {
            "model_name": [],
            "model_type": [],
            "algorithm": [],
            "classification_type": [],
            "embedding_type": [],
            "scaler_method": [],
            "scaler_param_str": [],
            "selection_method": [],
            "selection_param_str": [],
            "target_pair": [],
            "additional_structure": [],
            "additional_structure_str": [],
            "wire_config": [],
            "wire_config_str": [],
            "target_pair_str": [],
            "mean_test_score": [],
            "std_test_score": [],
            "params": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "loss_train_history": [],
        }
    )
    for result in result_list:
        y_test_hat = result.y_test_hat
        clf = result.clf
        model_configuration = result.model_configuration
        samples_tfd = result.samples_tfd
        model_name = result.model_name

        precision, recall, fscore, support = precision_recall_fscore_support(
            samples_tfd.y_test, y_test_hat, average="binary"  # TODO multiclass
        )
        accuracy = accuracy_score(samples_tfd.y_test, y_test_hat)
        tmp_result = model_configuration._asdict()
        tmp_result["model_name"] = model_name

        if not (type(model_configuration.target_pair[0]) == str):
            # if target pair are ints
            tmp_result[
                "target_pair_str"
            ] = f"{model_configuration.target_pair[0]}_{model_configuration.target_pair[1]}"
        else:
            tmp_result["target_pair_str"] = "_".join(model_configuration.target_pair)
        if model_configuration.additional_structure:
            tmp_result[
                "additional_structure_str"
            ] = f"{model_configuration.additional_structure[0]}_{model_configuration.additional_structure[1]}_{model_configuration.additional_structure[2]}"
            tmp_result["wire_config"] = model_configuration.additional_structure[2]
            tmp_result["wire_config_str"] = "-".join(
                [
                    str(item)
                    for item in model_configuration.additional_structure[2].values()
                ]
            )
        tmp_result["mean_test_score"] = clf.cv_results_["mean_test_score"][
            clf.best_index_
        ]
        tmp_result["std_test_score"] = clf.cv_results_["std_test_score"][
            clf.best_index_
        ]
        tmp_result["params"] = clf.cv_results_["params"][clf.best_index_]

        tmp_result["accuracy"] = accuracy
        tmp_result["precision"] = precision
        tmp_result["recall"] = recall
        tmp_result["f1"] = fscore
        tmp_result["loss_train_history"] = None  # set for quantum
        result_data = result_data.append(tmp_result, ignore_index=True)

    return result_data.copy()


def gather_result_list_1000(
    exp_id, path_experiments=f"/home/matt/dev/projects/quantum-cnn/experiments"
):
    path_single_experiment = f"{path_experiments}/{exp_id}"
    model_names = get_model_names(path_single_experiment)

    Results = namedtuple(
        "Results",
        [
            "model_name",
            "y_test_hat",
            "clf",
            "model_configuration",
            "samples_tfd",
            "pipeline",
        ],
    )
    result_list = []
    for model_name in model_names:
        y_test_hat = pd.read_csv(
            f"{path_single_experiment}/{model_name}-yhat.csv", index_col=0
        )
        clf = load(f"{path_single_experiment}/{model_name}-clf_results.joblib")
        model_configuration = load(
            f"{path_single_experiment}/{model_name}-model_configuration.joblib"
        )
        samples_tfd = load(f"{path_single_experiment}/{model_name}-samples_tfd.joblib")
        pipeline = load(f"{path_single_experiment}/{model_name}-pipeline.joblib")
        result_list = result_list + [
            Results(
                model_name,
                y_test_hat=y_test_hat,
                clf=clf,
                model_configuration=model_configuration,
                samples_tfd=samples_tfd,
                pipeline=pipeline,
            )
        ]
    return result_list


def gather_resultdf_1000(result_list):
    result_data = pd.DataFrame(
        {
            "model_name": [],
            "model_type": [],
            "algorithm": [],
            "classification_type": [],
            "embedding_type": [],
            "scaler_method": [],
            "scaler_param_str": [],
            "selection_method": [],
            "selection_param_str": [],
            "target_pair": [],
            "additional_structure": [],
            "additional_structure_str": [],
            "wire_config": [],
            "wire_config_str": [],
            "target_pair_str": [],
            "mean_test_score": [],
            "std_test_score": [],
            "params": [],
            "loss_train_history": [],
        }
    )
    for result in result_list:
        y_test_hat = result.y_test_hat
        clf = result.clf
        model_configuration = result.model_configuration
        samples_tfd = result.samples_tfd
        model_name = result.model_name        
        cf_matrix = get_cf_matrix(samples_tfd.y_test,y_test_hat)
        accuracy, TPR, TNR, precision, f1_score, gmean, informedness, fpr = get_confusion_matrix_stats(cf_matrix=cf_matrix)
        tmp_result = model_configuration._asdict()
        tmp_result["model_name"] = model_name

        if not (type(model_configuration.target_pair[0]) == str):
            # if target pair are ints
            tmp_result[
                "target_pair_str"
            ] = f"{model_configuration.target_pair[0]}_{model_configuration.target_pair[1]}"
        else:
            tmp_result["target_pair_str"] = "_".join(model_configuration.target_pair)
        if model_configuration.additional_structure:
            tmp_result[
                "additional_structure_str"
            ] = f"{model_configuration.additional_structure[0]}_{model_configuration.additional_structure[1]}_{model_configuration.additional_structure[2]}"
            tmp_result["wire_config"] = model_configuration.additional_structure[2]
            tmp_result["wire_config_str"] = "-".join(
                [
                    str(item)
                    for item in model_configuration.additional_structure[2].values()
                ]
            )
        # tmp_result["mean_test_score"] = clf.cv_results_["mean_test_score"][
        #     clf.best_index_
        # ]
        # tmp_result["std_test_score"] = clf.cv_results_["std_test_score"][
        #     clf.best_index_
        # ]
        # tmp_result["params"] = clf.cv_results_["params"][clf.best_index_]

        tmp_result["accuracy"] = accuracy
        tmp_result["TPR"] = TPR
        tmp_result["TNR"] = TNR
        tmp_result["precision"] = precision
        tmp_result["f1_score"] = f1_score
        tmp_result["gmean"] = gmean
        tmp_result["informedness"] = informedness
        tmp_result["fpr"] = fpr
        tmp_result["loss_train_history"] = None  # set for quantum
        result_data = result_data.append(tmp_result, ignore_index=True)

    return result_data.copy()


def get_cf_matrix(y_true, y_hat):
    """Get confusion matrix

    Args:
        y_true (np.array(int(0,1))): true labels as 0 or 1
        y_hat (np.array(int(0,1))): estimated labels as 0 or 1

    Returns:
        np.array(2*2): 2 by 2 matrix containing TN,FP,
                                                FN,TP
    """
    TP = np.sum(np.array(y_hat)[y_true == 1] == 1)
    TN = np.sum(np.array(y_hat)[y_true == 0] == 0)
    FP = np.sum(np.array(y_hat)[y_true == 0] == 1)
    FN = np.sum(np.array(y_hat)[y_true == 1] == 0)
    return np.array([[TN, FP], [FN, TP]])


def get_confusion_matrix_stats(cf_matrix):
    """Generate typical confusion matrix stats

    Args:
        cf_matrix (np.array(2 int * 2 int)): Confusion matrix as 2 dimensional numpy array, consisting of counts    

    Returns:
        tuple(floats): accuracy, TPR, TNR, precision, f1_score, gmean, informedness
    """
    TN, FP, FN, TP = cf_matrix[0, 0], cf_matrix[0, 1], cf_matrix[1, 0], cf_matrix[1, 1]
    accuracy = (TP + TN) / (TP + FP + FN + TN)  # accuracy
    precision = TP / (
        TP + FP
    )  # correctly classified positives out of all classified positives
    TPR = TP / (
        TP + FN
    )  # correctly classified positives out of all actual positives aslo called TPR, sensitivit, recall
    TNR = TN / (
        TN + FP
    )  # specificity, correctly classified negatives out of all negatives
    f1_score = (
        2 * precision * TPR / (precision + TPR)
    )  # measure of accuracy that considers both precision and recall
    fpr = FP / (
        FP + TN
    )  # Fraction of incorrectly classified positives out of all actual negatives (0's).
    gmean = np.sqrt(
        TPR * TNR
    )  # Imbalanced data metric describic ratio between positive and negative accuracy i.e
    # recall * TNR

    informedness = TPR + TNR - 1
    return accuracy, TPR, TNR, precision, f1_score, gmean, informedness, fpr


def gather_resultdf_991000(result_list):
    result_data = pd.DataFrame(
        {
            "model_name": [],
            "model_type": [],
            "algorithm": [],
            "classification_type": [],
            "embedding_type": [],
            "scaler_method": [],
            "scaler_param_str": [],
            "selection_method": [],
            "selection_param_str": [],
            "target_pair": [],
            "additional_structure": [],
            "additional_structure_str": [],
            "wire_config": [],
            "wire_config_str": [],
            "target_pair_str": [],
            "mean_test_score": [],
            "std_test_score": [],
            "params": [],
            "loss_train_history": [],
        }
    )
    for result in result_list:
        y_test_hat = result.y_test_hat
        clf = result.clf
        model_configuration = result.model_configuration
        samples_tfd = result.samples_tfd
        model_name = result.model_name
        cf_matrix = get_cf_matrix(samples_tfd.y_test,y_test_hat)
        accuracy, TPR, TNR, precision, f1_score, gmean, informedness, fpr = get_confusion_matrix_stats(cf_matrix=cf_matrix)

        tmp_result = model_configuration._asdict()
        tmp_result["c_step"]=result.model_configuration.additional_structure[2]["c_step"]
        tmp_result["n_wires"]=result.model_configuration.additional_structure[2]["n_wires"]
        tmp_result["p_step"]=result.model_configuration.additional_structure[2]["p_step"]
        tmp_result["pool_pattern"]=result.model_configuration.additional_structure[2]["pool_pattern"]
        tmp_result["wire_to_cut"]=result.model_configuration.additional_structure[2]["wire_to_cut"]

        if not (type(model_configuration.target_pair[0]) == str):
            # if target pair are ints
            tmp_result[
                "target_pair_str"
            ] = f"{model_configuration.target_pair[0]}_{model_configuration.target_pair[1]}"
        else:
            tmp_result["target_pair_str"] = "_".join(model_configuration.target_pair)
        if model_configuration.additional_structure:
            tmp_result[
                "additional_structure_str"
            ] = f"{model_configuration.additional_structure[0]}_{model_configuration.additional_structure[1]}_{model_configuration.additional_structure[2]}"
            tmp_result["wire_config"] = model_configuration.additional_structure[2]
            tmp_result["wire_config_str"] = "-".join(
                [
                    str(item)
                    for item in model_configuration.additional_structure[2].values()
                ]
            )
        tmp_result["mean_test_score"] = clf.cv_results_["mean_test_score"][
            clf.best_index_
        ]
        tmp_result["std_test_score"] = clf.cv_results_["std_test_score"][
            clf.best_index_
        ]
        tmp_result["params"] = clf.cv_results_["params"][clf.best_index_]

        tmp_result["accuracy"] = accuracy
        tmp_result["TPR"] = TPR
        tmp_result["TNR"] = TNR
        tmp_result["precision"] = precision
        tmp_result["f1_score"] = f1_score
        tmp_result["gmean"] = gmean
        tmp_result["informedness"] = informedness
        tmp_result["fpr"] = fpr

        tmp_result["loss_train_history"] = None  # set for quantum
        result_data = result_data.append(tmp_result, ignore_index=True)

    return result_data.copy()


def get_circuit_diagram(
    wire_combos, n_qbits=8, conv_color="0096ff", pool_color="ff7e79"
):
    qr = QuantumRegister(n_qbits, "q")
    q_circuit = QuantumCircuit(qr)

    disp_color = {}
    for layer, wires in wire_combos.items():
        if layer.split("_")[0].upper() == "P":
            disp_color[layer] = "#ff7e79"
        else:
            disp_color[layer] = "#0096ff"
        for wire_connection in wires:
            q_circuit.append(
                Gate(name=layer, num_qubits=2, params=[]),
                (qr[wire_connection[0]], qr[wire_connection[1]]),
            )
            q_circuit.barrier()

    return q_circuit.draw(
        output="mpl",
        plot_barriers=False,
        justify="none",
        style={"displaycolor": disp_color, "linecolor": "#000000"},
    )


def get_wire_combos_graph(
    wire_combos, n_qbits=8, conv_color="#0096ff", pool_color="#ff7e79"
):

    # labels = nx.draw_networkx_labels(graph, pos=pos)
    # nodes=nx.draw_networkx_nodes(graph,pos=pos, node_color="#ffffff")
    node_sizes = [1000 for ind in range(n_qbits)]
    n_graphs = {}
    for layer in wire_combos.keys():

        graph = nx.DiGraph()
        graph.add_nodes_from(range(n_qbits))
        graph.add_edges_from(wire_combos[layer])

        # Change order around a circle, this way you start at x=0 then move left around
        theta_0 = 2 / n_qbits
        theta_step = 1 / n_qbits
        pos = {
            ind: np.array(
                [
                    np.cos(2 * np.pi * (theta_0 + ind * theta_step)),
                    np.sin(2 * np.pi * (theta_0 + ind * theta_step)),
                ]
            )
            for ind in range(n_qbits)
        }
        if layer.split("_")[0].upper() == "P":
            node_color = pool_color
            # in the get_wire_combos function we add cut_wires at index 0, if that changes
            # this should update TODO
            cut_wires = [x[0] for x in wire_combos[layer]]
            node_sizes = [
                200 if (ind in cut_wires) else node_size
                for ind, node_size in zip(range(n_qbits), node_sizes)
            ]
        else:
            node_color = conv_color

        # cut_wires = [x[wire_to_cut] for x in wire_combos[layer]]
        # node_sizes = [100 if (ind in cut_wires) else 1000 for ind in range(n_qbits)]
        n_graphs[layer] = (graph, pos, node_sizes, node_color)

    return n_graphs


def plot_122_scatter(result_data, figsize=(10, 10)):
    groupby = ["additional_structure_str", "selection_method"]
    metric = "accuracy"
    filter_struc = lambda data, ind, val: data.apply(
        lambda row: row["additional_structure"][ind] == val, axis=1
    )
    filter_col_val = lambda data, col, val: data.apply(
        lambda row: row[col] == val, axis=1
    )
    U_5_rockreg = result_data[
        filter_col_val(result_data, "target_pair_str", "rock_reggae")
        & filter_struc(result_data, 0, "U_5")
    ].copy()
    # U_5_classpop = result_data[filter_col_val(result_data, "target_pair_str", "classical_pop") & filter_struc(result_data,0,"U_5")].copy()
    # U_SU4_rockreg = result_data[filter_col_val(result_data, "target_pair_str", "rock_reggae") & filter_struc(result_data,0,"U_SU4")].copy()
    # U_SU4_classpop = result_data[filter_col_val(result_data, "target_pair_str", "classical_pop") & filter_struc(result_data,0,"U_SU4")].copy()
    plot_U_5_rockreg = get_line_plot_data(U_5_rockreg, groupby, metric)
    # plot_U_5_classpop=get_line_plot_data(U_5_classpop, groupby, metric)
    # plot_U_SU4_rockreg=get_line_plot_data(U_SU4_rockreg, groupby, metric)
    # plot_U_SU4_classpop=get_line_plot_data(U_SU4_classpop, groupby, metric)

    plot_data = plot_U_5_rockreg
    plot_data["additional_structure_str"] = plot_data.apply(
        lambda row: "_".join(row["additional_structure_str"].split("_")[3:]), axis=1
    )
    plot_data.index = plot_data["additional_structure_str"]
    # plot_U_5_rockreg = plot_U_5_rockreg.drop("additional_structure_str", axis=0)
    # plot_U_SU4_classpop["wire_pattern"]=plot_U_5_classpop.apply(lambda row: "_".join(row["additional_structure_str"].split("_")[3:]), axis=1)
    # plot_U_SU4_rockreg["wire_pattern"]=plot_U_SU4_rockreg.apply(lambda row: "_".join(row["additional_structure_str"].split("_")[3:]), axis=1)
    # plot_U_SU4_classpop["wire_pattern"]=plot_U_SU4_classpop.apply(lambda row: "_".join(row["additional_structure_str"].split("_")[3:]), axis=1)
    markers = {"pca": "P", "tree": "v"}
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 1, figsize=figsize, sharey=True)
        axes.set_title("U_5, Rock vs Reggae")
        axes.set_xlabel("Wire Pattern")
        axes = sns.scatterplot(
            data=pd.melt(
                plot_data,
                "additional_structure_str",
                value_name="Accuracy",
                var_name="selection_method",
            ),
            x="additional_structure_str",
            y="Accuracy",
            hue="selection_method",
            markers=markers,
            style="selection_method",
            # palette=["#4c72b0","#dd8452"],
            s=100,
            # marker="o",
        )

        plt.xticks(rotation=90)
        axes.set(ylim=(0, 1))
        # axes.set_aspect('equal', adjustable='box')
        # 24 is the number of wire patterns, this helps make the plot square
        axes.yaxis.set_major_locator(ticker.MultipleLocator(1 / 24))


def generic_plot_201(
    plot_data,
    x,
    y,
    hue,
    figsize=(10, 10),
    title="U_5, Rock vs Reggae",
    x_label="Wire Pattern",
):
    markers = {"pca": "P", "tree": "v"}
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 1, figsize=figsize, sharey=True)
        axes.set_title(title)
        axes.set_xlabel(x_label)
        axes = sns.scatterplot(
            data=pd.melt(
                plot_data,
                x,
                value_name=y,
                var_name=hue,
            ),
            x=x,
            y=y,
            hue=hue,
            markers=markers,
            style=hue,
            # palette=["#4c72b0","#dd8452"],
            s=100,
            # marker="o",
        )

        plt.xticks(rotation=90)
        axes.set(ylim=(0, 1))
        # axes.set_aspect('equal', adjustable='box')
        # 24 is the number of wire patterns, this helps make the plot square
        axes.yaxis.set_major_locator(ticker.MultipleLocator(1 / plot_data.shape[0]))
        return fig


# %%
def plot_binary_pca_model(model_result, figsize=(16, 8)):
    selection_method = model_result.model_configuration.selection_method
    target_pair = model_result.model_configuration.target_pair
    var_exp = model_result.pipeline.named_steps[
        selection_method
    ].explained_variance_ratio_
    cum_var_exp = var_exp.cumsum()

    feature_names = [
        f"{selection_method}-{i}"
        for i in range(model_result.samples_tfd.X_test.shape[1])
    ] + ["genre"]
    plot_data = pd.DataFrame(
        np.c_[model_result.samples_tfd.X_test, model_result.samples_tfd.y_test],
        columns=feature_names,
    )

    plot_data["genre"] = plot_data.apply(
        lambda row: target_pair[1] if row["genre"] == 1 else target_pair[0], axis=1
    )
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].set_title("Principal Component Analysis - First 2 components")
        sns.scatterplot(
            ax=axes[0],
            data=plot_data,
            x="pca-0",
            y="pca-1",
            hue="genre",
            # markers=markers,
            # style="selection_method",
            # palette=["#4c72b0","#dd8452"],
            s=100,
            # marker="o",
        )
        axes[1].set_title("Cumulative Explained Variance")
        axes[1].bar(
            range(len(var_exp)),
            var_exp,
            alpha=0.5,
            align="center",
            label="individual explained variance",
        )
        axes[1].step(
            range(len(var_exp)),
            cum_var_exp,
            where="mid",
            label="cumulative explained variance",
        )
        axes[1].set_ylabel("Explained variance ratio")
        axes[1].set_xlabel("Principal components")
        return fig


def plot_binary_pair(model_result, X_test_columns):
    selection_method = model_result.model_configuration.selection_method
    target_pair = model_result.model_configuration.target_pair
    support_mask = model_result.pipeline.named_steps[
        selection_method
    ]._get_support_mask()
    feature_names = X_test_columns[support_mask]
    feature_names = list(feature_names) + ["genre"]
    plot_data = pd.DataFrame(
        np.c_[model_result.samples_tfd.X_test, model_result.samples_tfd.y_test],
        columns=feature_names,
    )
    plot_data["genre"] = plot_data.apply(
        lambda row: target_pair[1] if row["genre"] == 1 else target_pair[0], axis=1
    )
    sns.set_theme(style="ticks")
    with sns.axes_style("whitegrid"):
        plt = sns.pairplot(plot_data, hue="genre")
        # plt.fig.suptitle(title)

    return plt


# %%
def get_file_content(file_path):
    """
    Returns the text contained in the text file provided as a path, ex:
    file_content = get_file_content("../experiments/13/experiment.txt")
    Args:
        info_path (str): path to text file

    Returns:
        str:  contents of file
    """
    with open(file_path, "r") as f:
        if "json" in file_path:  # TODO Can do a much better test than this
            file_contents = json.load(f)
        else:
            file_contents = f.read()

    return file_contents


# experiment_info = get_file_content(
#     f"{experiments_path}/{experiment_id}/{experiment_filename}"
# )
# get_file_content("../experiments/13/experiment.txt")
# %%
