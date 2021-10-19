# %%
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from ast import literal_eval

from circuit_presets import (
    filter_embedding_options,
    EMBEDDING_OPTIONS,
    CIRCUIT_OPTIONS,
    POOLING_OPTIONS,
)


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
    all_combos = [pair.split("-") for pair in grouped_data.index]
    distinct_levels = {item for combo in all_combos for item in combo}
    display_table = pd.DataFrame(columns = distinct_levels, index=distinct_levels)
    for index, row in grouped_data.iterrows():
        target_pair = index.split("-")
        display_table.loc[target_pair[0], target_pair[1]] = row[0]
        display_table.loc[target_pair[1], target_pair[0]] = row[0]
        
    display_table.loc[f"Average"] = display_table.mean(axis=0)
    display_table[f"Average"] = display_table.mean(axis=1)
    return display_table

def plot_loss(data, each_var, group_var, group_filter=[], figsize=(30, 5)):
    grouped_data_train = data.groupby([each_var, group_var])["loss_train_history"].max()
    loss_history_train = grouped_data_train.copy()
    loss_history_train = loss_history_train.unstack(level=0)

    grouped_data_test = data.groupby([each_var, group_var])["loss_test_history"].max()
    loss_history_test = grouped_data_test.copy()
    loss_history_test = loss_history_test.unstack(level=0)

    sns.set(font_scale=1.2)
    for col in loss_history_train.columns:
        plot_data_train = pd.DataFrame()
        plot_data_test = pd.DataFrame()
        for index, row in loss_history_train.iterrows():
            if type(row[col]) == np.ndarray and check_filter_on_list(
                group_filter, index.split("-")
            ):
                plot_data_train[index] = row[col]
        plot_data_train["Iteration"] = plot_data_train.index

        plot_data_test = pd.DataFrame()
        for index, row in loss_history_test.iterrows():
            if type(row[col]) == np.ndarray and check_filter_on_list(
                group_filter, index.split("-")
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
                        var_name=group_var,
                    ),
                    x="Iteration",
                    y="MSE Cost",
                    hue=group_var,
                    markers=True,
                    dashes=False,
                )
                axes[0].set_title(f"{col} Train Cost Per Iteration")

                sns.lineplot(
                    ax=axes[1],
                    data=pd.melt(
                        plot_data_test,
                        "Iteration",
                        value_name="MSE Cost",
                        var_name=group_var,
                    ),
                    x="Iteration",
                    y="MSE Cost",
                    hue=group_var,
                    markers=True,
                    dashes=False,
                )
                axes[1].set_title(f"{col} Test Cost Per Iteration")


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
            "target_levels":[],
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
                    model_name = f"{config['preprocessing'].get('reduction_method', 'pca')}-{reduction_size}-{config.get('type', 'quantum')}-{embedding_option}-{circ_name}-{'-'.join(target_pair)}"
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
                            "target_levels":'-'.join(target_pair),
                            "y_hat":yhat_ytest["y_test"], # TODO super misleading column names, wrong order
                            "y_test":yhat_ytest["yhat_class"],
                            "accuracy": accuracy,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "loss_train_history": loss_train_history["Train_Cost"],
                            "loss_test_history": loss_test_history["Test_Cost"],
                        }
                        result_data = result_data.append(result, ignore_index=True)
    return result_data

# %%
# Testing
from ast import literal_eval
experiments_path = "../experiments"
experiment_filename = "experiment_config.json" #"experiment.txt"

experiment_id = 23
result_data = gather_experiment_results(f"{experiments_path}/{experiment_id}")

config_path = f"{experiments_path}/{experiment_id}/experiment_config.json"
with open(config_path, "r") as f:
    config = json.load(f)
# %%
distinct_levels = {item for combo in config["data"]["target_pairs"] for item in combo}
for level in distinct_levels:
    level_idx = result_data["target_levels"].str.contains(level)
    y_hats  = result_data.loc[level_idx,"y_hat"]
    level_list  = result_data.loc[level_idx,"target_levels_list"]
    predicing_index = [item.index(level) for item in level_list]
    y_hats.reset_index()
    
    for idx in y_hats.index:
        [literal_eval(prediction.replace(" ", ","))[predicing_index[idx]] for prediction in y_hats[idx]]
for x in result_data["target_levels"]:
    print(x)
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
    f = open(file_path, "r")
    file_contents = f.read()
    f.close()
    return file_contents


# get_file_content("../experiments/13/experiment.txt")
# %%
