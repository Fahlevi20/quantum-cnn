import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix_stats(cf_matrix):
    cf = np.array(cf_matrix)
    accuracy = np.trace(cf) / float(np.sum(cf))
    precision = cf[1, 1] / sum(cf[:, 1])
    recall = cf[1, 1] / sum(cf[1, :])
    f1_score = 2 * precision * recall / (precision + recall)
    stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
        accuracy, precision, recall, f1_score
    )
    return stats_text


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
