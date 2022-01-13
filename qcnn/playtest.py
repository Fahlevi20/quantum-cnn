# %%
import itertools as it
from math import log2
import re

target_levels = [
    "classical",
    "country",
    "rock",
    "pop",
    "hiphop",
    "jazz",
    "blues",
    "disco",
    "metal",
    "reggae",
]

target_pairs = [target_pair for target_pair in it.combinations(target_levels, 2)]
# %%
# Quicksort
def f(x):
    if len(x) <= 1:
        return x
    else:
        y = [a for a in x if a < x[0]]
        z = [a for a in x if a > x[0]]
        return f(y) + [x[0]] + f(z)


# %%
f([2, 1, 3, 10, 15])


# %%
import numpy as np

# can easily be extented to n_qbits=8
n_wires = 8
# During pooling layer apply measurement on index 1 (2nd wire of connected pair)
wire_to_cut = 1
step = 1
pool_pattern = "left"

if pool_pattern == "left":
    # 0 1 2 3 4 5 6 7
    # x x x x
    pool_filter = lambda arr: arr[0 : len(arr) // 2 : 1]  # Left
elif pool_pattern == "right":
    # 0 1 2 3 4 5 6 7
    #         x x x x
    pool_filter = lambda arr: arr[len(arr) : len(arr) // 2 - 1 : -1]  # Right
elif pool_pattern == "eo_even":
    # 0 1 2 3 4 5 6 7
    # x   x   x   x
    pool_filter = lambda arr: arr[0::2]  # eo even
elif pool_pattern == "eo_odd":
    # 0 1 2 3 4 5 6 7
    #   x   x   x   x
    pool_filter = lambda arr: arr[1::2]  # eo odd
elif pool_pattern == "inside":
    # 0 1 2 3 4 5 6 7
    #     x x x x
    pool_filter = lambda arr: arr[
        len(arr) // 2 - len(arr) // 4 : len(arr) // 2 + len(arr) // 4 : 1
    ]  # inside
elif pool_pattern == "outside":
    # 0 1 2 3 4 5 6 7
    # x x         x x
    pool_filter = lambda arr: [
        item
        for item in arr
        if not (
            item
            in arr[len(arr) // 2 - len(arr) // 4 : len(arr) // 2 + len(arr) // 4 : 1]
        )
    ]  # outside

# setup
from math import log2

# %%


def get_wire_combos(n_wires, step, pool_filter, wire_to_cut=1):
    wire_combos = {}
    wires = range(n_wires)
    for layer_ind, i in zip(
        range(int(log2(n_wires))), range(int(log2(n_wires)), 0, -1)
    ):
        conv_size = 2 ** i
        circle_n = lambda x: x % conv_size
        wire_combos[f"c_{layer_ind+1}"] = [
            (wires[x], wires[circle_n(x + step)]) for x in range(conv_size)
        ]
        if (i == 1) and (len(wire_combos[f"c_{layer_ind+1}"]) > 1):
            wire_combos[f"c_{layer_ind+1}"] = [wire_combos[f"c_{layer_ind+1}"][0]]

        wire_combos[f"p_{layer_ind+1}"] = pool_filter(wire_combos[f"c_{layer_ind+1}"])
        if len(wire_combos[f"p_{layer_ind+1}"]) == 0:
            wire_combos[f"p_{layer_ind+1}"] = [wire_combos[f"c_{layer_ind+1}"][0]]
        # for next iteration
        cut_wires = [x[wire_to_cut] for x in wire_combos[f"p_{layer_ind+1}"]]
        wires = [wire for wire in wires if not (wire in cut_wires)]
    return wire_combos


# circle_n = lambda x: x % n_wires
# c_1 = [(wires[x], wires[circle_n(x + step)]) for x in range(n_wires)]
# p_1 = c_1[0:4:1]
# # TODO can paramaterize
# cut_wires = [x[wire_to_cut] for x in p_1]


# wires = [wire for wire in wires if not (wire in cut_wires)]
# n_wires = len(wires)
# circle_n = lambda x: x % n_wires
# c_2 = [(wires[x], wires[circle_n(x + step)]) for x in range(n_wires)]
# p_2 = c_2[0:2:1]
# cut_wires = [x[wire_to_cut] for x in p_2]


# wires = [wire for wire in wires if not (wire in cut_wires)]
# n_wires = len(wires)
# circle_n = lambda x: x % n_wires
# c_3 = [(wires[x], wires[circle_n(x + step)]) for x in range(n_wires)]
# p_3 = c_3[0:1:1]


# %%
import os
import sys
import inspect
import numpy as np

import pandas as pd
from pprint import pprint
from reporting_functions import (
    get_file_content,
    get_result_table,
    get_result_table_target_pairs,
)
from joblib import dump, load
from collections import namedtuple


def get_model_result_list(experiment_config):
    path = f"{experiment_config.get('path')}/{experiment_config.get('ID')}"
    X_test = pd.read_csv(f"{path}/X_test.csv", index_col=0)
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
        pipeline = load(f"{path}/{model_name}-pipeline.joblib")
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


# %%
exp_id_list = [137]
result_dict = {}
for exp_id in exp_id_list:
    experiment_config = get_file_content(
        f"/home/matt/dev/projects/quantum-cnn/experiments/{exp_id}/experiment.json"
    )
    result_dict[exp_id] = get_model_result_list(experiment_config)
# %%
all_dict = {}
data = pd.DataFrame()
for exp_id, result in result_dict.items():
    tmp_dict = result[0].clf.best_estimator_.train_history_.copy()
    tmp_dict["exp_id"] = [exp_id] * len(tmp_dict["Iteration"])
    data = pd.concat([data, pd.DataFrame(tmp_dict)])
# filtered_results = [
#     result
#     for result in result_list
#     if result.model_configuration.additional_structure == 'custom_1'
# ]
# tmp_result = [result for result in result_list if result.model_name=="quantum-qcnn-binary-Angle-minmax-max_features=8_n_estimators=50-tree-feature_range=[0, 1.5707963267948966]-['rock', 'reggae']-('U_5', 'psatz1', [8, 1, 'outside'])"]


# %%


# %%
import matplotlib.pyplot as plt
import seaborn as sns

figsize = (30, 10)
plot_data = data.pivot("Iteration", "exp_id", "Cost")
# data = a.clf.best_estimator_.train_history_
with sns.axes_style("whitegrid"):
    fig, axes = plt.subplots(1, 1, figsize=figsize, sharey=True)
    sns.lineplot(
        ax=axes,
        data=plot_data,
        markers=True,
        dashes=False,
    )
    axes.set_title(f"Train Cost Per Iteration")


# array([-0.8038882 ,  0.38266347,  0.26455017, -0.37964004,  0.20714452,
#        -0.23700476,  0.03584148, -0.02753467, -1.18193767,  0.68775315,
#         0.57794982,  0.67696884,  0.20787334, -0.94388897, -0.19604747,
#         1.34532937, -0.4730406 ,  0.21910568, -1.46761429, -0.60966675,
#        -0.36326917, -1.21621954, -0.71328942,  0.19213284, -0.92900281,
#         1.66766682,  1.07659751,  0.16031539, -2.12708616,  1.74639568,
#        -0.92856249,  0.96872548, -1.59363407,  1.63219035, -0.97979345,
#         0.39311107])


# array([ 0.18223275, -0.1662073 , -0.23400546,  0.64319198, -0.66900911,
#        -0.87290196,  1.23428592, -0.55508484,  0.35165046, -0.20581643,
#         0.7524456 , -0.07387914,  0.0878729 ,  0.74592785, -1.84711082,
#         0.17036005, -0.70692405, -0.20232304,  0.38564801, -0.28972861,
#         0.89350956, -0.9022085 ,  0.55772628, -0.25817608, -0.40217651,
#        -1.12050764,  0.04375302, -0.84203446, -0.76195598,  0.66834117,
#        -0.19069866,  1.42414273,  1.11158516,  1.19345173, -1.53246746,
#        -1.02719161])

# self.coef_ = np.array(
#             [
#                 0.1193346,
#                 -1.29857603,
#                 0.18556851,
#                 0.33382759,
#                 -0.25776149,
#                 0.25705179,
#                 1.31814871,
#                 -0.35030913,
#                 -0.36404661,
#                 0.41125868,
#                 1.02868727,
#                 1.32078417,
#                 1.18355149,
#                 -0.63473673,
#                 0.44277276,
#                 0.50545649,
#                 -0.43902124,
#                 0.49003917,
#                 -0.87458557,
#                 -1.21007646,
#                 0.64970235,
#                 -1.07927684,
#                 -0.22957657,
#                 -0.75653132,
#                 0.32538887,
#                 0.03323765,
#                 -0.28561954,
#                 -1.69030752,
#                 1.89342474,
#                 0.29929749,
#                 -0.08119952,
#                 -1.34889213,
#                 -0.88747166,
#                 -0.96870005,
#                 0.20376645,
#                 -1.42554811,
#             ]
#         )
# %%
from sklearn.datasets import load_iris
from simple_estimator import Simple_Classifier
import numpy as np

X, y = load_iris(return_X_y=True)

index_01 = np.where(y != 2)

X = X[index_01]
y = y[index_01]
# %%
clf = Simple_Classifier().fit(X, y)
clf.predict(X[:2, :])

clf.predict_proba(X[:2, :])


clf.score(X, y)
# %%
"""
Gather and display  data encoding circuit results...
First result

exp:0=combinations encoding + all circuits + 1 genre pair choice of embedding
exp:1 - 8 =scaling, getting to minmax range
exp:4 scaling circuit, show angle does good
exp:9 all circuits hiphop-reggae, minmax(0, np.pi)
exp:10 all circuits disco-jazz, minmax(0, np.pi)
exp:11 all circuits rock-metal, minmax(0, np.pi)
exp:12-23 introducing config, testing and closer to 20 is minmaxscaling tests
exp:23=all genres, fixed U_5, Angle and scaling
exp:24-53=a lot of testing, introducing havlicek and different scaling
exp:53=all target pairs, default rest 60 iterations
exp:54=all target pairs, default rest 100 iterations
exp:90,91,92 encodings vary depth   20211127
exp:93=encodings vary scale         20211127
exp:94=1000 encodings iterations    20211127
exp:98,99=encodings vary scale      20211127
exp:97,100=ova encodings            20211127
exp:101=encodings 1000 iterations, vary circuits, 2 genres,
exp:102=ova
exp:103=ova
exp:104=pca encodings genres, circuits      20211102
exp:105=tree encodings genres, circuits     20211102
preprocessing experiment                    20211102
exp:107=image mnist data large experiment
exp:108=large classical
exp:109=ova
exp:110=ovo
exp:111=binary classical logreg
exp:112=ova
exp:114=classical, logreg svm all genres
exp:116=classical ovo
exp:117=classical ova
exp:118=quant+classical, binary structure
exp:119=quantum structure, one layer a time
exp:122=quantum large structure experiment      20211211
exp:124=[8,1,outside] all genre                 20211211
exp:[136,135,134,133,132] quantum fixed features correctly  20211211
exp:[139,140,141,142,143] classical fixed features correctly  20211211

exp:139=simple classical self built
"""


from reporting_functions import (
    get_file_content,
    confusion_matrix_stats,
    get_result_table_target_pairs,
    get_result_table,
    plot_loss,
    check_filter_on_list,
    gather_results_0_12,
)


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


def get_line_plot_data(data, groupby, metric):
    grouped_data = data.groupby(groupby)[metric].max()
    display(grouped_data)
    grouped_data_unstack = grouped_data.copy().unstack(level=-1)
    grouped_data_unstack[grouped_data_unstack.index.name] = grouped_data_unstack.index
    return grouped_data_unstack.copy()


# %%
from collections import namedtuple
import pandas as pd
from IPython.display import display
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from joblib import dump, load
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

path_experiments = f"/home/matt/dev/projects/quantum-cnn/experiments"
# %%
exp_id = 10
result_data = gather_results_0_12(exp_id, path_experiments=path_experiments)
display(plot_loss(result_data, ["embedding_type", "circuit"], figsize=(28, 5)))

# %%
"""Experiment 119
"""
path_experiments = f"/home/matt/dev/projects/quantum-cnn/experiments"
exp_id = 119
result_data = gather_results_118_135(exp_id)

display(get_experiment_config(path_experiments, exp_id))
display(
    get_result_table(
        result_data,
        ["algorithm", "additional_structure", "selection_method", "target_pair_str"],
        "accuracy",
    )
)
# display(plot_loss(result_data, ["embedding_type", "circuit"], group_filter=["U_5"], figsize=(28, 5)))
# %%
tmp_result_table = get_result_table(
    result_data,
    ["algorithm", "additional_structure", "selection_method", "target_pair_str"],
    "accuracy",
)

# %%
)




groupby = ["additional_structure", "selection_method"]
metric = "accuracy"
data_0 = result_data[result_data["target_pair_str"] == "rock_reggae"].copy()
data_1 = result_data[result_data["target_pair_str"] == "classical_pop"].copy()
plot_data_0 = get_line_plot_data(data_0, groupby, metric)
plot_data_1 = get_line_plot_data(data_1, groupby, metric)
# sns.set(font_scale=1.2)
with sns.axes_style("whitegrid"):
    fig, axes = plt.subplots(1, 2, figsize=(6, 6), sharey=True)
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


# %%
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
    axes[0].set_title(f"{col}-{'-'.join(group_filter)} Train Cost Per Iteration")

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
    axes[1].set_title(f"{col}-{'-'.join(group_filter)} Test Cost Per Iteration")
