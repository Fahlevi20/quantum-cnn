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
    get_line_plot_data,
    plot_119_accuracy_per_structure,
    gather_results_118_135,
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


# %%
from collections import namedtuple
import pandas as pd
from IPython.display import display
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from joblib import dump, load
import numpy as np
import itertools as it

import matplotlib.pyplot as plt
import seaborn as sns

path_experiments = f"/home/matt/dev/projects/quantum-cnn/experiments"
# %%
exp_id = 0
result_data = gather_results_0_12(exp_id, path_experiments=path_experiments)
plot_loss(result_data, ["embedding_type", "circuit"], figsize=(28, 5), save_plot=True)

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

# plot_119_accuracy_per_structure(result_data, figsize)
data_0 = result_data[result_data["target_pair_str"] == "rock_reggae"].copy()
data_1 = result_data[result_data["target_pair_str"] == "classical_pop"].copy()
plot_data_0 = get_line_plot_data(data_0, groupby, metric)
plot_data_1 = get_line_plot_data(data_1, groupby, metric)
plot_data_0[{"pca", "tree"}]
plot_data_1[{"pca", "tree"}]
# grouped_data = result_data.groupby(["additional_structure", "selection_method"])["accuracy"].max()
# grouped_data_unstack = grouped_data.copy().unstack(level=-1)
# grouped_data_unstack[grouped_data_unstack.index.name] = grouped_data_unstack.index


# %%
# graph
# TODO find way to show qbit is removed in make node small or something
# https://towardsdatascience.com/customizing-networkx-graphs-f80b4e69bedf
import networkx as nx
import matplotlib.pyplot as pl


# %%

# %%
# %%
from circuit_presets import get_wire_combos
from collections import namedtuple

# %%
from reporting_functions import get_model_result_list

exp_id = 203
result_data = gather_results_118_135(exp_id, path_experiments=path_experiments)
# display(get_experiment_config(path_experiments, exp_id))
# pd.set_option("display.max_rows", 100)
# result_table = get_result_table(
#     result_data,
#     ["algorithm", "additional_structure_str", "selection_method", "target_pair_str"],
#     "accuracy",
# )
# result_list = get_model_result_list(get_experiment_config(path_experiments, exp_id))
# %%
pair_data = get_result_table_target_pairs(
    result_data, "algorithm", "target_pair_str", "accuracy"
)
grouped_data = result_data.groupby(["target_pair_str"])["accuracy"].max()
grouped_data_df = grouped_data.to_frame()
grouped_data_df["level_0"] = [item.split("_")[0] for item in grouped_data.index]
grouped_data_df["level_1"] = [item.split("_")[1] for item in grouped_data.index]
# %%
import seaborn as sns

figsize = (10, 10)
pair_data = get_result_table_target_pairs(
    result_data, "algorithm", "target_pair_str", "accuracy"
)
plot_data = pair_data.fillna(1)


def plot_triangle_accuracies(plot_data, figsize=(10, 10)):
    mask = np.zeros_like(plot_data)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("whitegrid"):
        f, ax = plt.subplots(figsize=figsize)
        ax.set_title("Accuracy for pairs of genre's")
        sns.heatmap(
            plot_data,
            annot=True,
            fmt=".0%",
            ax=ax,
            vmin=0.3,
            cmap=sns.dark_palette("#28708a", reverse=False, as_cmap=True),
            mask=mask,
        )


# %%
groupby = ["algorithm", "additional_structure_str", "selection_method"]
top_5_models = (
    result_table["algorithm Average"].sort_values(ascending=False).head().index
)
filter_col_val = lambda data, col, val: data.apply(lambda row: row[col] == val, axis=1)

top_5_ids = [
    tuple(
        result_data[
            filter_col_val(result_data, "algorithm", model_config[0])
            & filter_col_val(result_data, "additional_structure_str", model_config[1])
            & filter_col_val(result_data, "selection_method", model_config[2])
        ]["model_name"]
    )
    for model_config in top_5_models
]
# %%

# %%
X_test = pd.read_csv(f"{path_experiments}/{exp_id}/X_test.csv", index_col=0)
columns = X_test.columns
tmp_model_ids = top_5_ids[3]
tmp_results = [result for result in result_list if result.model_name in tmp_model_ids]
tmp_results[0].model_configuration
model_result = tmp_results[1]

plot_binary_tree(tmp_results[0], columns)
plot_binary_tree(tmp_results[1], columns)
# %%


import seaborn as sns

sns.set_theme(style="ticks")

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")
# %%
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
    with sns.axes_style("whitegrid"):
        sns.pairplot(plot_data, hue="genre")

    return feature_names


# target_levels = pipe_Xy_df[data_utility.target].unique()
# %%
from reporting_functions import get_model_result_list
import seaborn as sns

result_list = get_model_result_list(get_experiment_config(path_experiments, exp_id))

# [detached from 60640.pts-0.hep1]
# %%
import matplotlib.pyplot as plt
from matplotlib import ticker


# %%
def plot_var_exp(pipeline, selection_method, figsize=(10, 10)):
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(1, 1, figsize=figsize, sharey=True)
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
        b = ax.set_xlabel("Principal components")
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
from circuit_presets import get_wire_combos
from reporting_functions import get_wire_combos_graph,get_circuit_diagram
import networkx as nx
import matplotlib.pyplot as plt

# (qcnn, U_5_psatz1_{'n_wires': 8, 'c_step': 1, 'pool_pattern': 'right', 'p_step': 0, 'wire_to_cut': 0}, tree)	0.976744	0.812500	0.894622
n_wires = 16
c_step = 1
p_step = 3
pool_pattern = "outside"
wire_to_cut = 1

wire_combos = get_wire_combos(n_wires, c_step, pool_pattern, p_step=p_step, wire_to_cut=wire_to_cut)


n_graphs = get_wire_combos_graph(wire_combos, n_qbits=n_wires)

for layer in n_graphs.keys():
    fig, ax = plt.subplots(figsize=(7, 7))
    tmp_g = n_graphs[layer]
    nx.draw(
        tmp_g[0],
        tmp_g[1],
        with_labels=True,
        node_size=tmp_g[2],
        edge_color="#000000",
        edgecolors="#000000",
        node_color=tmp_g[3],
        width=1.5,
    )
    fig.savefig(f"/home/matt/dev/projects/quantum-cnn/reports/20220112/{pool_pattern}/{n_wires}-{pool_pattern}-{c_step}-{p_step}-{wire_to_cut}-{layer}.svg")
    sub_wires = {key:value for key,value in wire_combos.items() if key==layer}
    fig_2 = get_circuit_diagram(sub_wires, n_qbits=n_wires)

    fig_2.savefig(f"/home/matt/dev/projects/quantum-cnn/reports/20220112/{pool_pattern}/{n_wires}-{pool_pattern}-{c_step}-{p_step}-{wire_to_cut}-{layer}-circuit.svg")
    # %%
