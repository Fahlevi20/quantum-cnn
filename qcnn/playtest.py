# %%
import itertools as it
from math import log2

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
        if len(wire_combos[f"p_{layer_ind+1}"])==0:
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
        "Results", ["model_name", "y_test_hat", "clf", "model_configuration", "samples_tfd", "pipeline"]
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
    tmp_dict["exp_id"] = [exp_id] * len(tmp_dict['Iteration'])
    data = pd.concat([data, pd.DataFrame(tmp_dict)])
# filtered_results = [
#     result
#     for result in result_list
#     if result.model_configuration.additional_structure == 'custom_1'
# ]
#tmp_result = [result for result in result_list if result.model_name=="quantum-qcnn-binary-Angle-minmax-max_features=8_n_estimators=50-tree-feature_range=[0, 1.5707963267948966]-['rock', 'reggae']-('U_5', 'psatz1', [8, 1, 'outside'])"]


# %%


# %%
import matplotlib.pyplot as plt
import seaborn as sns
figsize=(30, 10)
plot_data = data.pivot("Iteration", "exp_id", "Cost")
#data = a.clf.best_estimator_.train_history_
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

index_01 = np.where(y!=2)

X = X[index_01]
y = y[index_01]
# %%
clf = Simple_Classifier().fit(X, y)
clf.predict(X[:2, :])

clf.predict_proba(X[:2, :])


clf.score(X, y)
# %%
