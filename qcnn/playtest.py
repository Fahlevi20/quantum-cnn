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

experiment_config = get_file_content(
    "/home/matt/dev/projects/quantum-cnn/experiments/119/experiment.json"
)
path = f"{experiment_config.get('path')}/{experiment_config.get('ID')}"
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
print(len(result_list))
# %%
filtered_results = [
    result
    for result in result_list
    if result.model_configuration.additional_structure == 'custom_1'
]
len(filtered_results)

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
