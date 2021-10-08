#%%
import os
from operator import mod
import pandas as pd
import pennylane as qml
import circuit_presets
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# Custom
from data_utility import DataUtility
from qcnn_structure import (
    QcnnStructure,
    Layer,
    train_qcnn,
)
from circuit_presets import c_1, c_2, c_3, p_1, p_2, p_3
import embedding

#%%
data_path = "../data/archive/Data/features_30_sec.csv"
target = "label"
# Predicting 1 for the last item in the list
classes = ["metal", "rock"]

raw = pd.read_csv(data_path)
# Data cleaning / focusing only on chosen classes
filter_pat = "|".join(genre for genre in classes)
indices = raw["filename"].str.contains(filter_pat)
raw = raw.loc[indices, :].copy()

raw[target] = np.where(raw[target] == classes[1], 1, 0)
data_utility = DataUtility(raw, target=target, default_subset="modelling")

#%% 3
columns_to_remove = ["filename", "length"]
data_utility.update(columns_to_remove, "included", {"value": False, "reason": "manual"})

# Set train / test indices
X, y, Xy = data_utility.get_samples(raw)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)
data_utility.row_sample["train"] = X_train.index
data_utility.row_sample["test"] = X_test.index

#%%
# Setup
# TODO compact + pure amplitude


embedding_options = {
    8: ["Angle"],
    12: [f"Angular-Hybrid2-{i}" for i in range(1, 5)],
    16: [f"Amplitude-Hybrid2-{i}" for i in range(1, 5)] + ["Angle-Compact"],
    30: [f"Angular-Hybrid4-{i}" for i in range(1, 5)],
    32: [f"Amplitude-Hybrid4-{i}" for i in range(1, 5)],
}

# Name of circuit function from unitary.py along with param count
circuit_options = {
    "U_TTN": 2,
    "U_5": 10,
    "U_6": 10,
    "U_9": 2,
    "U_13": 6,
    "U_14": 6,
    "U_15": 4,
    "U_SO4": 6,
    "U_SU4": 15,
}

pooling_options = {"psatz1": 2, "psatz2": 0, "psatz3": 3}


#%%
# Configuration
EXPERIMENT_PATH = "../experiments"
# Ensure expirment doesn't get overridden
EXPERIMENT_ID = max([int(exp_str) for exp_str in os.listdir(EXPERIMENT_PATH)]) + 1
# EXPERIMENT_ID = 11
EXPERIMENT_CONTENT = (
    f"100 iterations\n"
    f"Embedding type Preprocessing\n"
    f"{'-'*28}\n"
    f"Angle\t\tminmax(0, np.pi)\n"  # f"Amplitude\tnormalized"
    f"Classes\n"
    f"{'-'*7}\n"
    f"y=1\t\trock\n"
    f"y=0\t\tmetal"
)
RESULT_PATH = f"{EXPERIMENT_PATH}/{EXPERIMENT_ID}"
REDUCTION_METHOD = "pca"


# experiment_embedding_list = [
#     "Amplitude-Hybrid4-4",
#     "Angle",
#     "Angular-Hybrid2-1",
#     "Amplitude-Hybrid2-1",
# ]
experiment_embedding_list = [
    # "Angle-Compact",
    "Angle",
    "Amplitude-Hybrid2-1",
    "Angular-Hybrid2-1",
]
experiment_circuit_list = ["U_5", "U_TTN", "U_6", "U_SO4", "U_SU4", "U_15", "U_14", "U_13", "U_9"]
# TODO make pretty
experiment_embeddings = {
    k: set(experiment_embedding_list) & set(v)
    for k, v in embedding_options.items()
    if len((set(experiment_embedding_list) & set(v))) > 0
}
experiment_circuits = {
    circ_name: circuit_options[circ_name] for circ_name in experiment_circuit_list
}
# Define preprocessing

# Define embedding # TODO experiment function, log time taken
print(f"Running expirement: {EXPERIMENT_ID}")
for reduction_size, embedding_set in experiment_embeddings.items():
    for embedding_option in embedding_set:
        for circ_name, circ_param_count in experiment_circuits.items():
            # Only minmax scale if angle
            if "Ang" in embedding_option:
                pipeline = Pipeline(
                    [
                        (
                            "scaler",
                            preprocessing.MinMaxScaler([0, np.pi]),
                        ),
                        ("pca", PCA(reduction_size)),
                    ]
                )
            else:
                pipeline = Pipeline(
                    [
                        # (
                        #     "scaler",
                        #     preprocessing.StandardScaler(),
                        # ),
                        ("pca", PCA(reduction_size)),
                    ]
                )

            # Define QCNN structure
            layer_dict = {
                "c_1": Layer(
                    c_1,
                    getattr(circuit_presets, circ_name),
                    "convolutional",
                    circ_param_count,
                    0,
                ),
                "p_1": Layer(
                    p_1,
                    getattr(circuit_presets, "psatz1"),
                    "pooling",
                    pooling_options["psatz1"],
                    1,
                ),
                "c_2": Layer(
                    c_2,
                    getattr(circuit_presets, circ_name),
                    "convolutional",
                    circ_param_count,
                    2,
                ),
                "p_2": Layer(
                    p_2,
                    getattr(circuit_presets, "psatz1"),
                    "pooling",
                    pooling_options["psatz1"],
                    3,
                ),
                "c_3": Layer(
                    c_3,
                    getattr(circuit_presets, circ_name),
                    "convolutional",
                    circ_param_count,
                    4,
                ),
                "p_3": Layer(
                    p_3,
                    getattr(circuit_presets, "psatz1"),
                    "pooling",
                    pooling_options["psatz1"],
                    5,
                ),
            }

            # Create QCNN structure
            qcnn_structure = QcnnStructure(layer_dict)
            model_name = (
                f"{REDUCTION_METHOD}-{reduction_size}-{embedding_option}-{circ_name}"
            )
            # Train and store results
            (
                y_hat,
                y_hat_class,
                loss_train_history,
                loss_test_history,
                params_history,
                cf_matrix,
            ) = train_qcnn(
                qcnn_structure,
                embedding_option,
                pipeline,
                raw,
                data_utility,
                experiment_content=EXPERIMENT_CONTENT,
                model_name=model_name,
                result_path=RESULT_PATH,
                steps=100,
            )
print("Experiment Done")

# # %%
# # Setup training run
# # TODO autoencoder option

# REDUCTION_METHOD = "pca"
# RESULT_PATH = "../results"

# for reduction_size, embed_list in embedding_options.items():
#     for embed_option in embed_list:
#         for circ_name, circ_param_count in circuit_options.items():
#             pipe = Pipeline(
#                 [("scaler", preprocessing.MinMaxScaler()), ("pca", PCA(reduction_size))]
#             )
#             embedding_type = embed_option

#             layer_dict = {
#                 "c_1": Layer(
#                     c_1,
#                     getattr(circuit_presets, circ_name),
#                     "convolutional",
#                     circ_param_count,
#                     0,
#                 ),
#                 "p_1": Layer(
#                     p_1,
#                     getattr(circuit_presets, "psatz1"),
#                     "pooling",
#                     pooling_options["psatz1"],
#                     1,
#                 ),
#                 "c_2": Layer(
#                     c_2,
#                     getattr(circuit_presets, circ_name),
#                     "convolutional",
#                     circ_param_count,
#                     2,
#                 ),
#                 "p_2": Layer(
#                     p_2,
#                     getattr(circuit_presets, "psatz1"),
#                     "pooling",
#                     pooling_options["psatz1"],
#                     3,
#                 ),
#                 "c_3": Layer(
#                     c_3,
#                     getattr(circuit_presets, circ_name),
#                     "convolutional",
#                     circ_param_count,
#                     4,
#                 ),
#                 "p_3": Layer(
#                     p_3,
#                     getattr(circuit_presets, "psatz1"),
#                     "pooling",
#                     pooling_options["psatz1"],
#                     5,
#                 ),
#             }
#             qcnn_structure = QcnnStructure(layer_dict)

#             STEPS = 200
#             LEARNING_RATE = 0.01
#             BATCH_SIZE = 25
#             COST_FN = "cross_entropy"

#             # Define optimizer
#             opt = qml.NesterovMomentumOptimizer(stepsize=LEARNING_RATE)

#             # Initialize starting paramaters
#             params = np.random.randn(qcnn_structure.paramater_count)

#             # Apply pipeline (this can happen more dynamically)
#             pipe.fit(X_train, y_train)
#             X_train_tfd = pipe.transform(X_train)
#             X_test_tfd = pipe.transform(X_test)
#             loss_train_history = []
#             loss_test_history = []
#             params_history = pd.DataFrame({"initial": params})
#             for it in range(STEPS):
#                 # TODO this is sampling with replacement
#                 batch_index = np.random.randint(0, len(X_train_tfd), (BATCH_SIZE,))
#                 X_batch = X_train_tfd[batch_index]
#                 y_batch = np.array(y_train)[batch_index]

#                 batch_test_index = np.random.randint(0, len(X_test_tfd), (BATCH_SIZE,))
#                 X_test_batch = X_test_tfd[batch_test_index]
#                 y_test_batch = np.array(y_test)[batch_test_index]

#                 params, cost_new = opt.step_and_cost(
#                     lambda v: cost(v, X_batch, y_batch, embedding_type, COST_FN), params
#                 )
#                 cost_test = cost(
#                     params, X_test_batch, y_test_batch, embedding_type, COST_FN
#                 )
#                 params_history[f"{it}"] = params
#                 loss_train_history.append(cost_new)
#                 loss_test_history.append(cost_test)
#                 if it % 10 == 0:
#                     print(
#                         f"{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}\nIteration: {it}\tCost:{cost_new}\t test_cost:{cost_test}"
#                     )
#             # save results
#             best_ind = np.where(loss_train_history == min(loss_train_history))[0][0]
#             best_params = params_history[f"{best_ind}"]
#             y_hat = [model(x, params, embedding_type, COST_FN) for x in X_test_tfd]
#             y_hat_class = get_y_label(y_hat)
#             cf_matrix = confusion_matrix(y_test, y_hat_class)

#             # store_results
#             store_results(
#                 RESULT_PATH,
#                 params_history,
#                 loss_train_history,
#                 loss_test_history,
#                 y_hat,
#                 y_hat_class,
#                 y_test,
#                 cf_matrix,
#             )


# def train_qcnn(
#     qcnn_structure,
#     embedding_type,
#     pipeline,
#     raw,
#     data_utility,
#     result_path=None,
#     steps=200,
#     learning_rate=0.01,
#     batch_size=25,
#     cost_fn="cross_entropy",
# ):
#     # Define optimizer
#     opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)

#     # Preprocessing
#     X_train, y_train, Xy_test, X_test, y_test, Xy_test = data_utility.get_samples(
#         raw, subsets=["train", "test"]
#     )
#     pipeline.fit(X_train, y_train)
#     X_train_tfd = pipe.transform(X_train)
#     X_test_tfd = pipe.transform(X_test)

#     # Initialize  paramaters
#     params = np.random.randn(qcnn_structure.paramater_count)
#     loss_train_history = {}
#     loss_test_history = {}
#     params_history = {}

#     for it in range(steps):
#         # Sample records for trainig run, TODO move to data_utility
#         batch_train_index = np.random.randint(X_train_tfd.shape[0], size=batch_size)
#         X_train_batch = X_train_tfd[batch_train_index]
#         y_train_batch = np.array(y_train)[batch_test_index]

#         # Sample test
#         batch_test_index = np.random.randint(X_train_tfd.shape[0], size=batch_size)
#         X_test_batch = X_test_tfd[batch_test_index]
#         y_test_batch = np.array(y_test)[batch_test_index]

#         # Run model and get cost
#         params, cost_train = opt.step_and_cost(
#             lambda v: cost(v, X_train_batch, y_train_batch, embedding_type, cost_fn),
#             params,
#         )
#         cost_test = cost(params, X_test_batch, y_test_batch, embedding_type, cost_fn)

#         # Store iteration results
#         params_history[it] = params
#         loss_train_history[it] = cost_train
#         loss_test_history[it] = cost_test
#         print(
#             f"{REDUCTION_METHOD}-{reduction_size}-{embedding_type}-{circ_name}\nIteration: {it}\tCost:{cost_train}\t test_cost:{cost_test}"
#         )

#     # Save results
#     best_ind = np.where(loss_train_history == min(loss_train_history))[0][0]
#     best_params = params_history[f"{best_ind}"]
#     y_hat = [model(x, best_params, embedding_type, COST_FN) for x in X_test_tfd]
#     y_hat_class = get_y_label(y_hat)
#     cf_matrix = confusion_matrix(y_test, y_hat_class)

#     # store_results if path is provided
#     if result_path:
#         store_results(
#             RESULT_PATH,
#             params_history,
#             loss_train_history,
#             loss_test_history,
#             y_hat,
#             y_hat_class,
#             y_test,
#             cf_matrix,
#         )
#     return (
#         y_hat,
#         y_hat_class,
#         loss_train_history,
#         loss_test_history,
#         params_history,
#         cf_matrix,
#     )


# # %%
# a = {1: [1, 2, 3], 2: [3, 2, 1]}
# for row in a:
#     print(row)

# # %%
# # Training run
# STEPS = 200
# LEARNING_RATE = 0.01
# BATCH_SIZE = 25
# COST_FN = "cross_entropy"

# # Define optimizer
# opt = qml.NesterovMomentumOptimizer(stepsize=LEARNING_RATE)

# # Initialize starting paramaters
# params = np.random.randn(qcnn_structure.paramater_count)

# # Apply pipeline (this can happen more dynamically)
# pipe.fit(X_train, y_train)
# X_train = pipe.transform(X_train)
# X_test = pipe.transform(X_test)
# loss_history = []
# params_history = pd.DataFrame({"initial": params})
# for it in range(STEPS):
#     # TODO this is sampling with replacement
#     batch_index = np.random.randint(0, len(X_train), (BATCH_SIZE,))
#     X_batch = X_train[batch_index]
#     y_batch = np.array(y_train)[batch_index]

#     params, cost_new = opt.step_and_cost(
#         lambda v: cost(v, X_batch, y_batch, embedding_type, COST_FN), params
#     )
#     params_history[f"{it}"] = params
#     loss_history.append(cost_new)
#     if it % 10 == 0:
#         print(f"{embedding_type} - Iteration: {it}\tCost:{cost_new}")
# # %%
# # save results
# best_ind = np.where(loss_history == min(loss_history))[0][0]
# params = params_history[f"{best_ind}"]
# y_hat = [model(x, params, embedding_type, COST_FN) for x in X_test]
# # TODO get best params
# import numpy as np


# def get_y_label(y_hat):
#     return [np.where(x == max(x))[0][0] for x in y_hat]


# y_hat_class = get_y_label(y_hat)
# cf_matrix = confusion_matrix(y_test, y_hat_class)
# print(cf_matrix)
# params_history.to_csv(
#     f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-param-history.csv"
# )
# pd.DataFrame(loss_history).to_csv(
#     f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-loss-history.csv"
# )
# pd.DataFrame(y_hat).to_csv(
#     f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-yhat.csv"
# )
# pd.DataFrame({"y_test": y_test, "yhat_class": y_hat_class}).to_csv(
#     f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-yhat-class-vs-y-test.csv"
# )
# pd.DataFrame(cf_matrix).to_csv(
#     f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-confusion-matrix.csv"
# )

# # Train model

# # Store results
# # %%

# %%
