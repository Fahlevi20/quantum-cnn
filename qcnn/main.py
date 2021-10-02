# %%
# Imports
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
    cross_entropy,
    square_loss,
    accuracy_test,
    get_y_label,
)
from circuit_presets import c_1, c_2, c_3, p_1, p_2, p_3
import embedding

# %%
# Load Data
data_path = "../data/archive/Data/features_30_sec.csv"
target = "label"
# Predicting 1 for the last item in the list
classes = ["pop", "classical"]

raw = pd.read_csv(data_path)
# Data cleaning / focusing only on chosen classes
filter_pat = "|".join(genre for genre in classes)
indices = raw["filename"].str.contains(filter_pat)
raw = raw.loc[indices, :].copy()

raw[target] = np.where(raw[target] == classes[1], 1, 0)
data_utility = DataUtility(raw, target=target, default_subset="modelling")
# %%
# Remove metadata columns
columns_to_remove = ["filename", "length"]
data_utility.update(columns_to_remove, "included", {"value": False, "reason": "manual"})

# %%
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
# %%
# Setup
# TODO compact + pure amplitude
DEVICE = qml.device("default.qubit", wires=8)

embedding_options = {
    8: ["Angle"],
    12: [f"Angular-Hybrid2-{i}" for i in range(1, 5)],
    16: [f"Amplitude-Hybrid2-{i}" for i in range(1, 5)],
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


@qml.qnode(DEVICE)
def model(data, params, embedding_type, cost_fn):
    embedding.data_embedding(data, embedding_type=embedding_type)
    qcnn_structure.evaluate(params)
    # TODO where does 4 come from / paramaterize?
    if cost_fn == "mse":
        result = qml.expval(qml.PauliZ(4))
    elif cost_fn == "cross_entropy":
        result = qml.probs(wires=4)
    return result


def cost(params, X_batch, y_batch, embedding_type, cost_fn):
    # Different for hierarchical
    y_hat = [model(x, params, embedding_type, cost_fn=cost_fn) for x in X_batch]

    if cost_fn == "mse":
        loss = square_loss(y_batch, y_hat)
    elif cost_fn == "cross_entropy":
        loss = cross_entropy(y_batch, y_hat)

    return loss


def store_results(
    result_path, params_history, loss_history, y_test, y_hat, y_hat_class, cf_matrix
):
    """
    Method to store results to a desired path
    """
    print(f"Storing resuts to:\n {result_path}")
    params_history.to_csv(
        f"../results/{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}-param-history.csv"
    )
    pd.DataFrame(loss_history).to_csv(
        f"../results/{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}-loss-history.csv"
    )
    pd.DataFrame(loss_test_history).to_csv(
        f"../results/{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}-loss-test-history.csv"
    )
    pd.DataFrame(y_hat).to_csv(
        f"../results/{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}-yhat.csv"
    )
    pd.DataFrame({"y_test": y_test, "yhat_class": y_hat_class}).to_csv(
        f"../results/{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}-yhat-class-vs-y-test.csv"
    )
    pd.DataFrame(cf_matrix).to_csv(
        f"../results/{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}-confusion-matrix.csv"
    )


def get_y_label(y_hat):
    return [np.where(x == max(x))[0][0] for x in y_hat]


# %%
# Setup training run
# TODO autoencoder option

REDUCTION_METHOD = "pca"
RESULT_PATH = "../results"

for reduction_size, embed_list in embedding_options.items():
    for embed_option in embed_list:
        for circ_name, circ_param_count in circuit_options.items():
            pipe = Pipeline(
                [("scaler", preprocessing.MinMaxScaler()), ("pca", PCA(reduction_size))]
            )
            embedding_type = embed_option

            layer_dict = {
                "c_1": Layer(
                    c_1,
                    getattr(circuit_presets, circ_name),
                    "convolutional",
                    circuit_options[circ_name],
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
                    circuit_options[circ_name],
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
                    circuit_options[circ_name],
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
            qcnn_structure = QcnnStructure(layer_dict)

            STEPS = 200
            LEARNING_RATE = 0.01
            BATCH_SIZE = 25
            COST_FN = "cross_entropy"

            # Define optimizer
            opt = qml.NesterovMomentumOptimizer(stepsize=LEARNING_RATE)

            # Initialize starting paramaters
            params = np.random.randn(qcnn_structure.paramater_count)

            # Apply pipeline (this can happen more dynamically)
            pipe.fit(X_train, y_train)
            X_train_tfd = pipe.transform(X_train)
            X_test_tfd = pipe.transform(X_test)
            loss_train_history = []
            loss_test_history = []
            params_history = pd.DataFrame({"initial": params})
            for it in range(STEPS):
                # TODO this is sampling with replacement
                batch_index = np.random.randint(0, len(X_train_tfd), (BATCH_SIZE,))
                X_batch = X_train_tfd[batch_index]
                y_batch = np.array(y_train)[batch_index]

                batch_test_index = np.random.randint(0, len(X_test_tfd), (BATCH_SIZE,))
                X_test_batch = X_test_tfd[batch_test_index]
                y_test_batch = np.array(y_test)[batch_test_index]

                params, cost_new = opt.step_and_cost(
                    lambda v: cost(v, X_batch, y_batch, embedding_type, COST_FN), params
                )
                cost_test = cost(
                    params, X_test_batch, y_test_batch, embedding_type, COST_FN
                )
                params_history[f"{it}"] = params
                loss_train_history.append(cost_new)
                loss_test_history.append(cost_test)
                if it % 10 == 0:
                    print(
                        f"{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}\nIteration: {it}\tCost:{cost_new}\t test_cost:{cost_test}"
                    )
            # save results
            best_ind = np.where(loss_train_history == min(loss_train_history))[0][0]
            best_params = params_history[f"{best_ind}"]
            y_hat = [model(x, params, embedding_type, COST_FN) for x in X_test_tfd]
            y_hat_class = get_y_label(y_hat)
            cf_matrix = confusion_matrix(y_test, y_hat_class)

            # store_results
            store_results(
                RESULT_PATH,
                params_history,
                loss_train_history,
                loss_test_history,
                y_hat,
                y_hat_class,
                y_test,
                cf_matrix,
            )


def sample_batch(data, size):
    batch_index = np.random.randint(data.shape[0], size=size)


def train_qcnn(
    qcnn_structure,
    embedding_type,
    pipeline,
    raw,
    data_utility,
    result_path=None,
    steps=200,
    learning_rate=0.01,
    batch_size=25,
    cost_fn="cross_entropy",
):
    X_train, y_train, Xy_test, X_test, y_test, Xy_test = data_utility.get_samples(
        raw, subsets=["train", "test"]
    )
    pipeline.fit(X_train, y_train)
    X_train_tfd = pipe.transform(X_train)
    X_test_tfd = pipe.transform(X_test)
    loss_train_history = {}
    loss_test_history = {}
    params_history = {}
    for it in range(steps):
        # Sample records for trainig run
        batch_train_index = np.random.randint(X_train_tfd.shape[0], size=batch_size)
        X_train_batch, y_train_batch = sample_batch(X_train_tfd, batch_size)
        X_test_batch, y_test_batch = sample_batch(X_test_tfd, batch_size)

        batch_test_index = np.random.randint(X_train_tfd.shape[0], size=batch_size)
        X_test_batch = X_test_tfd[batch_test_index]
        y_test_batch = np.array(y_test)[batch_test_index]

        # Run model and get cost
        params, cost_train = opt.step_and_cost(
            lambda v: cost(v, X_batch, y_batch, embedding_type, cost_fn), params
        )
        cost_test = cost(params, X_test_batch, y_test_batch, embedding_type, cost_fn)

        params_history[it] = params
        loss_train_history[it] = cost_train
        loss_test_history[it] = cost_test
        print(
            f"{REDUCTION_METHOD}-{reduction_size}-{embed_option}-{circ_name}\nIteration: {it}\tCost:{cost_train}\t test_cost:{cost_test}"
        )
    # save results
    best_ind = np.where(loss_train_history == min(loss_train_history))[0][0]
    best_params = params_history[f"{best_ind}"]
    y_hat = [model(x, params, embedding_type, COST_FN) for x in X_test_tfd]
    y_hat_class = get_y_label(y_hat)
    cf_matrix = confusion_matrix(y_test, y_hat_class)

    # store_results if path is provided
    if result_path:
        store_results(
            RESULT_PATH,
            params_history,
            loss_train_history,
            loss_test_history,
            y_hat,
            y_hat_class,
            y_test,
            cf_matrix,
        )
    return (
        y_hat,
        y_hat_class,
        loss_train_history,
        loss_test_history,
        params_history,
        cf_matrix,
    )


# %%
a = {1: [1, 2, 3], 2: [3, 2, 1]}
for row in a:
    print(row)

# %%
# Training run
STEPS = 200
LEARNING_RATE = 0.01
BATCH_SIZE = 25
COST_FN = "cross_entropy"

# Define optimizer
opt = qml.NesterovMomentumOptimizer(stepsize=LEARNING_RATE)

# Initialize starting paramaters
params = np.random.randn(qcnn_structure.paramater_count)

# Apply pipeline (this can happen more dynamically)
pipe.fit(X_train, y_train)
X_train = pipe.transform(X_train)
X_test = pipe.transform(X_test)
loss_history = []
params_history = pd.DataFrame({"initial": params})
for it in range(STEPS):
    # TODO this is sampling with replacement
    batch_index = np.random.randint(0, len(X_train), (BATCH_SIZE,))
    X_batch = X_train[batch_index]
    y_batch = np.array(y_train)[batch_index]

    params, cost_new = opt.step_and_cost(
        lambda v: cost(v, X_batch, y_batch, embedding_type, COST_FN), params
    )
    params_history[f"{it}"] = params
    loss_history.append(cost_new)
    if it % 10 == 0:
        print(f"{embedding_type} - Iteration: {it}\tCost:{cost_new}")
# %%
# save results
best_ind = np.where(loss_history == min(loss_history))[0][0]
params = params_history[f"{best_ind}"]
y_hat = [model(x, params, embedding_type, COST_FN) for x in X_test]
# TODO get best params
import numpy as np


def get_y_label(y_hat):
    return [np.where(x == max(x))[0][0] for x in y_hat]


y_hat_class = get_y_label(y_hat)
cf_matrix = confusion_matrix(y_test, y_hat_class)
print(cf_matrix)
params_history.to_csv(
    f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-param-history.csv"
)
pd.DataFrame(loss_history).to_csv(
    f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-loss-history.csv"
)
pd.DataFrame(y_hat).to_csv(
    f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-yhat.csv"
)
pd.DataFrame({"y_test": y_test, "yhat_class": y_hat_class}).to_csv(
    f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-yhat-class-vs-y-test.csv"
)
pd.DataFrame(cf_matrix).to_csv(
    f"../results/{REDUCTION_METHOD}-{REDUCTION_SIZE}-{EMBEDDING_OPTION}-confusion-matrix.csv"
)

# Train model

# Store results
# %%
