# %%
class QcnnStructure:
    # TODO add and remove layer functionality
    def __init__(self, layer_dict):
        """
        Initializes a quantum convolutional neural network
        """
        self.layer_dict = layer_dict
        # Sort the layers according to the order
        self.sort_layer_dict_by_order()
        total_param_count = 0
        param_indices = {}
        # Determine paramater indices per layer
        for layer_name, layer in self.layer_dict.items():
            param_indices[layer_name] = range(
                total_param_count, total_param_count + layer.param_count
            )
            total_param_count += layer.param_count

        self.paramater_count = total_param_count
        self.param_indices = param_indices.copy()

    def sort_layer_dict_by_order(self):
        """
        Sorts the layer dictionary by the order that's provided.
        """
        self.layer_dict = {
            layer_name: layer
            for layer_name, layer in sorted(
                self.layer_dict.items(), key=lambda x: x[1].layer_order
            )
        }

    def evaluate(self, params):
        for layer_name, layer in self.layer_dict.items():
            layer.layer_fn(layer.circuit, params[self.param_indices[layer_name]])


class Layer:
    """
    A generic layer consisting of some combination of variational circuits.
    Order doesn't have to be from 0, all layers get sorted purely by order value
    """

    def __init__(self, layer_fn, circuit, layer_type, param_count, layer_order):
        self.layer_fn = layer_fn
        self.circuit = circuit
        self.layer_type = layer_type
        self.param_count = param_count
        self.layer_order = layer_order


# TODO sort out imports
import os
import json
from operator import index, mod
import pandas as pd
import pennylane as qml
import circuit_presets
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# Custom
from data_utility import DataUtility
from circuit_presets import c_1, c_2, c_3, p_1, p_2, p_3
import embedding

DEVICE = qml.device("default.qubit", wires=8)


@qml.qnode(DEVICE)
def model(data, params, embedding_type, qcnn_structure=None, cost_fn="cross_entropy"):
    embedding.data_embedding(data, embedding_type=embedding_type)
    qcnn_structure.evaluate(params)
    # TODO where does 4 come from / paramaterize?
    if cost_fn == "mse":
        result = qml.expval(qml.PauliZ(4))
    elif cost_fn == "cross_entropy":
        result = qml.probs(wires=4)
    return result


def cost(params, X_batch, y_batch, embedding_type, qcnn_structure, cost_fn):
    # Different for hierarchical
    y_hat = [
        model(x, params, embedding_type, qcnn_structure=qcnn_structure, cost_fn=cost_fn)
        for x in X_batch
    ]

    if cost_fn == "mse":
        loss = square_loss(y_batch, y_hat)
    elif cost_fn == "cross_entropy":
        loss = cross_entropy(y_batch, y_hat)

    return loss


def filter_levels(data, feature, levels):
    """Returns all rows belonging to the list of levels for a specific feature

    Args:
        data (pd.DataFrame): data to filter rows
        feature (str): name of the feature for which the levels are concerned
        levels (list[da]): [description]
    """
    filter_pat = "|".join(level for level in levels)
    indices = data[feature].str.contains(filter_pat)
    return data.loc[indices, :].copy()


def store_results(
    config,
    model_name,
    params_history,
    best_params,
    loss_train_history,
    loss_test_history,
    y_hat,
    y_hat_class,
    y_test,
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
        json.dump(config, f,  indent=4)

    # print(f"Storing resuts to:\n {result_path}")
    pd.DataFrame(params_history).to_csv(f"{result_path}/{model_name}-param-history.csv")
    pd.DataFrame(best_params).to_csv(f"{result_path}/{model_name}-best-param.csv",
        index=False,)
    pd.DataFrame(loss_train_history).to_csv(
        f"{result_path}/{model_name}-loss-train-history.csv",
        index=False,
    )
    pd.DataFrame(loss_test_history).to_csv(
        f"{result_path}/{model_name}-loss-test-history.csv",
        index=False,
    )
    pd.DataFrame(y_hat).to_csv(f"{result_path}/{model_name}-yhat.csv")
    pd.DataFrame({"y_test": y_test, "yhat_class": y_hat_class}).to_csv(
        f"{result_path}/{model_name}-yhat-class-vs-y-test.csv"
    )
    pd.DataFrame(cf_matrix).to_csv(f"{result_path}/{model_name}-confusion-matrix.csv")


def train_qcnn(
    qcnn_structure,
    embedding_type,
    pipeline,
    target_levels,
    X_train,
    y_train,
    X_test,
    y_test,
    config,
    model_name="dummy"
):
    # Setup training job
    
    iterations = config["train"].get("iterations", 200)
    learning_rate = config["train"].get("learning_rate", 0.01)
    batch_size = config["train"].get("batch_size", 25)
    cost_fn = config["train"].get("cost_fn", "cross_entropy")
    test_size = config["train"].get("test_size", 0.3)
    random_state = config["train"].get("random_state", 42)
    save_results = False if config.get("path", None) is None else True 

    params = np.random.randn(qcnn_structure.paramater_count)
    loss_train_history = {"Iteration": [], "Cost": []}
    loss_test_history = {"Iteration": [], "Cost": []}
    params_history = {}
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)    

    # Preprocessing
    # Get test set first
    X_test_all = X_test.copy()
    y_test_all = y_test.copy()
    y_test_all = np.where(y_test_all == target_levels[1], 1, 0)

    ## Filter data
    x_train_filter_01 = np.where((y_train == target_levels[0]) | (y_train == target_levels[1]))
    x_test_filter_01 = np.where((y_test == target_levels[0]) | (y_test == target_levels[1]))

    X_train, X_test = X_train[x_train_filter_01], X_test[x_test_filter_01]
    y_train, y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]

    y_train = np.where(y_train == target_levels[1], 1, 0)
    y_test= np.where(y_test == target_levels[1], 1, 0)

    X_train = tf.image.resize(X_train[:], (784, 1)).numpy()
    X_test = tf.image.resize(X_test[:], (784, 1)).numpy()
    X_test_all = tf.image.resize(X_test_all[:], (784, 1)).numpy()
    X_train, X_test, X_test_all = tf.squeeze(X_train), tf.squeeze(X_test), tf.squeeze(X_test_all)

    pipeline.fit(X_train, y_train)

    # Transform data
    X_train_tfd = pipeline.transform(X_train)
    X_test_tfd = pipeline.transform(X_test)
    X_test_all_tfd = pipeline.transform(X_test_all) 

    for it in range(iterations):
        # Sample records for trainig run, TODO move to data_utility
        batch_train_index = np.random.randint(X_train_tfd.shape[0], size=batch_size)
        X_train_batch = X_train_tfd[batch_train_index]
        y_train_batch = np.array(y_train)[batch_train_index]

        # Sample test
        batch_test_index = np.random.randint(X_test_tfd.shape[0], size=batch_size)
        X_test_batch = X_test_tfd[batch_test_index]
        y_test_batch = np.array(y_test)[batch_test_index]

        # Run model and get cost
        params, cost_train = opt.step_and_cost(
            lambda params_current: cost(
                params_current,
                X_train_batch,
                y_train_batch,
                embedding_type,
                qcnn_structure,
                cost_fn,
            ),
            params,
        )
        cost_test = cost(
            params,
            X_test_batch,
            y_test_batch,
            embedding_type,
            qcnn_structure,
            cost_fn,
        )

        # Store iteration results
        params_history[it] = params
        loss_train_history["Iteration"].append(it)
        loss_train_history["Cost"].append(cost_train)
        loss_test_history["Iteration"].append(it)
        loss_test_history["Cost"].append(cost_test)

    # Save results
    best_iteration = loss_train_history["Iteration"][
        np.argmin(loss_train_history["Cost"])
    ]
    best_params = params_history[best_iteration]
    y_hat = [
        model(
            x,
            best_params,
            embedding_type,
            qcnn_structure=qcnn_structure,
            cost_fn=cost_fn,
        )
        for x in X_test_tfd
    ]
    # TODO can be optimized to use the already predicted ones
    y_hat_all = [
        model(
            x,
            best_params,
            embedding_type,
            qcnn_structure=qcnn_structure,
            cost_fn=cost_fn,
        )
        for x in X_test_all_tfd
    ]
    y_hat_class = get_y_label(y_hat)
    cf_matrix = confusion_matrix(y_test, y_hat_class)

    # store_results if path is provided
    if save_results:
        store_results(
            config,
            model_name,
            params_history,
            best_params,
            loss_train_history,
            loss_test_history,
            y_hat,
            y_hat_class,
            y_test,
            cf_matrix,
        )
    return (
        y_hat_all,
        cf_matrix,
    )


# TODO move to model evaluation
import autograd.numpy as anp
import numpy as np


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy

    return -1 * loss


def accuracy_test(predictions, labels, cost_fn, binary=False):
    if cost_fn == "mse":
        if binary == True:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 1:
                    acc = acc + 1
            return acc / len(labels)

        else:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 0.5:
                    acc = acc + 1
            return acc / len(labels)

    elif cost_fn == "cross_entropy":
        acc = 0
        for l, p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc = acc + 1
        return acc / len(labels)


def get_y_label(y_hat):
    return [np.where(x == max(x))[0][0] for x in y_hat]


# qcnn_generic["conv_layer"](2)
# %%

# %%
