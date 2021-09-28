# TODO move embedding to folder

# %% import libraries
# Computational
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import autograd.numpy as anp

# Semantics
from tabulate import tabulate

# Custom
import embedding
from data_load import load_image_data

# %% Data processing
def load_mnist():
    # Load data
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    classes_to_model = [0, 1]

    train_ind = np.where(np.isin(y_train, classes_to_model))
    test_ind = np.where(np.isin(y_test, classes_to_model))

    X_train, y_train, X_test, y_test = (
        X_train[train_ind],
        y_train[train_ind],
        X_test[test_ind],
        y_test[test_ind],
    )

    # Make y binary TODO maybe
    # Y_train = [1 if y == classes[0] else 0 for y in Y_train]
    # Y_test = [1 if y == classes[0] else 0 for y in Y_test]

    # %%
    # Normalize
    print(tabulate(X_train[1]))
    X_train, X_test = X_train[..., np.newaxis] / 255.0, X_test[..., np.newaxis] / 255.0
    print(tabulate(X_train[1]))

    # %%
    # Basically select only 0 1 classes, or provide functionality to dynamically select

    # Feature reduction look a t logic and generalize
    X_train = tf.image.resize(X_train[:], (784, 1)).numpy()
    X_test = tf.image.resize(X_test[:], (784, 1)).numpy()
    X_train, X_test = tf.squeeze(X_train), tf.squeeze(X_test)
    pca = PCA(32)

    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)


# %% Image example
# Load Image data
data_path = "../data/Data/features_30_sec.csv"
target = "label"
classes = ["pop", "classical"]
X_train, y_train, X_test, y_test = load_image_data(data_path, classes, target)

y_train = np.where(y_train == "classical", 1, 0)
y_test = np.where(y_test == "classical", 1, 0)
# Training
# %% Model definition / Structure setup
# define model / quantum circuit structure
DEVICE = qml.device("default.qubit", wires=8)

# Derived for qcnn
n_circuit_params = 15
n_pooling_params = 2
n_conv_layers = 3
n_pooling_layers = 3
"""
Since the paramater vector that gets optimized is one dimensional (might want to check if this has to be the case), we set up a data structure to track which paramaters belong
to which convolutional layers. Hierarchical has a different structure where it's: n_circuit_params * 7
"""
N_PARAMATERS = n_circuit_params * n_conv_layers + n_pooling_params * n_pooling_layers

# Every layer has the same number of parameters (in this specific implementation 16, which might have to be 15)
conv_param_indices = {
    f"conv_layer_{layer + 1}": range(
        layer * n_circuit_params, (layer + 1) * n_circuit_params
    )
    for layer in range(n_conv_layers)
}

# n_conv_layers * n_circuit_params is the base i.e. we need to start from there and add the pooling paramaters
pooling_param_indices = {
    f"pooling_layer_{layer + 1}": range(
        n_conv_layers * n_circuit_params + layer * n_pooling_params,
        n_conv_layers * n_circuit_params + (layer + 1) * n_pooling_params,
    )
    for layer in range(n_pooling_layers)
}

print(f"Convolutional paramater structure:\n {conv_param_indices}")
print(f"Pooling paramater structure:\n {pooling_param_indices}")
# if circuit == "QCNN":
#     total_params = U_params * 3 + 2 * 3
# elif circuit == "Hierarchical":
#     total_params = U_params * 7

# %% QCNN function to move to classes / modules


def qcnn_circuit(params, wires):  # 15 params
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])


def pooling_circuit(params, wires):  # 2 params
    qml.CRZ(params[0], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(params[1], wires=[wires[0], wires[1]])


# Convolutional layers
def conv_layer_1(params):
    qcnn_circuit(params, wires=[0, 7])
    for i in range(0, 8, 2):
        qcnn_circuit(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        qcnn_circuit(params, wires=[i, i + 1])


def conv_layer_2(params):
    qcnn_circuit(params, wires=[0, 6])
    qcnn_circuit(params, wires=[0, 2])
    qcnn_circuit(params, wires=[4, 6])
    qcnn_circuit(params, wires=[2, 4])


def conv_layer_3(params):
    qcnn_circuit(params, wires=[0, 4])


# Pooling layers
def pooling_layer_1(params):
    for i in range(0, 8, 2):
        pooling_circuit(params, wires=[i + 1, i])


def pooling_layer_2(params):
    pooling_circuit(params, wires=[2, 0])
    pooling_circuit(params, wires=[6, 4])


def pooling_layer_3(params):
    pooling_circuit(params, wires=[0, 4])


def qcnn_structure(params):
    # TODO ask, params gets double selected here
    # 16 + 16 + 16 for conv layers instead of 15 15 15?
    # Also all paramaters are optimized at once vs layer for layer??

    # TODO can be a for loop, but will be explicit in this case since their might be some importance in ordering + we need to think of how to generalize different structures
    # Generate structure dynamically

    conv_layer_1(params[conv_param_indices["conv_layer_1"]])
    pooling_layer_1(params[pooling_param_indices["pooling_layer_1"]])

    conv_layer_2(params[conv_param_indices["conv_layer_2"]])
    pooling_layer_2(params[pooling_param_indices["pooling_layer_2"]])

    conv_layer_3(params[conv_param_indices["conv_layer_3"]])
    pooling_layer_3(params[pooling_param_indices["pooling_layer_3"]])


@qml.qnode(DEVICE)
def qcnn(
    data,
    params,
    embedding_type="Amplitude-Hybrid4-3",
    cost_fn="cross_entropy",
):
    embedding.data_embedding(data, embedding_type=embedding_type)
    qcnn_structure(params)
    # TODO where does 4 come from / paramaterize?
    if cost_fn == "mse":
        result = qml.expval(qml.PauliZ(4))
    elif cost_fn == "cross_entropy":
        result = qml.probs(wires=4)
    return result


# %% Iterative training
STEPS = 200
LEARNING_RATE = 0.01
BATCH_SIZE = 25
EMBEDDING_TYPE = ["Amplitude-Hybrid4-1", "Amplitude-Hybrid4-2", "Amplitude-Hybrid4-4"]
COST_FN = "cross_entropy"

# Define optimizer
opt = qml.NesterovMomentumOptimizer(stepsize=LEARNING_RATE)

# Initialize starting paramaters
params = np.random.randn(N_PARAMATERS)

# Setup cost function
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


def cost(params, X_batch, y_batch, embedding_type, cost_fn):
    # Different for hierarchical
    y_hat = [qcnn(x, params, embedding_type, cost_fn=cost_fn) for x in X_batch]

    if cost_fn == "mse":
        loss = square_loss(y_batch, y_hat)
    elif cost_fn == "cross_entropy":
        loss = cross_entropy(y_batch, y_hat)

    return loss


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


# %%
# Train

for embedding_type in EMBEDDING_TYPE:
    loss_history = []
    params_history = pd.DataFrame({"initial": params})
    for it in range(STEPS):

        # TODO this is sampling with replacement
        batch_index = np.random.randint(0, len(X_train), (BATCH_SIZE,))
        X_batch = X_train[batch_index]
        y_batch = y_train[batch_index]
        # TODO there might be a reason for the list comprehensiopn
        # X_batch = [X_train[i] for i in batch_index]
        # Y_batch = [Y_train[i] for i in batch_index]

        params, cost_new = opt.step_and_cost(
            lambda v: cost(v, X_batch, y_batch, embedding_type, COST_FN), params
        )
        params_history[f"{it}"] = params
        loss_history.append(cost_new)
        if it % 10 == 0:
            print(f"{embedding_type} - Iteration: {it}\tCost:{cost_new}")
    # save results
    y_hat = [qcnn(x, params, COST_FN) for x in X_test]
    y_hat_class = get_y_label(y_hat)
    cf_matrix = confusion_matrix(y_test, y_hat_class)
    print(cf_matrix)
    params_history.to_csv(f"./results/{embedding_type}-pca32-param-history.csv")
    pd.DataFrame(loss_history).to_csv(
        f"./results/{embedding_type}-pca32-loss-history.csv"
    )
    pd.DataFrame(y_hat).to_csv(f"./results/{embedding_type}-pca32-yhat.csv")
    pd.DataFrame({"y_test": y_test, "yhat_class": y_hat_class}).to_csv(
        f"./results/{embedding_type}-pca32-yhat-class-vs-y-test.csv"
    )
    pd.DataFrame(cf_matrix).to_csv(
        f"./results/{embedding_type}-pca32-confusion-matrix.csv"
    )
#    return loss_history, params

# %%
# Evaluate

y_hat = [qcnn(x, params, COST_FN) for x in X_test]
accuracy = accuracy_test(y_hat, y_test, COST_FN)

print(accuracy)


def get_y_label(y_hat):
    return [np.where(x == max(x))[0][0] for x in y_hat]


# %%
# Unitaries = ['U_SU4']
# U_num_params = [15]
# Encodings = ['pca32-3']#, 'autoencoder32-3']
# dataset = "mnist" #'fashion_mnist'
# classes = [0,1]
# binary = False
# cost_fn = 'cross_entropy'
