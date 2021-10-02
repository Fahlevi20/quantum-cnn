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
