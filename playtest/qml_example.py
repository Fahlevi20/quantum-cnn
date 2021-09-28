# %% Imports
import pennylane as qml
from pennylane import numpy as np

# %%
# Creating a device.
"""
A device is any computational object that can apply quantum operations and return measurements. This can be actual hardware such as IBM QX4 or software simulaters such as Strawberry Fields.
We use a pure-state qubit simulator (default.qubit)
"""
device = qml.device("default.qubit", wires=1)

# %% Construct QNode
"""
a QNode is the combination of a quantum function and a device. We'll have to define a quantum function first. A quantum function is a subset of a standard python function. The restrictions are:
    - Q functions must contain quantum operations, 1 operation per line (not sure if this implies a line has to contain a q function) i.e. you might not be able to incorporate other logic here
    - Must return a single or tuple of measured observables (a classical quantity)
"""


@qml.qnode(device)
def circuit(params, *args, **kwargs):
    """
    This function will apply Rx, Ry rotations and then measure PauliZ
    """
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


# %% Execute
result = circuit([0.54, 0.12])
print(result)

# %% Get gradient
dcircuit = qml.grad(circuit, argnum=0)
print(dcircuit([0.54, 0.12]))

# %% Define cost


def cost(params, *args, **kwargs):
    return circuit(params, *args, **kwargs)


init_params = np.array([0.011, 0.012])
print(cost(init_params))

# %% Optimize

optimzer = qml.GradientDescentOptimizer(stepsize=0.4)

steps = 100
params = np.array([0.011, 0.012])

for i in range(steps):
    params = optimzer.step(cost, params)

    if (i + 1) % 5 == 0:
        print(f"Cost after step {i + 1}: {cost(params)}")
    
print(f"Optimized rotation angles: {params}")


# %%
test_list = [1, 2, 3]
test_tuple = (1, 2, 3)
test_list[0]
test_tuple[0]
