# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utility import DataUtility
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import time
from sklearn.model_selection import train_test_split

data_path = "../data/archive/Data/features_30_sec.csv"
target = "label"
# Predicting 1 for the last item in the list
classes = ["jazz", "disco"]

raw = pd.read_csv(data_path)
# Data cleaning / focusing only on chosen classes
filter_pat = "|".join(genre for genre in classes)
indices = raw["filename"].str.contains(filter_pat)
raw = raw.loc[indices, :].copy()

raw[target] = np.where(raw[target] == classes[1], 1, 0)
data_utility = DataUtility(raw, target=target, default_subset="modelling")


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

# pca = PCA()

# # set the tolerance to a large value to make the example faster
# logistic = LogisticRegression(max_iter=10000, tol=0.1)
# pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

# # Parameters of pipelines can be set using ‘__’ separated parameter names:
# param_grid = {
#     'pca__n_components': [5, 15, 30, 45, 64],
#     'logistic__C': np.logspace(-4, 4, 4),
# }

# search = GridSearchCV(pipe, param_grid, n_jobs=-1)
# search.fit(X_train, y_train)
# print("Best parameter (CV score=%0.3f):" % search.best_score_)
# print(search.best_params_)

# # Plot the PCA spectrum
# pca.fit(X_train)

# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
# ax0.plot(np.arange(1, pca.n_components_ + 1),
#          pca.explained_variance_ratio_, '+', linewidth=2)
# ax0.set_ylabel('PCA explained variance ratio')

# ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
#             linestyle=':', label='n_components chosen')
# ax0.legend(prop=dict(size=12))

# # For each number of components, find the best classifier results
# results = pd.DataFrame(search.cv_results_)
# components_col = 'param_pca__n_components'
# best_clfs = results.groupby(components_col).apply(
#     lambda g: g.nlargest(1, 'mean_test_score'))

# best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
#                legend=False, ax=ax1)
# ax1.set_ylabel('Classification accuracy (val)')
# ax1.set_xlabel('n_components')

# plt.xlim(-1, 70)

# plt.tight_layout()
# plt.show()
# # %%
# y_hat_class = search.predict(X_test)
# cf_matrix = confusion_matrix(y_test, y_hat_class)

# # %%
# model = LogisticRegression()
# model.fit(X_train, y_train)

# y_hat_class = model.predict(X_test)

# score = model.score(X_test, y_test)
# y_hat_class = search.predict(X_test)
# cf_matrix = confusion_matrix(y_test, y_hat_class)
# print(score)
# %%
pca = PCA(8)
logistic = LogisticRegression(max_iter=10000, tol=0.1)
pipeline = Pipeline(steps=[("pca", pca), ("logistic", logistic)])
pipeline.fit(X_train, y_train)

pipeline.score(X_test, y_test)
y_hat_class = pipeline.predict(X_test)
cf_matrix = confusion_matrix(y_test, y_hat_class)

pd.DataFrame(cf_matrix).to_csv(
    f"../results/pca-8-logreg-{'-'.join(classes)}-confusion-matrix.csv"
)
# %%


import numpy as np
import torch
import torch.nn as nn
t1 = time.time()
steps = 200
n_feature = 2
batch_size = 25
reduction_size = 8

input_size = reduction_size
final_layer_size = int(reduction_size / 4)

pipeline = Pipeline(
    [
        ("pca", PCA(reduction_size)),
    ]
)

CNN = nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=n_feature, kernel_size=2, padding=1),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2),
    nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=2, padding=1),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(n_feature * final_layer_size, 2),
)
X_train, y_train, Xy_test, X_test, y_test, Xy_test = data_utility.get_samples(
        raw, row_samples=["train", "test"]
    )
pipeline.fit(X_train, y_train)
X_train_tfd = pipeline.transform(X_train)
X_test_tfd = pipeline.transform(X_test)

loss_history = []
for it in range(steps):

    batch_train_index = np.random.randint(X_train_tfd.shape[0], size=batch_size)
    X_train_batch = X_train_tfd[batch_train_index]
    y_train_batch = np.array(y_train)[batch_train_index]

    # Sample test
    batch_test_index = np.random.randint(X_test_tfd.shape[0], size=batch_size)
    X_test_batch = X_test_tfd[batch_test_index]
    y_test_batch = np.array(y_test)[batch_test_index]

    X_train_batch_torch = torch.tensor(X_train_batch, dtype=torch.float32)
    X_train_batch_torch.resize_(batch_size, 1, input_size)
    y_train_batch_torch = torch.tensor(y_train_batch, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(CNN.parameters(), lr=0.01, betas=(0.9, 0.999))

    y_pred_batch_torch = CNN(X_train_batch_torch)

    loss = criterion(y_pred_batch_torch, y_train_batch_torch)
    loss_history.append(loss.item())
    if it % 10 == 0:
        print("[iteration]: %i, [LOSS]: %.6f" % (it, loss.item()))

    opt.zero_grad()
    loss.backward()
    opt.step()

    # X_test_torch = torch.tensor(X_test_batch, dtype=torch.float32)
    # X_test_torch.resize_(len(X_test_batch), 1, input_size)
    # Y_pred = CNN(X_test_torch).detach().numpy()
    # accuracy = accuracy_test(Y_pred, y_test)
    # print(accuracy)
    # N_params = get_n_params(CNN)\

t2 = time.time()
# %%
def get_y_label(y_hat):
    return [np.where(x == max(x))[0][0] for x in y_hat]
X_test_torch = torch.tensor(X_test_tfd, dtype=torch.float32)
X_test_torch.resize_(X_test_torch.shape[0], 1, input_size)
y_pred_torch = CNN(X_test_torch)
y_hat_class = get_y_label(y_pred_torch)



print(sum(y_test==get_y_label(y_pred_torch))/y_test.shape[0])
cf_matrix = confusion_matrix(y_test, y_hat_class)

pd.DataFrame(cf_matrix).to_csv(
    f"../results/pca-8-cnn-{'-'.join(classes)}-{steps}-confusion-matrix.csv"
)
print("--- %s seconds ---" % (t2 - t1))
# %%




