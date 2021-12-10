import numpy as np
import autograd.numpy as anp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import log_loss
import pennylane as qml
from embedding import apply_encoding



class Simple_Classifier(BaseEstimator, ClassifierMixin):
    """
    TODO get this to work
    from qcnn_estimator import Qcnn_Classifier
    from sklearn.utils.estimator_checks import check_estimator
    estimator = Qcnn_Classifier()
    check_estimator(estimator)
    """

    def __init__(
        self,
        n_iter=50,
        learning_rate=0.01,
        batch_size=25,
        optimizer="nesterov",
        cost="cross_entropy",
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.cost = cost

    def _more_tags(self):
        return {
            "binary_only": True,
        }

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, ensure_2d=False)
        # Set optimizer
        if self.optimizer == "nesterov":
            opt = qml.NesterovMomentumOptimizer(stepsize=self.learning_rate)
        else:
            raise NotImplementedError(
                f"There is no implementation for optimizer: {self.optimizer}"
            )

         # +1 for bias term
        self.coef_count_ = X.shape[1] + 1
        self.coef_ = np.random.randn(self.coef_count_)
        coefficients = self.coef_
        # Set paramaters that saves training information
        self.train_history_ = {"Iteration": [], "Cost": []}
        self.test_history_ = {"Iteration": [], "Cost": []}
        self.coef_history_ = {}

        for it in range(self.n_iter):
            # Sample a batch from the training set
            batch_train_index = np.random.randint(X.shape[0], size=self.batch_size)
            X_train_batch = X[batch_train_index]
            y_train_batch = np.array(y)[batch_train_index]

            # Run model and get cost
            coefficients, cost_train = opt.step_and_cost(
                lambda current_coef: self.coefficient_based_loss(
                    current_coef, X_train_batch, y_train_batch
                ),
                coefficients,
            )
            self.train_history_["Iteration"].append(it)
            self.train_history_["Cost"].append(cost_train)
            self.coef_history_[it] = coefficients

        best_iteration = self.train_history_["Iteration"][
            np.argmin(self.train_history_["Cost"])
        ]
        best_coefficients = self.coef_history_[best_iteration]
        # Set model coefficient corresponding to iteration that had lowest loss
        self.coef_ = best_coefficients.numpy()
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def coefficient_based_loss(self, coef, X, y):
        """Used for training"""
        self.coef_ = coef
        loss = self.score(X, y, return_loss=True, require_tensor=True)
        return loss

    def predict(self, X, cutoff=0.5):
        """A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "is_fitted_")
        y_hat = self.predict_proba(X)
        y_hat_clf = np.array([1 if y_p >= cutoff else 0 for y_p in y_hat])

        return y_hat_clf

    def predict_proba(self, X, require_tensor=False):
        X_1 = np.c_[np.ones(X.shape[0]), X]
        Xb = anp.dot(X_1, self.coef_)
        
        # TODO redundant
        if require_tensor:
            y_hat = sigmoid(anp.dot(X_1, self.coef_))
        else:
            y_hat = sigmoid(anp.dot(X_1, self.coef_))

        return y_hat

    def score(self, X, y, return_loss=False, **kwargs):
        """Returns the score (bigger is better) of the current model. Can specify to return
        the loss (which is the negative of the score). Reason to return loss is for things like training
        as to minimize loss
        """
        # Different for hierarchical

        y_pred = self.predict_proba(X, **kwargs)
        if self.cost == "mse":
            loss = square_loss(y, y_pred)
            # negates loss if return_loss is False so that larger values loss values
            # translate to smaller score values (bigger loss = worse score). If return_loss is True
            # Then loss is returned and below expression ends up being score=loss
            score = (-1 * (not (return_loss)) + return_loss % 2) * loss
        elif self.cost == "cross_entropy":
            loss = cross_entropy(y, y_pred)
            score = (-1 * (not (return_loss)) + return_loss % 2) * loss

        return score


def sigmoid(x):
    return 0.5 * (anp.tanh(x / 2.) + 1)

def cross_entropy(labels, predictions):
    # from sklearn.metrics import log_loss
    # log_loss(labels, [p._value for p in predictions])
    # TODO use pos_class implement ovr and then ensure labels is in a standardized format
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p)) + (1 - l) * anp.log(1 - p)
        loss = loss + c_entropy

    return -1 * loss

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

# %%
