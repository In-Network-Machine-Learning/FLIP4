from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
    RidgeClassifier,
)  # try to use different tools
import numpy as np
from supported_modles import Supported_modles
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier


class GlobalModel:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.accuracy = 0
        self.f1 = 0

    def init_global_model(self, model):
        self.model = model

    # update each agent model by current global model values
    def load_global_model(self, model):
        model.intercept_ = self.model.intercept_
        model.coef_ = self.model.coef_
        return model

    def update_global_model(self, applicable_clients, round_weights, model):
        # Average models parameters
        coefs = []
        intercept = []

        if model != Supported_modles.MLP_classifier:
            for client in applicable_clients:
                coefs.append(client.model.coef_)
                intercept.append(client.model.intercept_)

            self.model.coef_ = np.average(
                coefs, axis=0, weights=round_weights
            )  # weight
            self.model.intercept_ = np.average(
                intercept, axis=0, weights=round_weights
            )  # weight
        else:
            for client in applicable_clients:
                coefs.append(client.model.coefs_)
                intercept.append(client.model.intercepts_)

            self.model.coefs_ = np.average(
                coefs, axis=0, weights=round_weights
            )  # weight
            self.model.intercepts_ = np.average(
                intercept, axis=0, weights=round_weights
            )  # weight

    def f1_score(self, x_test, y_test):
        y_hat = self.model.predict(x_test)
        return f1_score(y_test, y_hat)
