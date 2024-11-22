from enum import Enum


class Supported_models(str, Enum):
    logistic_regression = "LogisticRegression"
    SGD_classifier = "SGDClassifier"
    MLP_classifier = "MLP_classifier"
    gradient_boosting_classifier = "GradientBoostingClassifier"
    NN_classifier = "NeuralNetworkClassifier"
    BNN_classifier = "BinaryNeuralNetworkClassifier"
