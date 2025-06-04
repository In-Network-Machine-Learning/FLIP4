########################################################################
# Copyright (c) Mingyuan Zang, Tomasz Koziak, Networks Technology and Service Platforms Group, Technical University of Denmark
# This work was partly done with Changgang Zheng at Computing Infrastructure Group, University of Oxford
# All rights reserved.
# E-mail: mingyuanzang@outlook.com
# Licensed under the Apache License, Version 2.0 (the 'License')
########################################################################

class Supported_models(str, Enum):
    logistic_regression = "LogisticRegression"
    SGD_classifier = "SGDClassifier"
    MLP_classifier = "MLP_classifier"
    gradient_boosting_classifier = "GradientBoostingClassifier"
    NN_classifier = "NeuralNetworkClassifier"
    BNN_classifier = "BinaryNeuralNetworkClassifier"
