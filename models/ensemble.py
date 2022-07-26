"""
Function for training and ensemble of linear regressor on bootstrap replica of the original training set
@author: Davide Cacciarelli
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def bootstrap_models(n_bootstrap, x_train, y_train, sampling_seed):
    """
    Training an enemble of models on bootstrapped replica of the training set
    :param n_bootstrap: number of models composing the ensemble
    :param x_train: predictors
    :param y_train: output variable
    :param sampling_seed: initial seed to be used for sampling instances (then it is increased at each step)
    :return: a list of fitted models
    """
    # list of models trained on bootstrapped replica of the training set
    models = []
    n = x_train.shape[0]

    # creating boostrap replica and fitting models on them
    for i in range(n_bootstrap):
        # initializing models
        model = LinearRegression()

        # getting random indices with replacement and subsetting train set
        np.random.seed(i + sampling_seed)
        train_idxs = np.random.choice(range(n), size=n, replace=True)
        x_train_bootstrap = pd.DataFrame(x_train.iloc[train_idxs, :], columns=x_train.columns)

        # fitting regression model
        model.fit(x_train_bootstrap, y_train.iloc[train_idxs])
        models.append(model)

    return models
