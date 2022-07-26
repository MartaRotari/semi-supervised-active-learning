"""
Function for random sampling of process variables and corresponding quality information
@author: Davide Cacciarelli
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def random_sampling(data, reps, n_obs_test, n_obs_val, min_size, seeds, max_size, alpha=0.95):
    """
    Time-based sampling
    :param data: dataset containing multiple runs
    :param reps: number of times the procedure runs,
    :param n_obs_test: number of observations in each run to be allocated to the test set
    :param n_obs_val: number of observations in each run to be allocated to the validation set (must be equal to the
    other methods even if here there is no validation)
    :param min_size: number of labeled observations initially available to the learner
    :param max_size: maximum size of the training set (budget = max_size - min_size)
    :param alpha: labeling rate
    :return: array of RMSE results for each learning step and for each run
    """
    all_results = []  # Storing results from all the runs
    simulation_run = 0

    while True:
        results = []  # Storing results of the current run

        # Allocating data to: test, train and stream
        current_series = data[data["RUN"] == simulation_run].drop("RUN", axis=1)
        test_set = current_series.sample(n=n_obs_test, random_state=seeds[simulation_run])
        current_series = current_series.drop(test_set.index)
        current_series = current_series.reset_index(drop=True)
        val_set = current_series.sample(n=n_obs_val, random_state=seeds[simulation_run])
        current_series = current_series.drop(val_set.index)
        current_series = current_series.reset_index(drop=True)
        train_set_labeled = pd.DataFrame(columns=list(test_set))
        stream_set = current_series

        rng = np.random.default_rng(seeds[simulation_run])
        sampling_index = rng.uniform(0, 1, size=len(stream_set))
        for i in range(len(stream_set)):
            if train_set_labeled.shape[0] == min_size:
                starting_point = i+1
                stream = stream_set.iloc[starting_point:, :]
                break
            if sampling_index[i] >= alpha:
                # selecting i-th sample and adding it to the labeled dataset
                sample_to_add = pd.DataFrame(np.array(stream_set.iloc[i, :]).reshape(1, -1),
                                             columns=list(train_set_labeled))
                train_set_labeled = train_set_labeled.append(sample_to_add, ignore_index=True)

        # Initializing regression class
        regr = LinearRegression(fit_intercept=True)

        scaler = StandardScaler()

        # Splitting into X, y
        x_train = train_set_labeled.drop(["y"], axis=1)
        y_train = train_set_labeled["y"]
        x_test = test_set.drop(["y"], axis=1)
        y_test = test_set["y"]

        # Fit and predict
        regr.fit(scaler.fit_transform(x_train), y_train)
        y_pred = regr.predict(scaler.transform(x_test))
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        results.append(rmse)

        for i in range(starting_point, len(stream)):
            if x_train.shape[0] == max_size - 1:  # stop if we reached the budget
                break
            # selecting i-th sample and adding it to the labeled dataset
            if sampling_index[i] >= alpha:
                sample_to_add = pd.DataFrame(np.array(stream.iloc[i, :]).reshape(1, -1),
                                             columns=list(train_set_labeled))
                train_set_labeled = train_set_labeled.append(sample_to_add, ignore_index=True)

                # Splitting into X, y
                x_train = train_set_labeled.drop(["y"], axis=1)
                y_train = train_set_labeled["y"]

                # Fit and predict
                regr.fit(scaler.fit_transform(x_train), y_train)
                y_pred = regr.predict(scaler.transform(x_test))
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results.append(rmse)

                if len(results) == max_size - min_size:
                    all_results.append(results)
                    simulation_run += 1

        if len(all_results) == reps:
            break

    return np.array(all_results)
