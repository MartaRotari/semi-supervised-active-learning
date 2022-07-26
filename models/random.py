"""
Function for random sampling of process variables and corresponding quality information
@author: Davide Cacciarelli
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from models.pca_fit_transform import DimensionalityReductionPCA


def random_sampling(data, reps, n_obs_test, n_obs_val, min_size, seeds, max_size, preprocessing=False, pc=0.85,
                    results_percentage=False, representative_test=False, alpha=0.95, store_predictions=False):
    """
    Time-based sampling
    :param data: dataset containing multiple runs
    :param reps: number of times the procedure runs,
    :param n_obs_test: number of observations in each run to be allocated to the test set
    :param n_obs_val: number of observations in each run to be allocated to the validation set (must be equal to the
    other methods even if here there is no validation)
    :param min_size: number of labeled observations initially available to the learner
    :param step: step size for sampling, must be equal to 1/alpha, where alpha is the far chosen for other methods
    :param max_size: maximum size of the training set (budget = max_size - min_size)
    :param results_percentage: if True shows the results as percentage of a model that has all the labels available
    :return: array of RMSE results for each learning step and for each run
    """
    all_results = []  # Storing results from all the runs
    simulation_run = 0

    while True:
        results = []  # Storing results of the current run
        predictions = []  # Storing predictions

        # Allocating data to: test, train and stream
        if representative_test:
            current_series = data[data["RUN"] == simulation_run].drop("RUN", axis=1)
            test_set = current_series.sample(n=n_obs_test, random_state=seeds[simulation_run])
            # test_set = current_series[current_series.index % 2 != 0].iloc[:n_obs_test, :]
            current_series = current_series.drop(test_set.index)
            current_series = current_series.reset_index(drop=True)
            # val_set = current_series.iloc[:n_obs_val, :]
            val_set = current_series.sample(n=n_obs_val, random_state=seeds[simulation_run])
            current_series = current_series.drop(val_set.index)
            current_series = current_series.reset_index(drop=True)
            train_set_labeled = pd.DataFrame(columns=list(test_set))
            # stream_set = current_series.iloc[n_obs_val:, :]
            stream_set = current_series

        else:
            current_series = data[data["RUN"] == simulation_run].drop("RUN", axis=1)
            test_set = current_series.iloc[0:n_obs_test, :]
            val_set = current_series.iloc[n_obs_test:n_obs_test + n_obs_val, :]
            train_set_labeled = pd.DataFrame(columns=list(test_set))
            stream_set = current_series.iloc[n_obs_test + n_obs_val:, :]

        # seed =
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

        if preprocessing:
            projection = DimensionalityReductionPCA(train_set=val_set, number_components=pc)
            projection.fit()
            test_set = projection.transform(test_set)
            # val_set = projection.transform(val_set)
            train_set_labeled = projection.transform(train_set_labeled)
            stream = projection.transform(stream)

        # Initializing regression class
        regr = LinearRegression(fit_intercept=True)

        scaler = StandardScaler()

        # Splitting into X, y
        x_train = train_set_labeled.drop(["y"], axis=1)
        y_train = train_set_labeled["y"]
        x_test = test_set.drop(["y"], axis=1)
        y_test = test_set["y"]
        # x_val = val_set.drop(["y"], axis=1)

        # Fit and predict
        regr.fit(scaler.fit_transform(x_train), y_train)
        y_pred = regr.predict(scaler.transform(x_test))
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # regr.fit(x_train, y_train)
        # y_pred = regr.predict(x_test)
        # rmse = np.sqrt(mean_squared_error(y_pred, y_test))
        if store_predictions:
            predictions.append(y_pred)

        # Benchmark RMSE: results we would get with 2000 labels available (approx. the number of obs spanned)
        if results_percentage:
            train_benchmark = stream.iloc[:2000, :]
            x_benchmark = train_benchmark.drop(["y"], axis=1)
            y_benchmark = train_benchmark["y"]
            regr_benchmark = LinearRegression()
            regr_benchmark.fit(x_benchmark, y_benchmark)
            y_pred_benchmark = regr_benchmark.predict(x_test)
            rmse_benchmark = np.sqrt(mean_squared_error(y_pred_benchmark, y_test))
            results.append(rmse_benchmark / rmse * 100)
        else:
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
                # regr.fit(x_train, y_train)
                # y_pred = regr.predict(x_test)
                # rmse = np.sqrt(mean_squared_error(y_pred, y_test))
                if results_percentage:
                    results.append(rmse_benchmark / rmse * 100)
                else:
                    results.append(rmse)
                if store_predictions:
                    predictions.append(y_pred)
                print("RAND", '\t', "Replication: {:3}".format(simulation_run + 1), '\t', "RMSE: {:5.3f}".format(rmse), '\t',
                      "Labeled samples: {:5}".format(x_train.shape[0]))

                if len(results) == max_size - min_size:
                    all_results.append(results)
                    simulation_run += 1

        if len(all_results) == reps:
            break

    if store_predictions:
        return y_test, predictions, train_set_labeled
    else:
        return np.array(all_results)
