"""
Function for sampling of process variables and corresponding quality information based on the T^2 statistics, i.e. the
Mahalanobis distance between incoming observations and the currently labeled dataset
@author: Davide Cacciarelli
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde, norm
from scipy.optimize import brentq


def t2_sampling(data, reps, n_obs_test, n_obs_val, min_size, max_size, alpha=0.95):
    """
    Sampling based on the Hotelling T^2 statistics
    :param data: dataset containing multiple runs
    :param reps: number of times the procedure runs
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

        try:
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
                    starting_point = i + 1
                    stream = stream_set.iloc[starting_point:, :]
                    break
                if sampling_index[i] >= alpha:
                    # selecting i-th sample and adding it to the labeled dataset
                    sample_to_add = pd.DataFrame(np.array(stream_set.iloc[i, :]).reshape(1, -1),
                                                 columns=list(train_set_labeled))
                    train_set_labeled = train_set_labeled.append(sample_to_add, ignore_index=True)

            # Initializing regression class
            regr = LinearRegression(fit_intercept=True)

            # Splitting into X, y
            x_train = train_set_labeled.drop(["y"], axis=1)
            y_train = train_set_labeled["y"]
            x_test = test_set.drop(["y"], axis=1)
            y_test = test_set["y"]

            # Fit and predict
            regr.fit(x_train, y_train)
            y_pred = regr.predict(x_test)
            rmse = np.sqrt(mean_squared_error(y_pred, y_test))
            results.append(rmse)

            # Getting Hotelling T2 statistics
            feat = np.array(x_train)  # Phase I data array corresponding to the currently labeled train set
            colmean = np.mean(feat, axis=0)
            matinv = np.linalg.inv(np.cov(feat.T))
            t2_statistics = np.array([(sample - colmean).T @ matinv @ (sample - colmean) for sample in feat])

            # Computing UCL 95% with KDE
            kde = gaussian_kde(t2_statistics)
            band_width = kde.covariance_factor() * t2_statistics.std()
            upper_control_limit = brentq(
                f=lambda x: sum(norm.cdf((x - t2_statistics) / band_width)) / len(t2_statistics) - alpha,
                a=-10, b=1e10, maxiter=1000)

            maxit = len(stream)
            for i in range(maxit):
                if x_train.shape[0] == max_size - 1:
                    break
                zi_yi = pd.DataFrame(stream.iloc[i, :]).T
                zi = zi_yi.iloc[:, :-1]

                zi = np.array(zi).reshape(-1, )
                t2_new_instance = np.array((zi - colmean).T @ matinv @ (zi - colmean))

                if t2_new_instance >= upper_control_limit:
                    train_set_labeled = train_set_labeled.append(zi_yi, ignore_index=True)
                    # Splitting into X, y
                    x_train = train_set_labeled.drop(["y"], axis=1)
                    y_train = train_set_labeled["y"]
                    regr.fit(x_train, y_train)
                    y_pred = regr.predict(x_test)
                    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
                    results.append(rmse)

                    # Updating Hotelling T2 statistics
                    colmean = np.mean(feat, axis=0)
                    matinv = np.linalg.inv(np.cov(feat.T))
                    t2_statistics = np.array([(sample - colmean).T @ matinv @ (sample - colmean) for sample in feat])

                    # Computing new UCL 95% with KDE
                    kde = gaussian_kde(t2_statistics)
                    band_width = kde.covariance_factor() * t2_statistics.std()
                    upper_control_limit = brentq(
                        f=lambda x: sum(norm.cdf((x - t2_statistics) / band_width)) / len(t2_statistics) - alpha,
                        a=-10, b=1e10, maxiter=1000)

            if len(results) == max_size - min_size:
                all_results.append(results)
            simulation_run += 1

        except Exception as e:
            print(e)
            simulation_run += 1

        if len(all_results) == reps:
            break

    return np.array(all_results)
