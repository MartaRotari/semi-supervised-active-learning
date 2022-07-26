"""
Function for sampling of process variables and corresponding quality information based on the expected model change
@author: Davide Cacciarelli
"""

import numpy as np
import pandas as pd
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde, norm
from scipy.optimize import brentq
from models.ensemble import bootstrap_models
from models.pca_fit_transform import DimensionalityReductionPCA


def compute_model_change(main_model, ensemble, x_new):
    """
    Computes the expected model change
    :param main_model: current linear regression model
    :param ensemble: committee of bootstrap models
    :param x_new: instances for which the expected model change should be computed
    :return: expected model change scores for selected instances
    """
    # getting predictions from the "main" model
    f_x_new = main_model.predict(x_new)

    # list to store the derivatives norm
    norm_of_derivatives = []

    # iterating through the ensemble of bootstrapped models
    for model in ensemble:
        # predictions from the bootstrapped models (to approximate predictive distribution)
        label = model.predict(x_new)
        derivative = np.array(f_x_new - label).reshape(-1, 1) * np.array(x_new)
        norm_of_derivatives.append(np.linalg.norm(derivative, axis=1))

    norm_of_derivatives = np.array(norm_of_derivatives).T
    score = np.sum(norm_of_derivatives, axis=1) / len(ensemble)

    return np.array(score)


def emc_sampling(data, reps, n_obs_test, n_obs_val, min_size, max_size, seeds, preprocessing=False,
                 pc=0.85, results_percentage=False, representative_test=False, alpha=0.95):
    """
    Sampling bsed on the expected model change
    :param pc: number of components for the PCA dimensionality reduction
    :param data: dataset containing multiple runs
    :param reps: number of times the procedure runs
    :param n_obs_test: number of observations in each run to be allocated to the test set
    :param n_obs_val: number of observations in each run to be allocated to the validation set
    :param min_size: number of labeled observations initially available to the learner
    :param max_size: maximum size of the training set (budget = max_size - min_size)
    :param seeds: a list of seeds to be used as starting list for each run for the bootstrap sampling
    :param results_percentage: if True shows the results as percentage of a model that has all the labels available
    :return: array of RMSE results for each learning step and for each run
    """
    all_results = []  # Storing results from all the runs
    simulation_run = 0

    while True:

        try:
            results = []  # Storing results of the current run

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

            if preprocessing:
                projection = DimensionalityReductionPCA(train_set=val_set, number_components=pc)
                projection.fit()
                test_set = projection.transform(test_set)
                val_set = projection.transform(val_set)
                train_set_labeled = projection.transform(train_set_labeled)
                stream = projection.transform(stream)

            # Initializing regression class
            regr = LinearRegression(fit_intercept=True)

            # Splitting into X, y
            x_train = train_set_labeled.drop(["y"], axis=1)
            y_train = train_set_labeled["y"]
            x_test = test_set.drop(["y"], axis=1)
            y_test = test_set["y"]
            x_val = val_set.drop(["y"], axis=1)

            # Fit and predict
            regr.fit(x_train, y_train)
            y_pred = regr.predict(x_test)
            rmse = np.sqrt(mean_squared_error(y_pred, y_test))

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

            # Getting expected model change score
            bootstrapped_models = bootstrap_models(10, x_train, y_train, sampling_seed=seeds[simulation_run])
            emcm = compute_model_change(regr, bootstrapped_models, x_val)

            # Computing UCL 95% with KDE
            kde = gaussian_kde(emcm)
            band_width = kde.covariance_factor() * emcm.std()
            upper_control_limit = brentq(
                f=lambda x: sum(norm.cdf((x - emcm) / band_width)) / len(emcm) - alpha,
                a=-10, b=1e10, maxiter=1000)

            maxit = len(stream)
            for i in range(maxit):
                if x_train.shape[0] == max_size - 1:
                    break
                zi_yi = pd.DataFrame(stream.iloc[i, :]).T
                zi = zi_yi.iloc[:, :-1]
                # start_time = time.time()
                emcm_zi = compute_model_change(regr, bootstrapped_models, zi)
                # time_to_decide = time.time() - start_time
                if emcm_zi >= upper_control_limit:
                    train_set_labeled = train_set_labeled.append(zi_yi, ignore_index=True)
                    # Splitting into X, y
                    x_train = train_set_labeled.drop(["y"], axis=1)
                    y_train = train_set_labeled["y"]
                    regr.fit(x_train, y_train)
                    y_pred = regr.predict(x_test)
                    rmse = np.sqrt(mean_squared_error(y_pred, y_test))
                    if results_percentage:
                        results.append(rmse_benchmark / rmse * 100)
                    else:
                        results.append(rmse)
                    print("EMC", '\t', "Replication: {:3}".format(simulation_run + 1), '\t',
                          "RMSE: {:5.3f}".format(rmse), '\t',
                          "Labeled samples: {:5}".format(x_train.shape[0]))

                    # Getting exepcted model change scores
                    bootstrapped_models = bootstrap_models(10, x_train, y_train, sampling_seed=seeds[simulation_run])
                    emcm = compute_model_change(regr, bootstrapped_models, x_val)

                    # Computing UCL 95% with KDE
                    kde = gaussian_kde(emcm)
                    band_width = kde.covariance_factor() * emcm.std()
                    upper_control_limit = brentq(
                        f=lambda x: sum(norm.cdf((x - emcm) / band_width)) / len(emcm) - alpha,
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
