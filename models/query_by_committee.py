"""
Function for sampling of process variables and corresponding quality information based on the ambiguity score, i.e. the
variance of the predictions made by an ensemble of models trained on bootstrapped replica of the training set
@author: Davide Cacciarelli
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde, norm
from scipy.optimize import brentq
from models.ensemble import bootstrap_models
from models.pca_fit_transform import DimensionalityReductionPCA


def compute_ambiguity(ensemble, x_new):
    """
    Computes the ambiguity of the committee on one instance
    :param ensemble: trained ensemble
    :param x_new: instance on which ambiguity is computed
    :return: variance of the predictions made by the committee members
    """
    # list to store the predictions
    predictions = []

    # iterating through the ensemble of bootstrapped models
    for model in ensemble:
        # predictions from the bootstrapped models (to approximate predictive distribution)
        predictions.append(model.predict(x_new))

    return np.var(predictions, axis=0)


def qbc_sampling(data, reps, n_obs_test, n_obs_val, min_size, max_size, seeds, representative_test=False,
                 preprocessing=False, pc=0.85, results_percentage=False, alpha=0.95, store_predictions=False):
    """
    Sampling based on the instance ambiguity
    :param data: dataset containing multiple runs
    :param reps: number of times the procedure runs
    :param n_obs_test: number of observations in each run to be allocated to the test set
    :param n_obs_val: number of observations in each run to be allocated to the validation set
    :param min_size: number of labeled observations initially available to the learner
    :param max_size: maximum size of the training set (budget = max_size - min_size)
    :param seeds: a list of seeds to be used as starting list for each run for the bootstrap sampling
    :param clustering: whether to pick the obs to build the first model through clustering
    :param radius: radius from cluster centroids that define the acceptance region for first samples
    :param preprocessing: only if not clustering, perform SS-PCR
    :param pc: number of principal components for PCA (can also be a float, indicating % of explained variance)
    :param results_percentage: if True shows the results as percentage of a model that has all the labels available
    :return: array of RMSE results for each learning step and for each run
    """
    all_results = []  # Storing results from all the runs
    simulation_run = 0
    predictions = []  # Storing predictions

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
            if store_predictions:
                predictions.append(y_pred)

            # Getting ambiguity score
            bootstrapped_models = bootstrap_models(10, x_train, y_train, sampling_seed=seeds[simulation_run])
            ambiguity = compute_ambiguity(bootstrapped_models, x_val)

            # Computing UCL 95% with KDE
            kde = gaussian_kde(ambiguity)
            band_width = kde.covariance_factor() * ambiguity.std()
            upper_control_limit = brentq(
                f=lambda x: sum(norm.cdf((x - ambiguity) / band_width)) / len(ambiguity) - alpha,
                a=-10, b=1e10, maxiter=1000)

            maxit = len(stream)
            for i in range(maxit):
                if x_train.shape[0] == max_size - 1:
                    break
                zi_yi = pd.DataFrame(stream.iloc[i, :]).T
                zi = zi_yi.iloc[:, :-1]
                ambiguity_zi = compute_ambiguity(bootstrapped_models, zi)
                if ambiguity_zi >= upper_control_limit:
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
                    if store_predictions:
                        predictions.append(y_pred)
                    print("QBC", '\t', "Replication: {:3}".format(simulation_run + 1), '\t',
                          "RMSE: {:5.3f}".format(rmse), '\t',
                          "Labeled samples: {:5}".format(x_train.shape[0]))

                    # Getting exepcted model change scores
                    bootstrapped_models = bootstrap_models(10, x_train, y_train, sampling_seed=seeds[simulation_run])
                    ambiguity = compute_ambiguity(bootstrapped_models, x_val)

                    # Computing UCL 95% with KDE
                    kde = gaussian_kde(ambiguity)
                    band_width = kde.covariance_factor() * ambiguity.std()
                    upper_control_limit = brentq(
                        f=lambda x: sum(norm.cdf((x - ambiguity) / band_width)) / len(ambiguity) - alpha,
                        a=-10, b=1e10, maxiter=1000)

            if len(results) == max_size - min_size:
                all_results.append(results)
            simulation_run += 1

        except Exception as e:
            print(e)
            simulation_run += 1

        if len(all_results) == reps:
            break

    if store_predictions:
        return y_test, predictions, train_set_labeled
    else:
        return np.array(all_results)
