# Online Active Learning for Soft Sensor Development using Semi-Supervised Autoencoders
Paper presented at the **ICML 2022 Workshop on Adaptive Experimental Design and Active Learning in the Real World**.

The full paper is available [here](https://realworldml.github.io/files/cr/paper12.pdf).

An orthogonal autoencoder is used to reduce the dimensionality of the input by nonlinear mapping.

<img src="https://user-images.githubusercontent.com/83544651/181049730-fa0e03f2-295f-492c-af62-087bda945eb8.png" width="80%" height="80%">

On top of this, we compare several strategies to acquire data points for linear regression, using the features extracted by the autoencoder as predictors.

<img src="https://user-images.githubusercontent.com/83544651/180979074-9145f42a-5106-4cfd-aa79-9de078f76827.png" width="80%" height="80%">

## Summary
This repo contains:
1. `data`: folder containing a file with instructions on how the TEP simulations were obtained and preprocessed.
2. `feature_extraction`: folder containing files for defining and training an autoencoder with an orthogonal regularization on its bottleneck.
3. `models`: folder containing files related to the different query strategies, namely `random.py`, `mahalanobis_distance.py`, `query_by_committee.py` and `expected_model_change.py`.

## Reproducing results
To ensure reproducibility, it is suggested to use environment `soft_sensor_environment.yml`. Two working notebooks to reproduce the results of the paper are provided:
1. `autoencoder_training.ipynb`: for thetraining of the autoencoer.
2. `results.ipynb`: the main results of the paper, showing how to obtain the learning curves related to semi-supervised learning and active learning.
