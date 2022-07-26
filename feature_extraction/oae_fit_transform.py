import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from feature_extraction.oae import OrthogonalAutoEncoder, TrainingModel


class DimensionalityReductionOAE:

    def __init__(self, initial_train_set):
        """ Initial_train_set : DataFrame, variable to be predicted should be named "y" """
        self.training_set = initial_train_set
        self.scaler = StandardScaler()

    def fit(self, encoding_layers=[30, 20, 10], penalty_term=1, nr_epochs=1000, patience=10, batch=1000,
            verbose=True):

        # Initialize autoencoder
        self.autoencoder = OrthogonalAutoEncoder(encoding_layers=encoding_layers)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.autoencoder = self.autoencoder.to(device)

        # Drop y and scale initial training set
        x = torch.Tensor(pd.DataFrame(self.scaler.fit_transform(self.training_set.drop(["y"], axis=1))).values)

        # Initialize training object and prepare data (sequences for LSTM)
        self.train = TrainingModel(penalty=penalty_term, learning_rate=1e-3, batch_size=batch, val_size=0.20,
                                   verbose=verbose)
        self.train.create_datasets(train_data=x)

        # Trained model and losses
        self.trained_autoencoder, losses = self.train.train_model(self.autoencoder, patience=patience, n_epochs=nr_epochs)

        # Plot of losses
        plt.plot(losses[train], color="c", label="Train Loss")
        plt.plot(losses[val], color="r", label="Valid Loss")
        plt.legend()

        return

    def transform(self, data):
        """ Data : DataFrame, it should contain both X and y """
        # Reducing dimensionality of process variables
        self.trained_autoencoder.eval()
        with torch.no_grad():
            x = torch.Tensor(pd.DataFrame(self.scaler.transform(data.drop(["y"], axis=1))).values)
            # Feeding X into OAE and concatenating y
            x_compressed, _, _ = self.trained_autoencoder(x)
            x_compressed = x_compressed.numpy()
            y = pd.DataFrame(data["y"]).reset_index(drop=True)
            x_compressed = pd.DataFrame(np.array(x_compressed)).reset_index(drop=True)
            x_compressed = pd.concat([x_compressed, y], axis=1, ignore_index=True)
            x_compressed.columns = ["Feature_" + str(x + 1) for x in range(x_compressed.shape[1] - 1)] + ["y"]

        return x_compressed
