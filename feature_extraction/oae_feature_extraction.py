import numpy as np
import pandas as pd
from feature_extraction.oae_fit_transform import DimensionalityReductionOAE
import os
import random
import torch


def dataset_preprocessing(data, reps):
    macro_df = []
    for simulation_run in range(0, reps+1):
        current_series = data[data["RUN"] == simulation_run].drop("RUN", axis=1)
        single_df = projection.transform(current_series)
        single_df["RUN"] = simulation_run
        macro_df.append(single_df)
        print(simulation_run)

    return pd.concat(macro_df)


# TEP: we use Stream9A, Stream9E (Purge) and Stream11D, Stream11E (Product)
df = pd.read_csv("/Users/dcac/Desktop/PhD/Data/TEP/Extended/tep_extended_compositions_1min.csv")
cols_to_drop = ['Stream9A', 'Stream9B', 'Stream9C', 'Stream9D', 'Stream9E', 'Stream9F', 'Stream9G', 'Stream9H',
                'Stream11D', 'Stream11E', 'Stream11F', 'Stream11G', 'Stream11H']
outcome = 'Stream9E'
cols_to_drop_updated = [col for col in cols_to_drop if col != outcome]
df = df.drop(cols_to_drop_updated, axis=1)
df.rename(columns={outcome: 'y', }, inplace=True)

# Taking last run to fit OAE
training_set = df[df["RUN"] == 59].drop("RUN", axis=1)

# Fitting OAE
projection = DimensionalityReductionOAE(initial_train_set=training_set)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
projection.fit(encoding_layers=[16, 160, 80, 40, 20, 10], penalty_term=0.1, nr_epochs=1000, patience=10, batch=1000, verbose=True)

# Test and save transformed datasets
new_dat = dataset_preprocessing(data=df, reps=58)
new_dat.to_csv("TEP_9E_OAE_"+str(i)+"_features.csv", index=False)
