import pandas as pd
"""
The process data is generated using the MATLAB code provided by Reinartz et al. (2022), the model for the simulation
is the Ricker mode, with no faults introduced during the operations. The initial seed is set equal to 42.
Model can be downloaded from here: https://www.researchgate.net/deref/https%3A%2F%2Fgithub.com%2Fdtuprodana%2FTEP
Paper can be found here: https://www.researchgate.net/deref/https%3A%2F%2Fonlinelibrary.wiley.com%2Fdoi%2Fabs%2F10.1002%2Fqre.2975 
"""

file_path = '/Users/dcac/Desktop/PhD/Data/TEP/Extended/TEP_30042022/Setup 1/Setup 1, run '
df = pd.read_csv(file_path+str(1)+".csv")

# We keep all the purge and product compositions
cols_to_drop = ['Time', 'BLANK1',

                # Manipulated variables
                'Stream2Valve', 'Stream3Valve', 'Stream1Valve', 'Stream4Valve',
                'CompressorRecycleValve', 'Stream9Valve', 'Stream10Valve', 'Stream11Valve', 'SteamFlowValve',
                'ReactorcCoolingWaterValve', 'CondenserCoolingWaterValve', 'AgitatorSpeedValve', 'BLANK2',

                # Extra controlled variables
                'ReactorPressure', 'ReactorLevel', 'SeparatorLevel', 'StripperLevel', 'Stream11', 'CompressorWork',

                # Reactor feed composition
                'Stream6A', 'Stream6B', 'Stream6C', 'Stream6D', 'Stream6E', 'Stream6F',

                # Extra purge composition (Stream9C and Stream9E have been used for AL)
                # 'Stream9A', 'Stream9B', 'Stream9C', 'Stream9D', 'Stream9E', 'Stream9F', 'Stream9G', 'Stream9H',

                # Product composition
                # 'Stream11D', 'Stream11E', 'Stream11F', 'Stream11G', 'Stream11H',
                'BLANK3',

                # Fault flags
                'IDV1', 'IDV2', 'IDV3', 'IDV4', 'IDV5', 'IDV6', 'IDV7', 'IDV8', 'IDV9', 'IDV10', 'IDV11', 'IDV12',
                'IDV13', 'IDV14', 'IDV15', 'IDV16', 'IDV17', 'IDV18', 'IDV19', 'IDV20', 'IDV21', 'IDV22', 'IDV23',
                'IDV24', 'IDV25', 'IDV26', 'IDV27', 'IDV28']

# Sampling rates: 52=1 min, 104=2min, 156=3min
datasets = []
for i in range(1, 61):
    df = pd.read_csv(file_path+str(i)+".csv")
    df = pd.DataFrame(df[df.index % 52 == 0])  # taking samples at higher sampling rates (156 for approx. 3 minutes)
    df = df.drop(cols_to_drop, axis=1)
    df = df.reset_index(drop=True)
    df["RUN"] = i-1
    datasets.append(df)
    print(i)
data = pd.concat(datasets)
data = data.to_csv("tep_extended_compositions_1min.csv", index=False)

# df = df.drop(["Stream9C"], axis=1)
# df.rename(columns={'Stream9E': 'y', }, inplace=True)
# df_reduced = df.drop(cols_to_drop, axis=1)
