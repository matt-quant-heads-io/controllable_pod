import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.utils import np_utils


def scale_data(data_col):
    scaler = MinMaxScaler()
    scaler.fit(data_col)
    return np.array(scaler.transform(data_col))


def get_train_data_from_csv(obs_size, data_size):
    pod_root_path = f"training_trajectories/traj_obs_{obs_size}_goal_size_{data_size}"

    dfs = []
    X = []

    for file in os.listdir(pod_root_path):
        print(f"compiling df {file}")
        df = pd.read_csv(f"{pod_root_path}/{file}")
        dfs.append(df)

    df = pd.concat(dfs)

    df = df.sample(frac=1).reset_index(drop=True)
    y_true = df[['target']]
    y = np_utils.to_categorical(y_true)
    df.drop('target', axis=1, inplace=True)
    y = y.astype('int32')

    # map conditions
    num_regions = scale_data(df[['num_regions']])
    df.drop('num_regions', axis=1, inplace=True)
    num_enemies = scale_data(df[['num_enemies']])
    df.drop('num_enemies', axis=1, inplace=True)
    nearest_enemy = scale_data(df[['nearest_enemy']])
    df.drop('nearest_enemy', axis=1, inplace=True)
    path_length = scale_data(df[['path_length']])
    df.drop('path_length', axis=1, inplace=True)

    maps_conditions = np.column_stack((num_regions, num_enemies, nearest_enemy, path_length))

    for idx in range(len(df)):
        x = df.iloc[idx, :].values.astype('float32').reshape((obs_size, obs_size, 8))
        X.append(x)

    X = np.array(X)

    return [X, maps_conditions], y
