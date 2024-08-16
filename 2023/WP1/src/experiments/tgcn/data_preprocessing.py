import numpy as np
import pandas as pd


def retrieve_data(path):
    adj_df = pd.read_csv(f'{path}/station_distance_matrix.csv', index_col=0)

    data = pd.read_csv(f'{path}/data.csv', index_col=(0,1))
    data.index = data.index.set_levels(pd.to_datetime(data.index.levels[1]), level=1)
    meta = pd.read_csv(f'{path}/meta.csv', index_col=0)

    data_nonna = data.groupby([pd.Grouper(level=0),
                               pd.Grouper(freq='H', level=1)]
                              ).max()

    data_nonna.index.get_level_values(1).value_counts()
    data_nonna = data_nonna[~data_nonna.isna().any(axis=1)]

    station_ids = data.index.get_level_values(0).unique()
    data_nonna = data_nonna.loc[station_ids]

    processed_data = process_data(data_nonna)
    adj_arr = process_adj(adj_df, station_ids)

    return processed_data, adj_arr


def process_data(data_df):

    n_nodes = len(data_df.index.get_level_values(0).unique())
    timesteps = len(data_df.index.get_level_values(1).unique())
    n_features = len(data_df.columns)

    data_array = np.zeros((n_features, timesteps, n_nodes))
    for i, col in enumerate(data_df.columns):
        temp = data_df[col].values.reshape((n_nodes, timesteps)).T
        print(col)
        data_array[i] = temp

    return data_array


def process_adj(adj_df, station_ids):
    adj_arr = adj_df[station_ids].loc[station_ids].values

    adj_arr = np.exp(-adj_arr / adj_arr.std())
    adj_arr[np.eye(len(adj_arr), dtype=bool)] -= 1

    return adj_arr


def create_regression_dataset(data: np.ndarray,
                              target: np.ndarray,
                              seq_length: int,
                              target_length: int):

    seq_data = data[:, :-target_length]
    target_data = target[seq_length:]

    iters = min(seq_data.shape[1] - seq_length,
                target_data.shape[0] - target_length)

    sequences = np.zeros((iters, seq_data.shape[0], seq_length, seq_data.shape[-1]))
    targets = np.zeros((iters, target_length, target_data.shape[-1]))

    for i in range(iters):

        sequences[i] = seq_data[:, i: i+seq_length]
        targets[i] = target_data[i:i+target_length, :]

    sequences = np.swapaxes(sequences, 0, 1)
    return sequences, targets


def split_data(data: np.ndarray, split: float = 0.20):
    ind = int(split * data.shape[1])

    return data[:, :-ind, :], data[:, -ind:, :]


if __name__ == '__main__':
    dat, adj = retrieve_data('data/custom/')
    seq, tar = create_regression_dataset(dat, dat[-1], 11, 3)

    print(seq.shape, tar.shape, adj.shape)