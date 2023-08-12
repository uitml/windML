import pandas as pd
import numpy as np

from sklearn.neighbors import kneighbors_graph
from dataclasses import dataclass

from torch_geometric.data import Data
from torch import tensor


def generate_knn_embedding_multiedge(distance_df: pd.DataFrame,
                           t: int, station_ids: np.ndarray = None,
                           k: int = 4,
                           temporal_strength: float = 0.5,
                           n_skips: int = 1
                           ) -> pd.DataFrame:
    if station_ids is not None:
        distance_df = distance_df[station_ids].loc[station_ids]

    station_order = distance_df.index

    coo_emb = kneighbors_graph(distance_df.values, k, mode='distance').tocoo()
    coo_emb.data = np.exp(-coo_emb.data / coo_emb.data.std())

    coo_emb.data = np.vstack((coo_emb.data, np.zeros(len(coo_emb.data)))).T

    shift_value = coo_emb.row.max() + 1
    unique_nodes = np.unique(coo_emb.row)

    orgrow = np.copy(coo_emb.row)
    orgcol = np.copy(coo_emb.col)
    orgdata = np.copy(coo_emb.data)



    for t in range(1, t):
        skiprow = unique_nodes + shift_value * (t-1)
        skipcol = unique_nodes + shift_value * t
        for skip in range(2, min(n_skips + 1, t)):
            skiprow = np.hstack((skiprow, unique_nodes + shift_value * (t-skip)))
            skipcol = np.hstack((skipcol, unique_nodes + shift_value * t))

        skipdata = np.zeros((len(skiprow), 2))
        skipdata[:, 1] += temporal_strength,

        coo_emb.row = np.hstack((coo_emb.row,
                                 skiprow,
                                 orgrow + shift_value * t
                                 ))
        coo_emb.col = np.hstack((coo_emb.col,
                                 skipcol,
                                 orgcol + shift_value * t
                                 ))

        coo_emb.data = np.vstack((coo_emb.data,
                                  skipdata,
                                  orgdata
                                  ))

    return coo_emb, station_order


def generate_temporal_feature_embedding(data_df: pd.DataFrame, station_ids: np.ndarray = None):
    if station_ids is not None:
        data_df = data_df.loc[station_ids]

    timestamps = data_df.index.levels[1]

    query = data_df.query(f'time == "{timestamps[0]}"')
    query = query.droplevel('time')

    feature_matrix = query.values

    for time in timestamps[1:]:
        query = data_df.query(f'time == "{time}"')
        query = query.droplevel('time')

        # temp_matrix = query.join(meta_df['heigh-asl (m)'].loc[query.index]).values

        feature_matrix = np.vstack((feature_matrix, query.values)) # np.vstack((feature_matrix, temp_matrix))

    return feature_matrix


def generate_target_embeding(data_df: pd.DataFrame,
                             target_colname: str,
                             lookahead: int = 3,
                             station_ids: np.ndarray = None):
    if station_ids is not None:
        data_df = data_df.loc[station_ids]

    start_stamps = data_df.index.levels[1][:-lookahead]
    end_stamps = data_df.index.levels[1][lookahead:]
    dates = data_df.index.get_level_values(1)

    mask = mask = (dates >= start_stamps[0]) & (dates < end_stamps[0])

    target_matrix = data_df[mask][target_colname].values.reshape(len(data_df.index.get_level_values(0).unique()), lookahead)

    for start, end in zip(start_stamps[1:], end_stamps[1:]):

        mask = (dates >= start) & (dates < end)

        temp_matrix = data_df[mask][target_colname].values.reshape(len(data_df.index.get_level_values(0).unique()), lookahead)

        target_matrix = np.vstack((target_matrix, temp_matrix))

    return target_matrix


def frost_api_to_gnn_dataset(data_df: pd.DataFrame,
                             distance_df: pd.DataFrame,
                             target_colname: str,
                             station_ids: np.ndarray = None,
                             lookahead: int = 3,
                             k: int = 4,
                             temporal_strength: float = 0.5,
                             n_skips: int = 1
                             ) -> object:

    if station_ids is None:
        station_ids = data_df.index.get_level_values(0).unique()

    timesteps = len(data_df.index.levels[-1]) - lookahead

    temporal_knn, station_order = generate_knn_embedding_multiedge(distance_df, timesteps, station_ids, k, temporal_strength)    

    sorted_data_df = data_df.loc[station_order]

    feature_embedding = generate_temporal_feature_embedding(sorted_data_df)
    feature_embedding = feature_embedding[:-len(station_ids) * lookahead]

    target_embedding = generate_target_embeding(sorted_data_df, target_colname)

    print(np.unique(temporal_knn.row).shape)
    print(feature_embedding.shape)
    print(target_embedding.shape)

    edge_coo = np.vstack((temporal_knn.row, temporal_knn.col))
    edge_feat = temporal_knn.data

    dataset = Data(x=tensor(feature_embedding).float(),
                   edge_index=tensor(edge_coo).long(),
                   edge_attr=tensor(edge_feat).float(),
                   y=tensor(target_embedding),
                   num_classes=target_embedding.shape[1]
                   )

    return dataset


if __name__ == '__main__': 
    pass