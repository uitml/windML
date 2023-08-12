import torch
import torch_geometric
from torch_geometric.transforms import NormalizeFeatures

from basic_gat import Mini_gnn
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simple_gnn_data_preprocessing import frost_api_to_gnn_dataset


def custom_MSE(out, target):
    '''
    generated custom mse to silence warnings from pytorch implementations
    '''
    return ((out - target) ** 2).mean()


def train(model, device, dataset, optimizer, loss_func, mask=None):
    '''
    Function to train the model

    params:
        model: The model to train (pytorch)
        device: trining device i.e 'cuda', 'cpu', 'mps'
        dataset: dataset to use for training
        optimizer: the optimizer to use
        loss_func: loss function for training 
        mask: can be passed for testing the model on a certain parto of the model.
    '''

    if mask is None:
        model.train()
        optimizer.zero_grad()

        outputs = model(dataset.x.to(device), dataset.edge_index.to(device), dataset.edge_attr.to(device))

        loss = loss_func(outputs, dataset.y.to(device))
        loss.backward()
        optimizer.step()
    else:
        model.eval()

        with torch.no_grad():
            outputs = model(dataset.x.to(device), dataset.edge_index.to(device), dataset.edge_attr.to(device))
            loss = loss_func(outputs[mask], dataset.y[:,0].to(device)[mask])

    return loss.item(), outputs


if __name__ == '__main__':
    path = 'data/'
    adj_df = pd.read_csv(f'{path}/station_distance_matrix.csv', index_col=0)
    data = pd.read_csv(f'{path}/data.csv', index_col=(0,1))
    data.index = data.index.set_levels(pd.to_datetime(data.index.levels[1]), level=1)   

    dataset = frost_api_to_gnn_dataset(data, adj_df, 'wind_speed', n_skips=5, temporal_strength=1)

    model = Mini_gnn(dataset)

    device = 'cpu'
    model.to(device)

    num_epochs = 300

    num_training_steps = num_epochs
    progress_bar = tqdm(range(num_training_steps))

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for e in range(num_epochs):
        loss, out = train(model, device, dataset, optim, custom_MSE)

        progress_bar.set_description(f"loss mse: {loss}")
        progress_bar.update(1)

    print(out[:5])
    print("\n------------\n")
    print(dataset.y[:5])