import torch
import numpy as np
from torch.optim import optimizer
import os

from tqdm.auto import tqdm

from model import TGCN_regression
from data_preprocessing import retrieve_data, create_regression_dataset, split_data


import matplotlib.pyplot as plt


def save_model_params(model, name):
    wd = os.getcwd()

    if not os.path.exists(os.path.join(wd, 'model_params')):
        os.mkdir(os.path.join(wd, 'model_params'))

    while os.path.exists(os.path.join(wd, 'model_params', name)):
        print(f"\n\nthe path '{name}' already exists")
        name = input("Give new savepath: ")

    full_path = os.path.join(wd, 'model_params', name)

    torch.save(model.state_dict(), full_path)

    return 1


def load_model_params(model, path):
    wd = os.getcwd()
    full_path = os.path.join(wd, path)

    model.load_state_dict(torch.load(full_path))

    return 1


def model_to_apple_gpu(model):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    if device == "cpu":
        return 0
    return device


def custom_MSE(out, target):
    return ((out - target) ** 2).mean()


def train_model(model, optimizer, data, targets):
    model.train()

    optimizer.zero_grad()

    out = model(data)

    loss = custom_MSE(out, targets)

    loss.backward()
    optimizer.step()

    return loss.item()


def test_model(model, data, targets):
    model.eval()
    with torch.no_grad():
        out = model(data)
    loss = custom_MSE(out, targets)

    return loss.item()


if __name__ == '__main__':

    sequence_length = 30
    e = 100

    dat, adj = retrieve_data('data')

    train, val = split_data(dat)

    train_seq, train_tar = create_regression_dataset(train, train[-1], sequence_length, 3)
    val_seq, val_tar = create_regression_dataset(val, val[-1], sequence_length, 3)

    pt_train_data = torch.from_numpy(train_seq[-1]).float()
    tar_train_data = torch.from_numpy(np.swapaxes(train_tar,1,2)).float()

    pt_val_data = torch.from_numpy(val_seq[-1]).float()
    tar_val_data = torch.from_numpy(np.swapaxes(val_tar,1,2)).float()

    model = TGCN_regression(adj, 64)

    e = 1000

    # ######TEST GPU AVAILABLE######

    device = model_to_apple_gpu(model)
    if not device:
        print('gpu not activated')
        exit()
    print("\n\n")
    print('model sent succesfully to gpu')

    pt_train_data = pt_train_data.to(device)
    tar_train_data = tar_train_data.to(device)

    pt_val_data = pt_val_data.to(device)
    tar_val_data = tar_val_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_training_steps = e
    progress_bar = tqdm(range(num_training_steps))

    saved_vals = np.zeros((2, e))
    for i in range(e):

        train_loss = train_model(model, optimizer, pt_train_data, tar_train_data)

        test_loss = test_model(model, pt_val_data, tar_val_data)

        saved_vals[0, i] = train_loss
        saved_vals[1, i] = test_loss

        progress_bar.set_description(f"(train - test)loss mse: {train_loss:.3f} -- {test_loss}")
        progress_bar.update(1)

    save_model_params(model, 'run.pt')

    plt.plot(saved_vals[0], label='train MSE')
    plt.plot(saved_vals[1], label='test MSE')

    plt.legend()
    plt.show()


