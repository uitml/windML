# Imports
import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap #pip install Basemap
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, PrefetchLoader
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing, GCNConv, GraphConv, GATConv
import torch.optim as SGD
from tqdm import tqdm



Goliat_lat = 71.3112
Goliat_lon = 22.25
# Choose k = 2, 4, or 8 for kNN-graph edges in output subgraph
output_k = 4

# Choose k = 2, 4, or 8 for kNN-graph edges in CARRA subgraph
carra_k = 4

#Directory paths
#Should only need to edit this path, the rest should be created automatically.
windgnn = r'C:\Users\SIGUR\OneDrive - UiT Office 365\Var24\prosjekt2'
csv = windgnn + '/csv/'
carra_folder = csv+ 'carra/'
s1_folder = csv+ 's1/'
output_edges = csv + 'output_edges/'
graph = csv+'graph/'
images = csv + 'images/'
models = csv + "models/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Check for and create model folder
if not os.path.exists(csv):
    os.makedirs(csv)
    print("Creating csv directory")
if not os.path.exists(models):
    os.makedirs(models)
    print("Creating models directory")
if not os.path.exists(images):
    os.makedirs(images)
    print("Creating images directory")
if not os.path.exists(graph):
    os.makedirs(graph)
    print("Creating graph directory")
if not os.path.exists(output_edges):
    os.makedirs(output_edges)
    print("Creating output_edges directory")





def initialize_output_features(output_grid, output_channels=1, method='constant'):
    """
    Initialize one column of features on the output grid so that messages can be passed to this feature from the CARRA grid.
    Args:
    - output_grid: output grid
    - output_channels: number of features to initialize (default = 1)
    - (initialization) method: 'constant' (default) or 'random'
    """

    if method == 'constant':
        return torch.full((output_grid.shape[0], output_channels), 0.5)
    elif method == 'random':
        return torch.rand(output_grid.shape[0], output_channels) * 0.1 + 0.45
    else:
        raise ValueError("Initialization method must be 'constant' or 'random'")

# Progress bar function cause why not
def progress_bar(n, total):
    scale = 120/total
    percent = round(n/total*100, 1)
    done = '█'*round(n*scale)
    todo = '-'*round((total - n)*scale)
    print(f"\r{percent}% [{done}{todo}]", end='')


# Function to check for NaNs in data (debugging purposes)
def check_for_nans(data_loader):
    for data in data_loader:
        if torch.isnan(data.x_dict['carra']).any() or torch.isnan(data.x_dict['output_grid']).any():
            print("Found NaNs in the input features")
        if torch.isnan(data.edge_attr_dict[('carra', 'to', 'output_grid')]).any():
            print("Found NaNs in the edge attributes")

def load_graph_data(timestamp, base_path='.', init_method='constant', output_channels=1, output_k=4, carra_k=4):
    """
    Function to correctly load all graph data for a given timestamp.
    Args:
    - timestamp: the timestamp as extracted from the filenames to then load the correct files
    - base_path: folder path to load the files from (default = '.')
    - init_method: specify initialization method for the output grid features ('constant' or 'random')
    - output_channels: specify number of features to add to output grid (default = 1)
    Returns:
    - Data object of the heterogeneous graph for given timestamp, including node types, edge types, edge weights, positions, and truth values.
    """
    time_step = timestamp
    if carra_k == 4:
        carra_base_path = base_path
        index = ''

    carra_x_path = os.path.join(graph, f's1_{timestamp}_carra_x_extended.pt')
    truth_x_path = os.path.join(graph, f's1_{timestamp}_truth_x_extended.pt')
    carra_edge_path = os.path.join(graph, f's1_{timestamp}_{index}carra_edge.pt')
    output_edge_path = os.path.join(csv, f'output_grid_{output_k}nn_edges.pt')
    carra_to_output_edges_path = os.path.join(output_edges, f's1_{timestamp}_carra_to_output_edges.pt')
    carra_to_output_attr_path = os.path.join(output_edges, f's1_{timestamp}_carra_to_output_attr.pt')

    carra_x = torch.load(carra_x_path)
    truth_x = torch.load(truth_x_path)
    carra_edge = torch.load(carra_edge_path)
    output_edge = torch.load(output_edge_path)
    carra_to_output_edges = torch.load(carra_to_output_edges_path)
    carra_to_output_attr = torch.load(carra_to_output_attr_path)

    # Turn carra-to-output edges from distances into weights and normalize
    carra_to_output_attr = 1.0 - (carra_to_output_attr / 2.5)


    # Define features, positions, and truth values
    carra_features = carra_x[:, 2:]  # All columns except lat/lon
    truth_features = initialize_output_features(truth_x, method=init_method, output_channels=output_channels)
    carra_lat_lon = carra_x[:, :2]
    truth_lat_lon = truth_x[:, :2]   # Only lat/lon columns
    truth_values = truth_x[:, 2]

    sin = np.sin(np.deg2rad(carra_features[:,1:2]*360))
    cos = np.cos(np.deg2rad(carra_features[:,1:2]*360))
    new_carra_features = np.concatenate((carra_features[:,0:1], sin, cos, carra_features[:,2:3], carra_features[:,3:4]), axis=1)
    carra_features = torch.tensor(new_carra_features)

    # Confuse yourself by constantly using two names for the same thing
    output_grid_features = truth_features

    # Create Data object
    data = Data(
        x_dict={
            'carra': carra_features,  # Node features for 'carra' excluding lat/lon
            'output_grid': output_grid_features  # Initialized node features for 'output_grid'
        },
        edge_index_dict={
            ('carra', 'to', 'carra'): carra_edge,  # Edge index for 'carra' subgraph
            ('output_grid', 'to', 'output_grid'): output_edge,  # Edge index for 'output_grid' subgraph
            ('carra', 'to', 'output_grid'): carra_to_output_edges  # Edge index for bipartite subgraph
        },
        edge_attr_dict={
            ('carra', 'to', 'output_grid'): carra_to_output_attr # Edge weights for edges from 'carra' to 'output_grid'
        },
        pos_dict={
            'carra': carra_lat_lon, # Position for 'CARRA' nodes (lat/lon)
            'output_grid': truth_lat_lon  # Positions for 'output_grid' nodes (lat/lon)
        },
        y_dict={
            'truth': truth_values # Ground truth values
        },
        t_dict={
            'timestamp': time_step
        }
    )
    return data

def split_dataset(data_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the dataset randomly into training, validation and test set
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."

    random.shuffle(data_list)

    train_size = int(train_ratio * len(data_list))
    val_size = int(val_ratio * len(data_list))

    train_data = data_list[:train_size]
    val_data = data_list[train_size:train_size + val_size]
    test_data = data_list[train_size + val_size:]

    return train_data, val_data, test_data

def scaled_MSE(prediction, target, epsilon=4, tau=0.425):
    if not (target.size() == prediction.size()):
        warnings.warn("Prediction and target size are not the same.")
    prediction.detach().numpy()
    target.detach().numpy()
    loss = 0
    for i in range(len(prediction)):
        if prediction[i] < target[i]:
            tau = 1 - tau
        beta = (epsilon
                + prediction[i])/(epsilon + target[i])
        loss += (prediction[i] - beta*target[i])**2 * tau
    return loss / len(prediction)

# Functions to load and test a trained model
def load_model(model, model_filename):
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    return model

def test_model(model, test_loader):
    model.eval()
    predictions = []
    timestamps = []
    with torch.no_grad():
        for data in test_loader:
            time_step = data.t_dict['timestamp']
            x_carra = data.x_dict['carra']
            x_output = data.x_dict['output_grid']
            edge_i_c2c = data.edge_index_dict[('carra', 'to', 'carra')]
            edge_i_c2o = data.edge_index_dict[('carra', 'to', 'output_grid')]
            edge_i_o2o = data.edge_index_dict[('output_grid', 'to', 'output_grid')]
            edge_attr_c2o = data.edge_attr_dict[('carra', 'to', 'output_grid')]
            out, timestamp = model(x_carra, x_output, edge_i_c2c, edge_i_c2o, edge_i_o2o, edge_attr_c2o, time_step)
            predictions.append(out.cpu().numpy())
            timestamps.append(timestamp)
    timestamps = np.array(timestamps)
    return predictions, timestamps


def import_data(force_reload=False):
    #This function will load the dataloader if it exists, otherwise it will create it using load_graph_data and split_dataset.
    #Set reload = True to force creation and overwriting of existing saved dataloader.
    dataloader_reload = force_reload
    if dataloader_reload == False and os.path.exists(csv+"dataloader/train_data.pt") and os.path.exists(csv+"dataloader/val_data.pt") and os.path.exists(csv+"dataloader/test_data.pt"):
        print("Dataloader found, loading")
        train_loader = torch.load(csv+"dataloader/train_data.pt")
        val_loader = torch.load(csv+"dataloader/val_data.pt")
        test_loader = torch.load(csv+"dataloader/test_data.pt")
    else:
        print("Dataloader not found, creating dataloader.")
        # Load output grid (51 x 51 points)
        output_grid = torch.load(csv+'output_grid_50_50.pt')
        # Load suitable timestamps (where output grid overlaps with SAR scenes)
        with open(csv+'timestamps_yes.txt', 'r') as f:
            suitable_timestamps = [line.strip() for line in f]
        # Load data list
        data_list = [load_graph_data(timestamp, output_k=output_k, carra_k=carra_k) for timestamp in suitable_timestamps]

        # Split dataset
        train_data, val_data, test_data = split_dataset(data_list)

        # Create DataLoader instances
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        #Check for and create data folder
        dl = csv+"dataloader/"
        if not os.path.exists(dl):
            os.makedirs(dl)
        #Save the data
        torch.save(train_data, dl+f'train_data.pt')
        torch.save(val_data, dl+f'val_data.pt')
        torch.save(test_data, dl+f'test_data.pt')
    
    trl = PrefetchLoader(train_loader, device=device)
    vl = PrefetchLoader(val_loader, device=device)
    tel = PrefetchLoader(test_loader, device=device)

    return trl, vl, tel

    # Note: ab 2022-07-01_16 down: 8310 edges, davor 8317.

# Function to initialize model weights with Xavier initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

def evaluate(model, loader, loss_fn):
    """
    Standard evaluation loop for WindGNN.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in loader:
            time_step = data.t_dict['timestamp']
            x_carra = data.x_dict['carra']
            x_output = data.x_dict['output_grid']
            edge_i_c2c = data.edge_index_dict[('carra', 'to', 'carra')]
            edge_i_c2o = data.edge_index_dict[('carra', 'to', 'output_grid')]
            edge_i_o2o = data.edge_index_dict[('output_grid', 'to', 'output_grid')]
            edge_attr_c2o = data.edge_attr_dict[('carra', 'to', 'output_grid')]
            out, _ = model(x_carra, x_output, edge_i_c2c, edge_i_c2o, edge_i_o2o, edge_attr_c2o, time_step)
            y = data.y_dict['truth']
            y = y.view(out.size()) # Ensure out and y have the same size
            y = y.to(device)
            loss = loss_fn(out, y) # scaled_MSE(out, y)
            total_loss += loss.item()
            timestamp = data.t_dict['timestamp']
    return total_loss / len(loader), timestamp


def train(model, train_loader, val_loader, optimizer, loss_fn, epochs=100, clip_value=10.0, version=None, sc = None):
    """
    Pretty standard training loop for WindGNN.
    """
    scheduler = sc
    model.train()
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        total_loss = 0
        # print(f"WORKING ON EPOCH {epoch + 1}:")

        for counter, data in enumerate(train_loader):
            optimizer.zero_grad()
            time_step = data.t_dict['timestamp']
            x_carra = data.x_dict['carra']
            x_output = data.x_dict['output_grid']
            edge_i_c2c = data.edge_index_dict[('carra', 'to', 'carra')]
            edge_i_c2o = data.edge_index_dict[('carra', 'to', 'output_grid')]
            edge_i_o2o = data.edge_index_dict[('output_grid', 'to', 'output_grid')]
            edge_attr_c2o = data.edge_attr_dict[('carra', 'to', 'output_grid')]

            out, _ = model(x_carra, x_output, edge_i_c2c, edge_i_c2o, edge_i_o2o, edge_attr_c2o, time_step)

            y = data.y_dict['truth']
            y = y.view(out.size())
            y.to(device)
            loss = loss_fn(out, y) # scaled_MSE(out, y)
            if torch.isnan(loss):
                print("NaNs detected in the loss")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            total_loss += loss.item()
            # progress_bar(counter+1, len(train_loader))
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        val_loss, _ = evaluate(model, val_loader, loss_fn) # scaled_MSE

        history['val_loss'].append(val_loss)
        scheduler.step()


        print(f'\nEpoch {epoch+1}/{epochs} completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the model and training history if a version is provided
    # if version:
    if epoch == epochs-1:
        model_filename = f"WindGNN_{version}.pth"
        history_filename = f"WindGNN_{version}_history.csv"

        torch.save(model.state_dict(), models+model_filename)
        history_df = pd.DataFrame(history)
        history_df.to_csv(models+history_filename, index=False)

        print(f"\n Training completed! Model saved as {model_filename}")
        print(f"Training history saved as {history_filename}")

    return history



