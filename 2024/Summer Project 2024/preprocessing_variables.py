"""
If you have processed parts of the data,
this will allow you to skip the files that have already been processed.
Set to false if you want to force full preprocessing of existing files.
"""
allow_skips = False

#If you want to use the full dataset, set this to True, 
full_dataset = False
#Otherwise, set the number of files to use, mostly for testing pipeline
n_files = 10

#knn (int): Number of nearest neighbors.
knn = 4 #Default 4

# Number of grid points in North and East direction (has to be divisible by 2, +1 each for center location)
grid_dimensions = (50,50) #Default (50,50)


#Define the general path for the folder of the project.
windgnn = r'C:\Users\SIGUR\OneDrive - UiT Office 365\Var24\prosjekt2'
csv = windgnn+"/csv/"
graphs = windgnn+"/csv/graph/"
intt = windgnn+"/csv/int/"
output_edges = windgnn+"/csv/output_edges/"
carra_folder = windgnn+"/csv/carra/"
s1_folder = windgnn+"/csv/s1/"



# Goliat coordinates
Goliat_lat = 71.3112 #Default 71.3112
Goliat_lon = 22.25 #Default 22.25

import os, torch
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
import re
from sklearn.neighbors import BallTree
import haversine as hav
from haversine import Unit
from sklearn.neighbors import NearestNeighbors


#Make sure the folders to store the data exist.
try:
    os.makedirs(windgnn+"/csv")
except FileExistsError:
    # directory already exists
    pass

try:
    os.makedirs(windgnn+"/csv/graph")
except FileExistsError:
    # directory already exists
    pass
try :
    os.makedirs(windgnn+"/csv/int")
except FileExistsError:
    # directory already exists
    pass
try :
    os.makedirs(windgnn+"/csv/output_edges")
except FileExistsError:
    # directory already exists
    pass