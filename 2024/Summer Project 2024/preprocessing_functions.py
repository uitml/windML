from preprocessing_variables import *

os.environ['TORCH'] = torch.__version__
print(torch.__version__)
print(torch.version.cuda)


# Progress bar function cause why not
def progress_bar(n, total):
    scale = 120/total
    percent = round(n/total*100, 1)
    done = '█'*round(n*scale)
    todo = '-'*round((total - n)*scale)
    print(f"\r{percent}% [{done}{todo}]", end='')

def coord_offset(lat, lon, dN, dE):
    """
    Offset a pair of lat, lon coordinates by meters

    Args:
    - lat, lon (floats) : Latitude and longitude of position to offset
    - dN, dE (floats) : Distances (in meters!) to offset position by, in North and East direction

    Returns:
    - lat0, lon0 (floats): Latitude and longitude of new position
    """

    # Earth's radius in meters
    R = 6378137

    # Coordinate offsets in radians
    dLat = dN/R
    dLon = dE/(R * np.cos(np.pi * lat/180))

    # Offset position, decimal degrees
    lat0 = lat + dLat * 180/np.pi
    lon0 = lon + dLon * 180/np.pi

    return lat0, lon0

coord_offset_vec = np.vectorize(coord_offset)

def create_grid(dimensions = (50,50), lat=Goliat_lat, lon=Goliat_lon, distN=1000, distE=1000):
    """
    Create grid around center point of lat, lon coordinates

    Args:
    - dimensions (tuple) : Tuple (x,y) with number of grid points per axis (+1 for center location), for North and East
    - lat, lon (floats) : Latitude and longitude of original position (Goliat station per default)
    - distN, distE (floats): Distances (in meters!) between grid points in North and East direction (1km per default)

    Returns:
    - mesh_df (pd.DataFrame) : DataFrame containing the columns 'lat' and 'lon' with the grid points
    """

    # Calculate number of grid points to go from center location in each direction
    points_N, points_E = dimensions
    n_N = int(points_N/2)
    n_E = int(points_E/2)

    # Calculate min and max offsets (in meters) in North and East direction
    min_N, max_N = -distN * n_N, distN * n_N
    min_E, max_E = -distE * n_E, distE * n_E

    # Create axes and a mesh grid
    axis_N = np.linspace(min_N, max_N, points_N+1)
    axis_E = np.linspace(min_E, max_E, points_E+1)

    X, Y = np.meshgrid(axis_N, axis_E)

    # Calculate offset coordinates
    mesh = coord_offset_vec(lat, lon, X, Y)

    #Transpose to get the (x,y) values for the offset grid's shape
    mesh_x_y_format = np.stack(mesh).transpose(1,2,0)

    # Reshape to get an array
    l = (points_N+1) * (points_E+1)
    mesh_array = np.reshape(mesh_x_y_format, (l, 2))

    # Create DataFrame
    mesh_df = pd.DataFrame({'lat': mesh_array[:,0], 'lon': mesh_array[:,1]})
    torch.save(mesh_df, windgnn+f'/csv/output_grid_{dimensions[0]}_{dimensions[1]}.pt')

    return mesh_df


def mask_s1(s1_df, output_grid):
    """
    Mask out entries from a SAR scene based on longitude and latitude ranges of the output grid to avoid unnecessary computations.

    Args:
    - s1_df (pd.DataFrame): Sentinel-1 DataFrame with columns 'lon', 'lat', 'wind_speed', 'wind_direction'.
    - output_grid (pd.DataFrame):

    Returns:
    - s1_masked (pd.DataFrame): DataFrame with entries outside the specified ranges masked out.
    """

    min_lon, max_lon = [min(output_grid['lon']), max(output_grid['lon'])]
    min_lat, max_lat = [min(output_grid['lat']), max(output_grid['lat'])]

    # Create boolean masks for longitude and latitude
    lon_mask = (s1_df['lon'] >= (min_lon - 0.1)) & (s1_df['lon'] <= (max_lon + 0.1))
    lat_mask = (s1_df['lat'] >= (min_lat - 0.1)) & (s1_df['lat'] <= (max_lat + 0.1))

    # Combine the masks to get the final mask
    final_mask = lon_mask & lat_mask

    # Apply the mask to the DataFrame
    s1_masked = s1_df[final_mask]

    return s1_masked

def interpolate_wind_speed(s1_file, output_dimensions, k):#, return_graph=False):
    """
    Interpolates the wind speed from a given Sentinel-1 scene onto the output grid.

    Args:
    - s1_file (pd.DataFrame) : DataFrame containing the Sentinel-1 SAR scene with columns 'lat', 'lon', 'wind_speed'
    - output_dimensions (tuple) : Tuple (x,y) defining the number of grid points (+1) in N and E direction of the output grid
    - k (int) : Number of nearest neighbours to consider for interpolation

    Returns:
    - output_wind_speed (pd.DataFrame) : Interpolated wind field on the output grid, given as 'lat', 'lon', 'wind_speed'
    """

    # Generate output grid
    output_grid = create_grid(dimensions=output_dimensions)

    # Calculate distances with haversine function in meters
    s1_coords = s1_file[['lat', 'lon']].values
    output_coords = output_grid[['lat', 'lon']].values

    distances = cdist(output_coords, s1_coords, lambda u, v: hav.haversine(u, v, unit='m'))

    # Find k nearest neighbors for each point in output grid
    indices = np.argsort(distances, axis=1)[:, :k]

    # Initialize array to store interpolated wind speed
    interpolated_wind_speed = np.zeros(indices.shape[0])

    # Interpolate wind speed
    for i, idx in enumerate(indices):
        neighbor_distances = distances[i][idx]
        weights = 1 / neighbor_distances
        weighted_speeds = weights * s1_file.iloc[idx]['wind_speed']
        interpolated_wind_speed[i] = np.sum(weighted_speeds) / np.sum(weights)

    # Add interpolated wind speeds to DataFrame
    output_grid.insert(2, 'wind_speed', interpolated_wind_speed)

    #if return_graph:
    x = torch.tensor(output_grid[['lat', 'lon', 'wind_speed']].values, dtype=torch.float)
    ground_truth = Data(x=x)
    ground_truth.edge_index = []
    torch.save(ground_truth, windgnn+f'/csv/int/ground_truth_{output_dimensions[0]}_{output_dimensions[1]}.pt')

    return ground_truth

def find_closest_carra_file(s1_filename):
    """
    Function finding the closest CARRA file for a given Sentinel-1 scene
    """
    # carra_folder = "csv/carra/"

    # Extract the date and hour from the Sentinel-1 filename
    s1_date, s1_hour = s1_filename.split('_')[1:3]
    s1_hour = int(s1_hour.split('.')[0])  # Remove file extension and convert to int

    # Determine the matching hour for the CARRA file
    carra_hour = '06' if s1_hour == 5 else '15'

    # Construct the CARRA filename
    carra_filename = f'carra_{s1_date}_{carra_hour}.csv'
    carra_filepath = os.path.join(carra_folder, carra_filename)

    # Check if the CARRA file exists
    if os.path.exists(carra_filepath):
        return carra_filepath
    else:
        print(carra_filepath)
        return None

def create_knn_graph(carra_file, features, k=knn):
    """
    Function constructing kNN graph of a given scene

    Args:
    - carra_file :
    - features :
    - k : number of neighbours to add edges to (default = 4)

    Returns:
    - x, edge_index :
    """

    positions = carra_file[['lat', 'lon']].values
    data_features = carra_file[features].values

    # Construct kNN graph
    adjacency_matrix = kneighbors_graph(positions, n_neighbors=k, mode='distance', include_self=True, metric='haversine')
    intermediate_matrix = np.array(adjacency_matrix.nonzero())

    edge_index = torch.tensor(intermediate_matrix, dtype=torch.long)

    # Create PyTorch tensors for features and positions
    x = torch.tensor(data_features, dtype=torch.float)

    return x, edge_index

def combine_graphs(carra_graph, ground_truth):
    """
    Function combining two kNN graphs of matching ground truth and CARRA scene into a heterogeneous graph
    Assuming carra_data and ground_truth are tuples of (x, edge_index) from create_knn_graph
    """

    carra_x, carra_edge_index = carra_graph
    truth_x, truth_edge_index = ground_truth

    # Create a heterogeneous graph
    data = HeteroData()
    data['carra'].x = carra_x
    data['carra'].edge_index = carra_edge_index
    data['truth'].x = truth_x
    data['truth'].edge_index = truth_edge_index

    # Add edges between 'carra' and 'truth' nodes here
    # …

    return data 

def create_graph_data(output_dimensions = (50,50)):
    """
    This function will create the graph data for the wind prediction model, using the Sentinel-1 and CARRA datasets.
    Args:  
    - output_dimensions (tuple) : Tuple (x,y) defining the number of grid points (+1) in N and E direction of the output grid
    """
    # Define the features for each dataset
    s1_features = ['wind_speed'] #Not used, but kept for clarity.
    carra_features = ['wind_speed', 'wind_direction', 'surface_pressure', 'temperature_2_m']

    #This loop will create the feature data from the sar Images.'
    imgdir = os.listdir(s1_folder)
    if not full_dataset:
        imgdir = imgdir[:n_files]
    for counter, s1_filename in enumerate(sorted(imgdir)):
        
        #Check if file truth_edge file already exists, to avoid spending time on creating existing files.
        if allow_skips:
            if os.path.exists(os.path.join(windgnn+"/csv/graph", f'{s1_filename.split(".")[0]}_truth_edge.pt')):
                progress_bar(counter, 832)
                continue


        # Find the closest CARRA file
        carra_filepath = find_closest_carra_file(s1_filename)
        # Load the matched files
        s1_df = pd.read_csv(os.path.join(s1_folder, s1_filename))

        carra_df = pd.read_csv(carra_filepath)

        # Create output grid
        output_grid = create_grid(dimensions=output_dimensions)

        # Mask all unused Sentinel-1 values
        s1_masked = mask_s1(s1_df, output_grid)

        # Compute interpolated wind speed field (ground truth)
        ground_truth = interpolate_wind_speed(s1_masked, output_dimensions, k=knn)#, return_graph=True)

        # Preprocess CARRA kNN graph
        carra_data = create_knn_graph(carra_df, carra_features)

        # Combine into a heterogeneous graph
        combined_data = combine_graphs(carra_data, ground_truth)

        # Save Data objects as files
        s1_filename = s1_filename.split(".")[0]
        torch.save(combined_data['carra'].x, graphs+f'{s1_filename}_carra_x.pt')
        torch.save(combined_data['carra'].edge_index, graphs+f'{s1_filename}_carra_edge.pt')
        torch.save(combined_data['truth'].x, graphs+f'{s1_filename}_truth_x.pt')
        torch.save(combined_data['truth'].edge_index, graphs+f'{s1_filename}_truth_edge.pt')

        # Add to the list for batching
        progress_bar(counter, 832)

def interpolate_carra(timestamp, output_dimension = (50,50), k=knn):
    if os.path.exists(windgnn+f'/csv/int/carra_interpolation_{timestamp}.pt'):
        pass
    else:
        output_grid = create_grid(dimensions=output_dimension)
        carra_file = find_closest_carra_file(os.path.join(s1_folder, f's1_{timestamp}.csv'))
        carra = pd.read_csv(carra_file)
        carra_interpolation = interpolate_wind_speed(mask_s1(carra, output_grid), output_dimensions=output_dimension, k=knn)
        torch.save(carra_interpolation, windgnn+f'/csv/int/carra_interpolation_{timestamp}.pt')

def find_suitable_timestamps(base_path='.'):
    suitable_timestamps = []
    unsuitable_timestamps = []


    pattern = re.compile(r's1_(\d{4}-\d{2}-\d{2}_\d+)_carra_x.pt')
    timestamps = {match.group(1) for filename in os.listdir(base_path) if (match := pattern.match(filename))}
    
    print("timestamps:", len(timestamps))

    for counter, timestamp in enumerate(sorted(timestamps)):
        s1_filename = f's1_{timestamp}.csv'
        s1_file = pd.read_csv(os.path.join(s1_folder, s1_filename))

        s1_positions = s1_file[['lat', 'lon']].values
        s1_tree = BallTree(np.radians(s1_positions), metric='haversine')

        truth_x_path = os.path.join(base_path, f's1_{timestamp}_truth_x.pt')
        truth_x = torch.load(truth_x_path)
        if isinstance(truth_x, tuple):
            if len(truth_x) == 2 and isinstance(truth_x[1], torch.Tensor):
                truth_x = truth_x[1]
            else:
                raise ValueError("truth_x is not in the expected format (tuple with a tensor as the second element).")
            torch.save(truth_x, truth_x_path)
        output_positions = truth_x[:, :2].numpy()

        suitable = True
        for output_position in output_positions:
            indices = s1_tree.query_radius([np.radians(output_position)], r=2 / 6371.0)
            if len(indices[0]) == 0:
                suitable = False
                break

        if suitable:
            suitable_timestamps.append(timestamp)
        else:
            unsuitable_timestamps.append(timestamp)

        progress_bar(counter, 832)

    with open(windgnn+'/csv/timestamps_yes.txt', 'w') as f:
        for ts in suitable_timestamps:
            f.write(ts + '\n')

    with open(windgnn+'/csv/timestamps_no.txt', 'w') as f:
        for ts in unsuitable_timestamps:
            f.write(ts + '\n')

    return suitable_timestamps, unsuitable_timestamps

def create_output_edges(dimensions, k):
    """
    Create kNN edges for the output grid nodes.
    
    Args:
        output_grid_positions (torch.Tensor): Positions of output grid nodes.
        k (int): Number of nearest neighbors.
        output_file (str): File to save the edges.
    
    Returns:
        None
    """
    output_grid = torch.load(csv+f'output_grid_{dimensions[0]}_{dimensions[1]}.pt')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(output_grid)
    _, indices = nbrs.kneighbors(output_grid)
    
    edges = []
    for i in range(indices.shape[0]):
        for j in range(1, k):  # start from 1 to avoid self-loop
            edges.append([i, indices[i, j]])
    
    output_file = windgnn+f"/csv/output_grid_{k}nn_edges.pt"
    edges = torch.tensor(edges, dtype=torch.long).t()
    torch.save(edges, output_file)
    print(f'Saved kNN edges to {output_file}')

def adjust_timestamp(timestamp):
    if timestamp.endswith('5'):
        return timestamp[:-1] + '06'
    elif timestamp.endswith('16'):
        return timestamp[:-2] + '15'
    return timestamp

def find_global_min_max(suitable_timestamps, base_path='.'):
    global_min_temp = float('inf')
    global_max_temp = float('-inf')
    global_min_pressure = float('inf')
    global_max_pressure = float('-inf')

    for timestamp in sorted(suitable_timestamps):
        adjusted_timestamp = adjust_timestamp(timestamp)
        carra_x_path = os.path.join(carra_folder, f'carra_{adjusted_timestamp}.csv')
        carra_x = pd.read_csv(carra_x_path)

        pressure = carra_x['surface_pressure']  # 'surface_pressure' is the third column
        temp = carra_x['temperature_2_m']  # 'temperature_2_m' is the last column

        global_min_pressure = min(global_min_pressure, pressure.min().item())
        global_max_pressure = max(global_max_pressure, pressure.max().item())
        global_min_temp = min(global_min_temp, temp.min().item())
        global_max_temp = max(global_max_temp, temp.max().item())

    return global_min_temp, global_max_temp, global_min_pressure, global_max_pressure

def carra_to_output(carra_x, output_x, radius=2.5):
    """
    Create edges from output grid nodes to CARRA nodes within a given radius.
    
    Args:
        carra_x (torch.Tensor): Features of CARRA nodes including lat and lon.
        output_x (torch.Tensor): Features of output grid nodes including lat and lon.
        radius (float): Radius to create edges in kilometers.
    
    Returns:
        torch.Tensor: Edge indices.
        torch.Tensor: Edge distances.
    """
    carra_lat = carra_x[:, 0].numpy()  # Assuming lat is second to last column
    carra_lon = carra_x[:, 1].numpy()  # Assuming lon is the last column
    output_lat = output_x[:, 0].numpy()
    output_lon = output_x[:, 1].numpy()

    carra_positions = np.vstack((carra_lat, carra_lon)).T
    output_positions = np.vstack((output_lat, output_lon)).T

    # Create a BallTree for CARRA positions with Haversine metric
    carra_tree = BallTree(np.radians(carra_positions), metric='haversine')
    
    # Query for neighbors within the radius (convert radius to radians by dividing by Earth's radius)
    radius_rad = radius / 6371.0

    indices = carra_tree.query_radius(np.radians(output_positions), r=radius_rad)
        
    edges = []
    distances = []

    for i, neighbors in enumerate(indices):
        if len(neighbors) == 0:
            print(f"No neighbors found for output node {i}")
        for neighbor in neighbors:
            edges.append([neighbor, i])  # from CARRA node to output grid node
            distance = hav.haversine(carra_positions[neighbor], output_positions[i], unit=Unit.KILOMETERS)
            distances.append(distance)
            #print(f"Edge from CARRA {neighbor} to output {i}, distance: {distance}")

    edges = torch.tensor(edges, dtype=torch.long).t()
    distances = torch.tensor(distances, dtype=torch.float)

    return edges, distances
def normalize_value(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def load_and_normalize_graph_data(timestamp, base_path=csv, output_n=4, min_temp=None, max_temp=None, min_pressure=None, max_pressure=None):
    carra_edge_path = os.path.join(graphs, f's1_{timestamp}_carra_edge.pt')
    carra_x_path = os.path.join(graphs, f's1_{timestamp}_carra_x.pt')
    truth_edge_path = os.path.join(csv, f'output_grid_{output_n}nn_edges.pt')
    truth_x_path = os.path.join(graphs, f's1_{timestamp}_truth_x.pt')
    
    carra_edge = torch.load(carra_edge_path)
    carra_x = torch.load(carra_x_path)
    truth_edge = torch.load(truth_edge_path)
    truth_x = torch.load(truth_x_path)

    carra_to_output_edges_path = os.path.join(output_edges, f's1_{timestamp}_carra_to_output_edges.pt')
    carra_to_output_attr_path = os.path.join(output_edges, f's1_{timestamp}_carra_to_output_attr.pt')
    extended_carra_x_path = os.path.join(graphs, f's1_{timestamp}_carra_x_extended.pt')
    extended_truth_x_path = os.path.join(graphs, f's1_{timestamp}_truth_x_extended.pt')

    if carra_x.shape[1] > 6:
        print(f"CARRA {timestamp} has too many columns! Check: {carra_x.shape}")
        carra_x = carra_x[2:]

    # Extract the tensor from the tuple
    if isinstance(truth_x, tuple):
        if len(truth_x) == 2 and isinstance(truth_x[1], torch.Tensor):
            truth_x = truth_x[1]
        else:
            raise ValueError("truth_x is not in the expected format (tuple with a tensor as the second element).")
        # Save the updated truth_x tensor
        # torch.save(truth_x, truth_x_path)
    # Adjust the timestamp for CARRA file
    adjusted_timestamp = adjust_timestamp(timestamp)
    torch.save(truth_x, truth_x_path)
    print(truth_x)

    # Load the CARRA lat/lon from the CSV file
    carra_filename = f'carra_{adjusted_timestamp}.csv'
    carra_file = pd.read_csv(os.path.join(carra_folder, carra_filename))
    
    carra_lat = torch.tensor(carra_file['lat'].values, dtype=torch.float32).unsqueeze(1)
    carra_lon = torch.tensor(carra_file['lon'].values, dtype=torch.float32).unsqueeze(1)
    
    #This is the file you want in the end, had to rename it as the original file was overwritten. Which caused bugs.
    # Append lat/lon to the existing carra_x features
    extended_carra_x = torch.cat([carra_lat, carra_lon, carra_x], dim=1)
    
    # Normalize values in carra_x
    extended_carra_x[:, 2] = torch.clamp(extended_carra_x[:, 2], max=25) / 25  # Normalize wind_speed
    extended_carra_x[:, 3] = extended_carra_x[:, 3] / 360  # Normalize wind_direction
    extended_carra_x[:, 4] = normalize_value(extended_carra_x[:, 4], min_pressure, max_pressure)  # Normalize surface_pressure
    extended_carra_x[:, 5] = normalize_value(extended_carra_x[:, 5], min_temp, max_temp)  # Normalize temperature_2_m
    # Normalize values in truth_x
    truth_x[:, 2] = torch.clamp(truth_x[:, 2], max=25) / 25  # Normalize wind_speed

    # Save the updated carra_x tensor
    torch.save(extended_carra_x, extended_carra_x_path)
    torch.save(truth_x, extended_truth_x_path)

    # Generate and save new edges and edge attributes
    carra_to_output_edges, carra_to_output_attr = carra_to_output(extended_carra_x, truth_x)
    
    # Normalize edge attributes
    carra_to_output_attr /= 2.5
    #Suddenly python decided to introduce float point errors.
    if torch.any(carra_to_output_attr > 1):
        # print(f"float error in {timestamp}, fixed by clamping values.")
        carra_to_output_attr = torch.clamp(carra_to_output_attr, max=1)
    torch.save(carra_to_output_edges, carra_to_output_edges_path)
    torch.save(carra_to_output_attr, carra_to_output_attr_path)

    return extended_carra_x, truth_x, carra_to_output_attr

def check_normalization(timestamp, carra_x, truth_x, carra_to_output_attr):
    # Check carra_x excluding the first two columns (lat, lon)
    if not torch.all((carra_x[:, 2:] >= 0) & (carra_x[:, 2:] <= 1)):
        print(f"Normalization error in 's1_{timestamp}_carra_x.pt'")
        out_of_range_values = carra_x[:, 2:][(carra_x[:, 2:] < 0) | (carra_x[:, 2:] > 1)]
        print("Out of range values in carra_x:", out_of_range_values)

    # Check truth_x excluding the first two columns (lat, lon)
    if not torch.all((truth_x[:, 2:] >= 0) & (truth_x[:, 2:] <= 1)):
        print(f"Normalization error in s1_{timestamp}_truth_x.pt")
        out_of_range_values = truth_x[:, 2:][(truth_x[:, 2:] < 0) | (truth_x[:, 2:] > 1)]
        print("Out of range values in truth_x:", out_of_range_values)

    # Check edge attributes
    if not torch.all((carra_to_output_attr >= 0) & (carra_to_output_attr <= 1)):
        print(f"Normalization error in 's1_{timestamp}_carra_to_output_attr.pt")
        #I get float error here where 1.0000 > 1, so I need to clamp the values.
        out_of_range_values = carra_to_output_attr[(carra_to_output_attr < 0) | (carra_to_output_attr > 1)]
        print("Out of range values in carra_to_output_attr:", out_of_range_values)

    # Print confirmation if all checks pass
    if torch.all((carra_x[:, 2:] >= 0) & (carra_x[:, 2:] <= 1)) and torch.all((truth_x[:, 2:] >= 0) & (truth_x[:, 2:] <= 1)) and torch.all((carra_to_output_attr >= 0) & (carra_to_output_attr <= 1)):
        print(f"All values in {timestamp} files are correctly normalized")


