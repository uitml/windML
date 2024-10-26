from preprocessing_functions import *

#Check preprocessing_variables before running this script
if __name__ == '__main__': 
    create_graph_data(output_dimensions = grid_dimensions)

    #Find the timestamps of images covering the entire AOI.
    suitable_timestamps, unsuitable_timestamps = find_suitable_timestamps(graphs)

    create_output_edges(dimensions=grid_dimensions, k=knn)

    global_min_temp, global_max_temp, global_min_pressure, global_max_pressure = find_global_min_max(suitable_timestamps)

    for timestamp in tqdm(suitable_timestamps):    
        extended_carra_x, truth_x, carra_to_output_attr = load_and_normalize_graph_data(timestamp, output_n=4, min_temp=global_min_temp, max_temp=global_max_temp, min_pressure=global_min_pressure, max_pressure=global_max_pressure)
        check_normalization(timestamp, extended_carra_x, truth_x, carra_to_output_attr)

        #This function isn't necessary for creating the graph data. 
        #Simply creates an interpolation that was previously used to compare the model to.
        interpolate_carra(timestamp, output_dimension=grid_dimensions, k = knn)

        
