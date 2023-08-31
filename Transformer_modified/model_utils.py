"""
Utility functions for model's processing
"""
import os
import pickle
import pyximport
import time

import networkx as nx
import numpy as np
import pandas as pd

pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos

def find_shortest_path_distance(data_path):
    """
    Based on Floyd-Warshall algorithm (implementation from Graphormer: `algos.pyx`), 
    compute all node's shortest path distance to all other nodes.
    (Result array will saved to local for convenience.)

    Args:
        data_path: path to dataset (social graph)
    """
    if 'shortest_path_result.npy' in os.listdir(data_path):
        print("Pre-computed shortest path matrix available...")
        shortest_path_result = np.load(data_path + '/shortest_path_result.npy')

        return shortest_path_result

    social_file = data_path + '/trustnetwork.csv'
    dataframe = pd.read_csv(social_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')
    social_graph_adj = nx.to_numpy_array(social_graph, dtype=np.int64)

    shortest_path_result, path = algos.floyd_warshall(social_graph_adj)

    # Ciao: (7317, 7317) with size 428.3 MB
    # Epinions: (18098, 18098) with size 2.6 GB
    start_time = time.time()
    np.save(data_path + '/shortest_path_result.npy', shortest_path_result)
    print(f"Finished saving .npy file, total time: {time.time() - start_time:.4f}s")
    print("#### Call this function again to get computed result. ####")

    return 0

#### For testing
if __name__ == "__main__":
    data_path = os.getcwd() + '/dataset/' + 'epinions'

    array = find_shortest_path_distance(data_path)
    print(array.shape)
    print(np.max(array))
    print(np.where(array == np.max(array)))
    # print(array[0][7020])
    pass
