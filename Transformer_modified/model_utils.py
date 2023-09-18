"""
Utility functions for model's processing
"""
import os
import pickle
import pyximport
import time
import torch

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

    print("Start SPD algorithm...")
    start_time = time.time()
    shortest_path_result, path = algos.floyd_warshall(social_graph_adj)
    print(f"Algorithm finished, time: {time.time() - start_time:.4f}s")     # ciao: 221.0372 s // epinions: 3134.4668s

    # Ciao: (7317, 7317) with size 428.3 MB
    # Epinions: (18098, 18098) with size 2.6 GB
    start_time = time.time()
    np.save(data_path + '/shortest_path_result.npy', shortest_path_result)
    print(f"Finished saving .npy file, total time: {time.time() - start_time:.4f}s")
    print("#### Call this function again to get computed result. ####")

    return 0

def generate_attn_pad_mask(seq_q, seq_k):
    """
    Generate attention mask
        0-padded data -> apply mask
        original data -> do not apply mask
    """
    # FIXME: (230911) 현재 mask 생성을 위한 입력으로 들어오는 shape은 다음과 같음.
        # Enc에선 [batch_size, seq_len_user, seq_len_user]
        # Dec에선 [batch_size, seq_len_item, seq_len_item]
    # print(seq_q.size())
    # print(seq_k.size())
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # [batch_size, 1, len_k(=len_q)]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    # [batch_size, len_q, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def generate_attn_subsequent_mask(seq):
    """
    Generate decoder's mask
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()

    return subsequent_mask


#### For testing
if __name__ == "__main__":
    # data_path = os.getcwd() + '/dataset/' + 'epinions'

    # array = find_shortest_path_distance(data_path)
    # # print(array.shape)
    # # print(np.max(array))
    # # print(np.where(array == np.max(array)))
    # # print(array[0][7020])
    # pass

    from dataset import MyDataset

    dataset = MyDataset(dataset='ciao')
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    _, data = next(enumerate(loader))

    print(data['user_seq'].shape)
    print(data['item_seq'].shape)
