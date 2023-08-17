"""
trustnetwork 에서 모든 사용자의 degree 정보를 담은 table 생성

trustnetwork 에서 random walk sequence 생성
    - 임의의 사용자 n명을 선택
    - 각 사용자마다 random walk length r 만큼의 subgraph sequence 생성
    - 생성한 sequence에서, 각 노드와 매칭되는 degree 정보를 degree table에서 GET
    - [노드, 노드, 노드], [degree, degree, dgree] 를 함께 구성 (like PyG's edge_index)
        => [[node1, node2, node3]]
"""
import os
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import loadmat

# np.random.seed(62)

# arg(or else) passing to DATASET later
DATASET = 'ciao'

data_path = os.getcwd() + '/dataset/' + DATASET


def mat_to_csv(data_path:str):
    """
    Convert .mat file into .csv file for using pandas.
        Ciao: rating.mat, trustnetwork.mat
        Epinions: rating.mat, trustnetwork.mat
    
    Args:
        data_path: Path to .mat file
    """
    # processed file check
    if ('rating.csv' or 'trustnetwork.csv') in os.listdir(data_path):
        print("Processed .csv file already exists...")
        return 0
    
    # load .mat file
    rating_file = loadmat(data_path + '/' + 'rating.mat')
    trust_file = loadmat(data_path + '/' + 'trustnetwork.mat')

    rating_file = rating_file['rating'].astype(np.int64)
    trust_file = trust_file['trustnetwork'].astype(np.int64)

    # convert to dataframe
    rating_df = pd.DataFrame(rating_file, columns=['user_id', 'product_id', 'category_id', 'rating', 'helpfulness'])
    trust_df = pd.DataFrame(trust_file, columns=['user_id_1', 'user_id_2'])

    ###### drop unused columns (TODO: Maybe used later)
    rating_df.drop(['category_id', 'helpfulness'], axis=1, inplace=True)
    ######

    rating_df.to_csv(data_path + '/rating.csv', index=False)
    trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)


def generate_user_degree_table(data_path:str):
    """
    Generate degree table from social graph(trustnetwork).
    """
    # processed file check
    if 'social_degree.csv' in os.listdir(data_path):
        print("Processed .csv file already exists...")
        return 0

    # user-user network
        # Ciao: 7317 users
    trust_file = data_path + '/trustnetwork.csv'
    dataframe = pd.read_csv(trust_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')

    degrees = {node: val for (node, val) in social_graph.degree()}
    degree_df = pd.DataFrame(degrees.items(), columns=['user_id', 'degree'])

    degree_df.to_csv(data_path + '/social_degree.csv', index=False)


def generate_social_random_walk_sequence(data_path:str, num_nodes:int=10, walk_length:int=5) -> dict:
    """
    Generate random walk sequence from social graph(trustnetwork).
    
    Args:
        data_path: path to data
        num_nodes: number of nodes to generate random walk sequence
        walk_length: length of random walk
    """
    trust_file = data_path + '/trustnetwork.csv'
    dataframe = pd.read_csv(trust_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')

    path_dict = {}

    # select target(anchor) nodes randomly.
    anchor_nodes = np.random.choice(social_graph.nodes(), size=num_nodes)
    
    # At first, there is no previous node, so set it to None.
    for nodes in anchor_nodes:
        path_dict[nodes] = [nodes]
        for _ in range(walk_length - 1):
            # Move to one of connected node randomly.
            if path_dict[nodes][-1] == nodes:
                next_node = find_next_social_node(graph=social_graph, previous_node=None, current_node=nodes, RETURN_PARAMS=0.0)
                path_dict[nodes].append(next_node)

            # If selected node was "edge node", there is no movable nodes, so pad it with 0(zero-padding).
            if path_dict[nodes][-1] == 0:
                path_dict[nodes].append(0)

            # Move to one of connected node randomly.
            else:
                next_node = find_next_social_node(graph=social_graph, previous_node=path_dict[nodes][-2], current_node=path_dict[nodes][-1], RETURN_PARAMS=0.0)
                path_dict[nodes].append(next_node)
        
        # Pop 1st element of list(since it is anchor node).
        del path_dict[nodes][0]

    return path_dict


def find_next_social_node(graph:nx.Graph(), previous_node, current_node, RETURN_PARAMS):
    """
    Find connected nodes, using transition probability. \n
        ###\n
            This code only finds un-visited nodes.\n
            (no re-visiting to previously visited node.)\n
        ###
    """
    select_prob = {}

    # Here, in social network we using, we don't have any edge weights.
    # So set transition(selecting neighbor node) probability to 1.
    for node in graph.neighbors(current_node):
        if node != previous_node:
            select_prob[node] = 1   

    # Set transition probabilites equally(for randomness).
    select_prob_sum = sum(select_prob.values())
    select_prob = {key: value / select_prob_sum * (1 - RETURN_PARAMS) for key, value in select_prob.items()}

    transitionable_nodes = [node for node in select_prob.keys()]
    transition_prob = [prob for prob in select_prob.values()]

    # If no nodes are selected, return 0 for zero-padding.
        # isolated node pair will return 0.
        # also "previous->current->(empty)" will return 0, too.
    if len(transitionable_nodes) == 0:
        return 0

    selected_node = np.random.choice(
        a = transitionable_nodes,
        p = transition_prob
    )

    return selected_node

# For checking
sequences = generate_social_random_walk_sequence(data_path, num_nodes=10, walk_length=20)
for key, value in sequences.items():
    print({f"{key} : {value}"})
    print(f"\t => length: {len(value)}")