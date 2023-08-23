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
from ast import literal_eval    # convert str type list to original type
from scipy.io import loadmat
from tqdm.auto import tqdm

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


def generate_user_degree_table(data_path:str) -> pd.DataFrame:
    """
    Generate & return degree table from social graph(trustnetwork).
    """
    # processed file check
    if 'degree_table_social.csv' in os.listdir(data_path):
        print("Processed .csv file already exists...")
        degree_df = pd.read_csv(data_path + '/degree_table_social.csv', index_col=[])
        return degree_df

    # user-user network
        # Ciao: 7317 users
    trust_file = data_path + '/trustnetwork.csv'
    dataframe = pd.read_csv(trust_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')

    degrees = {node: val for (node, val) in social_graph.degree()}
    degree_df = pd.DataFrame(degrees.items(), columns=['user_id', 'degree'])

    degree_df.to_csv(data_path + '/degree_table_social.csv', index=False)

    return degree_df


def generate_item_degree_table(data_path:str) -> pd.DataFrame:
    """
    Generate & return degree table from user-item graph(rating matrix).
    """
    # processed file check
    if 'degree_table_item.csv' in os.listdir(data_path):
        print("Processed .csv file already exists...")
        degree_df = pd.read_csv(data_path + '/degree_table_item.csv', index_col=[])
        return degree_df
    
    # user-item network
        # Ciao: 7375 user // 105114 items
    rating_file = data_path + '/rating.csv'
    dataframe = pd.read_csv(rating_file, index_col=[])

    # Since using NetworkX to compute bipartite graph's degree is time-consuming(because graph is too sparse),
    # we just use pandas for simple degree calculation.
    degree_df = dataframe.groupby('product_id')['user_id'].nunique().reset_index()
    # degree_df = dataframe.groupby('user_id')['product_id'].nunique().reset_index()

    degree_df.columns = ['product_id', 'degree']
    # degree_df.columns = ['user_id', 'degree']

    degree_df.to_csv(data_path + '/degree_table_item.csv', index=False)

    return degree_df


def generate_interacted_items_table(data_path:str) -> pd.DataFrame:
    """
    Generate & return user's interacted items & ratings table from user-item graph(rating matrix)
    """
    # processed file check
    if 'user_item_interaction.csv' in os.listdir(data_path):
        print("Processed .csv file already exists...")
        user_item_dataframe = pd.read_csv(data_path + '/user_item_interaction.csv', index_col=[])
        return user_item_dataframe

    rating_file = data_path + '/rating.csv'
    dataframe = pd.read_csv(rating_file, index_col=[])

    # For each user, find their interacted items and given rating.

    ## This will make dict with list of list: {user_id: [[product_id, ...], [rating, ...]], user_id: [[product_id, ...], [rating, ...]], ...} 
    # user_dict = {}
    # for _, data in dataframe.iterrows():
    #     if data['user_id'] not in user_dict:
    #         user_dict[data['user_id']] = [[], []]
    #     user_dict[data['user_id']][0].append(data['product_id'])
    #     user_dict[data['user_id']][1].append(data['rating'])

    ## This will make dict of dict: {user_id: {product_id:[...], rating:[...]}, user_id:{product_id:[...], rating:[...]}, ...}
    user_item_dataframe = dataframe.groupby('user_id').agg({'product_id': list, 'rating': list}).reset_index()
    user_item_dataframe.to_csv(data_path + '/user_item_interaction.csv', index=False)
    # user_item_dataframe = user_item_dataframe.set_index('user_id').to_dict(orient='index')

    return user_item_dataframe



def generate_social_random_walk_sequence(data_path:str, num_nodes:int=10, walk_length:int=5, save_flag=False, all_node=False, seed=False) -> list:
    """
    Generate random walk sequence from social graph(trustnetwork).
    Return:
        List containing dictionaries,\n 
        ex) [
            {user_id: [[user, ..., user], [degree, ..., degree]]}, 
            {user_id: [[user, ..., user], [degree, ..., degree]]},
            ...
            ]
    Args:
        data_path: path to data
        num_nodes: number of nodes to generate random walk sequence
        walk_length: length of random walk
        save_flag: save result locally (default=False)
        all_node: get all node's random walk sequence (default=False)
        seed: random seed, True or False (default=False)
    """
    trust_file = data_path + '/trustnetwork.csv'
    dataframe = pd.read_csv(trust_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')
    degree_table = generate_user_degree_table(data_path=data_path)

    all_path_list = []

    if all_node:
        num_nodes = len(social_graph.nodes())
    
    if seed:
        np.random.seed(62)

    # select target(anchor) nodes randomly. (without replacement)
    anchor_nodes = np.random.choice(social_graph.nodes(), size=num_nodes, replace=False)
    
    # At first, there is no previous node, so set it to None.
    for nodes in tqdm(anchor_nodes, desc="Generating random walk sequence..."):
        path_dict = {}
        path_dict[nodes] = [nodes]
        for _ in range(walk_length - 1):
            # Move to one of connected node randomly.
            if path_dict[nodes][-1] == nodes:
                next_node = find_next_social_node(graph=social_graph, previous_node=None, current_node=nodes, RETURN_PARAMS=0.0, seed=seed)
                path_dict[nodes].append(next_node)

            # If selected node was "edge node", there is no movable nodes, so pad it with 0(zero-padding).
            if path_dict[nodes][-1] == 0:
                path_dict[nodes].append(0)

            # Move to one of connected node randomly.
            else:
                next_node = find_next_social_node(graph=social_graph, previous_node=path_dict[nodes][-2], current_node=path_dict[nodes][-1], RETURN_PARAMS=0.0, seed=seed)
                path_dict[nodes].append(next_node)
        
        # Pop 1st element of list(since it is anchor node).
        del path_dict[nodes][0]
        
        # # Get each user's degree information from degree table.
        degree_list = []
        for node_list in path_dict.values():
            for node in node_list:
                if node != 0:
                    degree = degree_table['degree'].loc[degree_table['user_id'] == node].values[0]
                else:
                    # If node is 0 (zero-padded value), returns 0.
                    degree = 0
                degree_list.append(degree)
        
        # Add degree information to path_dict.
        path_dict = {key: [value, degree_list] for key, value in path_dict.items()}
        all_path_list.append(path_dict)

    if save_flag:
        # save result to .csv
        path = data_path + '/' + f"social_user_{num_nodes}_rw_length_{walk_length}_fixed_seed_{seed}.csv"

        keys, walks, degrees = [], [], []
        for paths in all_path_list:
            for key, value in paths.items():
                keys.append(key)
                walks.append(value[0])
                degrees.append(value[1])

        result_df = pd.DataFrame({
            'user_id':keys,
            'random_walk_seq':walks,
            'degree':degrees
        })
        result_df.sort_values(by=['user_id'], inplace=True)
        result_df.reset_index(drop=True, inplace=True)
        
        result_df.to_csv(path, index=False)
        
    return all_path_list


def find_next_social_node(graph:nx.Graph(), previous_node, current_node, RETURN_PARAMS, seed=False):
    """
    Find connected nodes, using transition probability. \n
        ###\n
            This code only finds un-visited nodes.\n
            (no re-visiting to previously visited node.)\n
        ###
    """
    if seed:
        np.random.seed(62)
    
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

def find_non_existing_user_in_social_graph(data_path):
    """
    There are some non-existing users in social network, while they exists in user-item network.
        rating: 7,375 user
        social: 7,317 user
        => 58 users are missing in social network.
    """
    rating = pd.read_csv(data_path + '/rating.csv', index_col=[])
    social = pd.read_csv(data_path + '/trustnetwork.csv', index_col=[])

    social_network = nx.from_pandas_edgelist(social, source='user_id_1', target='user_id_2')

    rating_user = rating['user_id'].unique()
    social_user = []
    for node in social_network.nodes():
        social_user.append(node)
    
    social_user = np.array(social_user)

    # print(social_user.shape, rating_user.shape)
    user_not_in_social_graph = np.setxor1d(rating_user, social_user)
    print(user_not_in_social_graph.tolist())
    print(len(user_not_in_social_graph))
    
    # get those user's interacted items
    for user in user_not_in_social_graph:
        interacted_items = rating['product_id'].loc[rating['user_id'] == user].values
        print(f"user {user}: {len(interacted_items)}")
    
    # TODO: return type?


def find_social_user_interacted_items(data_path:str, walk_list:list) -> dict:
    """
    Find (social) user's interacted items from user-item network. \n
        => 주어진 random walk sequence에서 sequence 상에 위치하는 사용자들이 상호작용을 한 item과 rating 정보를 get

    Return:
        list of items & ratings interacted with by all users in random walk sequence. \n
        => 주어진 random walk sequence에 해당하는 사용자(sequence node)가 상호작용한 아이템-평점 정보 리스트

    Args;
        data_path: path to data
        walk_list: random walk sequences, result of `generate_social_random_walk_sequence`. 
    """
    # Table which consists user's interacted item & rating information.
    user_item_table = generate_interacted_items_table(data_path)

    # Find random walk user's interacted items
        # this user value will used as key.
    interaction_dict = {}
    for walks in walk_list:                 # Overall random walk loop
        for key, value in walks.items():    # 1 random walk sequence loop
            for user_indexer in value[0]:   # Selected random walk sequence's node(user) loop
                interaction_dict[user_indexer] = [[], []]
                interacted_items = literal_eval(user_item_table['product_id'].loc[user_item_table['user_id'] == user_indexer].values[0])
                interacted_ratings = literal_eval(user_item_table['rating'].loc[user_item_table['user_id'] == user_indexer].values[0])
                interaction_dict[user_indexer][0] = interacted_items
                interaction_dict[user_indexer][1] = interacted_ratings
    
    # this dict will contain all user(in r.w. sequence's sequence node)'s interaction information from given walks.
    return interaction_dict


def find_selected_user_interacted_items(data_path:str, walk_list:list) -> dict:
    """
    Find selected user's interacted items from user-item network.
        => 주어진 random walk sequence에서 key에 해당하는 사용자들이 상호작용을 한 item과 rating 정보를 get

    Return:
        list of items & ratings interacted with users in random walk sequence's start. \n
        => 주어진 random walk sequence에서 시작점에 해당하는 사용자(start node)가 상호작용한 아이템-평점 정보

    Args;
        data_path: path to data
        walk_list: random walk sequences, result of `generate_social_random_walk_sequence`. 
    """
    user_item_table = generate_interacted_items_table(data_path)

    # Extract 'key' user(random walk sequence's starting node). 
    key_users = []
    for walks in walk_list:
        key_users.append(*walks)
    
    # Find those key user's interacted items & ratings.
    user_dict = {}
    for user in key_users:
        user_dict[user] = [[], []]
        interacted_items = literal_eval(user_item_table['product_id'].loc[user_item_table['user_id'] == user].values[0])
        interacted_ratings = literal_eval(user_item_table['rating'].loc[user_item_table['user_id'] == user].values[0])

        user_dict[user][0] = interacted_items
        user_dict[user][1] = interacted_ratings
    
    # This dict will contain all user(in r.w. sequence's start node)'s interaction information from given walks.
    return user_dict

## For checking
# sequence_list = generate_social_random_walk_sequence(data_path, num_nodes=10, walk_length=20, save_flag=True, all_node=True, seed=False)
# for sequences in sequence_list:
#     for key, value in sequences.items():
#         # print({f"{key} : {value}"})
#         print(key)
#         print('\t', value[0])
#         print('\t', value[1])
# interaction_df = find_interacted_items(data_path)
# print(interaction_df)

# walk_list = generate_social_random_walk_sequence(data_path, num_nodes=5, walk_length=2, save_flag=False, all_node=False, seed=True)
# key_user_item_list = find_selected_user_interacted_items(data_path, walk_list)
# walk_user_item_list = find_social_user_interacted_items(data_path, walk_list)

# # random walk sequence
# print(walk_list)
# print('\n')

# # random walk sequence에서 시작점에 해당하는 사용자 (input) 들이 상호작용한 item 목록
# print(key_user_item_list)
# print('\n')

# # random walk sequence에서 방문된 사용자(시작점 이후의 사용자)들이 상호작용한 item 목록
# print(walk_user_item_list)