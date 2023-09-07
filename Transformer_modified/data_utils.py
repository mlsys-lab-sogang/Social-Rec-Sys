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
import random
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ast import literal_eval    # convert str type list to original type
from scipy.io import loadmat
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.utils import shuffle

# arg(or else) passing to DATASET later
# DATASET = 'ciao'

# data_path = os.getcwd() + '/dataset/' + DATASET


def mat_to_csv(data_path:str, test=0.1):
    """
    Convert .mat file into .csv file for using pandas.
        Ciao: rating.mat, trustnetwork.mat
        Epinions: rating.mat, trustnetwork.mat
    
    Args:
        data_path: Path to .mat file
    """
    dataset_name = data_path.split('/')[-1]

    # processed file check
    # if ('rating.csv' or 'trustnetwork.csv') in os.listdir(data_path):
    #     print("Processed files already exists...")
    #     return 0
    
    # load .mat file & convert to dataframe
    if dataset_name == 'ciao':
        rating_file = loadmat(data_path + '/' + 'rating.mat')
        rating_file = rating_file['rating'].astype(np.int64)
        rating_df = pd.DataFrame(rating_file, columns=['user_id', 'product_id', 'category_id', 'rating', 'helpfulness'])

        # drop unused columns (TODO: Maybe used later)
        rating_df.drop(['category_id', 'helpfulness'], axis=1, inplace=True)
    
    elif dataset_name == 'epinions':
        rating_file = loadmat(data_path + '/' + 'rating_with_timestamp.mat')
        rating_file = rating_file['rating_with_timestamp'].astype(np.int64)
        rating_df = pd.DataFrame(rating_file, columns=['user_id', 'product_id', 'category_id', 'rating', 'helpfulness', 'timestamp'])

        # drop unused columns (TODO: Maybe used later)
        rating_df.drop(['category_id', 'helpfulness', 'timestamp'], axis=1, inplace=True)

    trust_file = loadmat(data_path + '/' + 'trustnetwork.mat')
    trust_file = trust_file['trustnetwork'].astype(np.int64)    
    trust_df = pd.DataFrame(trust_file, columns=['user_id_1', 'user_id_2'])

    ### train test split TODO: Change equation for split later on
    # TODO: make random_state a seed varaiable
    split_rating_df = shuffle(rating_df, random_state=42)
    num_test = int(len(split_rating_df) * test)
    rating_test_set = split_rating_df.iloc[:num_test]
    rating_valid_set = split_rating_df.iloc[num_test:2 * num_test]
    rating_train_set = split_rating_df.iloc[2 * num_test:]


    rating_df.to_csv(data_path + '/rating.csv', index=False)
    trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)

    rating_test_set.to_csv(data_path + '/rating_test.csv', index=False)
    rating_valid_set.to_csv(data_path + '/rating_valid.csv', index=False)
    rating_train_set.to_csv(data_path + '/rating_train.csv', index=False)


def generate_social_dataset(data_path:str, save_flag:bool = False, split:str='train'):
    """
    Generate social graph from train/test/validation dataset

    Args:
        data_path: path to dataset
        rating_file: path to rating file
    """
    rating_dataframe = pd.read_csv(f'{data_path}/rating_{split}.csv' , index_col=[])
    users = set(pd.unique(rating_dataframe['user_id']))
    trust_dataframe = pd.read_csv(data_path + '/trustnetwork.csv', index_col=[])
    social_graph = trust_dataframe[(trust_dataframe['user_id_1'].isin(users)) & (trust_dataframe['user_id_2'].isin(users))]
    if save_flag:
        social_graph.to_csv(data_path + f'/trustnetwork_{split}')
    return social_graph

def generate_user_degree_table(data_path:str, split:str='train') -> pd.DataFrame:
    """
    Generate & return degree table from social graph(trustnetwork).
    """
    # processed file check
    # if 'degree_table_social.csv' in os.listdir(data_path):
    #     print("Processed 'degree_table_social.csv' file already exists...")
    #     degree_df = pd.read_csv(data_path + '/degree_table_social.csv', index_col=[])
    #     return degree_df

    # user-user network
        # Ciao: 7317 users
        # Epinions: 18098 users
    trust_file = data_path + f'/trustnetwork_{split}.csv'
    dataframe = pd.read_csv(trust_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')

    degrees = {node: val for (node, val) in social_graph.degree()}
    degree_df = pd.DataFrame(degrees.items(), columns=['user_id', 'degree'])

    degree_df.sort_values(by='user_id', ascending=True, inplace=True)

    degree_df.to_csv(data_path + '/degree_table_social.csv', index=False)

    return degree_df


def generate_item_degree_table(data_path:str) -> pd.DataFrame:
    """
    Generate & return degree table from user-item graph(rating matrix).
    """
    # processed file check
    # if 'degree_table_item.csv' in os.listdir(data_path):
    #     print(f"Processed 'degree_table_item.csv' file already exists...")
    #     degree_df = pd.read_csv(data_path + '/degree_table_item.csv', index_col=[])
    #     return degree_df
    
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


def generate_interacted_items_table(data_path:str, item_length=4, all:bool=False, split:str='all') -> pd.DataFrame:
    """
    Generate & return user's interacted items & ratings table from user-item graph(rating matrix)

    Args:
        data_path: path to dataset
        item_length: number of interacted items to fetch
    """
    
    if split=='all':
        rating_file = data_path + '/rating.csv'
    else:
        rating_file = data_path + f'/rating_{split}.csv'
        
    dataframe = pd.read_csv(rating_file, index_col=[])

    if all==True:
        user_item_dataframe = dataframe.groupby('user_id').agg({'product_id': list, 'rating': list}).reset_index()
        user_item_dataframe.to_csv(data_path + '/user_item_interaction.csv', index=False)
        return user_item_dataframe
    
    # processed file check
    if f'user_item_interaction_item_length_{item_length}.csv' in os.listdir(data_path):
        print(f"Processed 'user_item_interaction_length_{item_length}.csv' file already exists...")
        user_item_dataframe = pd.read_csv(data_path + f'/user_item_interaction_item_length_{item_length}.csv', index_col=[])
        return user_item_dataframe
    
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
    # user_item_dataframe.to_csv(data_path + '/user_item_interaction.csv', index=False)
    # user_item_dataframe = user_item_dataframe.set_index('user_id').to_dict(orient='index')

    # Sample fixed number of interacted items.
        # TODO: pad with 0 for 'len(product_id) < item_length'
    user_item_dataframe['indices'] = user_item_dataframe.apply(lambda x: np.random.choice(len(x['product_id']), item_length, replace=False), axis=1)
    
    user_item_dataframe['product_id'] = user_item_dataframe.apply(lambda x: [x['product_id'][i] for i in x['indices']], axis=1)
    user_item_dataframe['rating'] = user_item_dataframe.apply(lambda x: [x['rating'][i] for i in x['indices']], axis=1)
    user_item_dataframe.drop(columns=['indices'], inplace=True)

    # This is for indexing 0, where random walk sequence has padded with 0.
    empty_data = [0, [0 for _ in range(item_length)], [0 for _ in range(item_length)]]
    user_item_dataframe.loc[-1] = empty_data
    user_item_dataframe.index = user_item_dataframe.index + 1
    user_item_dataframe.sort_index(inplace=True)
    
    user_item_dataframe.to_csv(data_path + f'/user_item_interaction_item_length_{item_length}.csv', index=False)

    return user_item_dataframe


def generate_social_random_walk_sequence(data_path:str, num_nodes:int=10, walk_length:int=5, save_flag=False, all_node=False, seed=False, split:str='train') -> list:
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
    trust_file = data_path + f'/trustnetwork_{split}.csv'
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
        # for _ in range(walk_length - 1):
        for _ in range(walk_length):
            # Move to one of connected node randomly.
            if path_dict[nodes][-1] == nodes:
                next_node = find_next_social_node(graph=social_graph, previous_node=None, current_node=nodes, RETURN_PARAMS=0.0, seed=seed)
                path_dict[nodes].append(next_node)

            # If selected node was "edge node", there is no movable nodes, so pad it with 0(zero-padding).
            elif path_dict[nodes][-1] == 0:
                path_dict[nodes].append(0)

            # Move to one of connected node randomly.
            else:
                next_node = find_next_social_node(graph=social_graph, previous_node=path_dict[nodes][-2], current_node=path_dict[nodes][-1], RETURN_PARAMS=0.0, seed=seed)
                path_dict[nodes].append(next_node)
            
            if len(path_dict[nodes]) > 20:
                continue
        
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

def find_non_existing_user_in_social_graph(data_path, print_flag=False, save_flag=False):
    """
    There are some non-existing users in social network, while they exists in user-item network.
        Ciao:
            rating: 7,375 user
            social: 7,317 user
            => 58 users are missing in social network.
        Epinions:
            rating: 22,164 user
            social: 18,098 user
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

    if print_flag:
        print(f"Users not exists (num: {len(user_not_in_social_graph)}): {user_not_in_social_graph.tolist()}")
        
        # get those user's interacted items
        for user in user_not_in_social_graph:
            interacted_items = rating['product_id'].loc[rating['user_id'] == user].values
            print(f"user {user}: {len(interacted_items)}")

    if save_flag:
        # Save `rating.csv`, eliminating non-existing users.
        rating = rating[~rating['user_id'].isin(user_not_in_social_graph.tolist())]
        rating.to_csv(data_path + '/rating.csv', index=False)
        return 0
    else:
        return user_not_in_social_graph.tolist()


if __name__ == "__main__":
    ##### For checking & debugging (will remove later)
    
    data_path = os.getcwd() + '/dataset/' + 'ciao' 
    rating_file = data_path + '/rating_test.csv'
    # generate_social_dataset(data_path, save_flag=True, split='train')
    mat_to_csv(data_path)
    # user_item_table = generate_interacted_items_table(data_path, all=True)
    
    # user_item_table = generate_interacted_items_table(data_path, all=True, split='train')
    

    # user_item_table['product_id'] = user_item_table.apply(lambda x: literal_eval(x['product_id']), axis=1)
    # user_item_table['rating'] = user_item_table.apply(lambda x: literal_eval(x['rating']), axis=1)
    # print(user_item_table['product_id'])
    # print(user_item_table['product_id'].to_dict())
    # dd = defaultdict(list)
    # print(user_item_table.to_dict(into=dd)['product_id'])
    # print(type(user_item_table.to_dict()['rating'][3440]))
    # mat_to_csv(data_path)
    # generate_social_random_walk_sequence(data_path=data_path, all_node=True, walk_length=20, save_flag=True)
    # walk_seq = generate_social_random_walk_sequence(data_path=data_path, num_nodes=5, walk_length=10, save_flag=False, all_node=False, seed=True)
    # print(walk_seq)
    # a = find_social_user_interacted_items(data_path, walk_list=walk_seq, item_length=4)
    # df = generate_interacted_items_table(data_path, 4)
    # print(df)
    quit()