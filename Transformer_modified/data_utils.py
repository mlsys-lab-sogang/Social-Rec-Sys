"""
trustnetwork 에서 모든 사용자의 degree 정보를 담은 table 생성

trustnetwork 에서 random walk sequence 생성
    - 임의의 사용자 n명을 선택
    - 각 사용자마다 random walk length r 만큼의 subgraph sequence 생성
    - 생성한 sequence에서, 각 노드와 매칭되는 degree 정보를 degree table에서 GET
    - [노드, 노드, 노드], [degree, degree, dgree] 를 함께 구성 (like PyG's edge_index)
        => [[node1, node2, node3]]
"""
import math
import os
import pickle
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
import torch
from scipy import sparse

# arg(or else) passing to DATASET later
# DATASET = 'ciao'

# data_path = os.getcwd() + '/dataset/' + DATASET


def mat_to_csv(data_path:str, test=0.1, seed=42):
    """
    Convert .mat file into .csv file for using pandas.
        Ciao: rating.mat, trustnetwork.mat
        Epinions: rating.mat, trustnetwork.mat
    
    Args:
        data_path: Path to .mat file
        test: percentage of test & valid dataset (default: 10%)
        seed: random seed (default=42)
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
    # print("before reset and filter")
    # quit()
    ### data filtering & id re-arrange ###
    # rating_df, trust_df = reset_and_filter_data(rating_df, trust_df)
    rating_df = pd.read_csv(data_path+'/rating.csv', index_col=[], dtype=int)#, names=['user_id', 'product_id', 'rating'])
    
    ### data filtering & id re-arrange ###
    # print("After reset and filter")
    # quit()
    # rating_matrix = rating_df.pivot_table(values='rating', index='user_id', columns='product_id').fillna(0).to_numpy()
    rating_matrix = sparse.lil_matrix((max(rating_df['user_id'].unique())+1, max(rating_df['product_id'].unique())+1), dtype=np.uint16)

    for index in rating_df.index:
        rating_matrix[rating_df['user_id'][index],rating_df['product_id'][index]] = rating_df['rating'][index]
    rating_matrix = rating_matrix.toarray()

    np.save(data_path + '/rating_matrix.npy', rating_matrix)    
    ### train test split TODO: Change equation for split later on
    # TODO: make random_state a seed varaiable
    split_rating_df = shuffle(rating_df, random_state=seed)
    num_test = int(len(split_rating_df) * test)
    rating_test_set = split_rating_df.iloc[:num_test]
    rating_valid_set = split_rating_df.iloc[num_test:2 * num_test]
    rating_train_set = split_rating_df.iloc[2 * num_test:]

    rating_df.to_csv(data_path + '/rating.csv', index=False)
    trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)

    rating_test_set.to_csv(data_path + '/rating_test.csv', index=False)
    rating_valid_set.to_csv(data_path + '/rating_valid.csv', index=False)
    rating_train_set.to_csv(data_path + '/rating_train.csv', index=False)

def reset_and_filter_data(rating_df:pd.DataFrame, trust_df:pd.DataFrame) -> pd.DataFrame:
    """
    Remove users not existing in social graph &
    Re-arrange data ids to be increasing by 1.
    
    This function is used in `mat_to_csv()`. 
    (`find_non_existing_user_in_social_graph()` is deprecated & merged into this function.)

    Args:
        rating_df: originally loaded `rating_df` (user-item interaction data)
        trust_df: originaly loaded `trust_df` (social data)
    """
    social_network = nx.from_pandas_edgelist(trust_df, source='user_id_1', target='user_id_2')

    social_ids = []
    for user_id in social_network.nodes():
        social_ids.append(user_id)
    
    user_item_ids = rating_df['user_id'].unique().tolist()

    # Find users not exists in social data
    non_users = np.setxor1d(user_item_ids, social_ids)

    # Remove users not exists in social data
        # Ciao: 7375 user (user-item) ==> 7317 user (social)
        # Epinions: 22164 user (user-item) ==> 18098 user (social)
    rating_df = rating_df[~rating_df['user_id'].isin(non_users.tolist())]

    # Generate user id mapping table
    mapping_table_user = {}
    new_id_user = 1
    for item in social_ids:
        mapping_table_user[item] = new_id_user
        new_id_user += 1
    
    # Generate item id mapping table
    mapping_table_item = {}
    new_id_item = 1
    for item in rating_df['product_id'].unique().tolist():
        mapping_table_item[item] = new_id_item
        new_id_item += 1
    # print("before replace")
    # Replace user id & item id, using id mapping table.
    # print(rating_df.head())
    # print(mapping_table_item)
    rating_df = rating_df.replace({'user_id': mapping_table_user, 'product_id':mapping_table_item})
    # print("after rating")
    trust_df = trust_df.replace({'user_id_1': mapping_table_user, 'user_id_2': mapping_table_user})
    # print("after repl")
    return rating_df, trust_df, 

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
        social_graph.to_csv(data_path + f'/trustnetwork_{split}.csv')
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
    # trust_file = data_path + f'/trustnetwork.csv'
    dataframe = pd.read_csv(trust_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')

    degrees = {node: val for (node, val) in social_graph.degree()}
    degree_df = pd.DataFrame(degrees.items(), columns=['user_id', 'degree'])

    degree_df.sort_values(by='user_id', ascending=True, inplace=True)

    degree_df.to_csv(data_path + f'/degree_table_social_{split}.csv', index=False)
    # degree_df.to_csv(data_path + f'/degree_table_social.csv', index=False)

    return degree_df


def generate_item_degree_table(data_path:str, split:str='train') -> pd.DataFrame:
    """
    Generate & return degree table from user-item graph(rating matrix).
    """
    # processed file check
    # if 'degree_table_item.csv' in os.listdir(data_path):
    #     print(f"Processed 'degree_table_item.csv' file already exists...")
    #     degree_df = pd.read_csv(data_path + '/degree_table_item.csv', index_col=[])
    #     return degree_df
    
    # user-item network
        # Ciao: 7375 user // 105114 items ==> 7317 user // 104975 items (after filtered)
    rating_file = data_path + f'/rating_{split}.csv'
    # rating_file = data_path + f'/rating.csv'
    dataframe = pd.read_csv(rating_file, index_col=[])

    # Since using NetworkX to compute bipartite graph's degree is time-consuming(because graph is too sparse),
    # we just use pandas for simple degree calculation.
    degree_df = dataframe.groupby('product_id')['user_id'].nunique().reset_index()
    # degree_df = dataframe.groupby('user_id')['product_id'].nunique().reset_index()

    degree_df.columns = ['product_id', 'degree']
    # degree_df.columns = ['user_id', 'degree']

    degree_df.to_csv(data_path + f'/degree_table_item_{split}.csv', index=False)
    # degree_df.to_csv(data_path + f'/degree_table_item.csv', index=False)

    return degree_df


def generate_interacted_items_table(data_path:str, item_length=4, all:bool=False, split:str='train') -> pd.DataFrame:
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
    degree_table = generate_item_degree_table(data_path=data_path, split=split)
    degree_table = dict(zip(degree_table['product_id'], degree_table['degree']))    # for id mapping.

    if all==True:
        user_item_dataframe = dataframe.groupby('user_id').agg({'product_id': list, 'rating': list}).reset_index()
        user_item_dataframe['product_degree'] = user_item_dataframe['product_id'].apply(lambda x: [degree_table[id] for id in x])

        # This is for indexing 0, where random walk sequence has padded with 0.
            # minimum number of interacted item is 4(before dataset splitting), so pad it to 4.
        empty_data = [0, [0 for _ in range(4)], [0 for _ in range(4)], [0 for _ in range(4)]]
        user_item_dataframe.loc[-1] = empty_data
        user_item_dataframe.index = user_item_dataframe.index + 1
        user_item_dataframe.sort_index(inplace=True)
        # user_item_dataframe.to_csv(data_path + '/user_item_interaction.csv', index=False)
        user_item_dataframe.to_csv(data_path + f'/user_item_interaction_{split}.csv', index=False)

        return user_item_dataframe
    
    # processed file check
    if f'user_item_interaction_item_length_{item_length}.csv' in os.listdir(data_path):
        print(f"Processed 'user_item_interaction_length_{item_length}.csv' file already exists...")
        user_item_dataframe = pd.read_csv(data_path + f'/user_item_interaction_item_length_{item_length}.csv', index_col=[])
        return user_item_dataframe
    
    # For each user, find their interacted items and given rating.
        ## This will make dict of dict: {user_id: {product_id:[...], rating:[...]}, user_id:{product_id:[...], rating:[...]}, ...}
    user_item_dataframe = dataframe.groupby('user_id').agg({'product_id': list, 'rating': list}).reset_index()

    # Sample fixed number of interacted items.
        # TODO: pad with 0 for 'len(product_id) < item_length'
    user_item_dataframe['indices'] = user_item_dataframe.apply(lambda x: np.random.choice(len(x['product_id']), item_length, replace=False), axis=1)
    
    user_item_dataframe['product_id'] = user_item_dataframe.apply(lambda x: [x['product_id'][i] for i in x['indices']], axis=1)
    user_item_dataframe['rating'] = user_item_dataframe.apply(lambda x: [x['rating'][i] for i in x['indices']], axis=1)
    user_item_dataframe.drop(columns=['indices'], inplace=True)

    # Fetch item's degree information from degree table.
    user_item_dataframe['product_degree'] = user_item_dataframe['product_id'].apply(lambda x: [degree_table[id] for id in x])

    # This is for indexing 0, where random walk sequence has padded with 0.
    empty_data = [0, [0 for _ in range(item_length)], [0 for _ in range(item_length)], [0 for _ in range(item_length)]]
    user_item_dataframe.loc[-1] = empty_data
    user_item_dataframe.index = user_item_dataframe.index + 1
    user_item_dataframe.sort_index(inplace=True)
    
    user_item_dataframe.to_csv(data_path + f'/user_item_interaction_item_length_{item_length}.csv', index=False)

    return user_item_dataframe

def generate_interacted_users_table(data_path:str, item_length=4, split:str='train') -> pd.DataFrame:
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

    user_item_dataframe = dataframe.groupby('product_id').agg({'user_id': list, 'rating': list}).reset_index()

    # This is for indexing 0, where random walk sequence has padded with 0.
        # minimum number of interacted item is 4(before dataset splitting), so pad it to 4.
    # empty_data = [0, [0 for _ in range(4)], [0 for _ in range(4)], [0 for _ in range(4)]]
    # user_item_dataframe.loc[-1] = empty_data
    # user_item_dataframe.index = user_item_dataframe.index + 1
    user_item_dataframe.sort_index(inplace=True)
    # user_item_dataframe.to_csv(data_path + '/user_item_interaction.csv', index=False)
    user_item_dataframe.to_csv(data_path + f'/item_user_interaction_{split}.csv', index=False)

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
    degree_table = generate_user_degree_table(data_path=data_path, split=split)

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
        path = data_path + '/' + f"social_user_{num_nodes}_rw_length_{walk_length}_fixed_seed_{seed}_split_{split}.csv"

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

def generate_input_sequence_data(data_path, seed:int, split:str='train', item_seq_len:int=250):
    """
    Prepare data to fed into Dataset class.

    data_path: path to dataset (/dataset/{ciao,epinions}/)
    split: data split type (train/valid/test)
    item_seq_len: pre-defined interacted item sequence length

    FIXME: 현재는 .csv로 저장 중. 추후 return을 한다면 아래와 같이 return을 할 수 있게 수정?

    Returns:
        user_seq:       고정된 길이의 사용자 랜덤워크 시퀀스, [num_user, seq_length]\n
        user_degree:    랜덤워크 시퀀스에서 출현한 사용자들의 degree 정보, [num_user, seq_length] \n
        item_seq:       랜덤워크 시퀀스에서 출현한 사용자들이 상호작용한 모든 아이템 리스트 \n
        item_rating:    랜덤워크 시퀀스에서 출현한 사용자들이 상호작용한 모든 아이템에 대한 rating 정보가 담긴 matrix \n
        item_degree:    선택된 아이템들의 degree 정보 (해당 아이템과 상호작용한 사용자의 수) \n
        spd_matrix:     현재 user_seq에 해당하는 사용자들의 SPD matrix (사전 생성한 전체 사용자의 SPD table에서 slicing한 matrix)
    """
    def slice_and_pad_list(input_list:list, slice_length:int):
        """
        Get list, and slice it by slice length, and pad with 0.
        """
        num_slices = math.ceil(len(input_list) / slice_length)

        # Pad input list with 0
        input_list += [0] * (slice_length * num_slices - len(input_list))

        # Create sliced & padded list
        result_list = [input_list[i:i + slice_length] for i in range(0, len(input_list), slice_length)]

        return result_list

    ## FIXME: 작성한 함수를 호출하도록 추후 수정
    files = os.listdir(data_path)
    for file_name in files:
        if 'social' in file_name and f'_split_{split}.csv' in file_name:
            user_path = file_name
    item_path = f'user_item_interaction_{split}.csv'
    spd_path = 'shortest_path_result.npy'
    item_rating_path = f'rating_{split}.csv'

    # Load dataset & convert data type
    user_df = pd.read_csv(data_path + '/' + user_path, index_col=[])
    user_df['random_walk_seq'] = user_df.apply(lambda x: literal_eval(x['random_walk_seq']), axis=1)
    user_df['degree'] = user_df.apply(lambda x: literal_eval(x['degree']), axis=1)

    item_df = pd.read_csv(data_path + '/' + item_path, index_col=[])
    item_df['product_id'] = item_df.apply(lambda x: literal_eval(x['product_id']), axis=1)
    item_df['rating'] = item_df.apply(lambda x: literal_eval(x['rating']), axis=1)
    item_df['product_degree'] = item_df.apply(lambda x: literal_eval(x['product_degree']), axis=1)

    # Load SPD table => 각 sequence마다 [seq_len_user, seq_len_user] 크기의 SPD matrix를 생성하도록.
    spd_table = torch.from_numpy(np.load(data_path + '/' + spd_path)).long()

    # Load rating table => 마찬가지로 각 sequence마다 [seq_len_user, seq_len_item] 크기의 rating matrix를 생성하도록.
    # rating_table = pd.read_csv(data_path + '/' + item_rating_path, index_col=[])
    rating_matrix = np.load(data_path + '/rating_matrix.npy')#pd.DataFrame(np.load(data_path + '/rating_matrix.npy'))
    # print(len(rating_matrix))
    # print(len(rating_matrix[0]))
    # print(rating_matrix)
    # quit()
    # print(rating_matrix)
    # quit()

    total_df = pd.DataFrame(columns=['user_id', 'user_sequences', 'user_degree', 'item_sequences', 'item_degree', 'item_rating', 'spd_matrix'])
    for _, data in tqdm(user_df.iterrows(), total=user_df.shape[0]):
        current_user = data['user_id']
        current_sequence = data['random_walk_seq']
        current_degree = data['degree']

        item_indexer = [int(x) for x in current_sequence]
        item_list, degree_list = [], []

        # 1개의 rw sequence에 있는 사용자들이 상호작용한 모든 아이템 & 해당 아이템들의 degree 가져와서
        for index in item_indexer:
            # item_df.iloc[index][]
            item_list.append(item_df.loc[item_df['user_id'] == index]['product_id'].values[0])
            degree_list.append(item_df.loc[item_df['user_id'] == index]['product_degree'].values[0])
        
        # 이를 flatten하고
        flat_item_list = [item for row in item_list for item in row]
        flat_degree_list = [item for row in degree_list for item in row]
        
        # 중복을 제거
        item_list_removed_duplicate = list(set(flat_item_list))

        # flat_item_list 원소와 대응되는 degree를 추출
        mapping_dict = {}
        for item, degree in zip(flat_item_list, flat_degree_list):
            if item not in mapping_dict:
                mapping_dict[item] = degree
        
        # 추출한 degree를 item_list_removed_duplicate에 대응하여 list 생성
        degree_list_removed_duplicate = [mapping_dict[item] for item in item_list_removed_duplicate]

        # 중복제거한 list를 정해진 길이 (item_seq_length) 만큼 자르기
        sliced_item_list = slice_and_pad_list(item_list_removed_duplicate, slice_length=item_seq_len)
        # sliced_rating_list = slice_and_pad_list(flat_rating_list, slice_length=item_seq_len)
        sliced_degree_list = slice_and_pad_list(degree_list_removed_duplicate, slice_length=item_seq_len)

        # 현재 user sequence에 해당하는 spd matrix를 생성
            # IndexError: index 7317 is out of bounds for dimension 0 with size 7317
            # spd matrix index가 0 ~ 7316 
            # sequence (user_id)가 1 ~ 7317
            # 따라서 -1을 해줘야 한다.
        spd_matrix = spd_table[torch.LongTensor(current_sequence).squeeze() - 1, :][:, torch.LongTensor(current_sequence).squeeze() - 1]

        # 자른 list와 위 정보들을 dataframe에 담아서 저장
        for item_list, degree_list in zip(sliced_item_list, sliced_degree_list):

            # 현재 선택된 user_seq에 있는 사용자들과 sliced item_seq에 대해 [seq_len_user, seq_len_item] 크기의 rating table 생성
            small_rating_matrix = torch.zeros((len(current_sequence), item_seq_len))
            # print(len(rating_matrix[0]))
            for i in range(small_rating_matrix.shape[0]):      # user loop (row)
                matrix_user = current_sequence[i]
                for j in range(small_rating_matrix.shape[1]):  # item loop (col)
                    matrix_item = item_list[j]
                    # print(rating_matrix)
                    # print(matrix_user, matrix_item)
                    # matrix_rating = rating_table.loc[(rating_table['user_id'] == matrix_user) & (rating_table['product_id'] == matrix_item)]['rating'].values
                    matrix_rating = rating_matrix[matrix_user][matrix_item]
                    # if len(matrix_rating) == 0:
                    #     continue
                    small_rating_matrix[i][j] = matrix_rating#[0]
            
            # total_df.loc[len(total_df)] = [torch.LongTensor(current_user), 
            #                                torch.LongTensor(current_sequence), 
            #                                torch.LongTensor(current_degree), 
            #                                torch.LongTensor(item_list), 
            #                                torch.LongTensor(degree_list), 
            #                                rating_matrix.to(torch.long), 
            #                                spd_matrix.to(torch.long)]
            total_df.loc[len(total_df)] = [current_user, 
                                            current_sequence, 
                                            current_degree, 
                                            item_list, 
                                            degree_list, 
                                            small_rating_matrix.numpy(),
                                            spd_matrix.numpy()]
    # print(total_df.head())
    # quit()
    ########################### FIXME: 초안모델 디버깅용으로 user_id=100 까지만 기록. ###########################
    ########################### FIXME: 그리고 전체 user_id 쓸때는 파일 이름 변경.    ###########################
        # if current_user == 100:
        #     break
    
    ## to_csv는 string으로 저장해버려서 array 중간이 ... 으로 저장됨.
    ## to_parquet, to_feather는 type이 다른 컬럼이 존재할 경우 error.
    ## pickle 저장이 가장 나을듯.
    # total_df.to_csv(data_path + f"/sequence_data_itemseq_{item_seq_len}_{split}.csv", index=False)
    # total_df.to_parquet(data_path + f"/sequence_data_itemseq_{item_seq_len}_{split}.parquet", engine='pyarrow',compression='gzip', index=False)
    # with open(data_path + f"/sequence_data_num_user_100_itemseq_{item_seq_len}_{split}.pkl", "wb") as file:
    with open(data_path + f"/sequence_data_seed_{seed}_itemlen_{item_seq_len}_{split}.pkl", "wb") as file:
        pickle.dump(total_df, file)
    ######################################################################################################


if __name__ == "__main__":
    ##### For checking & debugging (will remove later)

    data_path = os.getcwd() + '/dataset/' + 'epinions'
    # generate_input_sequence_data(data_path=data_path, split='train', item_seq_len=250)
    # user_sequences, user_degree, item_sequences, item_rating, item_degree = generate_sequence_data(data_path=data_path, split='train')
    # print(user_sequences.shape)
    # quit() 
    
    # data_path = os.getcwd() + '/dataset/' + 'ciao' 
    # rating_file = data_path + '/rating_test.csv'
    # generate_social_dataset(data_path, save_flag=True, split='train')
    # mat_to_csv(data_path)
    generate_input_sequence_data(data_path, seed=42)
    quit()
    # print(generate_interacted_items_table(data_path, all=True))
    # print(generate_sequence_data('ciao')[0][0])
    interacted_items = list(map(len, generate_sequence_data("ciao")))
    print(max(interacted_items))
    print(min(interacted_items))
    # print(generate_sequence_data('ciao')[1])#[0])
    
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