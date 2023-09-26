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

    elif dataset_name == 'ml-100k':
        
        rating_df = pd.read_csv(data_path+'/u.data', sep="\t", names=['user_id','item_id','rating','timestamp'])
        rating_df.drop(['timestamp'], axis=1, inplace=True)
    # trust_file = loadmat(data_path + '/' + 'trustnetwork.mat')
    # trust_file = trust_file['trustnetwork'].astype(np.int64)    
    # trust_df = pd.DataFrame(trust_file, columns=['user_id_1', 'user_id_2'])
    # print("before reset and filter")
    # quit()
    ### data filtering & id re-arrange ###
    # rating_df, trust_df = reset_and_filter_data(rating_df, trust_df)
    ### data filtering & id re-arrange ###
    # print("After reset and filter")
    # quit()

    ### train test split TODO: Change equation for split later on
    # TODO: make random_state a seed varaiable
    split_rating_df = shuffle(rating_df, random_state=seed)
    num_test = int(len(split_rating_df) * test)
    rating_test_set = split_rating_df.iloc[:num_test]
    rating_valid_set = split_rating_df.iloc[num_test:2 * num_test]
    rating_train_set = split_rating_df.iloc[2 * num_test:]

    rating_df.to_csv(data_path + '/rating.csv', index=False)
    # trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)

    rating_test_set.to_csv(data_path + '/rating_test.csv', index=False)
    rating_valid_set.to_csv(data_path + '/rating_valid.csv', index=False)
    rating_train_set.to_csv(data_path + '/rating_train.csv', index=False)

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
    # degree_table = generate_item_degree_table(data_path=data_path, split=split)
    # degree_table = dict(zip(degree_table['product_id'], degree_table['degree']))    # for id mapping.

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
    
    # # processed file check
    # if f'user_item_interaction_item_length_{item_length}.csv' in os.listdir(data_path):
    #     print(f"Processed 'user_item_interaction_length_{item_length}.csv' file already exists...")
    #     user_item_dataframe = pd.read_csv(data_path + f'/user_item_interaction_item_length_{item_length}.csv', index_col=[])
    #     return user_item_dataframe
    
    # # For each user, find their interacted items and given rating.
    #     ## This will make dict of dict: {user_id: {product_id:[...], rating:[...]}, user_id:{product_id:[...], rating:[...]}, ...}
    # user_item_dataframe = dataframe.groupby('user_id').agg({'product_id': list, 'rating': list}).reset_index()

    # # Sample fixed number of interacted items.
    #     # TODO: pad with 0 for 'len(product_id) < item_length'
    # user_item_dataframe['indices'] = user_item_dataframe.apply(lambda x: np.random.choice(len(x['product_id']), item_length, replace=False), axis=1)
    
    # user_item_dataframe['product_id'] = user_item_dataframe.apply(lambda x: [x['product_id'][i] for i in x['indices']], axis=1)
    # user_item_dataframe['rating'] = user_item_dataframe.apply(lambda x: [x['rating'][i] for i in x['indices']], axis=1)
    # user_item_dataframe.drop(columns=['indices'], inplace=True)

    # # Fetch item's degree information from degree table.
    # user_item_dataframe['product_degree'] = user_item_dataframe['product_id'].apply(lambda x: [degree_table[id] for id in x])

    # # This is for indexing 0, where random walk sequence has padded with 0.
    # empty_data = [0, [0 for _ in range(item_length)], [0 for _ in range(item_length)], [0 for _ in range(item_length)]]
    # user_item_dataframe.loc[-1] = empty_data
    # user_item_dataframe.index = user_item_dataframe.index + 1
    # user_item_dataframe.sort_index(inplace=True)
    
    # user_item_dataframe.to_csv(data_path + f'/user_item_interaction_item_length_{item_length}.csv', index=False)

    # return user_item_dataframe

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



if __name__ == "__main__":
    ##### For checking & debugging (will remove later)

    data_path = "../../datasets/ml-100k"
    mat_to_csv(data_path=data_path)
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