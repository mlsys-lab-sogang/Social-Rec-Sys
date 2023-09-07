import os
import numpy as np
import pandas as pd
from ast import literal_eval

import torch
from torch.utils.data import Dataset, DataLoader

import data_utils as utils

class MyDataset(Dataset):
    """
    Process & make dataset for DataLoader.
        __init__():     data load & convert to Tensor
        __getitem__():  data indexing (e.g. dataset[0])
        __len__():      length of dataset (e.g. len(dataset))
        load_data():    load data and preprocess data into ndarray
    """
    def __init__(self, dataset:str):
        """
        Args:
            dataset: raw dataset name (ciao or epinions)
        """
        self.data_path = os.getcwd() + '/dataset/' + dataset

        user_sequences, user_degree, item_sequences, item_rating = self.load_data()

        self.user_sequences = torch.LongTensor(user_sequences)      # (num_user, item_length)
        self.user_degree = torch.LongTensor(user_degree)            # (num_user, item_length)
        self.item_sequences = torch.LongTensor(item_sequences)      # (num_user, item_length)
        self.item_rating = torch.LongTensor(item_rating)            # (num_user, item_length)

        # assert len(self.user_sequences)+1 == len(self.user_degree)+1 == len(self.item_sequences) == len(self.item_rating), (
        #     f"All data should have same length: {len(self.user_sequences)}, {len(self.user_degree)}, {len(self.item_sequences)}, {len(self.item_rating)}"
        # )

    def __len__(self):
        return len(self.user_sequences)
    
    def __getitem__(self, idx):
        user_seq = self.user_sequences[idx]
        user_deg = self.user_degree[idx]

        # since we need all user in random walk sequence's interacted items, fetch it with sequence value as index.
        item_indexer = [int(x) for x in user_seq.numpy()]   # user_ids in user_seq
        item_list, rating_list = [], []
        for index in item_indexer:
            item_list.append(self.item_sequences[index])
            rating_list.append(self.item_rating[index])

        #########################Set으로 sequence의 모든 아이템 받기###############################
        item_set, rating_set = set(), set()
        for index in item_indexer:
            item_set.update(self.item_sequences[index])
            rating_set.update(self.item_rating[index])
        ######################################################################################

        item_seq = torch.stack(item_list, 0)
        item_rating = torch.stack(rating_list, 0)

        # TODO: Used dictionary to perform PyG-like data accessing.
        # return user_seq, user_deg, item_seq, item_rating
        return {'user_seq': user_seq, 'user_degree': user_deg, 'item_seq': item_seq, 'rating': item_rating}

    def load_data(self):
        """
        Data load & preprocess to ndarray
            # TODO: fix or handle later (using function, not reading .csv)
        """
        user_path = '/social_user_7317_rw_length_20_fixed_seed_False.csv'
        # user_path = '/social_user_18098_rw_length_20_fixed_seed_False.csv'
        item_df = utils.generate_interacted_items_table(data_path=self.data_path, item_length=1)

        #########################Set으로 sequence의 모든 아이템 받기###############################
        # item_df = utils.generate_interacted_items_table(data_path=self.data_path, all=True)
        ######################################################################################

        # load dataset & convert data type
            # values are saved as 'str', convert into original type, 'list'.
        user_df = pd.read_csv(self.data_path + user_path, index_col=[])
        user_df['random_walk_seq'] = user_df.apply(lambda x: literal_eval(x['random_walk_seq']), axis=1)
        user_df['degree'] = user_df.apply(lambda x: literal_eval(x['degree']), axis=1)

        item_df['product_id'] = item_df.apply(lambda x: literal_eval(x['product_id']), axis=1)
        item_df['rating'] = item_df.apply(lambda x: literal_eval(x['rating']), axis=1)
        
        # Since each row's element is list, convert it.
            # `user_sequences` & `user_degree` shape: (num_user, walk_length)
        user_sequences = user_df['random_walk_seq'].to_numpy(dtype=object)
        user_sequences = np.array([np.array(x) for x in user_sequences])
        user_degree = user_df['degree'].to_numpy(dtype=object)
        user_degree = np.array([np.array(x) for x in user_degree])

        # Since item_df's row element's type is 'list'(not 'str' like user_df), just convert it into ndarray.
        item_sequences = item_df['product_id'].to_numpy(dtype=object)
        item_sequences = np.array([np.array(x) for x in item_sequences])
        item_rating = item_df['rating'].to_numpy(dtype=object)
        item_rating = np.array([np.array(x) for x in item_rating])

        return user_sequences, user_degree, item_sequences, item_rating


if __name__ == "__main__":
    # for testing & debugging
    # dataset = MyDataset(dataset='ciao')
    dataset = MyDataset(dataset='ciao')
    print(dataset[0])
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for data in loader:
        # print(data[0].shape)
        # print(data[1].shape)
        # print(data[2].shape)
        # print(data[3].shape)
        # print(data['user_seq'].shape)
        # print(data['user_seq'][0].shape)
        quit()