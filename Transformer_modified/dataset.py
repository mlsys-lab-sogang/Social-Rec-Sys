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

        self.user_sequences = torch.FloatTensor(user_sequences)     # (num_user, walk_length)
        self.user_degree = torch.FloatTensor(user_degree)           # (num_user, walk_length)
        self.item_sequences = torch.FloatTensor(item_sequences)     # (num_user, item_length)
        self.item_rating = torch.FloatTensor(item_rating)           # (num_user, item_length)

        assert len(self.user_sequences) == len(self.user_degree) == len(self.item_sequences) == len(self.item_rating), (
            f"All data should have same length: {len(self.user_sequences)}, {len(self.user_degree)}, {len(self.item_sequences)}, {len(self.item_rating)}"
        )

    def __len__(self):
        return len(self.user_sequences)
    
    def __getitem__(self, idx):
        user_seq = self.user_sequences[idx]
        user_deg = self.user_degree[idx]
        item_seq = self.item_sequences[idx]
        item_rating = self.item_rating[idx]
    
        return user_seq, user_deg, item_seq, item_rating

    def load_data(self):
        """
        Data load & preprocess to ndarray
            # TODO: fix or handle later (using function, not reading .csv)
        """
        user_path = '/social_user_7317_rw_length_20_fixed_seed_False.csv'
        item_df = utils.generate_interacted_items_table(data_path=self.data_path, item_length=4)

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

        # TODO: since there are users doesn't exist in social graph, but exists in user-item graph, drop those users in item table.
        non_users = utils.find_non_existing_user_in_social_graph(data_path=self.data_path)
        item_df = item_df[~item_df['user_id'].isin(non_users)]
        item_df.reset_index(drop=True, inplace=True)

        # Since item_df's row element's type is 'list'(not 'str' like user_df), just convert it into ndarray.
        item_sequences = item_df['product_id'].to_numpy(dtype=object)
        item_sequences = np.array([np.array(x) for x in item_sequences])
        item_rating = item_df['rating'].to_numpy(dtype=object)
        item_rating = np.array([np.array(x) for x in item_rating])

        return user_sequences, user_degree, item_sequences, item_rating


if __name__ == "__main__":
    # for testing & debugging
    dataset = MyDataset(dataset='ciao')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in loader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        print(data[3].shape)
        quit()