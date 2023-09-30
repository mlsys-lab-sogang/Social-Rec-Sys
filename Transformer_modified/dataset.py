import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Process & make dataset for DataLoader
        __init__():     data load & convert to Tensor
        __getitem__():  data indexing (e.g. dataset[0])
        __len__():      length of dataset (e.g. len(dataset))
    """
    def __init__(self, dataset:str, split:str, seed:int, item_seq_len:int):
        """
        Args:
            dataset: raw dataset name (ciao // epinions)
            split: dataset split type (train // valid // test)
            seed: random seed used in dataset split
            item_seq_len: length of item list (processed in `data_utils.py`)
        """
        self.data_path = os.getcwd() + '/dataset/' + dataset


        #################### FIXME: user 100명에 대해서 test 
        # columns: user_id, user_sequences, user_degree, item_sequences, item_degree, item_rating, spd_matrix
        # with open(self.data_path + '/' + f'sequence_data_num_user_100_itemseq_250_{split}.pkl', 'rb') as file:
            # dataframe = pickle.load(file)
        
        # 모든 user
        with open(self.data_path + '/' + f'sequence_data_itemlen_250_{split}.pkl', 'rb') as file:
            dataframe = pickle.load(file)

        with open(self.data_path + '/' f'sequence_data_seed_{seed}_itemlen_{item_seq_len}_{split}.pkl', 'rb') as file:
            dataframe = pickle.load(file)
        ####################




        user_sequences = dataframe['user_sequences'].values
        user_sequences = np.array([np.array(x) for x in user_sequences])

        user_degree = dataframe['user_degree'].values
        user_degree = np.array([np.array(x) for x in user_degree])

        item_sequences = dataframe['item_sequences'].values
        item_sequences = np.array([np.array(x) for x in item_sequences])

        item_degree = dataframe['item_degree'].values
        item_degree = np.array([np.array(x) for x in item_degree])

        # shape: [total_samples(num_row), seq_len_user, seq_len_item]
        rating_matrix = dataframe['item_rating'].values
        rating_matrix = np.array([np.array(x) for x in rating_matrix])

        # shape: [total_samples(num_row), seq_len_user, seq_len_user]
        spd_matrix = dataframe['spd_matrix'].values
        spd_matrix = np.array([np.array(x) for x in spd_matrix])

        self.user_sequences = torch.LongTensor(user_sequences)
        self.user_degree = torch.LongTensor(user_degree)
        self.item_sequences = torch.LongTensor(item_sequences)
        self.item_degree = torch.LongTensor(item_degree)
        self.rating_matrix = torch.LongTensor(rating_matrix)
        self.spd_matrix = torch.LongTensor(spd_matrix)
    
    def __len__(self):
        # 전체 {train/valid/test}.csv의 길이 (dataframe의 전체 row 갯수)
        return len(self.user_sequences)

    def __getitem__(self, index):
        user_seq = self.user_sequences[index]
        user_deg = self.user_degree[index]
        item_seq = self.item_sequences[index]
        item_deg = self.item_degree[index]
        rating_table = self.rating_matrix[index]
        spd_table = self.spd_matrix[index]

        batch_data = {
            'user_seq': user_seq,
            'user_degree': user_deg,
            'item_list': item_seq,
            'item_degree': item_deg,
            'item_rating': rating_table,
            'spd_matrix': spd_table
        }

        return batch_data
        

if __name__ == "__main__":
    dataset = 'ciao'
    split = 'train'

    dataset = MyDataset(dataset, split)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for data in loader:
        print(data['user_seq'].shape)
        print(data['user_degree'].shape)
        print(data['item_list'].shape)
        print(data['item_degree'].shape)
        print(data['item_rating'].shape)
        print(data['spd_matrix'].shape)
        quit()