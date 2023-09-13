import os
import time
import numpy as np
import pandas as pd
from ast import literal_eval

import torch
from torch.utils.data import Dataset, DataLoader

import data_utils as utils

class MyDataset(Dataset):
    """
    Process & make dataset for DataLoader
        __init__():     data load & convert to Tensor
        __getitem__():  data indexing (e.g. dataset[0])
        __len__():      length of dataset (e.g. len(dataset))
    """
    def __init__(self, dataset:str, split:str):
        """
        Args:
            dataset: raw dataset name (ciao // epinions)
            split: dataset split type (train // valid // test)
        """
        self.data_path = os.getcwd() + '/dataset/' + dataset
        self.rating_table = pd.read_csv(self.data_path + '/' + f"rating_{split}.csv", index_col=[])

        using_file = f"sequence_data_{split}.csv"
        spd_file = self.data_path + '/' + 'shortest_path_result.npy'

        # start_time = time.time()

        # columns: user_id, user_sequences, user_degree, item_sequences, item_degree
        dataframe = pd.read_csv(self.data_path + '/' + using_file, index_col=[])
        
        # convert data types (list(str) => list(list))
        user_seq = dataframe.apply(lambda x: literal_eval(x['user_sequences']), axis=1).to_numpy(dtype=object)
        user_seq = np.array([np.array(x) for x in user_seq])
        
        user_deg = dataframe.apply(lambda x: literal_eval(x['user_degree']), axis=1).to_numpy(dtype=object)
        user_deg = np.array([np.array(x) for x in user_deg])

        item_seq = dataframe.apply(lambda x: literal_eval(x['item_sequences']), axis=1).to_numpy(dtype=object)
        item_seq = np.array([np.array(x) for x in item_seq])

        item_deg = dataframe.apply(lambda x: literal_eval(x['item_degree']), axis=1).to_numpy(dtype=object)
        item_deg = np.array([np.array(x) for x in item_deg])

        self.user_sequences = torch.LongTensor(user_seq)
        self.user_degree = torch.LongTensor(user_deg)
        self.item_sequences = torch.LongTensor(item_seq)
        self.item_degree = torch.LongTensor(item_deg)
        self.spd_table = torch.from_numpy(np.load(spd_file)).long()

        # end_time = time.time()
        # print(f"Data init time: {end_time - start_time:.4f}s")      # Ciao: 18.2 s 
    
    def __len__(self):
        # 전체 {train/valid/test}.csv의 길이
        return len(self.user_sequences)

    def __getitem__(self, index):
        user_seq = self.user_sequences[index]
        user_deg = self.user_degree[index]
        item_seq = self.item_sequences[index]
        item_deg = self.item_degree[index]

        # 현재 선택된 user_seq에 있는 사용자들에 대한 spd matrix 생성
        spd_table = self.spd_table[user_seq.squeeze(), :][:, user_seq.squeeze()]

        ###################### FIXME: data_utils -> generate_input_sequence_data() 로 이관해서 저장 ######################
        # 현재 선택된 user_seq에 있는 사용자들과 item_seq에 대해 [user_seq, item_seq] 크기의 rating table 생성
        rating_table = torch.zeros((len(user_seq), len(item_seq)))
        
        for i in range(rating_table.shape[0]):      # row
            current_user = user_seq[i].item()
            for j in range(rating_table.shape[1]):  # col
                current_item = item_seq[j].item()
                current_rating = self.rating_table.loc[(self.rating_table['user_id'] == current_user) & (self.rating_table['product_id'] == current_item)]['rating'].values
                if len(current_rating) == 0:
                    continue
                rating_table[i][j] = current_rating[0]
        ###################### FIXME: data_utils -> generate_input_sequence_data() 로 이관해서 저장 ######################

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