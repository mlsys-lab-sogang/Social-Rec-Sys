import os
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import loadmat

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    Process & make dataset for DataLoader.
        __init__():     data load & process
        __getitem__():  data indexing (e.g. dataset[0])
        __len__():      length of dataset (e.g. len(dataset))
    """
    def __init__(self, raw_data:str, flag='user'):
        """
        Args:
            raw_data: raw dataset name (ciao or epinions)
            flag: to use specific dataset ('user' or 'item'. default: 'user')
        """
        if flag == 'user':
            self.flag = flag
            filename = 'trustnetwork.csv'
        elif flag == 'item':
            self.flag = flag
            filename = 'rating.csv'

        # self.mat_to_csv(raw_data)
        matrix = self.load_file_to_pair(os.getcwd() + '/dataset/ciao/' + filename)

    def mat_to_csv(self, raw_data:str):
        """
        Convert .mat file into .csv file for using pandas.
            Ciao: rating.mat, trustnetwork.mat
            Epinions: rating.mat, trustnetwork.mat
        
        Args:
            mat_file: Path to .mat file
        """
        root_dir = os.getcwd() + '/dataset/'
        data_path = root_dir + raw_data

        # processed file check
        if (data_path + '/rating.csv' and data_path + '/trustnetwork.csv') in os.listdir(data_path):
            print("Processed .csv file already exists")
            return 0

        # load .mat file
        rating_file = loadmat(data_path + '/' + 'rating.mat')
        trust_file = loadmat(data_path + '/' + 'trustnetwork.mat')

        # get only usable data from .mat file
            # rating_file's columns: [user_id, product_id, category_id, rating, helpfulness]
            # trust_file's columns: [user_id_1, user_id_2]
        rating_file = rating_file['rating'].astype(np.int64)
        trust_file = trust_file['trustnetwork'].astype(np.int64)

        # convert into dataframe
        rating_df = pd.DataFrame(rating_file, columns=['user_id', 'product_id', 'category_id', 'rating', 'helpfulness'])
        trust_df = pd.DataFrame(trust_file, columns=['user_id_1', 'user_id_2'])

        # drop unused columns
        rating_df.drop(['category_id', 'helpfulness'], axis=1, inplace=True)

        rating_df.to_csv(data_path + '/rating.csv', index=False)#, header=False, sep='\t')
        trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)#, header=False, sep='\t')

    def load_file_to_pair(self, csv_file):
        if self.flag == 'user':
            """
            returns user list
            """
            dataframe = pd.read_csv(csv_file, index_col=[])
            # num_users = len(dataframe['user_id'].unique())
            user_matrix = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')
            user_matrix = nx.to_dict_of_lists(user_matrix)

            return user_matrix


        if self.flag == 'item':
            """
            returns item list, interacted with users
            """
            dataframe = pd.read_csv(csv_file, index_col=[])
            num_users = len(dataframe['user_id'].unique())
            num_items = len(dataframe['product_id'].unique())

            # # DoK matrix: user-item matrix에서 0이 아닌 위치를 기록
            # matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)

            # # input values to matrix
            # with open(csv_file, 'r') as f:
            #     next(f)
            #     line = f.readline()
            #     while line != None and line != "":
            #         arr = line.split(',')
            #         user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])

            #         # if rating exists, put it into matrix
            #         if rating > 0:
            #             matrix[user, item] = rating
            #         line = f.readline()


dataset = MyDataset(raw_data='ciao', flag='user')