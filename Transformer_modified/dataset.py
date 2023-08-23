import os
import networkx as nx
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import data_utils as utils

class MyDataset(Dataset):
    """
    Process & make dataset for DataLoader.
        __init__():     data load & process
        __getitem__():  data indexing (e.g. dataset[0])
        __len__():      length of dataset (e.g. len(dataset))
    """
    def __init__(self, dataset:str):
        """
        Args:
            dataset: raw dataset name (ciao or epinions)
        """
        self.root_dir = os.getcwd() + '/dataset/'
        self.data_path = self.root_dir + dataset        # /dataset/ciao || /dataset/epinions

        # generate random walk sequence
            # If `seed=True`, duplicated nodes will appear. (since seed is fixed for np.random.choice())
            # used for debugging & building
        random_walk_seq = utils.generate_social_random_walk_sequence(
            data_path = self.data_path,
            num_nodes = 10,
            walk_length = 20,
            seed = True,
            # all_node=True
        )

        # generate item interaction information 
            # number of interacted items of users(ciao): max 10,671, min 28, mean 260.4, std: 488.3
        # social_user_interact_items = utils.find_social_user_interacted_items(self.data_path, random_walk_seq)     # users in random walk sequence
        input_user_interact_items = utils.find_selected_user_interacted_items(self.data_path, random_walk_seq)    # users in random walk's start

        # encoder's input user list
            # shape: (num_nodes, )  
        self.enc_input_user = np.array(list(input_user_interact_items.keys()))
        
        # generate(fetch) each input user's random walk sequence nodes.
        # will be used in user-user relation encoding.
            # shape: (num_nodes. walk_length)
        user_stack = []
        for walks in random_walk_seq:
            for k, value in walks.items():
                user_stack.append(value[0])
        self.enc_input_friends = np.array(list(zip(*user_stack))).T

        print(self.enc_input_user.shape)
        print(self.enc_input_friends.shape)

        # TODO: Need to pre-define number of interacted items.
            # If not, ndarray's length will be shortest length of interacted item.
            # refer: https://stackoverflow.com/questions/53051560/stacking-numpy-arrays-of-different-length-using-padding
        # generate(fetch) each input user's interacted items
            # shape: [num_nodes, pre_defined_num_items]
        item_stack = []
        for key, value in input_user_interact_items.items():
            item_stack.append(value[0])
        for item in item_stack:
            print(len(item))
        # self.dec_input_items = np.array(list(zip(*item_stack))).T
        # print(self.dec_input_items)




dataset = MyDataset(dataset='ciao')