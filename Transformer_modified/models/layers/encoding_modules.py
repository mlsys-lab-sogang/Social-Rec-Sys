"""
Encoding & Embedding modules
"""
import numpy as np
import torch
import torch.nn as nn

class SocialNodeEncoder(nn.Module):
    """
    Embed node id to dense representation & Encode each node's degree information.
    (similar to positional encoding)
        num_nodes: number of nodes(users) in entire social graph
        max_degree: max degree in entire social graph
        d_model: embedding size
    """
    def __init__(self, num_nodes, max_degree, d_model):
        super(SocialNodeEncoder, self).__init__()

        # node id embedding table -> similar to word embedding table.
            # table size: (num_user_total, embed_dim)
        self.node_encoder = nn.Embedding(num_nodes, d_model, padding_idx=0)

        # Degree embedding table -> will be index by input's degree information.
            # table size: (max_degree, embed_dim)
        self.degree_encoder = nn.Embedding(max_degree, d_model, padding_idx=0)
    
    def forward(self, batched_data):
        """
        batched_data: batched data from DataLoader

        TODO: max degree in entire graph need to be pre-computed. => entire graph's degree information is in `degree_table_social.csv`.
        """
        x, degree = (
            batched_data["user_seq"],
            batched_data["user_degree"]
        )

        # Generate user_id embedding vector
        user_embedding = self.node_encoder(x)
        # print(user_embedding.shape)

        # Add degree embedding vector to user_id embedding vector
        degree_embedding = self.degree_encoder(degree)
        # print(degree_embedding.shape)

        input_embedding = user_embedding + degree_embedding
        # print(input_embedding.shape)

        return input_embedding

class SpatialEncoder(nn.Module):
    """
    Embed SPD(shortest path distance) information to dense representation, using pre-computed SPD matrix.
    (similar to spatial encoding)
        data_path: path to dataset
        spd_file: pre-computed SPD file (.npy)
    """
    def __init__(self, data_path, spd_file, num_heads):
        super(SpatialEncoder, self).__init__()

        spd_table = data_path + spd_file
        self.spatial_pos_table = torch.from_numpy(np.load(spd_table)).long()
        
        num_nodes = self.spatial_pos_table.size()[0]
        
        self.spatial_pos_encoder = nn.Embedding(num_nodes + 1, num_heads, padding_idx=0)

    def forward(self, batched_data):
        """
        batched_data: batched data from DataLoader
        """

        # TODO: sequence에 있는 사용자들의 거리 위치를 가져와야함.
        x = batched_data["user_seq"]


if __name__ == "__main__":
    import os

    data_path = os.getcwd() + '/dataset/ciao/'
    spd_file = 'shortest_path_result.npy'
    a = SpatialEncoder(data_path=data_path, spd_file=spd_file, num_heads=2)
    print(a)