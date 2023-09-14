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
        num_nodes: number of all nodes(users) in entire social graph
        max_degree: max degree in entire social graph
        d_model: embedding size
    """
    def __init__(self, num_nodes, max_degree, d_model):
        super(SocialNodeEncoder, self).__init__()

        # node id embedding table -> similar to word embedding table.
            # table size: [num_user_total + 1, embed_dim]
            # (id == index + 1)
        self.node_encoder = nn.Embedding(num_nodes + 1, d_model, padding_idx=0)

        # Degree embedding table -> will be index by input's degree information.
            # table size: [max_degree + 1, embed_dim]
            # (id == index + 1)
        self.degree_encoder = nn.Embedding(max_degree + 1, d_model, padding_idx=0)
    
    def forward(self, batched_data):
        """
        batched_data: batched data from DataLoader

        TODO: max degree in entire graph need to be pre-computed. => entire graph's degree information is in `degree_table_social.csv`.
        """
        x, degree = (
            batched_data["user_seq"],
            batched_data["user_degree"]
        )
        # print(f"Input seq min: {torch.min(x)}")
        # print(f"Input seq max: {torch.max(x)}")

        # Generate user_id embedding vector
        user_embedding = self.node_encoder(x)
        # print(user_embedding.shape)

        # Add degree embedding vector to user_id embedding vector
        degree_embedding = self.degree_encoder(degree)
        # print(degree_embedding.shape)

        input_embedding = user_embedding + degree_embedding

        return input_embedding

class SpatialEncoder(nn.Module):
    """
    Embed SPD(shortest path distance) information to dense representation, using pre-computed SPD matrix.
    (similar to spatial encoding)
        num_heads: number of heads in multi-head attention
        num_nodes: total number of users in social graph. (before splitting)
    """
    def __init__(self, num_heads, num_nodes):
        super(SpatialEncoder, self).__init__()
        
        self.num_heads = num_heads
        
        # # lookup table은 spatial-pos table에 있는 거리 값을 dense vector representation으로 변환
        #     # 현재 spatial_pos_table에 있는 값중 max값은 unreachable 거리이고, 이는 num_nodes + 1.
        # self.spatial_pos_encoder = nn.Embedding(num_nodes + 1, num_heads, padding_idx=0)

    def forward(self, batched_data):
        """
        batched_data: batched data from DataLoader
        """

        # 현재 user sequence에 있는 사용자들에 대한 SPD matrix는 Dataset class에서 pre-computed되어 있음.
            # [batch_size, seq_length_user, seq_length_user]
        spd_matrix = batched_data['spd_matrix']

        # [batch_size, seq_length, seq_length] ==> [num_heads, batch_size, seq_length, seq_length] ==> [batch_size, seq_length, seq_length, num_heads]
        attn_bias = spd_matrix.repeat(self.num_heads, 1, 1, 1).permute(1, 2, 3, 0)

        return attn_bias

class ItemNodeEncoder(nn.Module):
    """
    Embed node id to dense representation & Encode each node's degree information.
    (similar to positional encoding)
        num_nodes: number of all nodes(items) in entire user-item graph
            => max id value (if actual num_node is 100, but max id value is 120, num_nodes will be 120.)
        max_degree: max degree of items in entire user-item graph
        d_model: embedding size
    """
    def __init__(self, num_nodes, max_degree, d_model):
        super(ItemNodeEncoder, self).__init__()

        # node id embedding table -> similar to word embedding table.
            # table size: [num_item_total, embed_dim]
        self.node_encoder = nn.Embedding(num_nodes + 1, d_model, padding_idx=0)

        # Degree embedding table -> will be index by input's degree information
            # table size: [max_degree, embed_dim]
        self.degree_encoder = nn.Embedding(max_degree + 1, d_model, padding_idx=0)
    
    def forward(self, batched_data):
        """
        batched_data: batched data from DataLoader
        """
        # TODO: Dataset 에서 item_degree 정보도 batch data에 함께 담아주도록 수정 & item은 [batch_size, seq_length, interacted_item] 이었는데, 이를 [batch_size, seq_length, interacted_item*seq_length]
        # 즉, 시퀀스마다 고정된 수를 보는게 아니라 전체를 1개로 flatten [seq_length*interacted_item] 해서 전달
        x, degree = (
            batched_data['item_list'],
            batched_data['item_degree']
        )

        # Generate item_id embedding vector
        item_embedding = self.node_encoder(x)
        # print(item_embedding.shape)

        # Add degree embedding vector to item_id embedding vector
        degree_embedding = self.degree_encoder(degree)
        # print(degree_embedding.shape)

        input_embedding = item_embedding + degree_embedding

        return input_embedding
    
class RatingEncoder(nn.Module):
    """
    Encoder item's rating information to dense representation, using `batched_data['rating']`.
    """
    def __init__(self, num_heads):
        super(RatingEncoder, self).__init__()

        # TODO: Use embedding table later? embedding shape [num_rating, d_model] ?
        self.num_heads = num_heads

    def forward(self, batched_data):
        """
        batched_data: batched data from DataLoader
        """

        # [batch_size, seq_length_user, seq_length_item] 
        ###### FIXME: 정답인 rating 정보를 바로 주는건 말이 X. 따라서 상호작용 여부(0 or 1)로 주자.       
        item_rating = batched_data['item_rating']
        item_rating = torch.where(item_rating == 0, 0, 1)
        ######

        # Q*K^T 를 수행하면 [batch_size, num_heads, seq_length_item, seg_length_user]
        # 여기에 bias term으로 더해주므로 [batch_size, seq_length_user, seq_length_item] ==> [batch_size, num_heads, seq_length_user, seq_length_item] 이 되어야 함. -> decoder 부분에서 수행
            # [batch_size, seq_length_user, seq_length_item] 
            # ==> [num_heads, batch_size, seq_length_user, seq_length_item]
            # ==> [batch_size, seq_length_user, seq_length_item, num_heads]
        attn_bias = item_rating.repeat(self.num_heads, 1, 1, 1).permute(1, 2, 3, 0)

        return attn_bias


# if __name__ == "__main__":
#     import os

#     data_path = os.getcwd() + '/dataset/ciao'
#     spd_file = 'shortest_path_result.npy'
#     a = SpatialEncoder(data_path=data_path, spd_file=spd_file, num_heads=2)
#     print(a)