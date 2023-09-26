"""
Original name: UV_Aggregator
    Encoder가 Aggregator를 호출. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random

from Attention import Attention

# UserItem_Aggregator(item_embedding, opinion_embedding, user_embedding, embed_dim, cuda=device, user_modeling=True)
class UserItem_Aggregator(nn.Module):
    """
    Aggregate embeddings of neighbors
    User Modeling : Item aggregation (aggregate user's information from user-item) 
        Aggre_{items}
    Item Modeling : User aggregation (aggregate item's information from item-user)
        Aggre_{users}
    """
    def __init__(self, item_embedding, opinion_embedding, user_embedding, embed_dim, cuda="cpu", user_modeling=True):
        super(UserItem_Aggregator, self).__init__()
        self.user_modeling = user_modeling          # uv : User Modeling을 위한 item aggregation을 수행하는지, Item Modeling을 위한 user aggregation을 수행하는지에 대한 flag.

        self.item_embedding = item_embedding        # v2e : nn.Embedding 으로 생성한 item의 dense embedding vector
        self.opinion_embedding = opinion_embedding  # r2e : nn.Embedding 으로 생성한 rating의 dense embedding vector
        self.user_embedding = user_embedding        # u2e : nn.Embedding 으로 생성한 user의 dense embedding vector
        
        self.embed_dim = embed_dim
        self.device = cuda

        self.linear1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.linear2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.attention = Attention(self.embed_dim)


    def forward(self, nodes, user_item_pair, rating):
        """
        주어진 user-item network (user_item list, rating)를 이용해 feature aggregation을 수행.

            user_item_pair : history_uv -> C(i)와 B(j)를 의미. user와 상호작용한 모든 item의 수 // item과 상호작용한 모든 user의 수
            rating : history_r -> user가 item들에 매긴 평점들 // item이 user들로부터 받은 평점들 
        """
        embed_matrix = torch.empty(len(user_item_pair), self.embed_dim, dtype=torch.float).to(self.device)

        # 상호작용한 모든 item/user 들에 대한 loop. 즉 C(i), B(j)의 크기만큼 loop를 돈다.
        for i in range(len(user_item_pair)):
            history = user_item_pair[i]     # user-item 또는 item-user pair 1쌍을 가져온다.
            num_history_item = len(history) # 상호작용한 갯수 (user-item 이라면 item의 수, item-user라면 user의 수)
            true_rating_value = rating[i]   # 가져온 pair에 해당하는 rating을 가져온다.

            if self.user_modeling == True:
                """
                User Modeling을 위한 item aggregation을 수행하므로, user-item pair에서의 item embedding을 사용.
                """
                interacted_history_embedding = self.item_embedding.weight[history]      # user가 상호작용한 item들의 representation vector들을 get (item embedding q_a)
                user_item_representation = self.user_embedding.weight[nodes[i]]         # 현재 해당하는 user의 representation vector를 get
            else:
                """
                Item Modeling을 위한 user aggregation을 수행하므로, item-user pair에서의 user embedding을 사용.
                """
                interacted_history_embedding = self.user_embedding.weight[history]  # item과 상호작용한 user들의 representation vector들을 get (user embedding p_t)
                user_item_representation = self.item_embedding.weight[nodes[i]]     # 현재 해당하는 item의 representation vector를 get
            
            rating_representation = self.opinion_embedding.weight[true_rating_value]    # user-item // item-user pair에서 value들에 해당하는 rating 값을 get (opinion embedding e_r)

            x = torch.cat((interacted_history_embedding, rating_representation), 1)     # 수식(2),(15)의 CONCAT 부분.
            x = F.relu(self.linear1(x))
            interaction_representation = F.relu(self.linear2(x))          # 수식(2)의 x_ia, 수식(15)의 f_jt : opinion-aware interaction representation

            # TODO: Attention 호출
            # 수식(5)(6) & 수식(18)(19) attention score 계산
            # MLP를 거쳐 먼저 계산한 연결정보의 representation과 item/user embedding을 CONCAT하여 attention weight를 계산
            attention_weight = self.attention(interaction_representation, user_item_representation, num_history_item)

            # 실제 수식에서 attention_score * interaction_representation 을 하는 부분.
            weighted_aggregation_result = torch.mm(interaction_representation.t(), attention_weight)
            weighted_aggregation_result = weighted_aggregation_result.t()

            embed_matrix[i] = weighted_aggregation_result
        
        # embed_matrix의 구성요소는 attention weight가 적용된 representation이 들어있다.
        neighbor_features = embed_matrix

        return neighbor_features          