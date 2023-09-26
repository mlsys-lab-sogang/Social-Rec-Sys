"""
user-item graph를 이용, UserItem_Encoder의 출력인 h^I (User Modeling : user-item graph에서 표현되는 user latent factor)를 입력으로 받아,
user-user graph에서의 user latent factor h^S 를 생성. 
"""

import torch
import torch.nn as nn
from Attention import Attention

# Social_Aggregator(lambda nodes: user_item_encoding(nodes).t(), user_embedding, embed_dim, cuda=device)
class Social_Aggregator(nn.Module):
    """
    Social Aggregator : for aggregating embeddings of social neighbors.
        Aggregate embeddings of neighbors in social graph
            Social Modeling (Aggre_{neighbors})
        social network (user-user graph)에서 주어진 user와 바로 연결된 1-hop neighbor인 user들로부터 feature aggregation을 수행.
        이때 aggregation 시 weight를 주며, 이 weight는 attention mechanism을 통해 계산. (attention score)
            이 attention score는 결국 주어진 user와 연결된 이웃 user들간의 tie strength를 반영하고 있는 것으로 볼 수 있음. (2.3 User Modeling - Social Aggregation)
            (model their tie strengths, by relating social attention \beta with h^I and the target user embedding p_i, where the \beta can be seen as the strengths between users.)
    """
    def __init__(self, neighbor_user_latent_factor, user_embedding, embed_dim, cuda="cpu"):
        """
        neighbor_features 는 자신과 바로 연결된 user들의 item-space latent factor를 의미함. 
            즉, user-item graph에서 Aggre_{items}를 통해 생성한 user latent factor h^I를 의미.
        """
        super(Social_Aggregator, self).__init__()

        self.neighbor_features = neighbor_user_latent_factor    # 1-hop 이웃 user들의 h^I (UserItem_Encoder에서 생성된 latent factor)
        self.user_embedding = user_embedding                    # 주어진 user의 embedding p_i (given embedding)
        self.embed_dim = embed_dim

        self.attention = Attention(self.embed_dim)              # social weight \beta 계산을 위한 Attention Network

        self.device = cuda
    
    def forward(self, nodes, neighbor_nodes):
        """
        주어진 user-user network (user-user connectivity info, (item-space) user latent factor h^I)를 이용해 feature aggregation을 수행.

            user_item_pair : history_uv -> C(i)와 B(j)를 의미. user와 상호작용한 모든 item의 수 // item과 상호작용한 모든 user의 수
            rating : history_r -> user가 item들에 매긴 평점들 // item이 user들로부터 받은 평점들 
        """
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(nodes)):
            tmp_adj_info = neighbor_nodes[i]    # 현재 노드와 연결된 1-hop 이웃 노드들
            num_neighbors = len(tmp_adj_info)   # 현재 노드의 전체 이웃 수는 adj info의 길이와 동일.

            ###### TODO: user embedding vs. item-space user latent factor
            # fast: user embedding
                # 연결된 1-hop 이웃 user들의 item-space user latent factor h^I 가 아닌, 그들의 embedding vector를 사용
                # 말 그대로 이웃 user의 latent factor가 아닌 이웃 user의 embedding vector를 사용
                # e_u
            neighbors_representation = self.user_embedding.weight[list(tmp_adj_info)]

            # slow: item-space user latent factor (item aggregation)
                # 논문에서 언급한 그대로 이웃 user들의 item-space user latent factor h^I를 사용.
            # neighbors_feature = self.neighbor_features(torch.LongTensor(list(tmp_adj_info))).to(self.device)
            # neighbors_representation = torch.t(neighbors_feature)
            ###### TODO: user embedding vs. item-space user latent factor

            # 현재 사용자의 embedding vector p_i
                # u_rep
            user_representation = self.user_embedding.weight[nodes[i]]

            # TODO: Attention 호출
            # 수식(9)(10)(11) attention score 계산
            attention_weight = self.attention(neighbors_representation, user_representation, num_neighbors)

            # 실제 수식에서 attention_score * h^I를 하는 부분
            weighted_aggregation_result = torch.mm(neighbors_representation.t(), attention_weight)
            weighted_aggregation_result = weighted_aggregation_result.t()
            
            embed_matrix[i] = weighted_aggregation_result
        
        # embed_matrix의 구성요소는 attention weight가 적용된 reprsentation이 들어있다.
        neighbor_features = embed_matrix

        return neighbor_features