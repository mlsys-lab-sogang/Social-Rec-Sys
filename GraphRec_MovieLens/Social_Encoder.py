import torch
import torch.nn as nn
import torch.nn.functional as F

# Social_Encoder(lambda nodes: user_item_encoding(nodes).t(), embed_dim, social_connect_info, user_user_aggregation, base_model=user_item_encoding, cuda=device)
class Social_Encoder(nn.Module):
    """
    Generate final User latent factor, by combining h^I and h^S.
        h^I : item-space 상에서 사용자와 연결된 item과 rating 정보를 통해 얻은 해당 사용자의 latent factor
        h^S : social-sapce 상에서 사용자와 연결된 이웃 사용자들의 정보를 통해 얻은 해당 사용자의 latent factor
            이때 이웃 사용자들의 정보는 item-space 상에서 계산한 h^I를 사용. 
    """
    def __init__(self, neighbor_user_latent_factor, embed_dim, social_connect_info, aggregator, base_model=None, cuda="cpu"):
        super(Social_Encoder, self).__init__()

        # feature는 UserItem_Encoder에서 나온 결과, 즉 h^I를 의미.
        # 주어진 사용자와 연결된 1-hop 이웃 사용자들의 item-space에서의 user latent factor h^I를 의미.
        self.features = neighbor_user_latent_factor

        # social network, 즉 user-user graph를 의미.
        self.social_adj_lists = social_connect_info

        # Aggre_{neighbors} 를 담당하는 Aggregator. 
        # 1-hop 이웃 사용자들의 h^I와 자신의 p_i를 이용해 연결된 이웃 모두의 attention weight를 계산, 이웃 사용자들의 representation을 집계
        self.aggregator = aggregator

        if base_model != None:
            self.base_model = base_model
        
        self.embed_dim = embed_dim
        self.device = cuda

        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
    
    def forward(self, nodes):
        neighbor_list = []

        # 현재 노드와 연결된 1-hop 이웃 노드를 전부 GET
        for node in nodes:
            neighbor_list.append(self.social_adj_lists[int(node)])
        
        # TODO: Aggregator 호출
        # Aggregator는 user-user network를 입력으로 받아 연결된 neighbor user들의 feature aggregation을 수행.
        neighbor_features = self.aggregator.forward(nodes, neighbor_list)

        # 현재 주어진 embedding vector(item-space user latent factor h^I)에서 node 위치(이웃 사용자)에 대하여 embedding vector를 GET
        # (original) self-connection could be considered.
        self_features = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_features = self_features.t()

        # neighbor aggregation 결과와 자기 자신의 feature를 CONCAT -> self-connection을 고려하여 feature를 생성하는 셈.
        combined_feature = torch.cat([self_features, neighbor_features], dim=1)
        combined_feature = F.relu(self.linear1(combined_feature))

        # 따라서 return되는 값이 최종 생성되는 h^S 가 된다. 
        return combined_feature