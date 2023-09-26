"""
Original name: UV_Encoder
    Encoder가 최상단 모듈.
    즉, Attention ⊂ Aggregator, Aggregator ⊂ Encoder. (Attention ⊂ Aggregator ⊂ Encoder)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# UserItem_Encoder(user_embedding, embed_dim, history_user_item_list, history_user_rating_list, user_item_aggregation, cuda=device, user_modeling=True)
class UserItem_Encoder(nn.Module):
    def __init__(self, feature_vector, embed_dim, history_user_item_list, history_rating_list, aggregator, cuda="cpu", user_modeling=True):
        super(UserItem_Encoder, self).__init__()

        # feature vector는 처음에 주어지는 embedding. (user embedding, item embedding)
        self.feature_vector = feature_vector

        # user_item_list는 user-item 또는 item-user 형태.
        # Item Aggregation과 User Aggregation 모두 동일한 데이터를 사용하지만, 주어지는 dict의 형식이 다를 뿐임. ({user:[item, item, ...]} vs. {item:[user, user, ...]})
        self.history_user_item_list = history_user_item_list

        # rating list 또한 마찬가지로 user-item 의 rating과 item-user 의 rating. 주어지는 dict의 형식이 다를 뿐임.
        self.history_rating_list = history_rating_list

        # Aggregator
        # User modeling을 하는지(Item Aggregation, Aggre_{items}) vs. Item modeling을 하는지(User Aggregation, Aggre_{users})
        self.user_modeling = user_modeling
        self.aggregator = aggregator

        self.embed_dim = embed_dim
        self.device = cuda

        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
    
    def forward(self, nodes):
        tmp_user_item_list = []
        tmp_rating_list = []
        
        # 현재 주어진 데이터의 node(user/item index)만을 선택
        for node in nodes:
            tmp_user_item_list.append(self.history_user_item_list[int(node)])
            tmp_rating_list.append(self.history_rating_list[int(node)])

        # TODO: Aggregator 호출
        # 현재 주어진 user-item network에서 node 위치(선택된 사용자/아이템)에 대하여 feature aggregation을 수행 (UserItem_Aggregator 를 통해 수식의 Aggre_items 부분을 수행)
        # 한마디로 feature aggregation 수행
        neighbor_features = self.aggregator.forward(nodes, tmp_user_item_list, tmp_rating_list)

        # 현재 주어진 초기 embedding (user embedding // item embedding)에서 node 위치(선택된 사용자/아이템)에 대하여 그 자신의 embedding vector를 GET
        # (original) self-connection could be considered.
        self_features = self.feature_vector.weight[nodes]

        # neighbor aggregation 결과와 자기 자신의 feature를 CONCAT -> self-connection을 고려하여 feature를 생성하는 셈.
        combined_feature = torch.cat([self_features, neighbor_features], dim=1)
        combined_feature = F.relu(self.linear1(combined_feature))

        # 따라서 return되는 값이 최종 생성되는 h^I // z_j 가 된다. 
        return combined_feature
