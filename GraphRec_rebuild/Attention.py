"""
Attention 은 Aggregator에서 호출하여 사용

GAT의 기존 Attention 연산을 다른 variant에서의 연산으로 변경이 가능한지? (e.g. GATv2)
    https://nn.labml.ai/graphs/gatv2/index.html
    https://nn.labml.ai/graphs/gatv2/experiment.html
    https://github.com/tech-srl/how_attentive_are_gats

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Attention Network
        Parameterize attention score with two-layer neural network
        (attention score를 출력하는 최종 layer까지 합치면 3-layer.)
    """
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        
        # Bilinear는 x1^T * W * x2 + b 를 수행하는 layer (https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html)
        # 두 벡터를 입력으로 받아 1개의 출력을 생성하는 layer.
        ## 근데 사용되지 않는 것 같음.
        # self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)

        self.attention_layer1 = nn.Linear(self.embed_dim * 2, self.embed_dim)   # 1st layer
        self.attention_layer2 = nn.Linear(self.embed_dim, self.embed_dim)       # 2nd layer
        self.attention_layer3 = nn.Linear(self.embed_dim, 1)                    # 마지막 output layer. 단일 값으로 이것이 attention score를 출력으로 생성

        ## 이거도 안쓰임. functional로 사용중.
        # self.softmax = nn.Softmax(dim=0)                                        # output layer에서 출력된 attention score를 해당 softmax layer를 통해 normalize하여 최종 attention score를 생성.

    # att_w = self.att(o_history, uv_rep, num_histroy_item)
    def forward(self, computed_representation, given_embedding, num_neighbors):
        """
        computed_representation : 사전 계산된 reprsentation
            user-item에선 user-item interaction을 opinion-aware하게 represent한 vector x_{ia}
            user-user에선 이웃 노드들의 item-space reprsentation vector h^I
        given_embedding : 주어진 embedding vector
            user-item에선 user embedding p_i 또는 item embedding q_j
            user-user에선 user embedding p_i
        num_neighbors : 사용자/아이템 의 1-hop 이웃 수 (사용자 라면 상호작용한 아이템의 수 // 아이템 이라면 상호작용한 사용자의 수)
        """
        # torch.tensor.repeat는 주어진 tensor(앞쪽)를 정해진 크기(뒤쪽)의 횟수 만큼 늘려서 생성.
        # 즉 주어진 embedding을 (num_neighbors, 1) 크기로 생성. 
        given_embedding = given_embedding.repeat(num_neighbors, 1)

        # [x_ia CONCAT p_i] // [h_I CONCAT p_i] // [f_jt CONCAT q_j]
        x = torch.cat((computed_representation, given_embedding), dim=1)

        # 1st layer (sigma(W1 * [CONCATENATED_VECTOR] + b1))
        x = F.relu(self.attention_layer1(x))
        x = F.dropout(x, training=self.training)

        # 2nd layer (W2 * (1st_layer_output) + b2)
        x = F.relu(self.attention_layer2(x))
        x = F.dropout(x, training=self.training)

        # output layer 
        x = self.attention_layer3(x)

        # normalize output and make final attention weight
        attention_score = F.softmax(x, dim=0)

        return attention_score