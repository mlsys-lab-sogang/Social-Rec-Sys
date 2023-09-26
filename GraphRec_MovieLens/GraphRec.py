import torch
import torch.nn as nn
import torch.nn.functional as F


# GraphRec(final_user_representation, item_user_encoding, opinion_embedding)
class GraphRec(nn.Module):
    """
    GraphRec: Graph Neural Networks for Social Recommendation. 
    Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
    In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]
    """
    def __init__(self, final_user_latent_factor, item_latent_factor, opinion_embedding):
        super(GraphRec, self).__init__()
        
        self.user_latent_factor_encoder = final_user_latent_factor      # Social Encoder 클래스
        self.item_latent_factor_encoder = item_latent_factor            # UserItem Encoder(Item Modeling) 클래스

        self.embed_dim = final_user_latent_factor.embed_dim     # embedding size (default = 64)

        self.W_user_rating_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_user_rating_2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.W_item_rating_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.W_item_rating_2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.W_user_item_1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.W_user_item_2 = nn.Linear(self.embed_dim, 16)
        self.W_user_item_3 = nn.Linear(16, 1)

        self.opinion_embedding = opinion_embedding

        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)

        self.criterion = nn.MSELoss()
    
    def forward(self, nodes_user, nodes_item):
        """
        user node와 item node를 통한 forward
        """
        user_embedding = self.user_latent_factor_encoder(nodes_user)    # "UserItem(UserModeling) + Social"(h^I CONCAT h^S) Aggregate/Encode를 통한 최종 user latent vector h_i 생성
        item_embedding = self.item_latent_factor_encoder(nodes_item)    # "UserItem(ItemModeling)" Aggregate/Encoder를 통한 최종 item latent factor z_j 생성
        
        # User Modeling : concat 된 h_i를 2-layer FFN을 거쳐 최종 h_i를 생성.
        x_user = F.relu(self.bn1(self.W_user_rating_1(user_embedding)))
        x_user = F.dropout(x_user, training=self.training)
        x_user = self.W_user_rating_2(x_user)

        # Item Modeling : z_j를 2-layer FFN을 거쳐 최종 z_j를 생성.
        x_item = F.relu(self.bn2(self.W_item_rating_1(item_embedding)))
        x_item = F.dropout(x_item, training=self.training)
        x_item = self.W_item_rating_2(x_item)

        # 서로 다른 latent factor CONCAT
        # 이 latent factor는 user와 item을 represent하고 있음.
        x_concat = torch.cat((x_user, x_item), dim=1)

        # CONCAT한 vector를 3-layer FFN으로 feed
        x = F.relu(self.bn3(self.W_user_item_1(x_concat)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.W_user_item_2(x)))
        x = F.dropout(x, training=self.training)

        scores = self.W_user_item_3(x)
        predicted_score = scores.squeeze()

        return predicted_score
    
    def loss(self, nodes_user, nodes_item, labels_list):
        predicted_scores = self.forward(nodes_user, nodes_item)
        return self.criterion(predicted_scores, labels_list)