import numpy as np
import pickle
import time
import random
import argparse
import os
import math

from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from UserItem_Encoder import UserItem_Encoder
from UserItem_Aggregator import UserItem_Aggregator
from Social_Encoder import Social_Encoder
from Social_Aggregator import Social_Aggregator
from GraphRec import GraphRec

"""
지금 논문은 user-user와 user-item을 사용하는데, user-item을 각각
user-item, user-rating, item-user, item-rating 으로 세분화해서 사용헀음. 
 => user가 item에 남긴 rating의 representation, 즉 opinion embedding을 활용하기 위해서.

결국엔 rating 정보가 edge attribute인데, bipartite 에서 edge attribute를 고려해서 modeling하면 되는게 아닌가?
"""
"""
TODO: dataset의 sparsity check? `sparsity = 1.0 - count_nonzero(X) / X.size`
"""

# random seed
torch.manual_seed(1234)
random.seed(1234)

def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        # loader는 user, item, rating을 담고 있음.
        inbatch_user_nodes, inbatch_item_nodes, labels_list = data

        optimizer.zero_grad()

        loss = model.loss(inbatch_user_nodes.to(device), inbatch_item_nodes.to(device), labels_list.to(device))

        # 두 입력에 대한 loss를 한번에 계산하기 위해 retain_graph=True로 설정 
            # https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
        loss.backward(retain_graph=True)

        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0:
            print(f"[{epoch}, {i:05d}] loss: {running_loss/100:.3f}, The best rmse/mae: {best_rmse:.6f}/{best_mae:.6f}")
            running_loss = 0.0
    return 0

def test(model, device, test_loader):
    model.eval()

    tmp_pred_score = []
    target_score = []

    with torch.no_grad():
        for test_user, test_item, test_target in test_loader:
            # 마찬가지로 user, item, rating 을 담고 있음.
            test_user, test_item, test_target = test_user.to(device), test_item.to(device), test_target.to(device)

            val_output = model.forward(test_user, test_item)

            tmp_pred_score.append(list(val_output.data.cpu().numpy()))
            target_score.append(list(test_target.data.cpu().numpy()))

        tmp_pred_score = np.array(sum(tmp_pred_score, []))
        target_score = np.array(sum(target_score, []))

        expected_rmse = math.sqrt(mean_squared_error(tmp_pred_score, target_score))
        mae = mean_absolute_error(tmp_pred_score, target_score)

        return expected_rmse, mae

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--dataset', type=str, default='data_full/ciao/')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test set ratio to use. Default is 0.2(20%), so dataset will be split into train(0.8):test(0.2).')
    parser.add_argument('--toy_data', action='store_true', help='Use toy dataset for testing')      # If we use toy_dataset, we should specify this argument.
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"***Using {device}...***")

    embed_dim = args.embed_dim

    if args.toy_data:
        # toy dataset
        dir_data = './data/toy_dataset'
        path_data = dir_data + '.pickle'
    else:
        # Ciao or Epinions dataset
        # need to preprocess
        dir_data = args.dataset
        path_data = ''
        for file in os.listdir(dir_data):
            if file.endswith('.pickle'):
                path_data = dir_data + file
                print("\n Pickle file exists, no need to preprocess... \n")
        if path_data == '':
            print("\n dataset.pickle not found. Generating pickle file... \n")
            # preprocessor = DataPreprocess(dir_data, args.test_ratio)
            # preprocessor.preprocess()
            path_data = dir_data + 'dataset.pickle'

    print("\n ***** Data Loaded ***** \n")

    data_file = open(path_data, 'rb')
    # history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list, users, friends = pickle.load(data_file)
    history_user_item_list, history_user_rating_list, history_item_user_list, history_item_rating_list, train_user, train_item, train_rating, test_user, test_item, test_rating, social_connect_info, ratings_list = pickle.load(data_file)

    """
    #### user-item matrix로 부터 생성 (rating.mat)
    history_user_item_list : user's purchased history (item set in training set) (user-item 형태 : 사용자가 상호작용 한 아이템 목록) (<class 'collections.defaultdict'>)
        user가 어떤 item과 상호작용을 했는지를 dict로 표현.
        ex) {387: [47, 45], 279: [1248], 7: [24], ...}
            사용자 387은 47, 45를 구매했었다 (상호작용을 했다) // 사용자 279는 1248을 구매했었다 // ...

    history_user_rating_list : user's rating score (사용자가 부여한 점수들. 어떤 아이템에 준건지는 이거만 봐선 모름.) (<class 'collections.defaultdict'>)
        user가 어떤 rating들을 남겼는지 dict로 표현.
        이때 rating은 ratings_list에 있는 값으로, integer value로 mapping(encoding)이 완료된 값임. 
        ex) {59: [6, 4], 14: [5], 201: [5, 3, 0], 538: [5], ...}
            사용자 59는 6점, 4점의 평점을 남겼다 // 사용자 14는 5점의 평점을 남겼다 // 사용자 201은 5점, 3점, 0점의 평점을 남겼다 // ...

    history_item_user_list : user set (in training set) who have interaced with item (item-user 형태 : 아이템과 상호작용 한 사용자 목록) (<class 'collections.defaultdict'>)
        item이 어떤 user와 상호작용을 했는지를 dict로 표현.
        앞서 있던 user_item_list와는 다르게, 여기선 key가 item이고 value가 user. 이는 item feature 학습에 사용된다.
        ex) {1704: [643, 38], 1705: [104], 1706: [24], ...}
            아이템 1704는 사용자 643, 38이 구매했었다 // 아이템 1705는 사용자 104가 구매했었다 // ...

    history_item_rating_list : item's rating score (아이템이 부여받은 점수들. 어떤 사용자가 준건지는 이거만 봐선 모름.) (<class 'collections.defaultdict'>)
        item이 어떤 rating을 받았는지를 dict로 표현.
        마찬가지로 rating은 ratings_list에 있는 값으로, integer value로 mapping(encoding)이 완료된 값임.
        ex) {1704: [7, 1], 1705: [3], 1706: [4], ...}
            아이템 1704는 7점, 1점의 평점을 받았다 // 아이템 17005는 3점의 평점을 받았다 // ...
    #### 

    train_user, train_item, train_rating : train set (user, item, rating) (<class 'list'>) 
    test_user, test_item, test_rating : test set (user, item, rating) (<class 'list'>)
        >> 둘 모두 각각 list로 user index, item index, rating value가 들어있음.
        >> user-item-rating 을 sparse matrix로 표현하지 않고, <u,v,r> pair를 list로 저장한 것. 

    #### user-user matrix로부터 생성 (trustnetwork.mat)
    social_connect_info : user-user network로, 사용자와 연결된 사용자들의 목록을 가지고 있음 (<class 'collections.defaultdict'>)
        사용자(key)가 어떤 사용자들(value)과 연결되어 있는지를 dict로 표현. 
        ex) {0: {1, 2}, 1: {0}, 2: {0}, 3: {4, 201, 44, 84, 85}, 4: {3, 267, 44, 210, 84, 85}, 5: {6}, 6: {5, 558}, ...}
            사용자 0은 1,2와 연결되어 있다 // 사용자 1은 0과 연결되어 있다 // 사용자 2는 0과 연결되어 있다 // ...
    #### 
   
    ratings_list : rating value from 0.5 to 4.0 (8 opinion embeddings) (<class 'dict'>)
        이는 주어진 것으로, 말 그대로 각 rating score를 integer value로 encoding 한 것. 학습하는 것이 아님. 
        ex) {2.0: 0, 1.0: 1, 3.0: 2, 4.0: 3, 2.5: 4, 3.5: 5, 1.5: 6, 0.5: 7}

        
    toy dataset은 train이 705 user, 1941 item // test가 549 user, 355 item
    """
    
    train_dataset = torch.utils.data.TensorDataset(torch.LongTensor(train_user), torch.LongTensor(train_item), torch.FloatTensor(train_rating))
    test_dataset = torch.utils.data.TensorDataset(torch.LongTensor(test_user), torch.LongTensor(test_item), torch.FloatTensor(test_rating))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    num_users = history_user_item_list.__len__()
    num_items = history_item_user_list.__len__()
    num_ratings = ratings_list.__len__()

    # Define user/item/rating(opinion) embedding.
    # 단순히 index로 표현된 데이터를 dense vector로 표현. 
    user_embedding = nn.Embedding(num_users, embed_dim).to(device)      # u2e : user embedding vector p_i
    item_embedding = nn.Embedding(num_items, embed_dim).to(device)      # v2e : item embedding vector q_j
    opinion_embedding = nn.Embedding(num_ratings, embed_dim).to(device) # r2e : opinion embedding vector e_r

    #### User Modeling
    # User Modeling : Item Aggregation (user-item graph에 있는 user의 latent factor를 학습) -> 생성물: h^I
    user_item_aggregation = UserItem_Aggregator(    # agg_u_history
        item_embedding,             # given item embedding v2e (q_j)
        opinion_embedding,          # given opinion embedding r2e (e_r)
        user_embedding,             # given user embedding u2e (p_i)
        embed_dim, 
        cuda=device, 
        user_modeling=True          # Use this aggregator for User Modeling
    )
    user_item_encoding = UserItem_Encoder(      # enc_u_history
        user_embedding,             # given user embedding u2e (p_i)
        embed_dim,      
        history_user_item_list,     # user-item connectivity information ({user:[item, item, ...], ...}) : 사용자가 상호작용한 아이템 목록
        history_user_rating_list,   # user-rating information ({user: [rating, rating, ...], ...}) : 사용자가 매긴 rating 정보 목록
        user_item_aggregation,      # Aggregator : user-item aggregation을 수행하는 모듈 -> 연결정보, 의견정보를 Aggregator에 전달해 aggregation을 수행 (from item to user : item ==> user)
        cuda=device, 
        user_modeling=True          # Use this encoder for User Modeling.
    )

    # User Modeling : Social Aggregation (user-item graph에서 도출한 이웃들의 latent factor를 이용해 social network에서 주어진 user의 latent factor를 학습) -> 생성물: h^S
    user_user_aggregation = Social_Aggregator(      # agg_u_social
        lambda nodes: user_item_encoding(nodes).t(),    # user_item_encoding의 결과인 h^I를 이용
        user_embedding,                                 # social aggregation을 위해 user embedding p_i를 사용
        embed_dim, 
        cuda=device
    )
    final_user_representation = Social_Encoder(     # enc_u
        lambda nodes: user_item_encoding(nodes).t(),    # user_item_encoding의 결과인 h^I를 이용
        embed_dim, 
        social_connect_info,                            # user-user connectivity information : 사용자와 연결된 사용자 목록, 즉 shape가 (user,user)인 Adj. 
        user_user_aggregation,                          # Aggregator : user-user aggregation을 수행하는 모듈 -> 소셜 연결 정보를 Aggregator에 전달해 이웃 사용자의 aggregation을 수행
        base_model=user_item_encoding, 
        cuda=device
    )
    #### User Modeling

    #### Item Modeling
    # Item Modeling : User Aggregation (user-item graph에 있는 item의 latent factor를 학습)
    item_user_aggregation = UserItem_Aggregator(    # agg_v_history
        item_embedding,             # given item embedding v2e (q_j)
        opinion_embedding,          # given opinion embedding (e_r)
        user_embedding,             # given user embedding (p_i)
        embed_dim, 
        cuda=device, 
        user_modeling=False         # Use this aggregator for Item Modeling
    )
    item_user_encoding = UserItem_Encoder(          # enc_v_history
        item_embedding,             # given item embedding v2e (q_j)
        embed_dim, 
        history_item_user_list,     # item-user connectivity information ({item: [user, user, ...], ...}) : 아이템과 상호작용한 사용자 목록
        history_item_rating_list,   # item-rating information ({item: [rating, rating, ...], ...}) : 아이템이 받은 rating 목록
        item_user_aggregation,      # Aggregator : item-user aggregation을 수행하는 모듈 -> 연결정보, 의견정보를 Aggregator에 전달해 aggregation을 수행 (from user to item : user ==> item)
        cuda=device, 
        user_modeling=False         # Use this aggregator for Item Modeling
    )
    #### Item Modeling

    # print(user_item_encoding)           # User Modeling Network - item-space user modeling h^I
    # print('\n')
    # print(final_user_representation)    # User Modeling Network - social-space user modeling h^S
    # print('\n')
    # print(item_user_encoding)           # Item Modeling Network z_j
    
    # Define model
    model = GraphRec(final_user_representation, item_user_encoding, opinion_embedding).to(device)
    # print(model)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0    # used for early-stopping & validation set

    # add time-checking
    for epoch in range(1, args.epochs + 1):
        train_start_time = time.time()
        train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        train_end_time = time.time()
        total_train_time = train_end_time - train_start_time

        test_start_time = time.time()
        expected_rmse, mae = test(model, device, test_loader)
        test_end_time = time.time()
        total_test_time = test_end_time - test_start_time

        # Early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        
        print(f"Epoch {epoch:03d}, rmse: {expected_rmse:.4f}, mae: {mae:.4f}, train time (1 epoch) {total_train_time:.4f}s, test time {total_test_time:.4f}s")
        
        if endure_count > 5:
            break

if __name__ == "__main__":
    main()