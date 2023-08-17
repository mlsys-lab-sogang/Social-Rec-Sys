"""
trustnetwork 에서 모든 사용자의 degree 정보를 담은 table 생성

trustnetwork 에서 random walk sequence 생성
    - 임의의 사용자 n명을 선택
    - 각 사용자마다 random walk length r 만큼의 subgraph sequence 생성
    - 생성한 sequence에서, 각 노드와 매칭되는 degree 정보를 degree table에서 GET
    - [노드, 노드, 노드], [degree, degree, dgree] 를 함께 구성 (like PyG's edge_index)
        => [[node1, node2, node3]]
"""
import os
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import loadmat

# np.random.seed(62)

# arg(or else) passing to DATASET later
DATASET = 'ciao'

data_path = os.getcwd() + '/dataset/' + DATASET

def mat_to_csv(data_path:str):
    """
    Convert .mat file into .csv file for using pandas.
        Ciao: rating.mat, trustnetwork.mat
        Epinions: rating.mat, trustnetwork.mat
    
    Args:
        data_path: Path to .mat file
    """
    # processed file check
    if ('rating.csv' or 'trustnetwork.csv') in os.listdir(data_path):
        print("Processed .csv file already exists...")
        return 0
    
    # load .mat file
    rating_file = loadmat(data_path + '/' + 'rating.mat')
    trust_file = loadmat(data_path + '/' + 'trustnetwork.mat')

    rating_file = rating_file['rating'].astype(np.int64)
    trust_file = trust_file['trustnetwork'].astype(np.int64)

    # convert to dataframe
    rating_df = pd.DataFrame(rating_file, columns=['user_id', 'product_id', 'category_id', 'rating', 'helpfulness'])
    trust_df = pd.DataFrame(trust_file, columns=['user_id_1', 'user_id_2'])

    # drop unused columns (TODO: Maybe used later)
    rating_df.drop(['category_id', 'helpfulness'], axis=1, inplace=True)

    rating_df.to_csv(data_path + '/rating.csv', index=False)
    trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)

def generate_user_degree_table(data_path:str):
    """
    Generate degree table from social graph(trustnetwork).
    """
    # processed file check
    if 'social_degree.csv' in os.listdir(data_path):
        print("Processed .csv file already exists...")
        return 0

    # user-user network
        # Ciao: 7317 users
    trust_file = data_path + '/trustnetwork.csv'
    dataframe = pd.read_csv(trust_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')

    degrees = {node: val for (node, val) in social_graph.degree()}
    degree_df = pd.DataFrame(degrees.items(), columns=['user_id', 'degree'])

    degree_df.to_csv(data_path + '/social_degree.csv', index=False)

def generate_social_random_walk_sequence(data_path:str, num_nodes:int=10, walk_length:int=5):
    """
    Generate random walk sequence from social graph(trustnetwork).
    
    Args:
        data_path: path to data
        num_nodes: number of nodes to generate random walk sequence
        walk_length: length of random walk
    """
    trust_file = data_path + '/trustnetwork.csv'
    dataframe = pd.read_csv(trust_file, index_col=[])

    social_graph = nx.from_pandas_edgelist(dataframe, source='user_id_1', target='user_id_2')

    # sample_dict = {}
    # memory = []
    # anchor = 6926
    # sample_dict[anchor] = [anchor]
    # for _ in range(walk_length):
    #     print(sample_dict)
    #     if sample_dict[anchor][-1] == anchor:
    #         selected = find_next_social_node(social_graph, previous_node=None, current_node=anchor, RETURN_PARAMS=0.0)
    #         memory.append(selected)
    #         sample_dict[anchor].append(selected)
    #         print('\n\n')
    #         print(f"OOOOOKKKKKKKKKKK {sample_dict}")
    #         print(memory)
    #         print('\n\n')
            
    #     # elif (sample_dict[anchor][-1] == memory[0]) or (sample_dict[anchor][-1] == 0):
    #     #     print('\n')
    #     #     print('PAAAAAAASSSSSSSSSSSSSSSSSSSSs')
    #     #     # sample_dict[anchor].append(0)
    #     #     continue

    #     else:
    #         print('\n')
    #         print('GGGGEEEEEEEEEEETTTTTTTTTT')
    #         selected = find_next_social_node(social_graph, previous_node=anchor, current_node=memory[0], RETURN_PARAMS=0.0)
    #         sample_dict[anchor].append(selected)
    # del sample_dict[anchor][0]
    # # print(sample_dict)

    # if len(sample_dict[anchor]) != walk_length:
    #     len_flag = len(sample_dict[anchor])
    #     for _ in range(len_flag, walk_length):
    #         sample_dict[anchor].append(0)
    # print(sample_dict)

    # quit()

    path_dict = {}
    memory = []

    # select target(anchor) nodes randomly.
    anchor_nodes = np.random.choice(social_graph.nodes(), size=num_nodes)
    
    # At first, there is no previous node, so set it to None.
    for nodes in anchor_nodes:
        path_dict[nodes] = [nodes]
        for _ in range(walk_length):
            # 바로 이웃된 노드 추가
            if path_dict[nodes][-1] == nodes:
                next_node = find_next_social_node(graph=social_graph, previous_node=None, current_node=nodes, RETURN_PARAMS=0.0)
                path_dict[nodes].append(next_node)

                memory.append(next_node)
            else:
                next_node = find_next_social_node(graph=social_graph, previous_node=path_dict[nodes][-2], current_node=path_dict[nodes][-1], RETURN_PARAMS=0.0)
                path_dict[nodes].append(next_node)
        
        # pop 1st element of list (since it is anchor node)
        del path_dict[nodes][0]

        if len(path_dict[nodes]) != walk_length:
            # pad with 0 for fixed sequence length.
            length_flag = len(path_dict[nodes])
            for _ in range(length_flag, walk_length):
                path_dict[nodes].append(0)
            print(path_dict)
        else:
            continue
    print('\n\n')
    print(path_dict)
        # print(len(path_dict.keys()))
        # quit()

    

def find_next_social_node(graph:nx.Graph(), previous_node, current_node, RETURN_PARAMS):
    """
    Find connected nodes, using transition probability. 
    """
    select_prob = {}
    # print(f"@@@@@@@@@@@@@@@@@@ {current_node} @@@@@@@@@@@@@@@@@@@")
    for node in graph.neighbors(current_node):
        # select_prob[node] = 1
        # if len(list(graph.neighbors(current_node))) == 1:
        #     print(f"지ㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣ {node} ㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣㅣ")
        #     print(node)
        #     print(list((graph.neighbors(current_node))))
        #     # quit()
        #     return node
        # if (node == previous_node):
        #     # 현재 노드의 이웃 중에서 바로 직전의 노드는 제외
        #     print(f"88888888888888888888  {node}  88888888888888888888888")
        #     select_prob[node] = 0.0
        #     continue
        # print(f"################### {node} ###################")

        # Here, in social network we using, we don't have any edge weights. 
        # So set select probability to 1.
        select_prob[node] = 1
    # print(select_prob)

    # 이전에 방문한 노드(또는 시작 노드)로 돌아갈 확률은 0으로.
    if previous_node is not None:
        del select_prob[previous_node]
    else:
        RETURN_PARAMS = 0.0
    
    select_prob_sum = sum(select_prob.values())
    # {k: v/select_probabilities_sum*(1-RETURN_PARAMS) for k, v in select_probabilities.items()}
    select_prob = {key: value / select_prob_sum * (1 - RETURN_PARAMS) for key, value in select_prob.items()}

    # print(select_prob)
    # quit()
    # if select_prob[previous_node] == 0.0:
    #     print('AAAAAAAAAaa')
    #     return previous_node

    if previous_node is not None:
        select_prob[previous_node] = RETURN_PARAMS
    
    # print(previous_node)
    # print(select_prob)
    
    selected_node = np.random.choice(
        a = [node for node in select_prob.keys()],
        p = [prob for prob in select_prob.values()]
    )

    return selected_node


generate_social_random_walk_sequence(data_path)
