"""
data_utils.py에 있는 함수들을 사용해
.mat -> .csv -> degree table & interacted item table 등등 필요한 작업을 수행.

TODO: 적절히 순차적으로 분기할 수 있도록 작성. 현재는 주석단위로 처리하고있음.
"""
import os

import data_utils as utils


data_path = os.getcwd() + '/dataset/' + 'epinions' 

# utils.mat_to_csv(data_path)

# utils.generate_social_dataset(data_path, save_flag=True)

# utils.generate_user_degree_table(data_path)
# utils.generate_item_degree_table(data_path, split='test')

# utils.generate_interacted_items_table(data_path, all=True)
# utils.generate_interacted_items_table(data_path, all=True, split='test')
# utils.generate_interacted_items_table(data_path, all=False, item_length=4)

# utils.generate_social_random_walk_sequence(data_path, num_nodes=10, walk_length=20, save_flag=True, all_node=True, seed=False, split='test')

# utils.generate_input_sequence_data(data_path=data_path, split='test', item_seq_len=250)
