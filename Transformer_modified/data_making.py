"""
data_utils.py에 있는 함수들을 사용해
.mat -> .csv -> degree table & interacted item table 등등 필요한 작업을 수행.

TODO: 적절히 순차적으로 분기할 수 있도록 작성. 현재는 주석단위로 처리하고있음.
"""
import os

import data_utils as utils


data_path = os.getcwd() + '/dataset/' + 'epinions' 

# utils.mat_to_csv(data_path)

# utils.generate_social_dataset(data_path, save_flag=True, split='train')
# utils.generate_social_dataset(data_path, save_flag=True, split='test')
# utils.generate_social_dataset(data_path, save_flag=True, split='valid')

# utils.generate_user_degree_table(data_path)
# utils.generate_item_degree_table(data_path)


# utils.generate_interacted_items_table(data_path, all=True, split='train')
# utils.generate_interacted_items_table(data_path, all=True, split='test')
# utils.generate_interacted_items_table(data_path, all=True, split='valid')


# utils.generate_social_random_walk_sequence(data_path, walk_length=20, save_flag=True, all_node=True, seed=False, split='train')
# utils.generate_social_random_walk_sequence(data_path, walk_length=20, save_flag=True, all_node=True, seed=False, split='test')
# utils.generate_social_random_walk_sequence(data_path, walk_length=20, save_flag=True, all_node=True, seed=False, split='valid')

# utils.generate_input_sequence_data(data_path=data_path, split='train', item_seq_len=250)
utils.generate_input_sequence_data(data_path=data_path, split='test', item_seq_len=250)
# utils.generate_input_sequence_data(data_path=data_path, split='valid', item_seq_len=250)