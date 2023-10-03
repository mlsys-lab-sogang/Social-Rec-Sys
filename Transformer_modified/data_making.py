"""
data_utils.py에 있는 함수들을 사용해
.mat -> .csv -> degree table & interacted item table 등등 필요한 작업을 수행.

현재 사용중인 seed 목록
    42 -> 62 -> 1234 -> 12355 -> 731 -> 765 -> 583 -> 365 -> 462 -> 921
"""

import argparse
import os

import data_utils as utils

def get_args():
    parser = argparse.ArgumentParser(description='Data preparation(preprocess) for Transformer input')
    parser.add_argument("--dataset", type=str, default="ciao", help="ciao // epinions")
    parser.add_argument("--seed", type=int, default=42, help="random seed, used in dataset split")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="percentage of valid/test dataset")
    parser.add_argument("--random_walk_len", type=int, default=20, help="random walk seqeunce length (encoder's input length)")
    parser.add_argument("--item_seq_len", type=int, default=250, help="item list length (decoder's input length)")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    data_path = os.getcwd() + '/dataset/' + args.dataset

    ############# .mat 파일 전처리
    utils.mat_to_csv(data_path, test=args.test_ratio, seed=args.seed)
    #############

    ############# random walk sequence를 생성하기 위한 전처리
    utils.generate_social_dataset(data_path, save_flag=True, seed=args.seed, split='train')
    utils.generate_social_dataset(data_path, save_flag=True, seed=args.seed, split='test')
    utils.generate_social_dataset(data_path, save_flag=True, seed=args.seed, split='valid')

    utils.generate_user_degree_table(data_path, split='train', seed=args.seed)
    utils.generate_user_degree_table(data_path, split='test', seed=args.seed)
    utils.generate_user_degree_table(data_path, split='valid', seed=args.seed)

    utils.generate_item_degree_table(data_path, split='train', seed=args.seed)
    utils.generate_item_degree_table(data_path, split='test', seed=args.seed)
    utils.generate_item_degree_table(data_path, split='valid', seed=args.seed)

    utils.generate_interacted_items_table(data_path, all=True, split='train', seed=args.seed)
    utils.generate_interacted_items_table(data_path, all=True, split='test', seed=args.seed)
    utils.generate_interacted_items_table(data_path, all=True, split='valid', seed=args.seed)
    #############

    ############# random walk sequence 생성
    utils.generate_social_random_walk_sequence(data_path, walk_length=args.random_walk_len, save_flag=True, all_node=True, seed=False, data_split_seed=args.seed, split='train')
    utils.generate_social_random_walk_sequence(data_path, walk_length=args.random_walk_len, save_flag=True, all_node=True, seed=False, data_split_seed=args.seed, split='test')
    utils.generate_social_random_walk_sequence(data_path, walk_length=args.random_walk_len, save_flag=True, all_node=True, seed=False, data_split_seed=args.seed, split='valid')
    #############

    ############# 모델 입력을 위한 최종 데이터셋 구성
    utils.generate_input_sequence_data(data_path=data_path, seed=args.seed, split='train', item_seq_len=args.item_seq_len)
    utils.generate_input_sequence_data(data_path=data_path, seed=args.seed, split='test', item_seq_len=args.item_seq_len)
    utils.generate_input_sequence_data(data_path=data_path, seed=args.seed, split='valid', item_seq_len=args.item_seq_len)
    #############

if __name__ == '__main__':
    main()