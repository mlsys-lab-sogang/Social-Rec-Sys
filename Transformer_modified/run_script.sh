#!/usr/bin/env bash
export DATASET="$1"
export MODEL_NAME="$2"
export USER_SEQ_LEN="$3"
export ITEM_SEQ_LEN="$4"

python3 main.py --dataset "$DATASET" --seed 42 --name model_compare_seed_42_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 62 --name model_compare_seed_62_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 365 --name model_compare_seed_365_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 462 --name model_compare_seed_462_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 583 --name model_compare_seed_583_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 731 --name model_compare_seed_731_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 765 --name model_compare_seed_765_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 921 --name model_compare_seed_921_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 1234 --name model_compare_seed_1234_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s

python3 main.py --dataset "$DATASET" --seed 12355 --name model_compare_seed_12355_"$MODEL_NAME" --user_seq_len "$USER_SEQ_LEN" --item_seq_len "$ITEM_SEQ_LEN"
sleep 20s