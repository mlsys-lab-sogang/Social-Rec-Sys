#!/usr/bin/env bash
export DATASET="$1"

python3 main.py --dataset "$DATASET" --seed 42 --name model_compare_seed_42 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 62 --name model_compare_seed_62 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 365 --name model_compare_seed_365 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 462 --name model_compare_seed_462 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 583 --name model_compare_seed_583 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 731 --name model_compare_seed_731 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 765 --name model_compare_seed_765 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 921 --name model_compare_seed_921 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 1234 --name model_compare_seed_1234 --item_seq_len 250
sleep 1m

python3 main.py --dataset "$DATASET" --seed 12355 --name model_compare_seed_12355 --item_seq_len 250
sleep 1m