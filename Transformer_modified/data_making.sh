#!/usr/bin/env bash
export DATASET="$1"
export RANDOM_WALK_LEN="$2"
export ITEM_SEQ_LEN="$3"

python3 data_making.py --dataset "$DATASET" --first True --seed 42 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 42 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 62 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 62 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 365 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 365 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 462 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 462 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 583 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 583 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 731 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 731 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 765 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 765 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 921 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 921 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 1234 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 1234 data making finished"
sleep 15s

python3 data_making.py --dataset "$DATASET" --seed 12355 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"

echo "seed 12355 data making finished"

echo "Finished"