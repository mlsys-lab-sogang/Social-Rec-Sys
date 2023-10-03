#!/usr/bin/env bash
python3 data_making.py --dataset ciao --seed 42 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 42 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 62 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 62 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 1234 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 1234 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 12355 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 12355 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 731 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 731 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 765 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 765 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 583 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 583 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 365 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 365 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 462 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 462 data making finished"
sleep 1m

python3 data_making.py --dataset ciao --seed 921 --test_ratio 0.1 --random_walk_len 20 --item_seq_len 250

echo "seed 921 data making finished"

echo "Finished"