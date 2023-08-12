#!/usr/bin/env bash

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia

conda install pyg -c pyg
pip install dgl-cu116 dglgo -f https://data.dgl.ai/wheels/repo.html

pip install ogb==1.3.6