#!/bin/bash

GPU_ID=0

for run_id in `seq 1 1`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python main_pmnist.py \
    --savename save/pmnist/test_$run_id/
done