#!/bin/bash

GPU_ID=0
for run_id in `seq 1 1`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python main_mini_dataset.py \
    --savename save/mini/test_$run_id/ 
done