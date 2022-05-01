#!/bin/bash

GPU_ID=1
for run_id in `seq 1 1`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python main_five_dataset.py \
    --savename save/five_tasks/test_$run_id/ 
done