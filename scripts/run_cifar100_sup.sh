#!/bin/bash

GPU_ID=2
for run_id in `seq 1 1`; do
    CUDA_VISIBLE_DEVICES=$GPU_ID python main_cifar100_sup.py \
    --savename save/cifar_sup/test_$run_id/ 
done