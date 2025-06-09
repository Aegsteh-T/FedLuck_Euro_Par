#!/bin/bash

modes=("Sync" "ASync" "AFO" "FedBuff")
for mode in ${modes[*]};
do
    start_time=$(date +%s)
    echo "${mode} mode is training."
    python ../main.py \
        --mode ${mode} \
        --model "CNN1" \
        --dataset "MNIST" \
        --n_clients 5 \
        --epochs 10 \
        --res_path ../results/test \
        --li 15 \
        --bs 8 \
        --cr 0.01 \
        --comp topk \
        --err_feedback ${True}
    end_time=$(date +%s)
    cost_time=$[ $end_time - $start_time]
    echo "running ${mode} spends $(($cost_time/60))min $(($cost_time%60))s"
done