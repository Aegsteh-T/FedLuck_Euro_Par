#!/bin/bash

current_datetime=$(date "+%m%d_%H%M")
base_dir="results_Dec"

fmnist_iid_dir="${base_dir}/FMNIST_iid_${current_datetime}"
fmnist_non_iid_dir="${base_dir}/FMNIST_non_iid_${current_datetime}"
cifar10_iid_dir="${base_dir}/CIFAR10_iid_${current_datetime}"
cifar10_non_iid_dir="${base_dir}/CIFAR10_non_iid_${current_datetime}"
cifar100_iid_dir="${base_dir}/CIFAR100_iid_${current_datetime}"
cifar100_non_iid_dir="${base_dir}/CIFAR100_non_iid_${current_datetime}"
sc_iid_dir="${base_dir}/SC_iid_${current_datetime}"
sc_non_iid_dir="${base_dir}/SC_non_iid_${current_datetime}"

fmnist_iid_dir_fixed="${base_dir}/FMNIST_iid_fixed_${current_datetime}"
cifar10_iid_dir_fixed="${base_dir}/CIFAR10_iid_fixed_${current_datetime}"
cifar100_iid_dir_fixed="${base_dir}/CIFAR100_iid_fixed_${current_datetime}"
sc_iid_dir_fixed="${base_dir}/SC_iid_fixed_${current_datetime}"

# Function to run experiments
run_iid() {
    dataset="$1"
    dir="$2"
    
    python run.py --dataset "$dataset" --mode ASync --auto --dir "$dir" --iid # ASync with auto param
    # python run.py --dataset "$dataset" --mode AFO --dir "$dir" --iid  # AFO
    # python run.py --dataset "$dataset" --mode FedBuff --dir "$dir" --iid  # FedBuff
    # python run.py --dataset "$dataset" --mode ASync --dir "$dir" --iid # ASync
    # python run.py --dataset "$dataset" --mode Sync --dir "$dir" --iid  # Sync
    python run.py --dataset "$dataset" --mode Sync --fedavg_topk --dir "$dir" --iid # Sync with topk
    
}

run_niid() {
    dataset="$1"
    dir="$2"
    
    python run.py --dataset "$dataset" --mode ASync --auto --dir "$dir" # ASync with auto param
    python run.py --dataset "$dataset" --mode ASync --dir "$dir" # ASync
    python run.py --dataset "$dataset" --mode Sync --dir "$dir" # Sync
    python run.py --dataset "$dataset" --mode Sync --fedavg_topk --dir "$dir" # Sync with topk
    python run.py --dataset "$dataset" --mode AFO --dir "$dir"  # AFO
    python run.py --dataset "$dataset" --mode FedBuff --dir "$dir" # FedBuff
}

run_fixed_iid() {
    dataset="$1"
    dir="$2"
    # python run.py --dataset "$dataset" --mode ASync --auto --dir "$dir" --iid
    python run.py --dataset "$dataset" --mode ASync --auto --fixed_li --dir "$dir" --iid
    # python run.py --dataset "$dataset" --mode ASync --auto --fixed_cr --dir "$dir" --iid
}

run_fixed_iid() {
    dataset="$1"
    dir="$2"
    python run.py --dataset "$dataset" --mode ASync --auto --dir "$dir" --iid
    python run.py --dataset "$dataset" --mode ASync --auto --fixed_li --dir "$dir" --iid
    python run.py --dataset "$dataset" --mode ASync --auto --fixed_cr --dir "$dir" --iid
}


# FMNIST
# run_iid "FMNIST" "$fmnist_iid_dir"
# run_niid "FMNIST" "$fmnist_non_iid_dir"

# # CIFAR10
#run_iid "CIFAR10" "$cifar10_iid_dir"
# run_niid "CIFAR10" "$cifar10_non_iid_dir"

# # CIFAR100
run_iid "CIFAR100" "$cifar100_iid_dir"
# run_niid "CIFAR100" "$cifar100_non_iid_dir"

# # SC
# run_iid "SC" "$sc_iid_dir"
# run_niid "SC" "$sc_non_iid_dir"

#fixed param

# run_fixed_iid "FMNIST" "$fmnist_iid_dir_fixed"

# run_fixed_iid "CIFAR10" "$cifar10_iid_dir_fixed"

# run_fixed_iid "SC" "$sc_iid_dir_fixed"