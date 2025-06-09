import os
import subprocess
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def run():
    current_time = datetime.now().strftime('%m%d_%H%M')
    mode_verbose=mode
    if mode=='ASync' and use_FedLuck:
        if async_auto:
            mode_verbose='ASync_auto'
        elif async_auto_fixed_cr:
            mode_verbose='ASync_auto_fixed_cr'
        elif async_auto_fixed_li:
            mode_verbose='ASync_auto_fixed_li'
        else:
            raise ValueError('Invalid mode')
    name=f'{mode_verbose}_{li}_{cr}'
    res_path = os.path.join(
        DIR, name)
    os.makedirs(res_path, exist_ok=True)
    log_file = os.path.join(res_path,f"{name}.log")
    lr_cli = lr
    if os.path.isfile(log_file):
        os.remove(log_file)
    command = f"python3 -u main.py{' --async_auto' if async_auto else ''}{' --async_auto_fixed_cr' if async_auto_fixed_cr else ''}{' --async_auto_fixed_li' if async_auto_fixed_li else ''}{' --naive_auto' if naive_auto else ''}{' --iid' if iid else ''}{' --err_feedback' if err_feedback else ''} --kpow {kpow} --dpow {dpow} --n_clients {n_clients} --checkpoint_step {checkpoint_step} --acc {acc} --mode={mode} --model {MODEL} --dataset {DATASET} --bs {batch_size} --seed {seed} --comp topk --alpha_min {alpha_min} --alpha_max {alpha_max} --gpu_idx {gpu_idx} --lr_cli {lr_cli} --lr_server {lr_server} --epoch_interval {epoch_interval} --epochs {epochs} --li {li} --cr {cr} --res_path {res_path} --test_ratio {test_ratio}"
    print(command)
    with open(log_file, 'a') as f:
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            print(line.rstrip().decode('utf-8'), file=f)
            print(line.rstrip().decode('utf-8'))


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--dataset', type=str, default='FMNIST', help='Dataset name')
    parser.add_argument('--mode', type=str, default='ASync', help='Mode')
    parser.add_argument('--auto', action='store_true', help='Use FedLuck')
    parser.add_argument('--fedavg_topk', action='store_true', help='Use FedAvg top-k')
    parser.add_argument('--dir', type=str, default='results', help='Results directory')
    parser.add_argument('--iid', action='store_true', help='Use IID')
    parser.add_argument('--fixed_li', action='store_true', help='Fix li')
    parser.add_argument('--fixed_cr', action='store_true', help='Fix cr')
    
    args = parser.parse_args()
    
    # Access control variables
    DATASET = args.dataset
    mode = args.mode
    use_FedLuck = args.auto
    fedavg_topk = args.fedavg_topk
    iid = args.iid
    DIR=args.dir
    fixed_li=args.fixed_li
    fixed_cr=args.fixed_cr
    # DIR=f'{DATASET}_{"IID" if iid else "NIID"}'
    
    print(f'DATASET={DATASET},DIR={DIR},mode={mode},use_FedLuck={use_FedLuck},fedavg_topk={fedavg_topk},iid={iid}')
    
    # checks
    if use_FedLuck:
        assert mode=='ASync'
        
    if fedavg_topk:
        assert mode=='Sync'
        
    
    # shared settings
    
    gpu_idx = -1
    seed = 0 
    n_clients = 10
    kpow = 1
    dpow = 0.5
    err_feedback=False
    
    # experiment-specific settings
    if DATASET=='CIFAR10':
        MODEL='VGG11s_3'
        test_ratio=0.1
        epoch_interval = 3
        lr = 0.01
        batch_size = 32
        checkpoint_step=10
        acc = 0.90
        alpha_min = 0.15
        alpha_max = 0.15*4
        # beta will be calc by clients using bandwidth
        async_auto = False # use FedLuck
        async_auto_fixed_li=False
        async_auto_fixed_cr=False
        naive_auto=False
        if use_FedLuck:
            naive_auto=False
            assert not (fixed_li and fixed_cr)
            if fixed_li:
                async_auto_fixed_li=True
            elif fixed_cr:
                async_auto_fixed_cr=True
            else:
                async_auto = True
        lr_server = 1.0
        if mode=='ASync':
            li=8
            cr=0.1
            epochs = 2200
            if use_FedLuck:
                li=1
                cr=0.01
                if fixed_li:
                    li=7
                    epochs = 4000
                elif fixed_cr:
                    cr=0.1
        elif mode=='Sync':
            li=13
            cr=0.2 if fedavg_topk else 1.0
            epochs = 300 if fedavg_topk else 200
        elif mode=='AFO':
            li=20
            cr=1.0
            epochs = 2000
        elif mode=='FedBuff':
            li=13
            cr=1.0
            epochs = 600
    if DATASET=='CIFAR100':
        MODEL='ResNet18'
        test_ratio=0.1
        epoch_interval = 3
        lr = 0.01
        batch_size = 32
        checkpoint_step=10
        acc = 0.90
        alpha_min = 0.15
        alpha_max = 0.15*4
        # beta will be calc by clients using bandwidth
        async_auto = False # use FedLuck
        async_auto_fixed_li=False
        async_auto_fixed_cr=False
        naive_auto=False
        if use_FedLuck:
            naive_auto=False
            assert not (fixed_li and fixed_cr)
            if fixed_li:
                async_auto_fixed_li=True
            elif fixed_cr:
                async_auto_fixed_cr=True
            else:
                async_auto = True
        lr_server = 1.0
        if mode=='ASync':
            li=8
            cr=0.1
            epochs = 2200
            if use_FedLuck:
                li=1
                cr=0.01
                if fixed_li:
                    li=7
                    epochs = 4000
                elif fixed_cr:
                    cr=0.1
        elif mode=='Sync':
            li=13
            cr=0.2 if fedavg_topk else 1.0
            epochs = 300 if fedavg_topk else 200
        elif mode=='AFO':
            li=20
            cr=1.0
            epochs = 2000
        elif mode=='FedBuff':
            li=13
            cr=1.0
            epochs = 600
    if DATASET=='FMNIST':
        MODEL='CNN1'
        test_ratio=0.1
        epoch_interval = 1.5
        lr = 0.01
        batch_size = 64
        checkpoint_step=1
        acc = 0.90
        alpha_min = 0.1
        alpha_max = 0.1*4
        # beta will be calc by clients using bandwidth
        async_auto = False # use FedLuck
        async_auto_fixed_li=False
        async_auto_fixed_cr=False
        naive_auto=False
        if use_FedLuck:
            naive_auto=False
            assert not (fixed_li and fixed_cr)
            if fixed_li:
                async_auto_fixed_li=True
            elif fixed_cr:
                async_auto_fixed_cr=True
            else:
                async_auto = True
        lr_server = 0.7
        if mode=='ASync':
            li=10
            cr=0.3
            epochs = 600
            if use_FedLuck:
                li=1
                cr=1
                if fixed_li:
                    li=10
                elif fixed_cr:
                    cr=0.3
        elif mode=='Sync':
            li=20
            cr=0.5 if fedavg_topk else 1.0
            epochs = 150
        elif mode=='AFO':
            li=13
            cr=1.0
            epochs = 1000
        elif mode=='FedBuff':
            li=20
            cr=1.0
            epochs = 300
    if DATASET=='SC':
        MODEL='LSTM'
        test_ratio=0.2
        epoch_interval = 1.5
        lr = 0.01
        batch_size = 32
        checkpoint_step=10
        acc = 0.90
        alpha_min = 0.15
        alpha_max = 0.15*4
        # beta will be calc by clients using bandwidth
        async_auto = False # use FedLuck
        async_auto_fixed_li=False
        async_auto_fixed_cr=False
        naive_auto=False
        if use_FedLuck:
            naive_auto=False
            assert not (fixed_li and fixed_cr)
            if fixed_li:
                async_auto_fixed_li=True
            elif fixed_cr:
                async_auto_fixed_cr=True
            else:
                async_auto = True
            
        lr_server = 1.0
        if mode=='ASync':
            li=5
            cr=0.2
            epochs = 9000
            if use_FedLuck:
                li=1
                cr=1
                if fixed_li:
                    li=20
                elif fixed_cr:
                    cr=0.16
        elif mode=='Sync':
            li=8
            cr=0.5 if fedavg_topk else 1.0
            epochs = 1500
        elif mode=='AFO':
            li=5
            cr=1.0
            epochs = 7200
    run()