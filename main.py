from MFL.client.SyncClient import SyncClient
from MFL.client.AsyncClient import AsyncClient
from MFL.client.AFOClient import AFOClient
from MFL.client.FedBuffClient import FedBuffClient
from MFL.client.ERAFLClient import ERAFLClient
from MFL.client.FedLampClient import FedLampClient
from MFL.client.FedLGClient import FedLGClient

from MFL.server.AsyncServer import AsyncServer
from MFL.server.SyncServer import SyncServer
from MFL.server.AFOServer import AFOServer
from MFL.server.FedBuffServer import FedBuffServer
from MFL.server.ERAFLServer import ERAFLServer
from MFL.server.FedLampServer import FedLampServer
from MFL.server.FedLGServer import FedLGServer

from MFL.datasets import get_dataset, split_data, get_global_data

import MFL.tools.utils
from MFL.tools import jsonTool
import MFL.tools.resultTools as rt
import MFL.tools.delayTools as dt

import ctypes
import MFL.tools.utils
from MFL.tools import jsonTool
import multiprocessing
from multiprocessing import Manager
import MFL.tools.resultTools as rt
import MFL.tools.delayTools as dt

from MFL.server.AsyncServer import AsyncServer
from MFL.server.SyncServer import SyncServer
from MFL.server.AFOServer import AFOServer
from MFL.server.FedBuffServer import FedBuffServer

import os
import argparse

import random
import numpy as np
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# SYNC_BANDWIDTH = 1


def get_args():
    parser = argparse.ArgumentParser(description='federated learning')
    parser.add_argument('--mode', type=str, default='Sync', help='select the mode',
                        choices=['Sync', 'ASync', 'AFO', 'FedBuff', 'ERAFL', 'FedLamp', 'FedLG'])
    parser.add_argument('--model', type=str, help='the type of model', default='LSTM',
                        choices=['CNN1', 'CNN3', 'VGG11s', 'VGG11', 'VGG11s_3', 'ResNet18', 'logistic', 'LSTM','AlexNet','ResNet9'])
    parser.add_argument('--dataset', type=str, default='SC', help='set the dataset to be used',
                        choices=['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100', 'SC'])
    parser.add_argument('--n_clients', type=int, default=10,
                        help='the number of clients participating in training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='the number of global epochs')
    parser.add_argument('--res_path', type=str, default="./results/test",
                        help='The directory path to save result.')
    # we use uniform distribution now, this setting is oboselete
    # parser.add_argument('--bandwidth', type=float, default=1.5,
    #                     help='set client uplink bandwidth')
    parser.add_argument('--epoch_interval', type=float, default=1.5,
                        help='time length of each epoch')
    parser.add_argument('--lr_server', type=float, default=1,
                        help='server learning rate')
    parser.add_argument('--afo_alpha', type=float, default=0.4,
                        help='alpha for AFO moving avg')
    parser.add_argument('--acc', type=float, default=0.8,
                        help='set the target accuracy, program stops when fulfilled')
    parser.add_argument('--test_ratio', type=float, default=0.2,help='set the ratio for shrinking test dataset')
    
    parser.add_argument('--li', type=int, default=20,
                        help='the number of local iteration')
    parser.add_argument('--lr_cli', type=float, default=0.016,
                        help='client learning rate')
    parser.add_argument('--bs', type=int, default=64,
                        help='set the batch size')
    parser.add_argument('--cr', type=float, default=1,
                        help='set the uplink compression ratio')
    parser.add_argument('--comp', type=str, default='topk',
                        help='set the uplink compression method')
    parser.add_argument('--err_feedback', action='store_true',
                        help='use the error feedback')
    parser.add_argument('--async_auto', action='store_true',
                        help='auto decide li and cr')
    parser.add_argument('--async_auto_fixed_li', action='store_true',
                        help='auto decide cr')
    parser.add_argument('--async_auto_fixed_cr', action='store_true',
                        help='auto decide li')
    parser.add_argument('--naive_auto', action='store_true',
                        help='auto decide li and cr')
    parser.add_argument('--iid', action='store_true',
                        help='iid data distribution')
    parser.add_argument('--seed', type=int, default=0,
                        help='set seed')
    parser.add_argument('--gpu_idx',type=int,default=0,help='set gpu')
    parser.add_argument('--alpha_min',type=float,default=0.02,help='client alpha')
    parser.add_argument('--alpha_max',type=float,default=0.08,help='client alpha')
    parser.add_argument('--kpow',type=float,default=1,help='phi kpow')
    parser.add_argument('--dpow',type=float,default=1,help='phi dpow')
    parser.add_argument('--afo_rho',type=float,default=0.001,help='afo reg term')
    parser.add_argument('--checkpoint_step',type=int,default=0,help='check point step, 0 for disable')

    args = parser.parse_args()

    assert args.mode in ["Sync", "ASync", "AFO",
                         "FedBuff", "ERAFL", "FedLamp", "FedLG"], f"The mode is not in \"Sync\", \"ASync\", \"AFO\", \"FedBuff\"."
    assert args.model in ['CNN1', 'CNN3', 'VGG11s', 'VGG11', 'VGG11s_3', 'ResNet18',
                          'logistic', 'LSTM', 'AlexNet','ResNet9'], f"invalid model name"
    assert args.dataset in [
        'MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100', 'SC'], f"The dataset is not in ['MNIST', 'FMNIST', 'CIFAR10', 'CIFAR100']."
    assert args.n_clients > 0, f"The number of clients must be greater than 0."
    assert args.epochs > 0, f"The number of global epochs must be greater than 0."
    assert args.cr > 0 and args.cr <= 1, f"The compression ratio is {args.cr} not in the range of (0,1]."
    assert args.li > 0, f"The number of local iteration must be greater than 0."
    assert args.bs > 0, f"The batch size must be greater than 0."
    # assert args.bandwidth > 0, f"Bandwidth needs to be positive"
    assert args.epoch_interval > 0, f"Interval needs to be positive"
    assert args.lr_cli >0, "Need positive learning rate"
    assert args.lr_server>0, "Need positive learning rate"
    assert not np.isnan(args.alpha_min), "alpha min cannot be nan"
    assert not np.isnan(args.alpha_max), "alpha max cannot be nan"
    assert args.afo_alpha>=0 and args.afo_alpha<=1, "afo_alpha must be in [0,1]"
    assert args.acc>=0 and args.acc<=1, "acc must be in [0,1]"
    assert args.afo_rho>=0, "rho must be positive"
    assert args.test_ratio>=0 and args.test_ratio<=1, "test ratio must be in [0,1]"
    
    return args


args = get_args()


# load config
# read config json file and generate config dict
config_file = jsonTool.get_config_file(mode=args.mode)
config = jsonTool.generate_config(config_file)

# get global config
global_config = config["global"]
global_config["epoch"] = args.epochs
global_config["n_clients"] = args.n_clients
global_config["result_path"] = args.res_path
global_config["checkpoint_step"] = args.checkpoint_step

# global_config["bandwidth"]["param"]= args.bandwidth
global_config["epoch_time"]= args.epoch_interval
global_config["updater"]["params"]["lr_server"]=args.lr_server
global_config["seed"]=args.seed
global_config["alpha_min"]=args.alpha_min
global_config["alpha_max"]=args.alpha_max
global_config["afo_alpha"]=args.afo_alpha
global_config["acc_needed"]=args.acc
global_config['test_ratio']=args.test_ratio

# get client's config
client_config = config["client"]
client_config["model"] = global_config["model"] = args.model
client_config["dataset"] = global_config["dataset"] = args.dataset
client_config["local iteration"] = global_config["local iteration"] = args.li
client_config["loss function"] = global_config["loss function"]
client_config["batch_size"] = args.bs
client_config["optimizer"]["lr"] = args.lr_cli
client_config["seed"]=args.seed
client_config["afo_rho"]=args.afo_rho


client_config["epoch_time"]=args.epoch_interval
client_config["kpow"]=args.kpow
client_config["dpow"]=args.dpow
client_config['async_auto']=args.async_auto
client_config['async_auto_fixed_li']=args.async_auto_fixed_li
client_config['async_auto_fixed_cr']=args.async_auto_fixed_cr
client_config['naive_auto']=args.naive_auto


MFL.tools.utils.set_seed(args.seed)


# get training device according to os platform, gpu or cpu
device_name = MFL.tools.utils.get_least_busy_gpu_name() if args.gpu_idx==-1 else f'cuda:{args.gpu_idx}'
# device_name='cuda:0'
device=torch.device(device_name)
global_config["dev"]=device_name
client_config["dev"]=device_name
# gradient compression config
compressor_config = config["compressor"]
compressor_config["uplink"] = {"method": args.comp, "params": {
    "cr": args.cr, "error_feedback": args.err_feedback}}

# data distribution config

data_distribution_config = config["data_distribution"]
data_distribution_config["iid"] = args.iid

def get_data(dataset, n_clients):
    # dataset
    # load whole dataset
    dataset = get_dataset(dataset)
    train_set = dataset.get_train_dataset()  # get global training set
    split = split_data(data_distribution_config, n_clients, train_set)
    test_set = dataset.get_test_dataset()
    test_set = get_global_data(test_set)
    return split, test_set


def create_clients_server(n_clients, split, test_set):
    if args.mode == 'Sync':
        Client = SyncClient
        Server = SyncServer
    elif args.mode == 'ASync':
        Client = AsyncClient
        Server = AsyncServer
    elif args.mode == 'AFO':
        Client = AFOClient
        Server = AFOServer
    elif args.mode == 'FedBuff':
        Client = FedBuffClient
        Server = FedBuffServer
    elif args.mode == 'ERAFL':
        Client = ERAFLClient
        Server = ERAFLServer
    elif args.mode == 'FedLamp':
        Client = FedLampClient
        Server = FedLampServer
    elif args.mode == 'FedLG':
        Client = FedLGClient
        Server = FedLGServer
    
    
        
    
    

    clients = []
    cli_config_list=[dict(client_config) for _ in range(n_clients)] # each client has its own config copy
    # simulate bandwidths
    bandwidths = dt.generate_bandwidths(global_config)
    alpha_list=dt.generate_alpha(global_config)
    print(f'generated bandwidth {["{:.2f}".format(value) for value in bandwidths]}')
    print(f'generated alpha {["{:.2f}".format(value) for value in alpha_list]}')
    global_config["bandwidth"]["bandwidth_list"] = bandwidths
    global_config['config_copy']={'config':json.dumps(config)} # pass a config copy to server so it can save it to results
    bandwidth_avg=sum(bandwidths)/len(bandwidths)
    alpha_avg=sum(alpha_list)/len(alpha_list)
    for i in range(n_clients):
        cli_config_list[i]['global_bandwidth_avg']=bandwidth_avg
        cli_config_list[i]['global_alpha_avg']=alpha_avg
        cli_config_list[i]['alpha']=alpha_list[i]
    
    if args.mode == 'Sync' or args.mode == 'FedLamp':
        for i in range(n_clients):
            clients += [Client(cid=i,
                               dataset=split[i],
                               client_config=cli_config_list[i],
                               compression_config=compressor_config,
                               bandwidth=bandwidths[i],
                               device=device)]
        server = Server(global_config=global_config,
                        dataset=test_set,
                        compressor_config=compressor_config,
                        clients=clients,
                        device=device)
    else:
        for i in range(n_clients):
            clients += [Client(cid=i,
                               dataset=split[i],
                               client_config=cli_config_list[i],
                               compression_config=compressor_config,
                               bandwidth=bandwidths[i],
                               device=device)]
        server = Server(global_config=global_config,
                        dataset=test_set,
                        compressor_config=compressor_config,
                        clients=clients,
                        device=device)
    return clients, server


if __name__ == "__main__":
    # print config
    jsonTool.print_config(config)
    import json
    

    n_clients = args.n_clients
    split, test_set = get_data(args.dataset, n_clients)
    clients, server = create_clients_server(n_clients, split, test_set)

    for client in clients:
        client.set_server(server)

    # start training
    if args.mode == 'Sync' or args.mode == 'FedLamp':
        server.start()
    else:
        multiprocessing.set_start_method('spawn', force=True)
        MANAGER = Manager()  # multiprocessing manager
        # initialize STOP_EVENT, representing for if global training stops
        STOP_EVENT = MANAGER.Value(ctypes.c_bool, False)
        SELECTED_EVENT = MANAGER.list(
            [False for i in range(n_clients)])
        GLOBAL_QUEUE = MANAGER.Queue()
        GLOBAL_INFO = MANAGER.list([0])
        server.start(STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO)