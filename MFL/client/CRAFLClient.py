import copy

from ..model import get_model
from ..tools import jsonTool
import MFL.tools.utils
import MFL.tools.tensorTool as tl
import MFL.compressor as comp
from ..compressor import NoneCompressor, TopkCompressor
import numpy as np
from ..datasets import CustomerDataset, get_default_data_transforms

from multiprocessing import Process
import multiprocessing
import torch
import os
import sys
import time
os.chdir(sys.path[0])

mode = 'crafl'
config_file = jsonTool.get_config_file(mode=mode)
config = jsonTool.generate_config(config_file)
device = MFL.tools.utils.get_device(config["device"])

class CRAFLClient:
    def __init__(self, cid, dataset, client_config, compression_config, bandwidth, device):
        self.cid = cid          # the id of client

        # model
        self.model_name = client_config["model"]
        self.model = get_model(self.model_name).to(
            device)       # mechine learning model

        # config
        self.client_config = client_config
        self.compression_config = copy.deepcopy(compression_config)

        # model weight reference
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compressed = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}     # compressed gradient
        self.dW_compressed_cpu = {name: torch.zeros(
            value.shape) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}                # gradient
        # global model before local training
        self.W_old = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}
        self.A = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}                 # Error feedback

        # local iteration num
        self.local_iteration = client_config["local iteration"]
        self.lr = client_config["optimizer"]["lr"]      # learning rate
        self.momentum = client_config["optimizer"]["momentum"]  # momentum
        self.batch_size = client_config["batch_size"]       # batch size
        self.bandwidth = bandwidth          # simulate network bandwidth
        self.size_of_weight = tl.getModelSize(self.model)
        self.cr = compression_config["uplink"]["params"]["cr"]

        # dataset
        self.dataset_name = client_config["dataset"]
        # the dataset of client, a list with 2 elements, the first is all data, the second is all label
        self.dataset = dataset
        self.split_train_test(proportion=0.8)
        self.transforms_train, self.transforms_eval = get_default_data_transforms(
            self.dataset_name)
        self.train_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_train, self.y_train, self.transforms_train),
                                                        batch_size=self.batch_size,
                                                        shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval),
                                                       batch_size=self.batch_size,
                                                       shuffle=False)

        self.model_timestamp = 0        # timestamp, to compute staleness for server

        # loss function
        # loss function
        self.loss_fun_name = client_config["loss function"]
        self.loss_function = self.init_loss_fun()

        # optimizer
        self.optimizer_hp = client_config["optimizer"]      # optimizer
        self.optimizer = self.init_optimizer()

        # compressor
        self.compression_config = compression_config

        # training device
        self.device = device            # training device (cpu or gpu)

        # multiple process valuable
        self.selected_event = False     # indicate if the client is selected
        self.stop_event = False

    def __getstate__(self):
        """return a dict for current status"""
        state = self.__dict__.copy()
        res_keys = ['cid', 'W', 'dW', 'dW_compressed', 'W_old', 'A', 'local_iteration', 'dataset',
                    'lr', 'momentum', 'batch_size', 'bandwidth', 'model_stamp', 'selected_event', 'stop_event',
                    'client_config', 'compression_config']
        res_state = {}
        for key, value in state.items():
            if key in res_keys:
                res_state[key] = value
        return res_state

    def __setstate__(self, state):
        """return a dict for current status"""
        self.__dict__.update(state)
        return state

    def compress_weight(self, compression_config=None):
        accumulate = compression_config["params"]["error_feedback"]
        if accumulate:
            # compression with error accumulation
            tl.add(target=self.A, source=self.dW)
            tl.compress(target=self.dW_compressed, source=self.A,
                        compress_fun=comp.compression_function(compression_config))
            tl.subtract(target=self.A, source=self.dW_compressed)

        else:
            # compression without error accumulation
            tl.compress(target=self.dW_compressed, source=self.dW,
                        compress_fun=comp.compression_function(compression_config))

    def train_model(self):
        # print("Hello world")
        start_time = time.time()
        self.model.train()
        train_acc = 0.0
        train_loss = 0.0
        train_num = 0
        for epoch in range(self.local_iteration):
            # print(epoch)
            try:  # Load new batch of data
                features, labels = next(self.epoch_loader)
            except:  # Next epoch
                self.epoch_loader = iter(self.train_loader)
                features, labels = next(self.epoch_loader)
            features, labels = features.to(self.device), labels.to(self.device)
            # set accumulate gradient to zero
            self.optimizer.zero_grad()
            outputs = self.model.to(self.device)(
                features)                          # predict
            loss = self.loss_function(
                outputs, labels)              # compute loss
            # backward, compute gradient
            loss.backward()
            self.optimizer.step()                                   # update

            train_loss += loss.item()                               # compute total loss
            # get prediction label
            _, prediction = torch.max(outputs.data, 1)
            # compute training accuracy
            train_acc += torch.sum(prediction == labels.data)
            train_num += self.train_loader.batch_size
            time.sleep(0.02 * self.cid)
        train_acc = train_acc / train_num
        train_loss = train_loss / train_num
        end_time = time.time()

        # print("Client {}, Global Epoch {}, Train Accuracy: {} , Train Loss: {}, Used Time: {},cr: {},Local Iteration: {}\n".format(
        #     self.cid, self.model_timestamp, train_acc, train_loss, end_time - start_time,
        #     self.cr,
        #     self.local_iteration))

    def synchronize_with_server(self, GLOBAL_INFO):
        self.model_timestamp = GLOBAL_INFO[self.cid]['timestamp']
        W_G = GLOBAL_INFO[self.cid]['weight']
        tl.to_gpu(W_G, W_G)
        tl.copy_weight(target=self.W, source=W_G)
        self.local_iteration = GLOBAL_INFO[self.cid]["local iteration"]
        self.cr = GLOBAL_INFO[self.cid]["cr"]
        print('Client - {}, local iteration = {}, cr = {}'.format(self.cid,
              self.local_iteration, self.cr))

    def init_loss_fun(self):
        if self.loss_fun_name == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fun_name == 'MSE':
            return torch.nn.MSELoss()

    def init_optimizer(self):
        optimizer_name = self.optimizer_hp["method"]
        if optimizer_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(), self.lr, self.momentum)

    def split_train_test(self, proportion):
        # proportion is the proportion of the training set on the entire data set
        self.data = self.dataset[0]     # get raw data from dataset
        self.label = self.dataset[1]    # get label from dataset

        # package shuffle
        assert len(self.data) == len(self.label)
        randomize = np.arange(len(self.data))
        np.random.shuffle(randomize)
        data = np.array(self.data)[randomize]
        label = np.array(self.label)[randomize]

        # split train and test set
        # the number of training samples
        train_num = int(proportion * len(self.data))
        self.train_num = train_num
        self.test_num = len(self.data) - train_num
        self.x_train = data[:train_num]              # the data of training set
        self.y_train = label[:train_num]            # the label of training set
        self.x_test = data[train_num:]               # the data of testing set
        self.y_test = label[train_num:]             # the label of testing set

    def get_model_params(self):
        return self.model.state_dict()

    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def set_selected_event(self, bool):
        self.selected_event = bool

    def set_server(self, server):
        self.server = server


def get_client_from_temp(client_temp):
    client = CRAFLClient(cid=client_temp.cid,
                         dataset=client_temp.dataset,
                         client_config=client_temp.client_config,
                         compression_config=client_temp.compression_config,
                         bandwidth=client_temp.bandwidth,
                         device=device)
    return client


def run_client(client_temp, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO):
    # get a full attributed client
    client = get_client_from_temp(client_temp)
    cid = client.cid            # get cid for convenience
    # if the training process is going on
    while not STOP_EVENT.value:
        # if the client is selected by scheduler
        if SELECTED_EVENT[cid]:
            # synchronize
            client.synchronize_with_server(GLOBAL_INFO)

            # Training mode
            client.model.train()

            # W_old = W
            tl.copy_weight(client.W_old, client.W)
            
            # local training, SGD
            start_train_time = time.time()
            client.train_model()           # local training
            end_train_time = time.time()
            mu_i = (end_train_time - start_train_time) / client.local_iteration
            computation_consumption = mu_i * client.local_iteration

            # dW = W - W_old
            # gradient computation
            tl.subtract_(client.dW, client.W, client.W_old)

            # compress gradient
            client.compress_weight(
                compression_config=client.compression_config["uplink"])

            # set transmit dict
            tl.to_cpu(client.dW_compressed_cpu, client.dW_compressed)

            # transmit to server (simulate network bandwidth)
            beta = client.size_of_weight / client.bandwidth
            communication_consumption = client.cr * client.size_of_weight
            time.sleep(beta * client.cr)
            # send (cid,gradient,weight,timestamp) to server
            transmit_dict = {"cid": cid,
                             "client_gradient": client.dW_compressed_cpu,
                             "data_num": len(client.x_train),
                             "timestamp": client.model_timestamp,
                             "mu": mu_i,
                             "computation_consumption": computation_consumption,
                             "beta": beta,
                             "communication_consumption": communication_consumption}
            GLOBAL_QUEUE.put(transmit_dict)
            # set selected false, sympolize the client isn't on training
            SELECTED_EVENT[cid] = False
    print("Client {} Exit.\n".format(client.cid))
