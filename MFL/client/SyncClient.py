import torch
import os,sys
import copy
import time
os.chdir(sys.path[0])

import threading

from multiprocessing import Process
from ..model import get_model
import random
from ..datasets import CustomerDataset, get_default_data_transforms
import numpy as np

from ..compressor import NoneCompressor, TopkCompressor
import MFL.compressor as comp

import MFL.tools.tensorTool as tl
import MFL.tools.jsonTool as jsonTool

mode='Sync'
config_file = jsonTool.get_config_file(mode=mode)
config = jsonTool.generate_config(config_file)

class SyncClient():
    def __init__(self,cid, dataset, client_config, compression_config, bandwidth, device):
        self.cid = cid          # the id of client
        self.client_config=client_config
        # model
        self.model_name = client_config["model"]              
        self.model = get_model(self.model_name).to(device)       # mechine learning model

        self.W = {name : value for name, value in self.model.named_parameters()}            # model weight reference
        self.dW_compressed = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}     # compressed gradient
        self.dW = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}                # gradient
        self.W_old = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}             # global model before local training
        self.A = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}                 # Error feedback

        # hyperparameters
        self.local_iteration = client_config["local iteration"]      # local iteration num
        self.lr = client_config["optimizer"]["lr"]      # learning rate
        self.momentum = client_config["optimizer"]["momentum"]  # momentum
        self.batch_size = client_config["batch_size"]       # batch size
        self.bandwidth = bandwidth          # simulate network bandwidth
        self.size_of_weight = tl.getModelSize(self.model)
        self.cr = compression_config["uplink"]["params"]["cr"]

        # dataset
        self.dataset_name = client_config["dataset"]
        self.dataset = dataset      # the dataset of client, a list with 2 elements, the first is all data, the second is all label
        self.split_train_test(proportion=0.8)
        self.transforms_train, self.transforms_eval = get_default_data_transforms(self.dataset_name)
        self.train_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_train, self.y_train, self.transforms_train), 
                                                        batch_size=self.batch_size,
                                                        shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval), 
                                                        batch_size=self.batch_size,
                                                        shuffle=False)   


        # loss function
        self.loss_fun_name = client_config["loss function"]        # loss function
        self.loss_function = self.init_loss_fun()

        # optimizer
        self.optimizer_hp = client_config["optimizer"]      # optimizer
        self.optimizer = self.init_optimizer()

        # compressor
        self.compression_config = compression_config
        
        # training device
        self.device = device            # training device (cpu or gpu)

        self.t = 0
    
    def run(self):          # run the client process
        self.synchronize_with_server(self.server)
        # self.t += 1
        # d_dW = copy.deepcopy(self.dW)
        # dW_old = copy.deepcopy(self.dW)
        # Training mode
        start_train_time = time.time()
        self.model.train()
        end_train_time = time.time()
        computation_consumption = end_train_time - start_train_time

        # W_old = W
        tl.copy_weight(self.W_old,self.W)

        # local training, SGD
        self.train_model()           # local training

        # dW = W - W_old
        tl.subtract_(self.dW,self.W,self.W_old)     # gradient computation

        # if self.t > 1:
        #     tl.subtract_(d_dW, self.dW, dW_old)
        #     up = tl.norm_2(d_dW)
        #     down = tl.norm_2(self.dW)
        #     print('L = {}'.format(up / down))

        # compress gradient
        self.compress_weight(compression_config=self.compression_config["uplink"])
        beta = self.size_of_weight / self.bandwidth          # full model transmit time
        #print beta
        print(f'Client {self.cid} beta: {beta:.2f}')
        communication_consumption = self.cr * self.size_of_weight
        
        # set transmit dict
        transmit_dict = {"cid": self.cid,
                             "client_gradient": self.dW_compressed,
                             "data_num": len(self.x_train), 
                             "computation_consumption": self.local_iteration*self.client_config['alpha'],
                             "communication_time": beta*self.cr,
                             "communication_consumption": communication_consumption}
      
        # transmit to server
        self.server.receive(transmit_dict)    # send (cid,gradient,weight,timestamp) to server
    
    def compress_weight(self, compression_config=None):
        accumulate = compression_config["params"]["error_feedback"]
        if accumulate:
            # compression with error accumulation     
            tl.add(target=self.A, source=self.dW)
            tl.compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(compression_config, self.device))
            tl.subtract(target=self.A, source=self.dW_compressed)

        else: 
            # compression without error accumulation
            tl.compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(compression_config, self.device))
    
    def train_model(self):
        start_time = time.time()
        self.model.train()
        train_acc = 0.0
        train_loss = 0.0
        train_num = 0
        for epoch in range(self.local_iteration):
            try: # Load new batch of data
                features, labels = next(self.epoch_loader)
            except: # Next epoch
                self.epoch_loader = iter(self.train_loader)
                features, labels = next(self.epoch_loader)
            features, labels = features.to(self.device),labels.to(self.device)
            self.optimizer.zero_grad()                              # set accumulate gradient to zero
            outputs = self.model(features)                          # predict
            loss = self.loss_function(outputs, labels)              # compute loss
            loss.backward()                                         # backward, compute gradient
            self.optimizer.step()                                   # update

            train_loss += loss.item()                               # compute total loss
            _, prediction = torch.max(outputs.data, 1)              # get prediction label
            train_acc += torch.sum(prediction == labels.data)       # compute training accuracy
            train_num += self.train_loader.batch_size
            # time.sleep(0.02 * self.cid)
        
        train_acc = train_acc / train_num              # compute average accuracy and loss
        train_loss = train_loss / train_num
        end_time = time.time()
        print("Client {}, Train Accuracy: {} , Train Loss: {}, Used Time: {}\n".format(self.cid, train_acc, train_loss, end_time - start_time,self.compression_config["uplink"]["params"]["cr"]))
    
    def synchronize_with_server(self,server):
        tl.copy_weight(target=self.W, source=server.W)
    
    def init_loss_fun(self):
        if self.loss_fun_name == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fun_name == 'MSE':
            return torch.nn.MSELoss()
    
    def init_optimizer(self):
        optimizer_name = self.optimizer_hp["method"]
        if optimizer_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(),self.lr,self.momentum)
    
    def split_train_test(self,proportion): 
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
        train_num = int(proportion * len(self.data))       # the number of training samples
        self.train_num = train_num
        self.test_num = len(self.data) - train_num
        self.x_train = data[:train_num]              # the data of training set
        self.y_train = label[:train_num]            # the label of training set
        self.x_test = data[train_num:]               # the data of testing set
        self.y_test = label[train_num:]             # the label of testing set

    def send(self,server,transmit_dict):
        server.receive(transmit_dict)

    def get_model_params(self):
        return self.model.state_dict()
    
    def set_stop_event(self,stop_event):
        self.stop_event = stop_event
    
    def set_selected_event(self,bool):
        if bool == True:
            self.selected_event.set()
        else:
            self.selected_event.clear()
    
    def set_server(self,server):
        self.server = server