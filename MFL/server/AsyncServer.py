import torch
from ..model import get_model

import threading
import queue
import copy
import time
import schedule
import multiprocessing
from multiprocessing import Manager

import os
import sys

from ..compressor import NoneCompressor, TopkCompressor

from ..datasets import CustomerDataset, get_default_data_transforms

import MFL.server.ScheduleClass as sc

import MFL.tools.jsonTool as jsonTool
import MFL.tools.tensorTool as tl
import MFL.tools.resultTools as rt

from ..client.AsyncClient import run_client

import numpy as np
import MFL.tools.utils as utils

# load config
mode = 'ASync'
# config_file = jsonTool.get_config_file(mode=mode)
# config = jsonTool.generate_config(config_file)


class AsyncServer:
    def __init__(self, global_config, dataset, compressor_config, clients, device):
        # mutiple processes valuable
        self.global_stop_event = False

        # global_config
        self.global_config = global_config
        self.schedule_config = global_config["schedule"]

        # device
        self.device = device

        # model
        self.model_name = global_config["model"]
        self.model = get_model(self.model_name).to(device)  # mechine learning model
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.W_cpu = {name: torch.zeros(value.shape).to('cpu') for name, value in self.W.items()}  # used to transmit
        self.dW_compress = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(
            device) for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(device)
                  for name, value in self.W.items()}

        # receive queue
        self.parameter_queue = queue.Queue()

        # dataset
        self.dataset_name = global_config["dataset"]
        self.dataset = dataset

        # global iteration
        self.current_epoch = 0  # indicate the version of global model
        self.total_epoch = global_config["epoch"]

        # loss function
        # loss function
        self.loss_fun_name = global_config["loss function"]
        self.loss_func = self.init_loss_fun()

        self.compressor_config = compressor_config

        # results
        self.staleness_list = []
        self.loss_list = []
        self.accuracy_list = []
        self.gradient_num_list = []
        self.time_list = []  # used time
        self.communication_list = []  # communication bandwidth consumption
        self.real_alpha_list = []  # real alpha
        self.max_beta=0
        self.max_real_alpha=0
        self.start_time = time.time()
        
        # scheduling related
        self.last_update = None

        # global manager
        self.global_manager = AsyncGlobalManager(clients=clients,
                                                 dataset=dataset,
                                                 global_config=global_config,
                                                 stop_event=self.global_stop_event)

    def start(self, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO):  # start the whole training priod
        print("Start global training...\n")

        # processing pool
        client_pool = multiprocessing.Pool(len(self.global_manager.get_clients_list()))

        # load global model to GLOBAL_INFO
        tl.to_cpu(self.W_cpu, self.W)
        GLOBAL_INFO[0] = {"weight": self.W_cpu, "timestamp": self.current_epoch}

        # Start Training
        self.global_manager.start_clients(client_pool, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE,
                                          GLOBAL_INFO)  # start all clients for global training


        last=time.time()
        interval=self.global_config["epoch_time"]
        acc_needed=self.global_config.get("acc_needed",1.0)
        while self.current_epoch < self.total_epoch and (len(self.accuracy_list)==0 or self.accuracy_list[-1]<acc_needed):
            if(time.time()-last >= interval):
                # schedule.run_pending()
                last=time.time()
                self.update(STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE,GLOBAL_INFO)
                # print(f'update() takes {(time.time()-last):.2f}s')
                time.sleep(0.1) # since epoch_time is measured in seconds, sleeping for 0.1s is OK
                

        # stop global training
        STOP_EVENT.value = True
        client_pool.join()

        # Exit
        print("Global Updater Exit.\n")

    def print_update_interval(self):
        if self.last_update is not None:
            interval=time.time()-self.last_update
            expected=self.global_config["epoch_time"]
            if interval>expected+0.2:
                print(f'WARNING: update interval: {interval:.2f}s, expected: {self.global_config["epoch_time"]}, too slow')
        self.last_update=time.time()

    def update(self,STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE,GLOBAL_INFO):
        self.print_update_interval()
        time_start=time.time()
        if self.current_epoch >= self.total_epoch:  # if global training is going on
            return
        no_op=True
        if not GLOBAL_QUEUE.empty():            # if server has received some gradients from clients
            no_op=False
            client_gradients = []           # save multi local_W
            data_nums = []
            stalenesses = []
            gradient_num = 0
            communication_cost = 0
            print(f'Aggregating gradients from {GLOBAL_QUEUE.qsize()} clients')
            while not GLOBAL_QUEUE.empty():
                # get information from client,(cid, client_gradient, data_num, timestamp)
                transmit_dict = GLOBAL_QUEUE.get()
                # cid
                cid = transmit_dict["cid"]
                # client gradient
                client_gradient = transmit_dict["client_gradient"]
                tl.to_gpu(client_gradient, client_gradient,self.device)
                # number of data samples
                data_num = transmit_dict["data_num"]
                # timestamp of client gradient
                timestamp = transmit_dict["timestamp"]
                staleness = self.current_epoch - timestamp  # staleness
                gradient_num += 1

                # communication traffic consumption
                communication_consumption_cid = transmit_dict["communication_consumption"]
                communication_cost += communication_consumption_cid

                client_gradients.append(client_gradient)
                data_nums.append(data_num)
                stalenesses.append(staleness)
                self.staleness_list.append(staleness)
                real_alpha=transmit_dict['real_alpha']
                beta=transmit_dict['beta']
                self.real_alpha_list.append(real_alpha)
                #TODO: record these
                self.max_real_alpha=max(self.max_real_alpha,real_alpha)
                self.max_beta=max(self.max_beta,beta)
                
            time_collect=time.time()
            self.time_list.append(time.time() - self.start_time)
            utils.update_list_with_accumulation(self.communication_list, communication_cost)
            utils.update_list_with_accumulation(self.gradient_num_list, gradient_num)
            
            
            tl.weighted_average(target=self.dW,
                                sources=client_gradients,
                                weights=torch.Tensor(data_nums)) # global gradient
            tl.scale(self.dW,self.global_config["updater"]["params"]["lr_server"])
            # update global model
            tl.add(target=self.W, source=self.dW)
            # change global model
            tl.to_cpu(self.W_cpu,self.W)
            global_info = {'weight':self.W_cpu, 'timestamp':self.current_epoch}
            GLOBAL_INFO[0] = global_info
            time_update_model=time.time()
            self.eval_model()
            time_eval=time.time()

        self.current_epoch += 1
        print("Current Epoch: {}".format(self.current_epoch))
        
        # schedule
        participating_client_idxs = self.schedule(
            self.global_manager.clients_dict, self.schedule_config, SELECTED_EVENT)
        self.select_clients(participating_client_idxs, SELECTED_EVENT)

        # save result
        if self.global_config["checkpoint_step"]>0 and self.current_epoch % self.global_config["checkpoint_step"] == 0:
            self.save_model()
        self.save_result()
        time_finish=time.time()
        if not no_op:
            # collect|update model|eval|save2files
            print(f'Update time: {time_finish-time_start:.2f}s, break down: {time_collect-time_start:.2f}s|{time_update_model-time_collect:.2f}s|{time_eval-time_update_model:.2f}s|{time_finish-time_eval:.2f}s')

    def save_result(self):
        global_acc, global_loss = self.get_accuracy_and_loss_list()
        staleness_list = self.get_staleness_list()
        rt.save_results(self.global_config["result_path"],
                        dir_name=None,
                        config=self.global_config['config_copy'],
                        global_loss=global_loss,
                        global_acc=global_acc,
                        staleness=staleness_list,
                        gradient_num=self.gradient_num_list,
                        communication_cost=self.communication_list,
                        time=self.time_list,
                        real_alpha=self.real_alpha_list,)
    
    def save_model(self):
        if len(self.loss_list) == 0:
            return
        path=os.path.join(self.global_config["result_path"],
                          "checkpoints","{}.pt".format(self.current_epoch))
        rt.save_parameters(epoch=self.current_epoch,
                           net=self.model,
                           loss=self.loss_list[-1],
                           path=path)

    def init_loss_fun(self):
        if self.loss_fun_name == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fun_name == 'MSE':
            return torch.nn.MSELoss()

    def schedule(self, clients, schedule_config, SELECTED_EVENT, **kwargs):
        participating_clients = sc.idle_schedule(clients, schedule_config, SELECTED_EVENT)
        return participating_clients

    def select_clients(self, participating_clients_idxs, SELECTED_EVENT):
        for idx in participating_clients_idxs:
            SELECTED_EVENT[idx] = True

    def receive(self, transmit_dict):
        self.parameter_queue.put(transmit_dict)

    def eval_model(self):
        self.model.eval()
        data_loader = self.global_manager.test_loader
        test_correct = 0.0
        test_loss = 0.0
        test_num = 0
        for data in data_loader:
            features, labels = data
            features = features.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(features)  # predict
            _, id = torch.max(outputs.data, 1)
            test_loss += self.loss_func(outputs, labels).item()
            test_correct += torch.sum(id == labels.data).cpu().numpy()
            test_num += len(features)
        accuracy = test_correct / test_num
        loss = test_loss / test_num

        self.accuracy_list.append(accuracy)
        self.loss_list.append(loss)
        print("Server: Global Epoch {}, Test Accuracy: {} , Test Loss: {}".format(
            self.current_epoch, accuracy, loss))

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_staleness_list(self):
        return self.staleness_list


class AsyncGlobalManager:  # Manage clients and global information
    def __init__(self, clients, dataset, global_config, stop_event):
        # clients
        self.clients_num = len(clients)
        self.clients_list = clients
        self.clients_dict = {}
        self.register_clients(clients)

        # global infromation
        # global epoch/iteration
        self.global_epoch = global_config["epoch"]
        self.global_acc = []  # test accuracy
        self.global_loss = []  # training loss

        # global test dataset
        self.dataset_name = global_config["dataset"]
        # the test dataset of server, a list with 2 elements, the first is all data, the second is all label
        self.dataset = dataset

        self.x_test = dataset[0]
        self.y_test = dataset[1]
        
        if type(self.x_test) == torch.Tensor:
            self.x_test, self.y_test = self.x_test.numpy(), self.y_test.numpy()
        elif type(self.y_test) == list:
            self.y_test = np.array(self.y_test)

        # Create new x_test and y_test containing only part of the data
        from MFL.tools.utils import shrink_dataset
        self.x_test, self.y_test = shrink_dataset(self.x_test, self.y_test, global_config['test_ratio'])
        
        self.transforms_train, self.transforms_eval = get_default_data_transforms(self.dataset_name)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval),
                                                       batch_size=8,
                                                       shuffle=False)

        # multiple process valuable
        self.stop_event = stop_event  # False for initialization

    def find_client_by_cid(self, cid):  # find client by cid
        for client in self.clients:
            if client.cid == cid:
                return client
        return None

    def get_clients_dict(self):
        return self.clients_dict

    def get_clients_list(self):
        return self.clients_list

    def register_clients(self, clients):  # add multi-clients to server scheduler
        for client in clients:
            self.add_client(client)

    def add_client(self, client):  # add one client to server scheduler
        cid = client.cid
        if cid in self.clients_dict.keys():
            raise Exception("Client id conflict.")
        self.clients_dict[cid] = client

    # start all clients training
    def start_clients(self, client_pool, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO):
        clients_dict = self.get_clients_dict()
        for cid, client_thread in clients_dict.items():
            client_pool.apply_async(run_client, args=(
                client_thread, STOP_EVENT, SELECTED_EVENT, GLOBAL_QUEUE, GLOBAL_INFO),
                                    error_callback=err_call_back)  # add process to process pool
        client_pool.close()
        # client_pool.join()
        print("Start all client-threads\n")

    def stop_clients(self):
        clients_list = self.get_clients_dict()  # get all clients
        for cid, client_thread in clients_list.items():
            client_thread.set_stop_event(
                self.stop_event)  # start all clients


def err_call_back(err):
    print(f'Error: {str(err)}')
