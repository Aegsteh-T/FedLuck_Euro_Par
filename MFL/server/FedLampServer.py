import torch
from ..model import get_model
import numpy as np

import math
import threading
import queue
import copy
import schedule

from ..compressor import NoneCompressor, TopkCompressor

from ..datasets import CustomerDataset, get_default_data_transforms

import MFL.server.ScheduleClass as sc

import MFL.tools.jsonTool as jsonTool
import MFL.tools.tensorTool as tl
import MFL.tools.resultTools as rt


def update_list(lst, num):
    if len(lst) == 0:
        lst.append(num)
    else:
        lst.append(lst[-1] + num)


class FedLampServer:
    def __init__(self, global_config, dataset, compressor_config, clients, device):
        # global_config
        self.global_config = global_config
        self.schedule_config = global_config["schedule"]
        self.n_clients = len(clients)

        # device
        self.device = device

        # model
        self.model_name = global_config["model"]
        self.model = get_model(self.model_name).to(device)  # mechine learning model
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.dW_compress = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}
        self.dW = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        self.A = {name: torch.zeros(value.shape).to(device) for name, value in self.W.items()}

        # receive queue
        self.parameter_queue = queue.Queue()

        # dataset
        self.dataset_name = global_config["dataset"]
        self.dataset = dataset

        # global iteration
        self.current_epoch = -1  # indicate the version of global model
        self.total_epoch = global_config["epoch"]

        # loss function
        self.loss_fun_name = global_config["loss function"]  # loss function
        self.loss_func = self.init_loss_fun()

        self.compressor_config = compressor_config

        # results
        self.staleness_list = []
        self.loss_list = []
        self.accuracy_list = []
        self.time_list = []  # used time
        self.communication_list = []  # communication bandwith consumption

        # global manager
        self.global_manager = SyncGlobalManager(clients=clients,
                                                dataset=dataset,
                                                global_config=global_config)

        # algorithm variables
        self.C = 10000      # computing resource budget, input
        self.B = 10000      # bandwith resource budget, input

        self.h = 0          # global epoch, line 1
        self.C_used = 0     # computation resource used, line 1
        self.B_used = 0     # communication resource used, line 1
        self.T_used = 0     # clock time used, line 1

        self.v = 0.01           # v

        self.lis = [30 for i in range(self.n_clients)]       # local iterations of clients
        self.crs = [0.1 for i in range(self.n_clients)]       # compression rate of clients
        self.alphas = [1 / self.n_clients for i in range(self.n_clients)]     # aggregation weight of clients
        self.tau_e = 60         # maxmimal local iteration
        self.tau_s = 20          # minimal local iteration
        self.cis = [0 for i in range(self.n_clients)]     # computing resouce of clients
        self.bis = [0 for i in range(self.n_clients)]     # bandwith resouce of clients
        self.mus = [0 for i in range(self.n_clients)]     # computation time
        self.betas = [0 for i in range(self.n_clients)]   # communication time
        self.clockTimes = [0 for i in range(self.n_clients)]  # compute max time
        self.Lis = [0 for i in range(self.n_clients)]
        self.sigmais = [0 for i in range(self.n_clients)]

        self.pho = 100          # pho, optimization problem parameter 

    def start(self):  # start the whole training priod
        print('Testing computing and bandwith resource..')
        self.test_consum()

        print("Start global training...\n")

        self.update()

        # Exit
        print("Global Updater Exit.\n")

    def update(self):
        for epoch in range(self.total_epoch):
            if self.C_used > self.C or self.B_used > self.B:
                return
            # send model weight, local iteration, compression rate
            for client in self.global_manager.clients_list:
                client.run()

            client_gradients = []  # save multi local_W
            data_nums = []
            self.current_epoch += 1

            # aggregation 
            max_time = -1
            communication_cost = 0
            while not self.parameter_queue.empty():
                transmit_dict = self.parameter_queue.get()  # get information from client,(cid, client_gradient, data_num, timestamp)
                cid = transmit_dict["cid"]  # cid
                client_gradient = transmit_dict["client_gradient"]  # client gradient
                data_num = transmit_dict["data_num"]  # number of data samples

                # total computation time
                computation_time_cid = transmit_dict["computation_consumption"]
                # communication traffic consumption
                communication_consumption_cid = transmit_dict["communication_consumption"]
                communication_cost += communication_consumption_cid
                # communication time
                communication_time_cid = transmit_dict["beta"] * self.crs[cid]
                max_time = max(communication_time_cid + computation_time_cid, max_time)  # total time

                client_gradients.append(client_gradient)
                data_nums.append(data_num)
                # mu and beta
                self.mus[cid] = transmit_dict["mu"]
                self.betas[cid] = transmit_dict["beta"]
                if self.current_epoch >= 1:
                    self.Lis[cid] = transmit_dict["Li"]
                    self.sigmais[cid] = transmit_dict["sigmai"]
            
            tl.weighted_average(target=self.dW,
                                sources=client_gradients,
                                weights=torch.tensor(self.alphas))  # global gradient
            tl.add(target=self.W, source=self.dW)

            # update parameters, line 12 - 14
            self.C_used += np.sum([self.lis[cid] * self.cis[cid] for cid in range(self.n_clients)])
            self.B_used += np.sum([self.crs[cid] * self.bis[cid] for cid in range(self.n_clients)])
            self.T_used += np.max([self.lis[cid] * (self.mus[cid] + self.v * self.betas[cid]) for cid in range(self.n_clients)])
            # line 19 
            if self.current_epoch >= 1:
                #TODO: moving avg
                l = np.argmin([self.mus[cid] + self.v * self.betas[cid] for cid in range(self.n_clients)])
                L = np.mean(self.Lis)
                sigma = np.mean(self.sigmais)
                # search 
                tau = self.search_tau(L, sigma, l)
                self.lis = [tau * (self.mus[l] + self.v * self.betas[l]) // (self.mus[cid] + self.v * self.betas[cid]) for cid in range(self.n_clients)]
                self.crs = [self.lis[cid] * self.v for cid in range(self.n_clients)]
                #FIX: use sqrt(li)
                sqrt_lis=np.sqrt(self.lis)
                self.alphas = [self.n_clients * sqrt_lis[cid] / np.sum(sqrt_lis) for cid in range(self.n_clients)]
            
            print("lis: {}.\ncrs: {}".format(self.lis, self.crs))
            # record
            update_list(self.time_list, max_time)
            update_list(self.communication_list, communication_cost)

            data_nums = torch.Tensor(data_nums)

            self.eval_model()

            # save results
            global_acc, global_loss = self.get_accuracy_and_loss_list()
            staleness_list = self.get_staleness_list()
            rt.save_results(self.global_config["result_path"],
                            dir_name="{}_{}_{}_FedLamp".format(self.global_config["model"],
                                                              self.global_config["dataset"],
                                                              self.global_config["local iteration"]),
                            config=None,
                            global_loss=global_loss,
                            global_acc=global_acc,
                            staleness=staleness_list,
                            communication_cost=self.communication_list,
                            time=self.time_list
                            )
    
    def test_consum(self):
        for client in self.global_manager.clients_list:
            cid = client.cid
            ci, bi = client.test_consum()
            self.cis[cid] = ci
            self.bis[cid] = bi

    def init_loss_fun(self):
        if self.loss_fun_name == 'CrossEntropy':
            return torch.nn.CrossEntropyLoss()
        elif self.loss_fun_name == 'MSE':
            return torch.nn.MSELoss()

    def scatter_init_model(self):
        for cid, client in self.global_manager.get_clients_dict().items():
            client.synchronize_with_server(self)
            model_timestamp = copy.deepcopy(self.current_epoch)["t"]
            client.model_timestamp = model_timestamp

    def schedule(self, clients, schedule_config, **kwargs):
        participating_clients = sc.schedule(clients, schedule_config)
        return participating_clients

    def select_clients(self, participating_clients):
        for client in participating_clients:
            client.set_selected_event(True)

    def receive(self, transmit_dict):
        self.parameter_queue.put(transmit_dict)

    def eval_model(self, retu=False):
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
        print("Server: Global Epoch {}, Test Accuracy: {} , Test Loss: {}".format(self.current_epoch, accuracy, loss))
        return 

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_staleness_list(self):
        return self.staleness_list
    
    def search_tau(self, L, sigma, l):
        for tau in range(self.tau_s, self.tau_e + 1):
            # compute g(H, tau)
            g = 2 * L * 0.012 / (self.v * tau * math.sqrt(tau * self.total_epoch * self.n_clients))
            g += self.v * tau * (sigma ** 2) / math.sqrt(tau * self.total_epoch * self.n_clients)
            g += (self.v ** 2) * tau * (sigma ** 2) / self.total_epoch
            # make sure g <= pho
            if g > self.pho:
                continue
            # compute local iterations
            lis = [tau * (self.mus[l] + self.v * self.betas[l]) // (self.mus[cid] + self.v * self.betas[cid]) for cid in range(self.n_clients)]
            # compute resource consumption
            c = np.sum([lis[cid] * self.cis[cid] for cid in range(self.n_clients)])
            b = np.sum([self.v * lis[cid] * self.bis[cid] for cid in range(self.n_clients)])
            rest_epoch = self.total_epoch - self.current_epoch
            rest_C = self.C - self.C_used
            rest_B = self.B - self.B_used
            if rest_epoch * c > rest_C or rest_epoch * b > rest_B:
                continue
            # if satisify min tau, return
            return tau
        return self.tau_s



class SyncGlobalManager:  # Manage clients and global information
    def __init__(self, clients, dataset, global_config):
        # clients
        self.clients_num = len(clients)
        self.clients_list = clients
        self.clients_dict = {}
        self.register_clients(clients)

        # global infromation
        self.global_epoch = global_config["epoch"]  # global epoch/iteration
        self.global_acc = []  # test accuracy
        self.global_loss = []  # training loss

        # global test dataset
        self.dataset_name = global_config["dataset"]
        self.dataset = dataset  # the test dataset of server, a list with 2 elements, the first is all data, the second is all label
        self.x_test = dataset[0]
        self.y_test = dataset[1]
        if type(self.x_test) == torch.Tensor:
            self.x_test, self.y_test = self.x_test.numpy(), self.y_test.numpy()
        elif type(self.y_test) == list:
            self.y_test = np.array(self.y_test)
        # print(self.y_test.shape)
        self.transforms_train, self.transforms_eval = get_default_data_transforms(self.dataset_name)
        self.test_loader = torch.utils.data.DataLoader(CustomerDataset(self.x_test, self.y_test, self.transforms_eval),
                                                       batch_size=8,
                                                       shuffle=False)

    def find_client_by_cid(self, cid):  # find client by cid
        for client in self.clients:
            if client.cid == cid:
                return client
        return None

    def get_clients_dict(self):
        return self.clients_dict

    def register_clients(self, clients):  # add multi-clients to server scheduler
        for client in clients:
            self.add_client(client)

    def add_client(self, client):  # add one client to server scheduler
        cid = client.cid
        if cid in self.clients_dict.keys():
            raise Exception("Client id conflict.")
        self.clients_dict[cid] = client

    def start_one_client(self, cid):
        clients_dict = self.get_clients_dict()  # get all clients
        for c in clients_dict.keys():
            if c == cid:
                clients_dict[c].start()
