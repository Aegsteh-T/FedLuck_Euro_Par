'''
simulate delay
'''
import numpy as np
import random

def generate_delays(global_config):
    # delay config, including mode, params
    delay_config = global_config["delay"]
    n_clients = global_config["n_clients"]          # the number of clients
    mode = delay_config["mode"]
    param = delay_config["param"]
    delays = []
    if mode == "base":
        for i in range(n_clients):
            # delays.append(param * 5 * i + 1)
            delays.append(2.4 * param + i)
    return delays


def generate_bandwidths(global_config):
    # delay config, including mode, params
    bandwidth_config = global_config["bandwidth"]
    n_clients = global_config["n_clients"]          # the number of clients
    mode = bandwidth_config["mode"]
    
    bandwidths = []
    if mode == "base":
        bandwidth0 = bandwidth_config["param"]
        for i in range(n_clients):
            bandwidths.append(bandwidth0 / np.float_power(i + 1, 0.25))
    elif mode == "exp":
        min_band = 0.1
        max_band = 1
        for i in range(n_clients):
            bandwidths.append(random.uniform(min_band, max_band))
    elif mode == "uniform":
        min_bw=bandwidth_config["min"]
        max_bw=bandwidth_config["max"]
        # bandwidths = [random.uniform(min_bw, max_bw) for _ in range(n_clients)]
        bandwidths=np.linspace(min_bw,max_bw,n_clients).tolist()
        bandwidths[1]=bandwidth_config["straggler"]
    else:
        raise NotImplementedError
    return bandwidths

def generate_alpha(global_config):
    alpha_min = global_config["alpha_min"]
    alpha_max = global_config["alpha_max"]
    n_clients = global_config["n_clients"]          # the number of clients
    return np.linspace(alpha_min, alpha_max, n_clients).tolist()[::-1]