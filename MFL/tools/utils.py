import os,sys
import torch
import GPUtil
import numpy as np
import random

def shrink_dataset(x,y, ratio):
    new_size = max(int(len(x)*ratio),1)
    indices = np.random.choice(len(x), new_size, replace=False)
    x_shrink = x[indices]
    y_shrink = y[indices]
    return x_shrink, y_shrink

def set_seed(seed):    
    # Set the seed for Python's random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_gpu_name():
    platform = sys.platform
    if platform == 'darwin':
        return 'mps'
    else:
        return 'cuda'

def get_least_busy_gpu_name():
    deviceIDs = GPUtil.getAvailable(order = 'memory')
    assert len(deviceIDs) > 0
    idx=deviceIDs[0]
    # print(f'selected: cuda:{idx}')
    return f'cuda:{idx}'

def get_device(device="gpu"):
    assert device=='gpu' or device=='cpu'
    if device == "gpu":
        return torch.device(get_gpu_name())
    else:
        return torch.device('cpu')
    
def update_list_with_accumulation(lst, num):
    if len(lst) == 0:
        lst.append(num)
    else:
        lst.append(lst[-1] + num)
