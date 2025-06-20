import os
import sys
import json
import torch

os.chdir(sys.path[0])


def save_to_file(path, result):
    with open(path, 'w') as f:
        for i in result:
            f.write(str(i) + '\n')


def save_results(root=None, dir_name=None, config=None, global_loss=None,
                 global_acc=None, staleness=None, gradient_num=None,
                 communication_cost=None, computation_cost=None, time=None, real_alpha=None,
                 ):
    if dir_name is not None:
        dir_path = os.path.join(root, dir_name)  # experiment path
    else:
        dir_path = root
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not config is None:  # write config to json
        config_file_name = 'config.json'
        config_file_path = os.path.join(dir_path, config_file_name)
        with open(config_file_path, 'w') as f:
            json.dump(config, f)

    if not global_loss is None:
        global_loss_name = "global_loss.txt"
        global_loss_path = os.path.join(dir_path, global_loss_name)
        save_to_file(global_loss_path, global_loss)

    if not global_acc is None:
        global_acc_name = "global_acc.txt"
        global_acc_path = os.path.join(dir_path, global_acc_name)
        save_to_file(global_acc_path, global_acc)

    if not staleness is None:
        staleness_name = "staleness.txt"
        staleness_path = os.path.join(dir_path, staleness_name)
        save_to_file(staleness_path, staleness)

    if not gradient_num is None:
        gradient_num_name = "gradient_num.txt"
        gradient_num_path = os.path.join(dir_path, gradient_num_name)
        save_to_file(gradient_num_path, gradient_num)

    if not communication_cost is None:
        communication_cost_path = os.path.join(
            dir_path, "communication_cost.txt")
        save_to_file(communication_cost_path, communication_cost)

    if not computation_cost is None:
        computation_cost_path = os.path.join(dir_path, "computation_cost.txt")
        save_to_file(computation_cost_path, computation_cost)

    if not time is None:
        time_path = os.path.join(dir_path, 'time.txt')
        save_to_file(time_path, time)
        
    if not real_alpha is None:
        alpha_path = os.path.join(dir_path, 'real_alpha.txt')
        save_to_file(alpha_path, real_alpha)

def save_parameters(epoch, net,  loss, path, optimizer=None):
    save_dict = {'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'loss': loss,}

    if optimizer != None:
        save_dict['optimizer_state_dict'] = optimizer
    
    root = path[:path.rfind('/')]

    if not os.path.exists(root):
        os.makedirs(root)
    
    torch.save(save_dict, path)