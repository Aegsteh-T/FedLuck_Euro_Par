import torch
import math

def model2tensor(model):
    # Transform a model's entire weights into a 1D tensor
    return torch.cat([param.view(-1) for param in model.parameters()])

def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - \
            subtrahend[name].data.clone()


def copy_weight(target, source):
    for name in target:
        target[name].data = source[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack(
            [source[name].data for source in sources]), dim=0).clone()


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()

def scale(target, scalar):
    for name in target:
        target[name].data *= scalar


def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight/summ*n for weight in weights]
        target[name].data = torch.mean(torch.stack(
            [m*source[name].data for source, m in zip(sources, modify)]), dim=0).clone()


def afo_average(target, client_source, alpha):
    for name in target:
        target[name].data = (1 - alpha) * target[name].data.clone() + \
            alpha * client_source[name].data.clone()


def afo_gradient_avg(target, gradient, alpha):
    for name in target:
        target[name].data = target[name].data.clone() + alpha * \
            gradient[name].data.clone()


def compress(target, source, compress_fun):
    '''
    compress_fun : a function f : tensor (shape) -> tensor (shape)
    '''
    for name in target:
        target[name].data = compress_fun(source[name].data.clone())


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def to_cpu(target, source):
    for name in target:
        target[name] = source[name].detach().cpu().clone()


def to_gpu(target, source, device):
    for name in target:
        target[name] = source[name].cuda(device).clone()
    res = 0.0

def norm_2(W):
    norm = 0.0
    for name in W:
        norm += torch.sum(W[name] ** 2)
    return math.sqrt(norm)

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    #return (param_size, param_sum, buffer_size, buffer_sum, all_size)
    #print model name and size
    print(f'model name: {model.__class__.__name__}, model size: {all_size:.2f} MB')
    return all_size

def cos(weight1, weight2):
    ans = 0.0
    for name in weight1:
        ans += torch.sum(torch.mul(weight1[name], weight2[name]))
    norm1 = norm_2(weight1)
    norm2 = norm_2(weight2)
    ans /= (norm1 * norm2)
    return ans