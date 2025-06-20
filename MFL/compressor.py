from abc import ABC, abstractmethod
from functools import partial
import torch
import numpy as np

def sparsify(tensor, compression_rate):
    tensor = tensor.flatten()  # flatten the tensor
    k = max(int(tensor.numel() * compression_rate), 1)  # compute k, elements greater than k-th ele will be saved
    if tensor.device.type == "mps":
        values, indices = torch.topk(tensor.cpu().abs(), k, sorted=False)
        values = values.to("mps:0")
        indices = indices.to("mps:0")
    else:
        values, indices = torch.topk(tensor.abs(), k, sorted=False)  # get topk elements' values and indices
    values = torch.gather(tensor, 0, indices)  # get all topk values
    return values, indices


def desparsify(values_and_indices, num):
    values, indices = values_and_indices
    tensor_decompress = torch.zeros(num, dtype=values.dtype, layout=values.layout,
                                    device=values.device)  # get a flatten tensor
    tensor_decompress.scatter_(0, indices, values)
    return tensor_decompress

class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)


class NoneCompressor(Compressor):
    def __init__(self):
        super().__init__()

    def compress(self, tensor, name):
        return tensor, name

    def decompress(self, tensor, name):
        return tensor


class TopkCompressor(Compressor):
    def __init__(self, compression_rate):
        super().__init__()
        self.compression_rate = compression_rate

    def compress(self, tensor, name):
        values_and_indices = sparsify(tensor, self.compression_rate)  # get compressed values and indices
        num_and_size = tensor.numel(), tensor.size()
        return values_and_indices, num_and_size

    def decompress(self, values_and_indices, num_and_size):
        num, size = num_and_size
        tensor_decompress = desparsify(values_and_indices, num)
        tensor_decompress = tensor_decompress.view(size)
        return tensor_decompress


def approx_v(T, p, frac):
    if frac < 1.0:
        n_elements = T.numel()
        n_sample = min(int(max(np.ceil(n_elements * frac), np.ceil(100 / p))), n_elements)
        n_top = int(np.ceil(n_sample * p))
        if n_elements == n_sample:
            i = 0
        else:
            i = np.random.randint(n_elements - n_sample)

        topk, _ = torch.topk(T.flatten()[i:i + n_sample], n_top)
        if topk[-1] == 0.0 or topk[-1] == T.max():
            return approx_v(T, p, 1.0)
    else:
        n_elements = T.numel()
        n_top = int(np.ceil(n_elements * p))
        if T.device.type == "mps":
            topk, _ = torch.topk(T.flatten().cpu().abs(), n_top)
            topk = topk.to("mps:0")
            _ = _.to("mps:0")
        else:
            topk, _ = torch.topk(T.flatten(), n_top)
    # print("n_top:")
    # print(n_top)

    return topk[-1], topk


def random_k(T, hp):
    hp_ = {"cr": 0.001, 'approx': 1.0}
    hp_.update(hp)

    if hp_["cr"] >= 1.0:
        return T


def none(T, hp, device):
    '''
    Identity
    '''
    return T


def topk(T, hp, device):
    '''
    "Deep Gradient Compression: Reducing the communication Bandwidth for Distributed Training, Lin et al."
    '''
    hp_ = {"cr": 0.001, 'approx': 1.0}
    hp_.update(hp)

    if hp_["cr"] >= 1.0:
        return T
    print('topk cr=',hp_["cr"])
    T_abs = torch.abs(T)

    v, _ = approx_v(T_abs, hp_["cr"], hp_["approx"])

    out = torch.where(T_abs >= v, T, torch.Tensor([0.0]).to(device))

    return out


# 硬阈值稀疏化

def ht(T, hp):
    hp_ = {"cr": 0.001, 'approx': 1.0}
    hp_.update(hp)
    if hp_["cr"] >= 1.0:
        return T
    T_abs = torch.abs(T)
    v = hp_["cr"]
    out = torch.where(T_abs >= v, T, torch.Tensor([0.0]).to(device))

    return out


def stc(T, hp):
    '''
    "Sparse Binary Compression: Towards Distributed Deep Learning with minimal Communication, Sattler et al."
    '''
    hp_ = {"cr": 0.001, 'approx': 1.0}
    hp_.update(hp)

    T_abs = torch.abs(T)

    v, topk = approx_v(T_abs, hp_["p"], hp_["approx"])
    mean = torch.mean(topk)

    out_ = torch.where(T >= v, mean, torch.Tensor([0.0]).to(device))
    out = torch.where(T <= -v, -mean, out_)

    return out


def randomk(T, hp):
    hp_ = {"cr": 0.001, 'approx': 1.0}
    hp_.update(hp)

    shape = T.size()
    tensor = T.flatten()
    numel = torch.numel(tensor)

    k = max(1, int(np.ceil(numel * hp_["cr"])))
    indices = torch.randperm(numel, device=tensor.device)[:k]
    values = tensor[indices]

    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=tensor.device)
    tensor_decompressed.scatter_(0, indices, values)
    out = tensor_decompressed.view(shape)

    return out


def signsgd(T, hp):
    """
    signSGD: Compressed Optimisation for non-convex Problems, Bernstein et al.

    """
    return T.sign()


def qsgd(T, hp):
    hp_ = {"cr": 0.125, 'approx': 1.0}
    hp_.update(hp)

    s = 2 ** (32 * hp_["cr"])  # the number of distributed values

    shape = T.size()  # get shape
    tensor = T.flatten()  # flatten the tensor

    norm = tensor.norm()  # get 2-norm
    norm = norm.flatten()
    abs_gradient = tensor.abs()  # get abs value to compute

    level_float = s / norm * abs_gradient  # |v_i| * s / ||v||
    previous_level = level_float.floor()  # l/s
    prob = torch.empty_like(tensor).uniform_()  # the prob of next level
    is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
    new_level = (previous_level + is_next_level)

    sign = tensor.sign()
    tensor_compressed = new_level * sign

    decode_output = tensor_compressed
    tensor_decompressed = norm / s * decode_output
    tensor_decompressed = tensor_decompressed.view(shape)

    return tensor_decompressed


def compression_function(compression_config, device):
    '''
    Returns a function that maps a tensor to a tensor of the same shape
    '''
    name = compression_config["method"]
    hp = compression_config["params"]
    return partial(globals()[name], hp=hp,device=device)


def none(T, hp, device):
    return T


###############################################################################################
# COUNTING BITS
###############################################################################################


def get_bits(T, compression_method, approx=False):
    """
    Returns the number of bits that are required to communicate the Tensor T, which was compressed with compresion_method
    """

    B_val = {"none": 32, "dgc": 32, "stc": 1, "signsgd": 1}[compression_method]

    # dense methods
    if compression_method in ["none", "signsgd"]:
        k = T.numel()
        B_pos = 0

    # sparse methods non-optimal encoding
    elif compression_method in ["dgc"]:
        k = torch.sum(T != 0.0).item()
        B_pos = 16

    # sparse methods golomb encoding
    elif compression_method in ["stc"]:
        k = torch.sum(T != 0.0).item()
        N = T.numel()

        q = (k + 1) / (N + 1)
        golden = (np.sqrt(5) + 1) / 2

        if q == 1:
            return B_val * T.numel()
        if q == 0:
            return 0

        b_star = 1 + np.floor(np.log2(np.log(golden - 1) / np.log(1 - q)))

        if approx:
            B_pos = b_star + 1 / (1 - (1 - q) ** (2 ** b_star)) + 1
        else:
            idc = torch.nonzero(T.view(-1))
            distances = idc[:] - torch.cat([torch.Tensor([[-1]]).long().to("cuda"), idc[:-1]])
            B_pos = torch.mean(torch.ceil(distances.float() / 2 ** b_star)).item() + (b_star + 1)

    elif compression_method in ["randomk"]:
        k = torch.sum(T != 0.0).item()
        B_pos = 16

    b_total = (B_pos + B_val) * k

    return b_total


def get_update_size(dW, compression_method):
    """
    Returns the number of bits that are required to communicate the entire network dW, which was compressed with compresion_method
    """

    update_size = sum([get_bits(T, compression_method[0]) for T in dW.values()])

    return update_size
