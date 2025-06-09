from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torch
# import torchtext
# import torchaudio
import numpy as np
import os,sys

import scipy.io.wavfile as wav
import librosa
from python_speech_features import mfcc

class CustomerDataset(Dataset):
    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms
        self.data = self.inputs
        self.targets = self.labels

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]

class MNIST:
    def __init__(self):
        # get dataset
        self.train_datasets = datasets.MNIST(root='data/', train=True, download=True)
        self.test_datasets = datasets.MNIST(root='data/', train=False, download=True)

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets


class FashionMNIST:
    def __init__(self):
        # 获取数据集
        train_datasets = datasets.FashionMNIST(root='data/', train=True,
                                               transform=transforms.ToTensor(), download=True)
        test_datasets = datasets.FashionMNIST(root='data/', train=False,
                                              transform=transforms.ToTensor(), download=True)
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets


class CIFAR10:
    def __init__(self):
        # 获取数据集
        train_datasets = datasets.CIFAR10(root='data/', train=True,
                                          transform=transforms.ToTensor(), download=True)
        test_datasets = datasets.CIFAR10(root='data/', train=False,
                                         transform=transforms.ToTensor(), download=True)
        train_datasets.data = train_datasets.data.transpose((0, 3, 1, 2))
        test_datasets.data = test_datasets.data.transpose((0, 3, 1, 2))
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets

class CIFAR100:
    def __init__(self):
        # 获取数据集
        train_datasets = datasets.CIFAR100(root='data/', train=True,
                                          transform=transforms.ToTensor(), download=True)
        test_datasets = datasets.CIFAR100(root='data/', train=False,
                                         transform=transforms.ToTensor(), download=True)
        train_datasets.data = train_datasets.data.transpose((0, 3, 1, 2))
        test_datasets.data = test_datasets.data.transpose((0, 3, 1, 2))
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets

class IMDB:
    def __init__(self):
        TEXT = torch.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
        LABEL =torch.data.LabelField(dtype=torch.float)
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
        LABEL.build_vocab(train_data)
        self.train_datasets = train_data
        self.test_datasets = test_data

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets

# Speech Commands
class SC:
    def __init__(self):
        x_train, y_train, x_test, y_test = get_speechcommands()
        # torchaudio.datasets.SPEECHCOMMANDS(
        #     root="data",                         # 你保存数据的路径
        #     url = 'speech_commands_v0.02',         # 下载数据版本URL
        #     folder_in_archive = 'SpeechCommands',  
        #     download = True                        # 这个记得选True 
        # )
        self.train_datasets = CustomerDataset(x_train, y_train)
        self.test_datasets = CustomerDataset(x_test, y_test)

    def get_test_dataset(self):
        return self.test_datasets

    def get_train_dataset(self):
        return self.train_datasets


def get_default_data_transforms(name, train=True, verbose=False):
    name = name.lower()
    transforms_train = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        # (0.24703223, 0.24348513, 0.26158784)
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]),
        'geo': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]),
        'kws': None,
        'sc': None
    }
    transforms_eval = {
        'mnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.06078,), (0.1957,))
        ]),
        'fmnist': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'cifar10': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),  #
        'cifar100': transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))]),
        'geo': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]),
        'kws': None,
        'sc': None
    }

    if verbose:
        print("\nData preprocessing: ")
        for transformation in transforms_train[name].transforms:
            print(' -', transformation)
        print()

    return (transforms_train[name], transforms_eval[name])


def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))


def print_split(clients_split):
    n_labels = 10
    print("Data split:")
    for i, client in enumerate(clients_split):
        split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
        print(" - Client {}: {}".format(i, split))
    print()


# shuffle two list together, so that shuffling operation won't destroy the one-to-one map between data and label
def together_shuffle(x_data, y_data):
    assert len(x_data) == len(y_data)
    randomize = np.arange(len(x_data))
    np.random.shuffle(randomize)
    x_data = np.array(x_data)[randomize]
    y_data = np.array(y_data)[randomize]
    return x_data, y_data


# input: data distribution config, train set and test set
# output: each client train set and test set
def split_data(data_distribution_config, n_clients, train_set):
    if data_distribution_config["iid"] == True:  # iid
        split = iid_split(n_clients, train_set)
    elif data_distribution_config["customize"] == False:  # Non-IID and auto generate(non customize)
        split = niid_dirichlet_split(n_clients, 1.0, train_set)
    elif data_distribution_config["customize"] == True:  # Non-IID and customize
        split = niid_customize_split(n_clients, train_set, data_distribution_config["cus_distribution"])
    return split


def get_global_data(test_set):
    x_test = test_set.data  # get data of test samples
    y_test = test_set.targets
    test_set = (x_test, y_test)
    return test_set


# split data uniformly
def iid_split(n_clients, train_set):
    num_train = len(train_set)  # get number of training samples
    x_train = train_set.data  # get data of training samples
    y_train = train_set.targets  # get label of training samples

    clients_sample_num = int(num_train / n_clients)  # the number of client samples
    x_train, y_train = together_shuffle(x_train, y_train)  # shuffle

    split = []  # data split, each element is a tuple (x_data, y_data)

    for i in range(n_clients):
        client_x_data = x_train[clients_sample_num * i: clients_sample_num * (i + 1)]  # get a slice of data
        client_y_data = y_train[clients_sample_num * i: clients_sample_num * (i + 1)]
        # print(client_y_data.shape)
        split += [(client_x_data, client_y_data)]

    print_split(split)
    return split


def niid_dirichlet_split(n_clients, alpha, train_set):
    '''
    Dirichlet distribution with parameter alpha, dividing the data index into n_clients subsets
    '''
    # total classes num
    x_train = train_set.data  # get data of training samples
    y_train = train_set.targets  # get label of training samples
    try:
        n_classes = y_train.max() + 1
    except:
        n_classes = np.max(y_train) + 1

    # shuffle
    x_train, y_train = together_shuffle(x_train, y_train)

    # [alpha] * n_clients is as follows：
    # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # Record the ratio of each client to each category
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    # Record the sample subscript corresponding to each category
    class_idcs = [np.argwhere(y_train == y).flatten()
                  for y in range(n_classes)]

    # Define an empty list as the final return value
    client_idcs = [[] for _ in range(n_clients)]
    # Record the indexes of N clients corresponding to the sample collection
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split, According to the proportion, the samples of category k are divided into N subsets
        # for i, idcs In order to traverse the index of the sample set corresponding to the i-th client
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    split = []

    for i in range(n_clients):
        client_x_data = x_train[client_idcs[i]]  # get a slice of data
        client_y_data = y_train[client_idcs[i]]
        client_x_data, client_y_data = together_shuffle(client_x_data, client_y_data)
        split += [(client_x_data, client_y_data)]

    print_split(split)
    return split


def niid_class_split(n_clients, train_set, distribution):
    # total classes num
    x_train = train_set.data  # get data of training samples
    y_train = train_set.targets  # get label of training samples
    n_classes = y_train.max() + 1

    # generate ratio matrix
    ratio_matrix = []
    for i in range(n_clients):
        shift = i  # the first type of data
        class_num = distribution[i]  # the number of class of i-th client
        class_list = [0 for j in range(
            n_classes)]  # the class list of the i-the client, like:[1,1,0,0,0] i.e. i-th client has 0,1 two class
        for j in range(class_num):
            class_list[(shift + j) % n_classes] = 1
        ratio_matrix.append(class_list)

    # get split
    split = niid_matrix_split(train_set, n_clients, ratio_matrix)
    return split


def ratio_matrix_to_num_matrix(labels, ratio_matrix):
    """
    :param labels: Labels for all data in this dataset
    :param ratio_matrix: the scale matrix of the data distribution to obtain
    :return: The actual data matrix num_matrix of the data distribution to obtain
    """
    # Get the labels for each label
    mask = np.unique(labels)
    mask = sorted(mask)

    # Get the number of data for each label
    labels_num = []
    labels = labels.cpu().numpy()
    for v in mask:
        labels_num.append(np.sum(labels == v))

    # Get the total number of proportions, and the data volume of a proportion
    ratio_sum = np.sum(ratio_matrix, axis=0)  # Get the total number of proportions of each labeled data
    one_ratio_num = labels_num // ratio_sum  # the data volume of a proportion

    # get data number matrix
    num_matrix = []
    for i in range(len(ratio_matrix)):  # for each client
        client_data_num = []  # Amount of data per client, ist
        for j in range(len(ratio_sum)):  # for each class
            data_num = one_ratio_num[j] * ratio_matrix[i][
                j]  # Calculate the amount of data of the jth class of the i-th client
            client_data_num.append(data_num)
        num_matrix.append(client_data_num)

    return num_matrix


def niid_matrix_split(train_set, n_clients, ratio_matrix, shuffle=True):
    data = train_set.data
    labels = train_set.targets
    num_matrix = ratio_matrix_to_num_matrix(labels, ratio_matrix)
    n_labels = len(num_matrix[0])

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]  # data_idcs[i] Represents all index numbers of the data of the i-th category
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)  # shuffle

    clients_split = []
    for i in range(n_clients):  # for each client
        client_idcs = []  # Store all index numbers of client i's data
        client_data_num = num_matrix[
            i]  # Get the number of data of each type of client i, client_data_num[c] indicates the number of data of type c of client i
        for c in range(n_labels):  # for each class
            if client_data_num[c] == 0:  # If the class requires 0 data, continue looping
                continue
            take = int(client_data_num[c])
            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

        client_x_data = data[client_idcs]  # get a slice of data
        client_y_data = labels[client_idcs]
        client_x_data, client_y_data = together_shuffle(client_x_data, client_y_data)
        clients_split += [(client_x_data, client_y_data)]

    print_split(clients_split)
    return clients_split


def niid_customize_split(n_clients, train_set, distribution):
    if type(distribution[0]) is int:
        # if the distribution is the number of class of each client
        # like: [2,2,2,2,2,2,2,2,2,2]
        # i.e. each client has two class
        split = niid_class_split(n_clients, train_set, distribution)
    else:
        # if the distribution is the ratio matrix
        split = niid_matrix_split(train_set, n_clients, distribution)
    return split


def get_dataset(dataset_name):
    if dataset_name == 'MNIST':
        return MNIST()
    elif dataset_name == 'FMNIST':
        return FashionMNIST()
    elif dataset_name == 'CIFAR10':
        return CIFAR10()
    elif dataset_name == 'CIFAR100':
        return CIFAR100()
    elif dataset_name == 'IMDB':
        return IMDB()
    elif dataset_name == 'SC':
        return SC()

def get_speechcommands():
    DATA_PATH = 'data/'
    data_root = os.path.join(DATA_PATH,'SpeechCommands')                # SPEECHCOMMANDS数据集的根目录
    npz_path = os.path.join(data_root,'speechcommands.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
        # cnn 用：
        # x_train = np.resize(x_train,(x_train.shape[0],1,32,32))
        # x_test = np.resize(x_test,(x_test.shape[0],1,32,32))
        return x_train, y_train, x_test, y_test

    # rename_speechcommands()

    labels_and_num = []         # 存储全部标签和其数量
    for label in os.listdir(data_root):
        if label.__contains__('_'):                 # 如果是_background_noise_则跳过
            continue
        label_path = os.path.join(data_root,label)
        if not os.path.isdir(label_path):
            continue
        labels_and_num.append((label,len(os.listdir(label_path))))
    labels_and_num = sorted(labels_and_num,key=lambda x:x[1],reverse=True)
    labels_and_num = labels_and_num[:10]

    labels = [label_and_num[0] for label_and_num in labels_and_num]         # 获取训练集中10种数据的标签
    label_to_id = {label:idx for idx,label in enumerate(labels)}            # 获取标签与id标签的映射关系

    train_size = 3000               # 一个标签下的训练集大小
    test_size = 300                 # 一个标签下的测试集大小

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for label in labels:
        x_train_one_label = []              # 一个标签下的训练数据与测试数据
        y_train_one_label = []
        x_test_one_label = []
        y_test_one_label = []
        if label.__contains__('_'):                 # 如果是_background_noise_则跳过
            continue
        label_path = os.path.join(data_root,label)
        if not os.path.isdir(label_path):
            continue
        label_wavs_path = os.listdir(label_path)        # 获取该标签的所有数据的路径
        label_wavs_path = [os.path.join(label_path,wav_path) for wav_path in label_wavs_path]
        for wav_path in label_wavs_path:              # 对于每个文件
            label_id = label_to_id[label]  # 获取数字标签
            x_data = wav_read_mfcc(wav_path)       # 获取数据文件
            if x_data.shape == (99,13):                     # 固定大小
                x_data = x_data.flatten()[:(32*32)]
                x_data = np.resize(x_data,(1,32,32))                                       # cnn用
                if len(x_train_one_label) < train_size:           # 如果训练集没满
                    x_train_one_label.append(x_data)                # 则放入训练数据与测试数据
                    y_train_one_label.append(label_id)
                elif len(x_test_one_label) < test_size:           # 如果测试集没满
                    x_test_one_label.append(x_data)
                    y_test_one_label.append(label_id)
                else:                       # 如果测试集和训练集都满了
                    break
        x_train.extend(x_train_one_label)
        y_train.extend(y_train_one_label)
        x_test.extend(x_test_one_label)
        y_test.extend(y_test_one_label)

    # shuffle
    x_train,y_train = together_shuffle(x_train,y_train)
    x_test,y_test = together_shuffle(x_test,y_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    np.savez(npz_path,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
    return x_train,y_train,x_test,y_test

def together_shuffle(x_data,y_data):
    assert len(x_data) == len(y_data)
    randomize = np.arange(len(x_data))
    np.random.shuffle(randomize)
    x_data = np.array(x_data)[randomize]
    y_data = np.array(y_data)[randomize]
    return x_data,y_data

def wav_read_mfcc(file_name):
    try:
        fs, audio = wav.read(file_name)
        # return fs: 采样频率，data：数据
        processed_audio = mfcc(audio,samplerate=fs,nfft=2048)
    except ValueError:
        audio,fs = librosa.load(file_name)
        # return y: 数据 sr：采样频率
        processed_audio = mfcc(audio,samplerate=fs,nfft=2048)
    return processed_audio