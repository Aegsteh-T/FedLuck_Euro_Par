from MFL.model.CNN import CNN1, CNN3, VGG11s, VGG11, VGG11s_3, ResNet18, AlexNet,ResNet9_3_10
from MFL.model.LinearModel import logistic
from MFL.model.LSTM import LSTM

def get_model(model_name):
    if model_name == 'CNN1':
        return CNN1()
    elif model_name == 'CNN3':
        return CNN3()
    elif model_name == 'VGG11s':
        return VGG11s()
    elif model_name == 'VGG11':
        return VGG11()
    elif model_name == 'VGG11s_3':
        return VGG11s_3()
    elif model_name == 'ResNet18':
        return ResNet18()
    elif model_name =='logistic':
        return logistic()
    elif model_name == 'LSTM':
        return LSTM()
    elif model_name == 'AlexNet':
        return AlexNet()
    elif model_name == 'ResNet9':
        return ResNet9_3_10()