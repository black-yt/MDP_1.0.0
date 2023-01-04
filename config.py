# coding: UTF-8
import warnings
import torch

class DefaultConfig(object):



    model = 'PyGCN'
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = 'gpu'
    else:
        device = 'cpu'
    load_model_path = None

    network = 'cora'
    batch_size = 8
    num_workers = 3
    max_epoch = 10
    lr = 0.005
    lr_decay = 1
    weight_decay = 1e-5
    train_rate = 0.05
    val_rate = 0.05
    droput = 0.5

    privacy_budget = 30
    NC = 2


    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()