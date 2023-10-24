import math
from torch import nn, optim
import learn2learn as l2l
import torch


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def create_model(method, device, inner_lr, outer_lr, ways, dataset):
    if method == 'MAML':
        return create_model_MAML(device, inner_lr, outer_lr, ways, dataset)
    elif method == 'ITD-BiO':
        return create_model_ITD_BiO(device, inner_lr, outer_lr, ways, dataset)
    elif method == 'ANIL':
        return create_model_ANIL(device, inner_lr, outer_lr, ways, dataset)
    elif method == 'OAGD':
        return create_model_OAGD(device, inner_lr, outer_lr, ways, dataset)
    else:
        raise NotImplementedError(method, " not implemented!")

def create_model_MAML(device, inner_lr, outer_lr, ways, dataset):
    if dataset == 'fc100': dimension = 256
    elif dataset == 'miniimagenet': dimension = 1600
    else: raise NotImplementedError(dataset, " not implemented!")

    features = l2l.vision.models.ConvBase(hidden=64, channels=3, max_pool=True)
    meta_model = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, dimension)),
                                     torch.nn.Linear(dimension, ways))
    # model = torch.nn.Sequential(features, Lambda(lambda x: x.view(25, -1)))
    meta_model.to(device)
    meta_model = l2l.algorithms.MAML(meta_model, lr=inner_lr, first_order=False)
    optimizer = optim.SGD(meta_model.parameters(), outer_lr)

    return meta_model, features, optimizer, meta_model.parameters()


def create_model_ITD_BiO(device, inner_lr, outer_lr, ways, dataset):
    if dataset == 'fc100': dimension = 256
    elif dataset == 'miniimagenet': dimension = 1600
    else: raise NotImplementedError(dataset, " not implemented!")

    features = l2l.vision.models.ConvBase(hidden=64, channels=3, max_pool=True)
    features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, dimension)))
    features.to(device)
    head = torch.nn.Linear(dimension, ways)
    meta_model = l2l.algorithms.MAML(head, lr=inner_lr)
    meta_model.to(device)
    all_parameters = list(features.parameters())
    optimizer = torch.optim.SGD(all_parameters, lr=outer_lr)

    return meta_model, features, optimizer, all_parameters


def create_model_ANIL(device, inner_lr, outer_lr, ways, dataset):
    if dataset == 'fc100': dimension = 256
    elif dataset == 'miniimagenet': dimension = 1600
    else: raise NotImplementedError(dataset, " not implemented!")

    features = l2l.vision.models.ConvBase(hidden=64, channels=3, max_pool=True)
    features = torch.nn.Sequential(features, Lambda(lambda x: x.view(-1, dimension)))
    features.to(device)
    head = torch.nn.Linear(dimension, ways)
    meta_model = l2l.algorithms.MAML(head, lr=inner_lr)
    meta_model.to(device)

    # Setup optimization
    all_parameters = list(features.parameters()) + list(meta_model.parameters())
    optimizer = torch.optim.SGD([{'params': list(meta_model.parameters()), 'lr': outer_lr},
                                 {'params': list(features.parameters()), 'lr': outer_lr}])

    return meta_model, features, optimizer, all_parameters


def create_model_OAGD(device, inner_lr, outer_lr, ways, dataset):
    return


def get_cnn_omniglot(hidden_size, n_classes):
    def conv_layer(ic, oc, ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True # When this is true is called the "transductive setting"
                           )
        )

    net =  nn.Sequential(
        conv_layer(1, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size, n_classes)
    )

    initialize(net)
    return net


def get_cnn_miniimagenet(hidden_size, n_classes):
    def conv_layer(ic, oc):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=False  # When this is true is called the "transductive setting"
                           )
        )

    net = nn.Sequential(
        conv_layer(3, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size*5*5, n_classes)
    )

    initialize(net)
    return net


def initialize(net):
    # initialize weights properly
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            #m.weight.data.normal_(0, 0.01)
            #m.bias.data = torch.ones(m.bias.data.size())
            m.weight.data.zero_()
            m.bias.data.zero_()

    return net