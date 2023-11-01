import math
from torch import nn, optim
import learn2learn as l2l
import torch
import higher
from torch.nn import functional as F


class Task:
    """
    Handles the train and validation loss for a single task
    """
    def __init__(self, reg_param, meta_model, data, batch_size=None):
        device = next(meta_model.parameters()).device

        # stateless version of meta_model
        self.fmodel = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)

        self.n_params = len(list(meta_model.parameters()))
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None

    def bias_reg_f(self, bias, params):
        # l2 biased regularization
        return sum([((b - p) ** 2).sum() for b, p in zip(bias, params)])

    def train_loss_f(self, params, hparams):
        # biased regularized cross-entropy loss where the bias are the meta-parameters in hparams
        out = self.fmodel(self.train_input, params=params)
        return F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.bias_reg_f(hparams, params)

    def val_loss_f(self, params, hparams):
        # cross-entropy loss (uses only the task-specific weights in params
        out = self.fmodel(self.test_input, params=params)
        val_loss = F.cross_entropy(out, self.test_target)/self.batch_size
        self.val_loss = val_loss.item()  # avoid memory leaks

        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss


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

    meta_model = get_cnn(hidden_size=64, n_classes=ways, input_channel=3).to(device)
    outer_opt = torch.optim.SGD(lr=outer_lr, params=meta_model.parameters())

    return meta_model, [], outer_opt, []


def get_cnn(hidden_size, n_classes, input_channel):
    def conv_layer(ic, oc, ):
        return nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.BatchNorm2d(oc, momentum=1., affine=True,
                           track_running_stats=True # When this is true is called the "transductive setting"
                           )
        )

    net =  nn.Sequential(
        conv_layer(input_channel, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        conv_layer(hidden_size, hidden_size),
        nn.Flatten(),
        nn.Linear(hidden_size*25, n_classes)
    )

    initialize(net)
    return net


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