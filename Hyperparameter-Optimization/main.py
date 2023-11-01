import argparse
import os
import yaml
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from utils.metrics import print_num_params
from core.trainer import train_epoch, eval_epoch, train_epoch_online, train_epoch_OGD, train_epoch_SGD
from core.utils import loss_adjust_cross_entropy, cross_entropy,loss_adjust_cross_entropy_cdt
from core.utils import get_init_dy, get_init_ly, get_train_w, get_val_w
from dataset.cifar10 import load_cifar10, load_mnist
from dataset.tadpole import load_tadpole
from dataset.adult import load_adult
from models.MLP import build_model_mlp, build_model_cnn, get_cnn_small, get_cnn_large
import torch.optim as optim
import torch.nn as nn
import time
import random
import pickle as pkl
assert torch.backends.cudnn.enabled


def loader_to_list(dataloader):
    batches = []
    for batch in dataloader:
        batches.append(batch)
    return batches


def main(seed, args):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True

    device = args["device"]
    dataset = args["dataset"]

    if dataset == 'tadpole':
        num_classes = 2  # MCI vs AD
        train_loader, val_loader, test_loader, eval_train_loader, \
            eval_val_loader, num_train_samples, num_val_samples, img_size = load_tadpole(args['batch_size'])
        model = build_model_mlp(img_size, num_classes, device)
    elif dataset == 'adult':
        num_classes = 2
        train_loader, val_loader, test_loader, eval_train_loader, \
            eval_val_loader, num_train_samples, num_val_samples, img_size = load_adult(args['batch_size'])
        model = build_model_mlp(img_size, num_classes, device)
    elif dataset == 'mnist':
        num_classes = 10
        # model = ResNet32(num_classes=num_classes)
        # model = build_model_cnn(num_classes, input_channel=1)
        model = get_cnn_small(hidden_size=64, n_classes=num_classes)
        train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples = load_mnist(
            train_size=args["train_size"], val_size=args["val_size"],
            balance_val=args["balance_val"], batch_size=args["low_batch_size"],
            train_rho=args["train_rho"],
            image_size=28, path=args["datapath"])
    elif dataset == 'cifar10': # the train size is (9931, 32, 32, 3), val (2487, 32, 32, 3), test (10000, 32, 32, 3)
        num_classes = 10
        # model = ResNet32(num_classes=num_classes)
        model = build_model_cnn(num_classes)
        train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples = load_cifar10(
            train_size=args["train_size"], val_size=args["val_size"],
            balance_val=args["balance_val"], batch_size=args["low_batch_size"],
            train_rho=args["train_rho"],
            image_size=32, path=args["datapath"])

    args["num_classes"] = num_classes

    print_num_params(model)

    if args["checkpoint"] != 0:
        model = torch.load(f'{args["save_path"]}/epoch_{args.checkpoint}.pth')
        # model.load_state_dict(torch.load(f'{args["save_path"]}/epoch_{args.checkpoint}.pth'))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    dy = get_init_dy(args, num_train_samples)
    ly = get_init_ly(args, num_train_samples)
    w_train = get_train_w(args, num_train_samples)
    w_val = get_val_w(args, num_val_samples)

    print(f"w_train: {w_train}\nw_val: {w_val}")
    print('ly', ly, '\n dy', dy)

    print("train data size", len(train_loader.dataset), len(train_loader))

    # learning rate (step size) schedular
    up_start_epoch = args["up_configs"]["start_epoch"]
    if "low_lr_multiplier" in args:
        def warm_up_with_multistep_lr_low(epoch):
            return (epoch + 1) / args["low_lr_warmup"] \
                if epoch < args["low_lr_warmup"] \
                else args["low_lr_multiplier"][len([m for m in args["low_lr_schedule"] if m <= epoch])]
    else:
        def warm_up_with_multistep_lr_low(epoch):
            return (epoch + 1) / args["low_lr_warmup"] \
                if epoch < args["low_lr_warmup"] \
                else 0.1 ** len([m for m in args["low_lr_schedule"] if m <= epoch])
    def warm_up_with_multistep_lr_up(epoch):
        return (epoch - up_start_epoch + 1) / args["up_lr_warmup"] \
            if epoch - up_start_epoch < args["up_lr_warmup"] \
            else 0.1 ** len([m for m in args["up_lr_schedule"] if m <= epoch])

    train_optimizer = optim.SGD(params=model.parameters(),
                                lr=args["inner_lr"], momentum=0.9, weight_decay=1e-4)
    val_optimizer = optim.SGD(params=[{'params': dy}, {'params': ly}],
                              lr=args["outer_lr"], momentum=0.9, weight_decay=1e-4)
    train_lr_scheduler = optim.lr_scheduler.LambdaLR(
        train_optimizer, lr_lambda=warm_up_with_multistep_lr_low)
    val_lr_scheduler = optim.lr_scheduler.LambdaLR(
        val_optimizer, lr_lambda=warm_up_with_multistep_lr_up)

    if args["save_path"] is None:
        args["save_path"] = f'./results/{int(time.time())}'
    if not os.path.exists(args["save_path"]):
        os.makedirs(args["save_path"])

    assert (args["checkpoint"] == 0)

    # torch.save(model, f'{args["save_path"]}/init_model.pth')

    train_loader = loader_to_list(train_loader)
    val_loader = loader_to_list(val_loader)

    train_acc_list = []
    train_bacc_list = []
    train_loss_list = []
    test_acc_list = []
    test_bacc_list = []
    test_loss_list = []
    running_time_list = []
    start_time = time.time()
    for i in range(args["checkpoint"], args["epoch"] + 1):

        start_per_iter = time.time()

        # if i % args["checkpoint_interval"] == 0:
        #     torch.save(model, f'{args["save_path"]}/epoch_{i}.pth')

        if i % args["eval_interval"] == 0:
            if args["up_configs"]["dy_init"] == "CDT":
                print("CDT")
                text, train_loss, train_err, balanced_train_err, classwise_train_err, train_acc, train_bacc = eval_epoch(
                    eval_train_loader, model,
                    loss_adjust_cross_entropy_cdt, i, ' train_dataset', args,
                    params=[dy, ly, w_train])

            else:
                text, train_loss, train_err, balanced_train_err, classwise_train_err, train_acc, train_bacc = eval_epoch(
                    eval_train_loader, model,
                    loss_adjust_cross_entropy, i, ' train_dataset', args,
                    params=[dy, ly, w_train])

            text, val_loss, val_err, balanced_val_err, classwise_val_err, val_acc, val_bacc = eval_epoch(
                eval_val_loader, model,
                cross_entropy, i, ' val_dataset', args, params=[dy, ly, w_val])


            text, test_loss, test_err, balanced_test_err, classwise_test_err, test_acc, test_bacc = eval_epoch(
                test_loader, model,
                cross_entropy, i, ' test_dataset', args, params=[dy, ly])


            train_acc_list.append(train_acc)
            train_bacc_list.append(train_bacc)
            train_loss_list.append(train_loss)
            test_acc_list.append(test_acc)
            test_bacc_list.append(test_bacc)
            test_loss_list.append(test_loss)


        print('ly', ly, '\n dy', dy, '\n')

        if args['method'] == 'autobalance':
            print('train AutoBalance...')
            train_epoch(i, model, args,
                        low_loader=train_loader, low_criterion=loss_adjust_cross_entropy,
                        low_optimizer=train_optimizer, low_params=[dy, ly, w_train],
                        up_loader=val_loader, up_optimizer=val_optimizer,
                        up_criterion=cross_entropy, up_params=[dy, ly, w_val])
        elif args['method'] == 'OAGD':
            print('train OAGD...')
            train_epoch_online(i, model, args,
                               low_loader=train_loader, low_criterion=loss_adjust_cross_entropy,
                               low_optimizer=train_optimizer, low_params=[dy, ly, w_train],
                               up_loader=val_loader, up_optimizer=val_optimizer,
                               up_criterion=cross_entropy, up_params=[dy, ly, w_val])
        elif args['method'] == 'OGD':
            print('train OGD...')
            train_epoch_OGD(i, model, args,
                            low_loader=train_loader, low_criterion=loss_adjust_cross_entropy,
                            low_optimizer=train_optimizer, low_params=[dy, ly, w_train])
        elif args['method'] == 'SGD':
            print('train SGD...')
            train_epoch_SGD(i, model, args,
                            low_loader=train_loader, low_criterion=loss_adjust_cross_entropy,
                            low_optimizer=train_optimizer, low_params=[dy, ly, w_train])


        train_lr_scheduler.step()
        val_lr_scheduler.step()

        end_time = time.time()
        running_time_list.append(end_time - start_time)

        print('time per iteration', end_time - start_per_iter)
        print('total running time', end_time - start_time)


    # torch.save(model, f'{args["save_path"]}/final_model.pth')

    return train_acc_list, train_bacc_list, train_loss_list, \
           test_acc_list, test_bacc_list, test_loss_list, \
           running_time_list

if __name__ ==  '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tadpole')  # adult, mnist, tadpole, cifar10
    parser.add_argument('--method', type=str, default='OAGD')  # autobalance, OAGD, SGD, OGD
    parser.add_argument('--win_size', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)  # 128
    parser.add_argument('--gamma', type=float, default=0.9)  # gamma is for calculating the momentum for smoothing
    parser.add_argument('--inner_lr', type=float, default=0.1)
    parser.add_argument('--outer_lr', type=float, default=0.001)
    args = parser.parse_args()
    method = args.method
    win_size = args.win_size
    dataset = args.dataset
    batch_size = args.batch_size
    gamma = args.gamma
    inner_lr = args.inner_lr
    outer_lr = args.outer_lr
    config = f'configs/{args.dataset}.yaml'

    print('dataset: ', dataset)
    print('method: ', method)
    print('window size: ', win_size)
    print('batch size: ', batch_size)
    print('gamma: ', gamma)
    print('inner_lr: ', inner_lr)
    print('outer_lr: ', outer_lr)

    with open(config, mode='r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args['method'] = method
    args['win_size'] = win_size
    args['batch_size'] = batch_size
    args['gamma'] = gamma
    args['inner_lr'] = inner_lr
    args['outer_lr'] = outer_lr
    print('Device: ', args['device'])

    seeds = [42]  # 42, 52, 62, 72, 82

    results = {}
    train_acc_list, train_bacc_list, train_loss_list = [], [], []
    test_acc_list, test_bacc_list, test_loss_list = [], [], []
    run_time_list = []
    for idx, seed in enumerate(seeds):
        train_acc, train_bacc, train_loss, test_acc, test_bacc, test_loss, run_time = main(seed, args)
        train_acc_list.append(train_acc)
        train_bacc_list.append(train_bacc)
        train_loss_list.append(train_loss)
        test_acc_list.append(test_acc)
        test_bacc_list.append(test_bacc)
        test_loss_list.append(test_loss)
        run_time_list.append(run_time)

    results['train_acc'] = train_acc_list
    results['train_bacc'] = train_bacc_list
    results['train_loss'] = train_loss_list
    results['test_acc'] = test_acc_list
    results['test_bacc'] = test_bacc_list
    results['test_loss'] = test_loss_list
    results['run_time'] = run_time_list

    low_lr = args['inner_lr']
    up_lr = args['outer_lr']

    if not os.path.isdir("results"):
        os.makedirs("results")

    if method == 'OAGD':
        with open(f'results/{dataset}_{method}_win_size_{win_size}'
                  f'_batch_size_{batch_size}_low_lr_{low_lr}_up_lr_{up_lr}'
                  f'_gamma_{gamma}_late_lr_decay.pkl', 'wb') as f:
            pkl.dump(results, f)
    else:
        with open(f'results/{dataset}_{method}_batch_size_{batch_size}'
                  f'_low_lr_{low_lr}_up_lr_{up_lr}_late_lr_decay.pkl', 'wb') as f:
            pkl.dump(results, f)

    print('\ndone!')



