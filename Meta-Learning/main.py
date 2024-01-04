import os
import time
import random
import numpy as np
import matplotlib

matplotlib.use('pdf')
import torch
from torch import nn

import pickle
import argparse
from dataset.data_loader import load_data
from core.trainer import train_one_epoch_baseline, evaluate_one_epoch_baseline, \
                    train_one_epoch_OAGD, evaluate_one_epoch_OAGD
from models.models import create_model


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# def accuracy(predictions, targets):
#     predictions = predictions.argmax(dim=1).view(targets.shape)
#     return (predictions == targets).sum().float() / targets.size(0)


def main(args, ways=5, shots=5, cuda=1, seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda = bool(cuda)
    if cuda and torch.cuda.device_count():
        print('using CUDA')
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('using CPU')

    train_tasks, valid_tasks, test_tasks = load_data(args.dataset)

    # Create model
    meta_model, features, optimizer, all_parameters = \
        create_model(args.method, device, args.inner_lr, args.outer_lr, ways, args.dataset)

    loss = nn.CrossEntropyLoss(reduction='mean')

    training_accuracy = []
    training_error = []
    test_accuracy = []
    test_error = []
    running_time = []

    momentum_list = [args.gamma ** i for i in range(args.win_size)]
    # W = sum(momentum_list)
    # momentum_list = [momentum / W for momentum in momentum_list]

    start_time = time.time()

    for iteration in range(args.iters):
        start_per_iter = time.time()

        optimizer.zero_grad()

        if args.method != 'OAGD':
            meta_train_error, meta_train_accuracy = \
                train_one_epoch_baseline(args, iteration, meta_model, features, all_parameters,
                                         loss, optimizer, train_tasks, device)
            meta_test_error, meta_test_accuracy = \
                evaluate_one_epoch_baseline(args, meta_model, features, loss, test_tasks, device)
        else:
            meta_train_error, meta_train_accuracy = \
                train_one_epoch_OAGD(args, iteration, meta_model, optimizer, train_tasks, device, momentum_list)
            meta_test_error, meta_test_accuracy = \
                evaluate_one_epoch_OAGD(args, test_tasks, meta_model)

        training_accuracy.append(meta_train_accuracy)
        training_error.append(meta_train_error)
        test_accuracy.append(meta_test_accuracy)
        test_error.append(meta_test_error)

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error)
        print('Meta Train Accuracy', meta_train_accuracy)
        print('Meta Test Error', meta_test_error)
        print('Meta Test Accuracy', meta_test_accuracy)

        # Average the accumulated gradients and optimize
        # if args.method != 'OAGD':
        #     num_tasks = 32  # for all the baselines, the number of tasks is 32
        # else:
        #     num_tasks = args.win_size
        # for p in all_parameters:
        #     p.grad.data.mul_(1.0 / num_tasks)
        # optimizer.step()

        end_time = time.time()
        running_time.append(end_time - start_time)
        print('time per iteration', end_time - start_per_iter)
        print('total running time', end_time - start_time)

    return training_accuracy, training_error, test_accuracy, test_error, running_time


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='OAGD')
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--win_size', type=int, default=10)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--hg_mode', type=str, default='fixed_point', metavar='N',
                        help='hypergradient approximation: CG or fixed_point')
    parser.add_argument('--dataset', type=str, default='fc100', metavar='N',
                        help='fc100 or miniimagenet')
    parser.add_argument('--method', type=str, default='OAGD',
                        help='OAGD, ANIL, ITD-BiO or MAML')
    parser.add_argument('--outer_lr', type=float, default=0.001)
    parser.add_argument('--inner_lr', type=float, default=0.1)
    parser.add_argument('--inner_stp', type=int, default=20)
    parser.add_argument('--reg', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1,
                        help='gamma 1 means do the simple average over all the windows'
                             'gamma 0 means ony focus on the current window')

    args = parser.parse_args()

    print('method: ', args.method)
    print('dataset: ', args.dataset)
    print('total iterations: {}'.format(args.iters))
    if args.method == 'OAGD':
        print('window size: ', args.win_size)
        print('gamma for momentum: ', args.gamma)
        print('hg_mode: ', args.hg_mode)
    print('outer lr: ', args.outer_lr)
    print('inner lr: ', args.inner_lr)
    print('inner step: ', args.inner_stp)
    if args.method != 'MAML':
        print('regularization: ', args.reg)
    train_accuracy = []
    train_error = []
    test_accuracy = []
    test_error = []
    run_time = []

    seeds = [42, 52, 62, 72, 82]

    for idx, seed in enumerate(seeds):
        print('\n-------------------{}th seed: {}-------------------\n'.format(idx, seed))
        training_accuracy, training_error, testing_accuracy, testing_error, running_time = main(
                                                            args, ways=5, shots=5, cuda=1, seed=42)
        train_accuracy.append(training_accuracy)
        train_error.append(training_error)
        test_accuracy.append(testing_accuracy)
        test_error.append(testing_error)
        run_time.append(running_time)

    results = {'train_accuracy': train_accuracy,
              'train_error': train_error,
              'test_accuracy': test_accuracy,
              'test_error': test_error,
              'run_time': run_time
    }

    if args.method == 'OAGD':
        file_name = 'method_' + args.method + '_dataset_' + args.dataset \
                + '_iters_' + str(args.iters) + '_outer_lr_' + str(args.outer_lr) \
                + '_inner_lr_' + str(args.inner_lr) + '_inner_stp_' + str(args.inner_stp) \
                +  '_reg_' + str(args.reg) + '_win_size_' + str(args.win_size) \
                + '_mode_' + args.hg_mode + '_gamma_' + str(args.gamma) + '_Adam_memory_random_sampling'
    else:
        file_name = 'method_' + args.method + '_dataset_' + args.dataset \
                + '_iters_' + str(args.iters) + '_outer_lr_' + str(args.outer_lr) \
                + '_inner_lr_' + str(args.inner_lr) + '_inner_stp_' + str(args.inner_stp) \
                + '_reg_' + str(args.reg) + '_Adam'

    if not os.path.isdir("results"):
        os.makedirs("results")

    with open('results/' + file_name + '.pkl', 'wb') as f:
        pickle.dump(results, f)

    print('\ndone!')
