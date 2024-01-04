import os
import random

import numpy as np
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision as tv

import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from sklearn.model_selection import train_test_split
from copy import deepcopy

import pickle
import argparse
# from torchmeta.datasets.helpers import omniglot, miniimagenet
# from torchmeta.utils.data import BatchMetaDataLoader

def load_data(dataset, ways=5, shots=5):
    if not os.path.isdir("data"):
        os.makedirs("data")

    if dataset == 'fc100':
        return load_fc100(ways, shots)
    elif dataset == 'omniglot':
        return load_omniglot(ways, shots)
    elif dataset == 'miniimagenet':
        return load_miniimagenet(ways, shots)
    elif dataset == 'cifar100':
        return load_cifar100(ways, shots)
    else:
        raise NotImplementedError(dataset, " not implemented!")


def load_cifar100(ways, shots):
    current_folder = os.getcwd()

    train_dataset = l2l.vision.datasets.CIFARFS(root=f'{current_folder}/data', mode='train', download=True)
    valid_dataset = l2l.vision.datasets.CIFARFS(root=f'{current_folder}/data', mode='validation', download=True)
    test_dataset = l2l.vision.datasets.CIFARFS(root=f'{current_folder}/data', mode='test', download=True)

    train_tasks = data2task(train_dataset, ways, shots, num_tasks=20000)
    valid_tasks = data2task(valid_dataset, ways, shots, num_tasks=600)
    test_tasks = data2task(test_dataset, ways, shots, num_tasks=600)

    return train_tasks, valid_tasks, test_tasks

def load_omniglot(ways, shots):
    current_folder = os.getcwd()

    dataset = l2l.vision.datasets.FullOmniglot(root=f'{current_folder}/data', download=True)

    rem_dataset, test_dataset = \
        train_test_split(dataset, test_size=0.1, random_state=42)
    train_dataset, valid_dataset = \
        train_test_split(rem_dataset, test_size=0.1, random_state=42)

    train_tasks = data2task(train_dataset, ways, shots, num_tasks=20000)
    valid_tasks = data2task(valid_dataset, ways, shots, num_tasks=600)
    test_tasks = data2task(test_dataset, ways, shots, num_tasks=600)

    return train_tasks, valid_tasks, test_tasks


def load_miniimagenet(ways, shots):

    current_folder = os.getcwd()

    train_dataset = l2l.vision.datasets.MiniImagenet(root=f'{current_folder}/data', mode='train', download=True)
    valid_dataset = l2l.vision.datasets.MiniImagenet(root=f'{current_folder}/data', mode='validation', download=True)
    test_dataset = l2l.vision.datasets.MiniImagenet(root=f'{current_folder}/data', mode='test', download=True)

    train_tasks = data2task(train_dataset, ways, shots, num_tasks=20000)
    valid_tasks = data2task(valid_dataset, ways, shots, num_tasks=600)
    test_tasks = data2task(test_dataset, ways, shots, num_tasks=600)

    return train_tasks, valid_tasks, test_tasks


def load_fc100(ways, shots):

    current_folder = os.getcwd()

    # Create Datasets, train (36000, 32, 32, 3), val (12000, 32, 32, 3), test (12000, 32, 32, 3)
    train_dataset = l2l.vision.datasets.FC100(root=f'{current_folder}/datasource',
                                              transform=tv.transforms.ToTensor(),
                                              mode='train',
                                              download=True)
    valid_dataset = l2l.vision.datasets.FC100(root=f'{current_folder}/datasource',
                                              transform=tv.transforms.ToTensor(),
                                              mode='validation',
                                              download=True)
    test_dataset = l2l.vision.datasets.FC100(root=f'{current_folder}/datasource',
                                             transform=tv.transforms.ToTensor(),
                                             mode='test',
                                             download=True)

    train_tasks = data2task(train_dataset, ways, shots, num_tasks=20000)
    valid_tasks = data2task(valid_dataset, ways, shots, num_tasks=600)
    test_tasks = data2task(test_dataset, ways, shots, num_tasks=600)

    return train_tasks, valid_tasks, test_tasks


def data2task(dataset, ways, shots, num_tasks):

    dataset = l2l.data.MetaDataset(dataset)
    transforms = [
        FusedNWaysKShots(dataset, n=ways, k=2 * shots),
        LoadData(dataset),
        RemapLabels(dataset),
        ConsecutiveLabels(dataset),
    ]
    tasks = l2l.data.TaskDataset(dataset,
                                 task_transforms=transforms,
                                 num_tasks=num_tasks)
    return tasks