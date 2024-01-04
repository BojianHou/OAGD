from math import ceil
from PIL.Image import BICUBIC
from PIL import Image
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine
from torchvision.transforms import ToTensor, Normalize

from torch.utils.data import Subset, Dataset, Sampler, ConcatDataset, DataLoader, WeightedRandomSampler
import torchvision.utils as vutils
import numpy as np
import random
import torch
import itertools


class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # Acrually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch


class CustomDataset(Dataset):
    """CustomDataset with support of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    def __len__(self):
        return len(self.data)


class CustomDataset_mnist(Dataset):
    """CustomDataset with support of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    def __len__(self):
        return len(self.data)


def load_mnist(train_size = 4000, train_rho = 0.01, val_size = 1000, val_rho = 0.01,
               image_size = 32, batch_size = 128, num_workers = 2,
               path = './data', num_classes = 10, balance_val = False):

    trans_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root=path, train=True, transform=trans_mnist, download=True)
    test_dataset = MNIST(root=path, train=False, transform=trans_mnist, download=True)
    train_x, train_y = np.array(train_dataset.data), np.array(train_dataset.targets)
    test_x, test_y = np.array(test_dataset.data), np.array(test_dataset.targets)
    total_size = 5000
    num_total_samples = []
    num_train_samples = []
    num_val_samples = []

    if not balance_val:
        train_mu = train_rho ** (1. / 9.)
        val_mu = val_rho ** (1. / 9.)
        for i in range(num_classes):
            num_total_samples.append(ceil(total_size * (train_mu ** i)))
            num_train_samples.append(ceil(train_size * (train_mu ** i)))
            num_val_samples.append(ceil(val_size * (val_mu ** i)))
            # num_val_samples.append(num_total_samples[-1]-num_train_samples[-1])
            # num_val_samples.append(round(val_size*(val_mu**i)))
    elif balance_val:
        train_mu = train_rho ** (1. / 9.)
        for i in range(num_classes):
            num_val_samples.append(val_size)
            num_total_samples.append(ceil(total_size * (train_mu ** i)))
            num_train_samples.append(ceil(train_size * (train_mu ** i)))

    train_index = []
    val_index = []
    # print(train_x,train_y)
    print(num_train_samples,num_val_samples)
    for i in range(num_classes):
        train_index.extend(np.where(train_y == i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y == i)[0][-num_val_samples[i]:])

    # total_index = []
    # total_index.extend(train_index)
    # total_index.extend(val_index)
    # total_index = list(set(total_index))
    # random.shuffle(total_index)
    # train_x, train_y = train_x[total_index], train_y[total_index]

    random.shuffle(train_index)
    random.shuffle(val_index)

    train_data, train_targets = np.expand_dims(train_x[train_index], axis=-1), train_y[train_index]
    val_data, val_targets = np.expand_dims(train_x[val_index], axis=-1), train_y[val_index]

    test_data, test_targets = np.expand_dims(test_x[:1000], axis=-1), test_y[:1000]

    train_dataset = CustomDataset_mnist(train_data, train_targets, trans_mnist)
    val_dataset = CustomDataset_mnist(val_data, val_targets, trans_mnist)
    train_eval_dataset = CustomDataset_mnist(train_data, train_targets, trans_mnist)
    val_eval_dataset = CustomDataset_mnist(val_data, val_targets, trans_mnist)
    test_dataset = CustomDataset_mnist(test_data, test_targets, trans_mnist)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = True, drop_last = False, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = True, drop_last = False, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)

    eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)
    eval_val_loader = DataLoader(val_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)

    return train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples


def load_mnist_new(train_size = 4000, train_mu = 0.6, val_size = 1000, val_mu = 0.6,
                   batch_size = 128, num_workers = 2,
                   path = './data', num_classes = 10, balance_val = True):

    trans_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root=path, train=True, transform=trans_mnist, download=True)
    test_dataset = MNIST(root=path, train=False, transform=trans_mnist, download=True)
    train_x, train_y = np.array(train_dataset.data), np.array(train_dataset.targets)
    test_x, test_y = np.array(test_dataset.data), np.array(test_dataset.targets)
    total_size = 5000
    num_train_samples = []
    num_val_samples = []

    if not balance_val:
        # train_mu = train_rho ** (1. / 9.)
        # val_mu = val_rho ** (1. / 9.)
        for i in range(num_classes):
            num_train_samples.append(ceil(train_size * (train_mu ** i)))
            num_val_samples.append(ceil(val_size * (val_mu ** i)))
            # num_val_samples.append(num_total_samples[-1]-num_train_samples[-1])
            # num_val_samples.append(round(val_size*(val_mu**i)))
    elif balance_val:
        # train_mu = train_rho ** (1. / 9.)
        for i in range(num_classes):
            num_val_samples.append(val_size)
            num_train_samples.append(ceil(train_size * (train_mu ** i)))

    train_index = []
    val_index = []
    # print(train_x,train_y)
    print(num_train_samples, num_val_samples)
    for i in range(num_classes):
        train_index.extend(np.where(train_y == i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y == i)[0][-num_val_samples[i]:])

    # total_index = []
    # total_index.extend(train_index)
    # total_index.extend(val_index)
    # total_index = list(set(total_index))
    # random.shuffle(total_index)
    # train_x, train_y = train_x[total_index], train_y[total_index]

    random.shuffle(train_index)
    random.shuffle(val_index)

    train_data, train_targets = np.expand_dims(train_x[train_index], axis=-1), train_y[train_index]
    val_data, val_targets = np.expand_dims(train_x[val_index], axis=-1), train_y[val_index]

    test_data, test_targets = np.expand_dims(test_x[:1000], axis=-1), test_y[:1000]

    train_dataset = CustomDataset_mnist(train_data, train_targets, trans_mnist)
    val_dataset = CustomDataset_mnist(val_data, val_targets, trans_mnist)
    train_eval_dataset = CustomDataset_mnist(train_data, train_targets, trans_mnist)
    val_eval_dataset = CustomDataset_mnist(val_data, val_targets, trans_mnist)
    test_dataset = CustomDataset_mnist(test_data, test_targets, trans_mnist)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = True, drop_last = False, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = True, drop_last = False, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)

    eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)
    eval_val_loader = DataLoader(val_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)

    return train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples



def load_mnist_balance(train_size = 4000, train_rho = 0.01, val_size = 1000, val_rho = 0.01,
               image_size = 32, batch_size = 128, num_workers = 2,
               path = './data', num_classes = 10, balance_val = False):

    trans_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root=path, train=True, transform=trans_mnist, download=True)
    test_dataset = MNIST(root=path, train=False, transform=trans_mnist, download=True)
    train_x, train_y = np.array(train_dataset.data), np.array(train_dataset.targets)
    test_x, test_y = np.array(test_dataset.data), np.array(test_dataset.targets)
    total_size = 5000
    num_total_samples = []
    num_train_samples = []
    num_val_samples = []

    for i in range(num_classes):
        num_val_samples.append(val_size)
        num_total_samples.append(total_size)
        num_train_samples.append(train_size)

    train_index = []
    val_index = []
    # print(train_x,train_y)
    # print(num_train_samples,num_val_samples)
    for i in range(num_classes):
        train_index.extend(np.where(train_y == i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y == i)[0][-num_val_samples[i]:])

    # total_index = []
    # total_index.extend(train_index)
    # total_index.extend(val_index)
    # total_index = list(set(total_index))
    # random.shuffle(total_index)
    # train_x, train_y = train_x[total_index], train_y[total_index]

    random.shuffle(train_index)
    random.shuffle(val_index)

    train_data, train_targets = np.expand_dims(train_x[train_index], axis=-1), train_y[train_index]
    val_data, val_targets = np.expand_dims(train_x[val_index], axis=-1), train_y[val_index]

    test_data, test_targets = np.expand_dims(test_x[:1000], axis=-1), test_y[:1000]

    train_dataset = CustomDataset_mnist(train_data, train_targets, trans_mnist)
    val_dataset = CustomDataset_mnist(val_data, val_targets, trans_mnist)
    train_eval_dataset = CustomDataset_mnist(train_data, train_targets, trans_mnist)
    val_eval_dataset = CustomDataset_mnist(val_data, val_targets, trans_mnist)
    test_dataset = CustomDataset_mnist(test_data, test_targets, trans_mnist)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = True, drop_last = False, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = True, drop_last = False, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)

    eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)
    eval_val_loader = DataLoader(val_eval_dataset, batch_size=batch_size, num_workers=num_workers,
                    shuffle = False, drop_last = False, pin_memory = True)

    return train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples

def loader_to_list(dataloader):
    batches = []
    for batch in dataloader:
        batches.append(batch)
    return batches


def intersperse(list1, list2):
  """Intersperses two lists together."""

  interleaved_list = []
  for i in range(min(len(list1), len(list2))):
    interleaved_list.append(list1[i])
    interleaved_list.append(list2[i])

  return interleaved_list


def load_mnist_dist_shift(train_size = 4000, train_rho = 0.01, val_size = 1000, val_rho = 0.01,
               image_size = 32, batch_size = 128, num_workers = 2,
               path = './data', num_classes = 10, balance_val = False):
    train_loader_balance, val_loader_balance, test_loader_balance, eval_train_loader_balance, \
        eval_val_loader_balance, num_train_samples_balance, num_val_samples_balance = load_mnist_balance(
        train_size=train_size, val_size=val_size,
        balance_val=balance_val, batch_size=batch_size,
        train_rho=train_rho, image_size=28, path=path)
    train_loader_balance = loader_to_list(train_loader_balance)
    val_loader_balance = loader_to_list(val_loader_balance)
    test_loader_balance = loader_to_list(test_loader_balance)
    eval_train_loader_balance = loader_to_list(eval_train_loader_balance)
    eval_val_loader_balance = loader_to_list(eval_val_loader_balance)

    train_loader_imbalance, val_loader_imbalance, test_loader_imbalance, \
        eval_train_loader_imbalance, eval_val_loader_imbalance, \
        num_train_samples_imbalance, num_val_samples_imbalance = load_mnist(
        train_size=train_size, val_size=val_size,
        balance_val=balance_val, batch_size=batch_size,
        train_rho=train_rho, image_size=28, path=path)
    train_loader_imbalance = loader_to_list(train_loader_imbalance)
    val_loader_imbalance = loader_to_list(val_loader_imbalance)
    test_loader_imbalance = loader_to_list(test_loader_imbalance)
    eval_train_loader_imbalance = loader_to_list(eval_train_loader_imbalance)
    eval_val_loader_imbalance = loader_to_list(eval_val_loader_imbalance)

    train_loader = intersperse(train_loader_balance, train_loader_imbalance)
    # val_loader = intersperse(val_loader_balance, val_loader_imbalance)
    test_loader = intersperse(test_loader_balance, test_loader_imbalance)
    eval_train_loader = intersperse(eval_train_loader_balance, eval_train_loader_imbalance)
    # eval_val_loader = intersperse(eval_val_loader_balance, eval_val_loader_imbalance)

    return train_loader, val_loader_balance, test_loader, eval_train_loader, eval_val_loader_balance, \
        num_train_samples_balance, num_val_samples_balance


def load_mnist_shift_every_100(batch_size):
    train_loader0, val_loader0, test_loader0, eval_train_loader0, eval_val_loader0, num_train_samples0, num_val_samples0 = \
        load_mnist_new(train_size=4000, train_mu=0.4, val_size=1000, val_mu=0.4,
                   batch_size=batch_size, num_workers=2,
                   path='./data', num_classes=10, balance_val=True)
    train_loader0 = loader_to_list(train_loader0)
    val_loader0 = loader_to_list(val_loader0)
    test_loader0 = loader_to_list(test_loader0)
    eval_train_loader0 = loader_to_list(eval_train_loader0)
    eval_val_loader0 = loader_to_list(eval_val_loader0)

    train_loader1, val_loader1, test_loader1, eval_train_loader1, eval_val_loader1, num_train_samples1, num_val_samples1 = \
        load_mnist_new(train_size=4000, train_mu=0.6, val_size=1000, val_mu=0.4,
                       batch_size=batch_size, num_workers=2,
                       path='./data', num_classes=10, balance_val=True)
    train_loader1 = loader_to_list(train_loader1)
    val_loader1 = loader_to_list(val_loader1)
    test_loader1 = loader_to_list(test_loader1)
    eval_train_loader1 = loader_to_list(eval_train_loader1)
    eval_val_loader1 = loader_to_list(eval_val_loader1)

    train_loader2, val_loader2, test_loader2, eval_train_loader2, eval_val_loader2, num_train_samples2, num_val_samples2 = \
        load_mnist_new(train_size=4000, train_mu=0.8, val_size=1000, val_mu=0.8,
                       batch_size=batch_size, num_workers=2,
                       path='./data', num_classes=10, balance_val=True)
    train_loader2 = loader_to_list(train_loader2)
    val_loader2 = loader_to_list(val_loader2)
    test_loader2 = loader_to_list(test_loader2)
    eval_train_loader2 = loader_to_list(eval_train_loader2)
    eval_val_loader2 = loader_to_list(eval_val_loader2)

    train_loader3, val_loader3, test_loader3, eval_train_loader3, eval_val_loader3, num_train_samples3, num_val_samples3 = \
        load_mnist_new(train_size=4000, train_mu=1.0, val_size=1000, val_mu=1.0,
                       batch_size=batch_size, num_workers=2,
                       path='./data', num_classes=10, balance_val=True)
    train_loader3 = loader_to_list(train_loader3)
    val_loader3 = loader_to_list(val_loader3)
    test_loader3 = loader_to_list(test_loader3)
    eval_train_loader3 = loader_to_list(eval_train_loader3)
    eval_val_loader3 = loader_to_list(eval_val_loader3)

    # there are 53, 78, 140, 313 batches in train_loader0, train_loader1, train_loader2, train_loader3 respectively
    train_loader = train_loader0[:100] + train_loader1[:100] + train_loader2[:100] + train_loader3[:100]
    # there are 79 batches in each val_loader, here val_loader is balanced
    val_loader = val_loader0
    # there are 8 batches in each test_loader
    test_loader = test_loader0

    eval_train_loader = eval_train_loader0[:100] + eval_train_loader1[:100] + \
                        eval_train_loader2[:100] + eval_train_loader3[:100]
    eval_val_loader = eval_val_loader0

    return train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples3, num_val_samples3





# def check_imbalance(dataloader, imbalance_factor):
#     class_counts = {i: 0 for i in range(10)}  # Assuming there are 10 classes in MNIST
#     total_samples = 0
#
#     for _, labels in dataloader:
#         for label in labels:
#             class_counts[label.item()] += 1
#             total_samples += 1
#
#     print(f"Imbalance Factor: {imbalance_factor}")
#     print("Class Distribution in each batch:")
#     for class_label, count in class_counts.items():
#         print(f"Class {class_label}: {count} samples, {count / total_samples * 100:.2f}%")
#     print("-" * 50)
