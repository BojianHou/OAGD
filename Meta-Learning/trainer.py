import torch
import numpy as np


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def split_data(data, labels, shots, ways, device):

    data, labels = data.to(device), labels.to(device)

    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    return adaptation_data, adaptation_labels, evaluation_data, evaluation_labels


def fast_adapt(method, batch, learner, loss, shots, ways, inner_steps, reg_lambda, device):
    data, labels = batch

    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels \
        = split_data(data, labels, shots, ways, device)

    if method == 'ITD-BiO' or method == 'ANIL':
        for step in range(inner_steps):
            l2_reg = 0
            for p in learner.parameters():
                l2_reg += p.norm(2)
            train_error = loss(learner(adaptation_data), adaptation_labels) + reg_lambda * l2_reg
            learner.adapt(train_error)
    elif method == 'MAML':
        for step in range(inner_steps):
            train_error = loss(learner(adaptation_data), adaptation_labels)
            train_error /= len(adaptation_data)
            learner.adapt(train_error)

    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


def train_one_epoch_baseline(method, meta_model, train_parameters,
                             loss, optimizer, train_tasks, inner_steps, reg_lambda,
                             device, num_tasks=32, shots=5, ways=5):

    # This is the function for training metamodel in one epoch for the baseline methods ITD-BiO, ANIL and MAML
    # Note that the argument "meta_model" can be different from "train_parameters"
    # For example, a metamodel consists of two parts, i.e., "features" and "head"
    # "features" is to process data while "head" is to do the classification
    # **In ITD-BiO, "meta_model" is just the head while the train_parameters only contain the parameters of features
    # **In MAML, "meta_model" is the combination of both "features" and "head", and the "train_parameters" are exactly the parameters of the meta_model
    # **In ANIL, "meta_model" is just the "head" while the "train_parameters" still contain the parameters of the "features" and "head"

    optimizer.zero_grad()
    meta_train_error = 0.0
    meta_train_accuracy = 0.0

    for task in range(num_tasks):
        learner = meta_model.clone()
        batch = train_tasks.sample()

        evaluation_error, evaluation_accuracy = fast_adapt(method, batch, learner, loss,
                                                           shots, ways, inner_steps, reg_lambda, device)

        evaluation_error.backward()
        meta_train_error += evaluation_error.item()
        meta_train_accuracy += evaluation_accuracy.item()

    # Average the accumulated gradients and optimize
    for p in train_parameters:
        p.grad.data.mul_(1.0 / num_tasks)

    optimizer.step()

    return meta_train_error / num_tasks, meta_train_accuracy / num_tasks


def evaluate_one_epoch_baseline(method, meta_model, loss, train_tasks, inner_steps, reg_lambda,
                             device, num_tasks=32, shots=5, ways=5):
    meta_eval_error = 0.0
    meta_eval_accuracy = 0.0

    for task in range(num_tasks):
        learner = meta_model.clone()
        batch = train_tasks.sample()

        evaluation_error, evaluation_accuracy = fast_adapt(method, batch, learner, loss, shots, ways, inner_steps,
                                                           reg_lambda, device)
        meta_eval_error += evaluation_error.item()
        meta_eval_accuracy += evaluation_accuracy.item()

    return meta_eval_error / num_tasks, meta_eval_accuracy / num_tasks


def train_one_epoch_OAGD(method, meta_model, train_parameters,
                             loss, optimizer, train_tasks, inner_steps, reg_lambda,
                             device, num_tasks=32, shots=5, ways=5):

    return

def evaluate_one_epoch_OAGD(method, meta_model, loss, test_tasks,
                                inner_steps, reg, device):
    return