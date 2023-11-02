import torch
import numpy as np
from models.models import Task
from hypergrad.hypergradients import *
from hypergrad.diff_optimizers import *


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


def fast_adapt(method, batch, learner, features, loss, shots, ways, inner_steps, reg_lambda, device):
    data, labels = batch

    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels \
        = split_data(data, labels, shots, ways, device)

    if method == 'ITD-BiO' or method == 'ANIL':
        adaptation_data = features(adaptation_data)
        evaluation_data = features(evaluation_data)
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


def train_one_epoch_baseline(args, cur_iter, meta_model, features, train_parameters,
                             loss, optimizer, train_tasks,
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

    # baseline offline models will be trained on cumulated data observed until current iteration
    num_tasks = cur_iter + 1

    for task_idx in range(num_tasks):
        learner = meta_model.clone()
        # batch = train_tasks.sample()
        batch = train_tasks[task_idx]

        evaluation_error, evaluation_accuracy = fast_adapt(args.method, batch, learner, features, loss,
                                                           shots, ways, args.inner_stp, args.reg, device)

        evaluation_error.backward()
        meta_train_error += evaluation_error.item()
        meta_train_accuracy += evaluation_accuracy.item()

    # Average the accumulated gradients and optimize
    for p in train_parameters:
        p.grad.data.mul_(1.0 / num_tasks)

    optimizer.step()

    return meta_train_error / num_tasks, meta_train_accuracy / num_tasks


def evaluate_one_epoch_baseline(args, meta_model, features, loss, task_loader,
                             device, num_tasks=32, shots=5, ways=5):
    meta_eval_error = 0.0
    meta_eval_accuracy = 0.0

    for task in range(num_tasks):
        learner = meta_model.clone()
        batch = task_loader.sample()

        evaluation_error, evaluation_accuracy = fast_adapt(args.method, batch, learner, features, loss,
                                                           shots, ways, args.inner_stp,
                                                           args.reg, device)
        meta_eval_error += evaluation_error.item()
        meta_eval_accuracy += evaluation_accuracy.item()

    return meta_eval_error / num_tasks, meta_eval_accuracy / num_tasks


def inner_loop(hparams, params, optim, n_steps, log_interval=None, create_graph=False):
    params_history = [optim.get_opt_params(params)]

    for t in range(n_steps):
        params_history.append(optim(params_history[-1], hparams, create_graph=create_graph))

        if log_interval and (t % log_interval == 0 or t == n_steps-1):
            print('t={}, Loss: {:.6f}'.format(t, optim.curr_loss.item()))

    return params_history


def get_inner_opt(train_loss, inner_lr=0.1):
    inner_opt_class = GradientDescent
    inner_opt_kwargs = {'step_size': inner_lr}
    return inner_opt_class(train_loss, **inner_opt_kwargs)


def train_one_epoch_OAGD(args, cur_iter, meta_model, optimizer, train_tasks,
                             device, momentum_list, shots=5, ways=5):

    optimizer.zero_grad()
    meta_train_error = 0.0
    meta_train_accuracy = 0.0
    for task_idx in range(args.win_size):

        if cur_iter - task_idx < 0: pass
        data, labels = train_tasks[cur_iter-task_idx]  # current task
        # data, labels = train_tasks.sample()
        data, labels = data.to(device), labels.to(device)
        # Separate data into adaptation/evaluation sets
        adaptation_data, adaptation_labels, evaluation_data, evaluation_labels \
            = split_data(data, labels, shots, ways, device)

        # single task set up
        task = Task(args.reg, meta_model, (adaptation_data, adaptation_labels,
                                            evaluation_data, evaluation_labels), batch_size=args.win_size)
        inner_opt = get_inner_opt(task.train_loss_f, args.inner_lr)

        # single task inner loop
        params = [p.detach().clone().requires_grad_(True) for p in meta_model.parameters()]
        last_param = inner_loop(meta_model.parameters(), params, inner_opt, args.inner_stp)[-1]

        # single task hypergradient computation
        if args.hg_mode == 'CG':
            # This is the approximation used in the paper CG stands for conjugate gradient
            cg_fp_map = GradientDescent(loss_f=task.train_loss_f, step_size=1)  # original 1.
            CG(last_param, list(meta_model.parameters()), K=5,
               fp_map=cg_fp_map, outer_loss=task.val_loss_f, momentum=momentum_list[task_idx])
        elif args.hg_mode == 'fixed_point':
            fixed_point(last_param, list(meta_model.parameters()), K=5, fp_map=inner_opt,
                        outer_loss=task.val_loss_f, momentum=momentum_list[task_idx])

        meta_train_error += task.val_loss
        meta_train_accuracy += task.val_acc

    optimizer.step()

    return meta_train_error / args.win_size, meta_train_accuracy / args.win_size


def evaluate_one_epoch_OAGD(args, task_loader, meta_model, shots=5, ways=5):
    meta_model.train()
    device = next(meta_model.parameters()).device

    meta_eval_error = 0.0
    meta_eval_accuracy = 0.0
    for idx in range(args.win_size):
        data, labels = task_loader.sample()
        data, labels = data.to(device), labels.to(device)
        # Separate data into adaptation/evaluation sets
        adaptation_data, adaptation_labels, evaluation_data, evaluation_labels \
            = split_data(data, labels, shots, ways, device)

        task = Task(args.reg, meta_model, (adaptation_data, adaptation_labels, evaluation_data, evaluation_labels))
        inner_opt = get_inner_opt(task.train_loss_f, args.inner_lr)

        params = [p.detach().clone().requires_grad_(True) for p in meta_model.parameters()]
        last_param = inner_loop(meta_model.parameters(), params, inner_opt, args.inner_stp)[-1]

        task.val_loss_f(last_param, meta_model.parameters())

        meta_eval_error += task.val_loss
        meta_eval_accuracy += task.val_acc

    return meta_eval_error / args.win_size, meta_eval_accuracy / args.win_size