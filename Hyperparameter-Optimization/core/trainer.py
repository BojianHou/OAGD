import torch.nn.functional as F
from utils.metrics import topk_corrects
import torch
from torch.autograd import grad
import numpy as np

from core.utils import gather_flat_grad,neumann_hyperstep_preconditioner
from core.utils import get_trainable_hyper_params,assign_hyper_gradient
from sklearn.metrics import balanced_accuracy_score


def get_current_batches_bak(cur_epoch, window_size, dataloader):
    # get a small batches from all the batches in the dataloader according to the current epoch
    selected_batches = []
    cur_batch = cur_epoch % len(dataloader)
    for idx, batch in enumerate(dataloader):
        if idx <= cur_batch and idx >= max(0, cur_batch-window_size+1):
            selected_batches.append(batch)

    return selected_batches


def get_current_batches(cur_batch, window_size, dataloader):
    # get a small batches from all the batches in the dataloader according to the current epoch
    # here dataloader is a list of batches

    cur_batch = cur_batch % len(dataloader)

    return dataloader[max(0, cur_batch-window_size+1):(cur_batch+1)]


def accumulate_batches_over_time_bak(time_step, dataloader):

    total_batches = len(dataloader)
    num_repeats = time_step // total_batches
    remainder = time_step % total_batches

    # Repeat the DataLoader as many times as needed and concatenate
    accumulated_batches = [batch for _ in range(num_repeats) for batch in dataloader]

    # Handle the remaining batches if there's a remainder
    for i, batch in enumerate(dataloader):
        if i > remainder:
            break
        accumulated_batches.append(batch)

    return accumulated_batches


def accumulate_batches_over_time(time_step, dataloader):

    total_batches = len(dataloader)
    num_repeats = time_step // total_batches
    remainder = time_step % total_batches

    final_loader = []
    for _ in range(num_repeats):
        final_loader += dataloader
    final_loader += dataloader[:(remainder+1)]

    return final_loader


def train_epoch(
    cur_epoch, model, args,
    low_loader, low_criterion , low_optimizer, low_params=None,
    up_loader=None, up_optimizer=None, up_criterion=None, up_params=None,
    ):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    num_classes=args["num_classes"]
    group_size=args["group_size"]
    ARCH_EPOCH=args["up_configs"]["start_epoch"]
    ARCH_END=args["up_configs"]["end_epoch"]
    ARCH_EPOCH_INTERVAL=args["up_configs"]["epoch_interval"]
    ARCH_INTERVAL=args["up_configs"]["iter_interval"]
    ARCH_TRAIN_SAMPLE=args["up_configs"]["train_batches"]
    ARCH_VAL_SAMPLE=args["up_configs"]["val_batches"]
    device=args["device"]
    is_up=(cur_epoch >= ARCH_EPOCH) and (cur_epoch <= ARCH_END) and \
        ((cur_epoch+1) % ARCH_EPOCH_INTERVAL) == 0

    low_loader = accumulate_batches_over_time(cur_epoch, low_loader)

    if is_up:
        print('lower lr: ',low_optimizer.param_groups[0]['lr'],'  upper lr: ',up_optimizer.param_groups[0]['lr'])
        up_iter = iter(up_loader)
        low_iter_alt=iter(low_loader)
    else:
        print('lr: ',low_optimizer.param_groups[0]['lr'])
    
    model.train()
    total_correct=0.
    total_sample=0.
    total_loss=0.
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), 2*((num_classes-1)//group_size)+1
    use_reg=True

    d_train_loss_d_w = torch.zeros(num_weights,device=device)

    for cur_iter, (low_data, low_targets) in enumerate(low_loader):

        low_data, low_targets = low_data.to(device=device, non_blocking=True), low_targets.to(device=device, non_blocking=True)
        # Update architecture
        if is_up:
            model.train()
            up_optimizer.zero_grad()
            if cur_iter%ARCH_INTERVAL==0:
                for _ in range(ARCH_TRAIN_SAMPLE):
                    try:
                        low_data_alt, low_targets_alt = next(low_iter_alt)
                    except StopIteration:
                        low_iter_alt = iter(low_loader)
                        low_data_alt, low_targets_alt = next(low_iter_alt) 
                    low_data_alt, low_targets_alt = low_data_alt.to(device=device, non_blocking=True), low_targets_alt.to(device=device, non_blocking=True)
                    low_optimizer.zero_grad()
                    low_preds=model(low_data_alt)
                    low_loss=low_criterion(low_preds,low_targets_alt,low_params,group_size=group_size) 
                    d_train_loss_d_w+=gather_flat_grad(grad(low_loss,model.parameters(),create_graph=True))
                    #print(cur_iter_alt)
                d_train_loss_d_w/=ARCH_TRAIN_SAMPLE
                d_val_loss_d_theta = torch.zeros(num_weights,device=device)
                #d_val_loss_d_theta, direct_grad = torch.zeros(num_weights).cuda(), torch.zeros(num_hypers).cuda()

                for _ in range(ARCH_VAL_SAMPLE):
                    try:
                        up_data, up_targets = next(up_iter)
                    except StopIteration:
                        up_iter = iter(up_loader)
                        up_data, up_targets = next(up_iter) 
                #for _,(up_data,up_targets) in enumerate(up_loader):
                    up_data, up_targets = up_data.to(device=device, non_blocking=True), up_targets.to(device=device, non_blocking=True)
                    model.zero_grad()
                    low_optimizer.zero_grad()
                    up_preds = model(up_data)
                    up_loss = up_criterion(up_preds,up_targets,up_params,group_size=group_size)
                    d_val_loss_d_theta += gather_flat_grad(grad(up_loss, model.parameters(), retain_graph=use_reg))
                    # if use_reg:
                    #     direct_grad+=gather_flat_grad(grad(up_loss, get_trainable_hyper_params(up_params), allow_unused=True))
                    #     direct_grad[direct_grad != direct_grad] = 0
                d_val_loss_d_theta/=ARCH_VAL_SAMPLE
                #direct_grad/=ARCH_VAL_SAMPLE
                preconditioner = d_val_loss_d_theta
                
                preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, 0.5, 5, model)
                indirect_grad = gather_flat_grad(
                    grad(d_train_loss_d_w, get_trainable_hyper_params(up_params), grad_outputs=preconditioner.view(-1),allow_unused=True))
                hyper_grad=-indirect_grad#+direct_grad
                up_optimizer.zero_grad()
                assign_hyper_gradient(up_params,hyper_grad,num_classes)
                up_optimizer.step()
                d_train_loss_d_w = torch.zeros(num_weights,device=device)

        # Perform the forward pass
        low_preds = model(low_data)

        # Compute the loss
        loss = low_criterion(low_preds, low_targets, low_params,group_size=group_size)
        # Perform the backward pass
        low_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        low_optimizer.step()

        # Compute the errors
        mb_size = low_data.size(0)
        ks = [1] 
        top1_correct = topk_corrects(low_preds, low_targets, ks)[0]
        
        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct+=top1_correct
        total_sample+=mb_size
        total_loss+=loss*mb_size
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss/total_sample}   ACC = {total_correct/total_sample}')


def train_epoch_online_bak(
        cur_epoch, model, args,
        low_loader, low_criterion, low_optimizer, low_params=None,
        up_loader=None, up_optimizer=None, up_criterion=None, up_params=None,
):
    """Performs one epoch of online bilevel optimization."""

    # Enable training mode
    num_classes = args["num_classes"]
    group_size = args["group_size"]
    device = args["device"]

    print('lower lr: ', low_optimizer.param_groups[0]['lr'], '  upper lr: ', up_optimizer.param_groups[0]['lr'])
    up_iter = iter(up_loader)
    low_iter = iter(low_loader)

    model.train()

    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), 2 * ((num_classes - 1) // group_size) + 1
    use_reg = True

    if 'time_step' in args:
        time_step = args['time_step']
    else:
        time_step = 10

    d_train_loss_d_w = torch.zeros(num_weights, device=device)

    for cur_iter in range(time_step):  # time step

        print('time step: ', cur_iter)
        low_data, low_targets = next(low_iter)
        low_data, low_targets = low_data.to(device=device, non_blocking=True), low_targets.to(device=device,
                                                                                              non_blocking=True)

        # upper (outer) level
        model.train()
        up_optimizer.zero_grad()
        low_optimizer.zero_grad()
        low_preds = model(low_data)
        low_loss = low_criterion(low_preds, low_targets, low_params, group_size=group_size)
        d_train_loss_d_w += gather_flat_grad(grad(low_loss, model.parameters(), create_graph=True))

        d_val_loss_d_theta = torch.zeros(num_weights, device=device)
        up_data, up_targets = next(up_iter)
        up_data, up_targets = up_data.to(device=device, non_blocking=True), up_targets.to(device=device,
                                                                                          non_blocking=True)
        model.zero_grad()
        low_optimizer.zero_grad()
        up_preds = model(up_data)
        up_loss = up_criterion(up_preds, up_targets, up_params, group_size=group_size)
        d_val_loss_d_theta += gather_flat_grad(grad(up_loss, model.parameters(), retain_graph=use_reg))

        preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, 0.5,
                                                          5, model)
        indirect_grad = gather_flat_grad(
            grad(d_train_loss_d_w, get_trainable_hyper_params(up_params), grad_outputs=preconditioner.view(-1),
                 allow_unused=True))
        hyper_grad = -indirect_grad  # +direct_grad
        up_optimizer.zero_grad()
        assign_hyper_gradient(up_params, hyper_grad, num_classes)
        up_optimizer.step()
        d_train_loss_d_w = torch.zeros(num_weights, device=device)

        # inner level
        inner_step = 5
        total_correct = 0.
        total_sample = 0.
        total_loss = 0.
        for inner_idx in range(inner_step):
            # Perform the forward pass
            low_preds = model(low_data)

            # Compute the loss
            loss = low_criterion(low_preds, low_targets, low_params, group_size=group_size)
            # Perform the backward pass
            low_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            low_optimizer.step()

            # Compute the errors
            mb_size = low_data.size(0)
            ks = [1]
            top1_correct = topk_corrects(low_preds, low_targets, ks)[0]

            # Copy the stats from GPU to CPU (sync point)
            loss = loss.item()
            top1_correct = top1_correct.item()
            total_correct += top1_correct
            total_sample += mb_size
            total_loss += loss * mb_size

        print(f'Time Step {cur_iter} :  '
              f'Loss = {total_loss / total_sample}   '
              f'ACC = {total_correct / total_sample * 100.}')


def train_epoch_online(  # online autobalance
        cur_epoch, model, args,
        low_loader, low_criterion, low_optimizer, low_params=None,
        up_loader=None, up_optimizer=None, up_criterion=None, up_params=None
):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    num_classes = args["num_classes"]
    group_size = args["group_size"]
    ARCH_EPOCH = args["up_configs"]["start_epoch"]
    ARCH_END = args["up_configs"]["end_epoch"]
    ARCH_EPOCH_INTERVAL = args["up_configs"]["epoch_interval"]
    ARCH_INTERVAL = args["up_configs"]["iter_interval"]
    ARCH_TRAIN_SAMPLE = args["up_configs"]["train_batches"]
    ARCH_VAL_SAMPLE = args["up_configs"]["val_batches"]
    device = args["device"]
    is_up = (cur_epoch >= ARCH_EPOCH) and (cur_epoch <= ARCH_END) and \
            ((cur_epoch + 1) % ARCH_EPOCH_INTERVAL) == 0

    window_size = args['win_size']
    low_loader = get_current_batches(cur_epoch, window_size, low_loader)
    up_loader  = get_current_batches(cur_epoch, window_size, up_loader)

    if is_up:
        print('lower lr: ', low_optimizer.param_groups[0]['lr'], '  upper lr: ', up_optimizer.param_groups[0]['lr'])
        up_iter = iter(up_loader)
        # low_iter_alt = iter(low_loader)
    else:
        print('lr: ', low_optimizer.param_groups[0]['lr'])

    model.train()
    total_correct = 0.
    total_sample = 0.
    total_loss = 0.
    num_weights, num_hypers = sum(p.numel() for p in model.parameters()), 2 * ((num_classes - 1) // group_size) + 1
    use_reg = True

    # d_train_loss_d_w = torch.zeros(num_weights, device=device)

    for cur_iter, (low_data, low_targets) in enumerate(low_loader):

        low_data, low_targets = low_data.to(device=device, non_blocking=True), low_targets.to(device=device,
                                                                                              non_blocking=True)
        # Perform the forward pass
        low_preds = model(low_data)

        # Compute the loss
        loss = low_criterion(low_preds, low_targets, low_params, group_size=group_size)
        # Perform the backward pass
        low_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        low_optimizer.step()

        # Compute the errors
        mb_size = low_data.size(0)
        ks = [1]
        top1_correct = topk_corrects(low_preds, low_targets, ks)[0]

        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct += top1_correct
        total_sample += mb_size
        total_loss += loss * mb_size
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss / total_sample}   ACC = {total_correct / total_sample * 100.}')

    # momentum for window smoothing
    gamma = args['gamma']  # 1, 0.9, 0.5
    momentum_list = [gamma ** i for i in range(len(low_loader))]
    W = sum(momentum_list)
    momentum_list = [momentum / W for momentum in momentum_list]
    momentum_list.reverse()  # the last one is for the current batch

    # Update architecture
    if is_up:
        model.train()
        up_optimizer.zero_grad()

        # we have 2 trainable hyperparameters, each has num_classes dimensions
        indirect_grad = torch.zeros(2 * num_classes, device=device)

        # win_id=0 means current batch, win_idx=1 means last batch, win_idx=2 means last second batch
        for win_idx in range(len(low_loader)):  # from the previous batch to the current batch

            low_data_alt, low_targets_alt = low_loader[win_idx]

            low_data_alt, low_targets_alt = low_data_alt.to(device=device, non_blocking=True), \
                low_targets_alt.to(device=device, non_blocking=True)
            low_optimizer.zero_grad()
            low_preds = model(low_data_alt)
            low_loss = low_criterion(low_preds, low_targets_alt, low_params, group_size=group_size)
            d_train_loss_d_w = gather_flat_grad(grad(low_loss, model.parameters(), create_graph=True))

            # up_data, up_targets = up_loader[win_idx]
            try:
                up_data, up_targets = next(up_iter)
            except StopIteration:
                up_iter = iter(up_loader)
                up_data, up_targets = next(up_iter)
                # print('length of up loader: ', len(up_loader))

            up_data, up_targets = up_data.to(device=device, non_blocking=True), \
                up_targets.to(device=device, non_blocking=True)
            model.zero_grad()
            low_optimizer.zero_grad()
            up_preds = model(up_data)
            up_loss = up_criterion(up_preds, up_targets, up_params, group_size=group_size)
            d_val_loss_d_theta = gather_flat_grad(grad(up_loss, model.parameters(), retain_graph=use_reg))

            preconditioner = neumann_hyperstep_preconditioner(d_val_loss_d_theta, d_train_loss_d_w, 0.5, 5, model)
            indirect_grad += momentum_list[win_idx] * \
                             gather_flat_grad(grad(d_train_loss_d_w, get_trainable_hyper_params(up_params),
                                                   grad_outputs=preconditioner.view(-1), allow_unused=True))

        # hyper_grad = - indirect_grad / window_size  # +direct_grad
        hyper_grad = - indirect_grad
        up_optimizer.zero_grad()
        assign_hyper_gradient(up_params, hyper_grad, num_classes)
        up_optimizer.step()
        # d_train_loss_d_w = torch.zeros(num_weights, device=device)


def train_epoch_SGD(
        cur_epoch, model, args,
        low_loader, low_criterion, low_optimizer, low_params=None,
):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    num_classes = args["num_classes"]
    group_size = args["group_size"]

    device = args["device"]

    print('lr: ', low_optimizer.param_groups[0]['lr'])

    low_loader = accumulate_batches_over_time(cur_epoch, low_loader)

    model.train()
    total_correct = 0.
    total_sample = 0.
    total_loss = 0.

    for cur_iter, (low_data, low_targets) in enumerate(low_loader):

        low_data, low_targets = low_data.to(device=device, non_blocking=True), low_targets.to(device=device,
                                                                                              non_blocking=True)
        # Perform the forward pass
        low_preds = model(low_data)

        # Compute the loss
        loss = low_criterion(low_preds, low_targets, low_params, group_size=group_size)
        # Perform the backward pass
        low_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        low_optimizer.step()

        # Compute the errors
        mb_size = low_data.size(0)
        ks = [1]
        top1_correct = topk_corrects(low_preds, low_targets, ks)[0]

        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct += top1_correct
        total_sample += mb_size
        total_loss += loss * mb_size
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss / total_sample}   ACC = {total_correct / total_sample * 100.}')


def train_epoch_OGD(
        cur_epoch, model, args,
        low_loader, low_criterion, low_optimizer, low_params=None,
):
    """Performs one epoch of bilevel optimization."""
    # Enable training mode
    group_size = args["group_size"]

    device = args["device"]

    print('lr: ', low_optimizer.param_groups[0]['lr'])

    low_loader = get_current_batches(cur_epoch, 1, low_loader)  # window size is 1

    model.train()
    total_correct = 0.
    total_sample = 0.
    total_loss = 0.

    for cur_iter, (low_data, low_targets) in enumerate(low_loader):

        low_data, low_targets = low_data.to(device=device, non_blocking=True), low_targets.to(device=device,
                                                                                              non_blocking=True)
        # Perform the forward pass
        low_preds = model(low_data)

        # Compute the loss
        loss = low_criterion(low_preds, low_targets, low_params, group_size=group_size)
        # Perform the backward pass
        low_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        low_optimizer.step()

        # Compute the errors
        mb_size = low_data.size(0)
        ks = [1]
        top1_correct = topk_corrects(low_preds, low_targets, ks)[0]

        # Copy the stats from GPU to CPU (sync point)
        loss = loss.item()
        top1_correct = top1_correct.item()
        total_correct += top1_correct
        total_sample += mb_size
        total_loss += loss * mb_size
    # Log epoch stats
    print(f'Epoch {cur_epoch} :  Loss = {total_loss / total_sample}   ACC = {total_correct / total_sample * 100.}')


@torch.no_grad()
def eval_epoch(data_loader, model, criterion, cur_epoch, text, args, params=None):
    num_classes=args["num_classes"]
    group_size=args["group_size"]
    device = args["device"]
    model.eval()
    correct=0.
    total=0.
    loss=0.
    class_correct=np.zeros(num_classes,dtype=float)
    class_total=np.zeros(num_classes,dtype=float)
    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)#.cuda()
    y_true, y_pred = [], []
    for cur_iter, (data, targets) in enumerate(data_loader):
        # if cur_iter%5==0:
        #     print(cur_iter,len(data_loader))
        # data, targets = data.cuda(), targets.cuda(non_blocking=True)
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
         
        preds = logits.data.max(1)[1]
        for t, p in zip(targets.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        mb_size = data.size(0)
        loss += criterion(logits,targets,params,group_size=group_size).item() * mb_size
        y_true.append(targets.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())
    class_correct = confusion_matrix.diag().cpu().numpy()
    class_total = confusion_matrix.sum(1).cpu().numpy()
    total = confusion_matrix.sum().cpu().numpy()
    correct = class_correct.sum()

    text=f'{text}: Epoch {cur_epoch} :  Loss = {loss/total}   ACC = {correct/total} ' \
         f'Balanced ACC = {balanced_accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))}' \
         f'\n Class wise = {(class_correct/class_total)}'
    print(text)
    return text, loss/total, 1.-correct/total, \
        1.-float(np.mean(class_correct/class_total)), \
        (1-class_correct/class_total).tolist(), \
        correct / total, \
        balanced_accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))

