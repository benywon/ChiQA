# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 
import math  


def listMLE(y_pred, y_true, eps=1e-10, padded_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias=bias)
    if bias:
        nn.init.constant_(m.bias, 0)
    return m 


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return (1.0 - x) / (1.0 - warmup)


def warmup_fix(step, warmup_step):
    return min(1.0, step / warmup_step)


def inverse_square_root(step, warmup_step):
    weight = 1 / math.sqrt(max(step, warmup_step))
    return weight 


def get_lr_schedule_fn(lr_scheduler_name):
    SCHEDULES = {
        'warmup_cosine': warmup_cosine,
        'warmup_constant': warmup_constant,
        'warmup_linear': warmup_linear,
        'warmup_fix': warmup_fix,
        'inverse_square_root': inverse_square_root
    }
    if lr_scheduler_name not in SCHEDULES:
        return SCHEDULES['warmup_fix']
    return SCHEDULES[lr_scheduler_name]