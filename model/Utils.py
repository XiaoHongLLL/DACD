import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformer.Layers import get_non_pad_mask

def softplus(x, beta):

    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

def log_likelihood(model, event_time, time_gap, event_type):
    pass
    non_pad_mask = get_non_pad_mask(event_type)
    return 0, 0

def type_loss(prediction, types, loss_func):
    pass
    truth = types[:, 1:] - 1

    if prediction.ndim == 4:
        truth = truth.unsqueeze(2).expand(-1, -1, prediction.size(2))
        loss = loss_func(prediction.reshape(-1, prediction.size(-1)), truth.reshape(-1))
        loss = loss.view(truth.size())
        loss = loss.mean(2)
    else:
        loss = loss_func(prediction.reshape(-1, prediction.size(-1)), truth.reshape(-1))
        loss = loss.view(truth.size())

    return torch.sum(loss)

def evaluate_samples(t_sample, gt_t, type_sample, event_type, opt):
    pass


    if opt.normalize == 'log':
        clamp_val = float(getattr(opt, 'clamp_threshold', 6.0))
        t_sample = torch.clamp(t_sample, min=-clamp_val, max=clamp_val)


    if opt.normalize == 'log':


        gt_log_arg = torch.clamp(gt_t * opt.var_log_data + opt.mean_log_data, min=-80.0, max=80.0)
        sample_log_arg = torch.clamp(t_sample * opt.var_log_data + opt.mean_log_data, min=-80.0, max=80.0)
        gt_t_real = torch.exp(gt_log_arg)
        t_sample_real = torch.exp(sample_log_arg)
        if getattr(opt, 'eval_time_scale', 'legacy') == 'physical':
            gt_t_real = gt_t_real * opt.mean_data
            t_sample_real = t_sample_real * opt.mean_data

    elif opt.normalize == 'normal':
        gt_t_real = gt_t * opt.mean_data
        t_sample_real = t_sample * opt.mean_data
    else:
        gt_t_real = gt_t
        t_sample_real = t_sample


    t_sample_real = torch.clamp(t_sample_real - 0.5, min=0.0)


    gt_t_real = torch.clamp(gt_t_real - 0.5, min=0.0)


    if getattr(opt, 'eval_time_scale', 'legacy') == 'physical':
        max_val = getattr(opt, 'time_raw_max', getattr(opt, 'time_max', 1e7)) * 5.0
    else:
        max_val = getattr(opt, 'time_rel_max', getattr(opt, 'time_max', 1e7)) * 5.0
    t_sample_real = torch.clamp(t_sample_real, max=max_val)


    non_pad_mask = get_non_pad_mask(event_type)
    valid_mask = non_pad_mask[:, 1:].squeeze(2)


    target_quantiles = opt.eval_quantile
    t_pred_quantiles = torch.quantile(t_sample_real, target_quantiles, dim=-1)


    hits_all = (gt_t_real.unsqueeze(0) <= t_pred_quantiles) * valid_mask.unsqueeze(0)
    batch_hit_counts = hits_all.sum(dim=(1, 2))


    t_median = torch.quantile(t_sample_real, 0.5, dim=-1)
    batch_il_sum = (t_median * valid_mask).sum().item()


    num_samples = t_sample_real.size(-1)
    term1 = torch.abs(t_sample_real - gt_t_real.unsqueeze(-1)).mean(dim=-1)

    if num_samples > 100:

        t_mean = t_sample_real.mean(dim=-1, keepdim=True)
        term2 = torch.abs(t_sample_real - t_mean).mean(dim=-1) * 2
    else:

        t_sample_perm = t_sample_real[:, :, torch.randperm(num_samples)]
        term2 = torch.abs(t_sample_real - t_sample_perm).mean(dim=-1)

    crps_map = term1 - 0.5 * term2
    batch_crps_sum = (crps_map * valid_mask).sum().item()


    truth = event_type[:, 1:] - 1

    if isinstance(type_sample, tuple) or (isinstance(type_sample, torch.Tensor) and type_sample.ndim > 2):
        type_pred = type_sample if isinstance(type_sample, torch.Tensor) else type_sample[0]
        if type_pred.ndim > 2:
             type_pred = type_pred.mode(dim=-1).values
    else:
        type_pred = type_sample

    batch_correct_type = (type_pred.eq(truth) * valid_mask).sum().item()


    se = (t_median - gt_t_real) ** 2
    batch_sse_sum = (se * valid_mask).sum().item()


    batch_total_events = valid_mask.sum().item()

    return {
        'hit_counts': batch_hit_counts,
        'il_sum': batch_il_sum,
        'crps_sum': batch_crps_sum,
        'correct_type': batch_correct_type,
        'total_events': batch_total_events,
        'sse_sum': batch_sse_sum
    }

class LabelSmoothingLoss(nn.Module):
    pass
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 <= label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        pass
        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)

        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss.sum()
