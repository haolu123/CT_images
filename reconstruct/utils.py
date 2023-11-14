import torch
import torch.nn as nn
import torchsort

def spearman_hard(traget1, pred):
    pred = get_rank(pred, [i for i in range(pred.shape[1])]).float()
    traget1 = get_rank(traget1, [i for i in range(traget1.shape[1])]).float()
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    traget1 = traget1 - traget1.mean()
    traget1 = traget1 / traget1.norm()
    return (pred * traget1).sum()

def weighted_spearmanr(pred, target, target_sort, k=2, s=5, **kw):
    pred_rank = torchsort.soft_rank(pred, **kw)
    target_rank = torchsort.soft_rank(target, **kw)
    
    target_sort = target_sort.flatten()
    W = torch.sqrt( (target_sort[pred_rank.flatten().long()-1] - target_sort[target_rank.flatten().long()-1]) ** 2 ).requires_grad_(False)
    # w = torch.tensor([0,0,0,0,19])
    pred_rank = pred_rank - pred_rank.mean()
    pred_rank = pred_rank / pred_rank.norm()
    target_rank = target_rank - target_rank.mean()
    target_rank = target_rank / target_rank.norm()
    
    W = 1/(1 + torch.exp(-k*(W-s)))

    return (pred_rank * target_rank * W).sum(), W

def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

def get_rank(x, indices):
   vals = x[list(range(len(x))), indices]
   return (x < vals[:, None]).long().sum(1)

def spearman_hard_eval(traget1, pred):
    pred = get_rank_eval(pred).float()
    traget1 = get_rank_eval(traget1).float()
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    traget1 = traget1 - traget1.mean()
    traget1 = traget1 / traget1.norm()
    return (pred * traget1).sum()

def get_rank_eval(x):
   vals = x[0, :]
   return (x < vals[:, None]).long().sum(1)