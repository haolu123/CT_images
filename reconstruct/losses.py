import torch
import torch.nn as nn
import torchsort

class SpearmanLoss(nn.Module):
    """
        Spearman Loss
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, pred: torch.tensor, target: torch.tensor, **kw) -> torch.tensor:
        """
            pred: (B, 1)
            target: (B, 1)
        """
        pred = torchsort.soft_rank(pred, **kw)
        target = torchsort.soft_rank(target, **kw)
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()
        return -(pred * target).sum()

class WeightedSpearmanLoss(nn.Module):
    """
        Spearman Loss
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_rank_eval(self, x):
        vals = x[0, :]
        return (x < vals[:, None]).long().sum(1)
    
    def forward(self, pred: torch.tensor, target: torch.tensor, target_sort: torch.tensor, k=2, s=5, **kw) -> torch.tensor:
        
        pred_rank = torchsort.soft_rank(pred, **kw)
        target_rank = torchsort.soft_rank(target, **kw)
        
        # w = torch.tensor([0,0,0,0,19])
        pred_rank = pred_rank - pred_rank.mean()
        pred_rank = pred_rank / pred_rank.norm()
        target_rank = target_rank - target_rank.mean()
        target_rank = target_rank / target_rank.norm()
        

        pred_rank_hard = self._get_rank_eval(pred)
        target_rank_hard = self._get_rank_eval(target)
        pred_rank_hard = pred_rank_hard.detach()
        target_rank_hard = target_rank_hard.detach()
        target_sort = target_sort.flatten().detach()
        W = torch.sqrt( (target_sort[pred_rank_hard.flatten().long()-1] - target_sort[target_rank_hard.flatten().long()-1]) ** 2 )
        W = 1/(1 + torch.exp(-k*(W-s)))
        W = W / W.sum()*len(W)
        W.requires_grad_(False)
        W_detached = W.detach()
        return -(pred_rank * target_rank * W_detached).sum()

class Bachnorm(nn.Module):
    '''
        Batchnorm layer
    '''

    def __init__(self) -> None:
        super(Bachnorm, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            x: (B, 1)
        '''
        mean = x.mean(dim=(0), keepdim=True)
        std = x.std(dim=(0), keepdim=True)
        return (x - mean) / (std + 1e-5)