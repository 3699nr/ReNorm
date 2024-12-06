# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from typing import List

import torch
from torch.optim.lr_scheduler import *


def bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < a[mid]: hi = mid
        else: lo = mid+1
    return lo



class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch) ## 二分查找返回右索引+1。当self.last_epoch >= warmup_iters，warmup_factor=1,然后lr在最原始的基础上x0.1
            for base_lr in self.base_lrs
        ]
    





class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            warmup_factor: float = 0.1,
            warmup_epochs: int = 10,
            warmup_method: str = "linear",
            last_epoch: int = -1,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_epoch(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )
        return [
            base_lr 
            * warmup_factor 
            ** bisect_right(self.milestones, self.last_epoch) for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


def _get_warmup_factor_at_epoch(
        method: str, epoch: int, warmup_epochs: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        epoch (int): epoch at which to calculate the warmup factor.
        warmup_epochs (int): the number of warmup epochs.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if epoch >= warmup_epochs:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = epoch / warmup_epochs
        return warmup_factor * (1 - alpha) + alpha
    elif method == "exp":
        return warmup_factor ** (1 - epoch / warmup_epochs)
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
