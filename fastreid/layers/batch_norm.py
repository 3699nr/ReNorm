# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

# try:
#     from apex import parallel
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run model with syncBN")

__all__ = ["IBN", "get_norm"]


class DomainFreeze2d_share(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True,
                 last_gamma=False):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.last_gamma = last_gamma

        self.weight = nn.Parameter(torch.ones(num_features),requires_grad=False)  
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=False)

        
        self.mean_weight = nn.Parameter(torch.ones(2),requires_grad=False) 

        self.softmax = nn.Softmax(0)

        self.register_buffer('running_mean_1', torch.zeros(num_features))  
        self.register_buffer('running_var_1', torch.zeros(num_features))
        self.register_buffer('running_mean_2', torch.zeros(num_features))  
        self.register_buffer('running_var_2', torch.zeros(num_features))
        self.register_buffer('running_mean_3', torch.zeros(num_features))  
        self.register_buffer('running_var_3', torch.zeros(num_features))
        self.use_train = True
        self.reset_parameters()

    def set_eval(self):
        self.use_train = False
        self.weight.requires_grad=False
        self.bias.requires_grad=False
        self.mean_weight.requires_grad=False
        # self.var_weight.requires_grad=False
        
    def set_train(self):
        self.use_train = True
        self.weight.requires_grad_(True)
        self.bias.requires_grad_(True)
        self.mean_weight.requires_grad_(True)
        # self.var_weight.requires_grad_(True)

    def reset_parameters(self):
        self.running_mean_1.zero_()  
        self.running_mean_2.zero_() 
        self.running_mean_3.zero_() 
        self.running_var_1.zero_()
        self.running_var_2.zero_()
        self.running_var_3.zero_()

        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x, cur_idx):
        self._check_input_dim(x)   # input : [N, C, H, W]

        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        mean_in = x.mean(-1)    ## IN, [N, C]
        var_in = x.var(-1)
        temp = var_in + mean_in ** 2

        if self.training and self.use_train:
            mean_bn = mean_in.mean(0)  
            #var_bn = temp.mean(0) - mean_bn ** 2
            var_bn = x.var(axis=(0, 2))
            if self.using_moving_average:
                if cur_idx == 0:
                    self.running_mean_1.mul_(self.momentum)
                    self.running_mean_1.add_((1 - self.momentum) * mean_bn.data)   
                    self.running_var_1.mul_(self.momentum)
                    self.running_var_1.add_((1 - self.momentum) * var_bn.data)
                elif cur_idx == 1:
                    self.running_mean_2.mul_(self.momentum)
                    self.running_mean_2.add_((1 - self.momentum) * mean_bn.data)   
                    self.running_var_2.mul_(self.momentum)
                    self.running_var_2.add_((1 - self.momentum) * var_bn.data)
                elif cur_idx == 2:
                    self.running_mean_3.mul_(self.momentum)
                    self.running_mean_3.add_((1 - self.momentum) * mean_bn.data)   
                    self.running_var_3.mul_(self.momentum)
                    self.running_var_3.add_((1 - self.momentum) * var_bn.data)
            else:
                self.running_mean_1.add_(mean_bn.data)
                self.running_var_1.add_(mean_bn.data ** 2 + var_bn.data)
        else:
            if cur_idx == 0:
                mean_bn = torch.autograd.Variable(self.running_mean_1, requires_grad=False)  
                var_bn = torch.autograd.Variable(self.running_var_1, requires_grad=False)
            elif cur_idx == 1:
                mean_bn = torch.autograd.Variable(self.running_mean_2, requires_grad=False) 
                var_bn = torch.autograd.Variable(self.running_var_2, requires_grad=False)
            elif cur_idx == 2:
                mean_bn = torch.autograd.Variable(self.running_mean_3, requires_grad=False)  
                var_bn = torch.autograd.Variable(self.running_var_3, requires_grad=False)
            else:
                mean_bn = (self.running_mean_3 + self.running_mean_2 + self.running_mean_1) / 3
                var_bn = (self.running_var_3 + self.running_var_2 + self.running_var_1) / 3


        if self.training and self.use_train:
            mean_weight = self.softmax(self.mean_weight) 
            # var_weight = self.softmax(self.var_weight)

            mean = mean_weight[0] * mean_in.unsqueeze(-1) + mean_weight[1] * mean_bn.reshape(1, -1, 1)
            var = mean_weight[0] * var_in.unsqueeze(-1) + mean_weight[1] * var_bn.reshape(1, -1, 1)
            x = (x-mean) / (var+self.eps).sqrt()
            x = x.view(N, C, H, W)
            return x * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)
        else:

            weight_1 = torch.autograd.Variable(self.weight, requires_grad=False)
            bias_1 = torch.autograd.Variable(self.bias, requires_grad=False)
            mean_weight_1 = self.softmax(torch.autograd.Variable(self.mean_weight, requires_grad=False))
            mean = mean_weight_1[0] * mean_in.unsqueeze(-1) + mean_weight_1[1] * mean_bn.reshape(1, -1, 1)
            var = mean_weight_1[0] * var_in.unsqueeze(-1) + mean_weight_1[1] * var_bn.reshape(1, -1, 1)
            x = (x-mean) / (var+self.eps).sqrt()
            x = x.view(N, C, H, W)
            return x * weight_1.reshape(1, -1, 1, 1) + bias_1.reshape(1, -1, 1, 1)
        

class EmulationNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.num_features = num_features
        self.aff_mean_weight = nn.Parameter(torch.ones(2),requires_grad=True)
        self.aff_var_weight = nn.Parameter(torch.ones(2),requires_grad=True)
        self.softmax = nn.Softmax(0)
        # self.weight = nn.Parameter(torch.ones(num_features),requires_grad=False)  
        # self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=False)
        self.softmax = nn.Softmax(0)

    def forward(self, x, mean, var):
       # input : [N, C, H, W]
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1)    ## IN, [N, C]
        var_in = x.var(-1)

        aff_mean = self.softmax(self.aff_mean_weight)
        aff_var = self.softmax(self.aff_var_weight)

        mean_1 = aff_mean[0]*mean_in.unsqueeze(-1) + aff_mean[1]*mean.reshape(1, -1, 1)
        var_1 = aff_var[0]*var_in.unsqueeze(-1) + aff_var[1]*var.reshape(1, -1, 1)
        x_ = (x-mean_1) / (var_1+self.eps).sqrt()
        x_ = x_.view(N, C, H, W)
        return x_

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0, **kwargs):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class SyncBatchNorm(nn.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: nn.init.constant_(self.weight, weight_init)
        if bias_init is not None: nn.init.constant_(self.bias, bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


class IBN(nn.Module):
    def __init__(self, planes, bn_norm, **kwargs):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, **kwargs)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits=1, **kwargs):
        super().__init__(num_features, **kwargs)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            self.running_mean = self.running_mean.repeat(self.num_splits)
            self.running_var = self.running_var.repeat(self.num_splits)
            outputs = F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0)
            return outputs
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


class FrozenBatchNorm(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.
    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.
    Other pre-trained backbone models may contain all 4 parameters.
    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        if version is not None and version < 3:
            logger = logging.getLogger(__name__)
            logger.info("FrozenBatchNorm {} is upgraded to version 3.".format(prefix.rstrip(".")))
            # In version < 3, running_var are used without +eps.
            state_dict[prefix + "running_var"] -= self.eps

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.
        Args:
            module (torch.nn.Module):
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.
        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res



def get_norm(norm, out_channels, **kwargs):
    """
    Args:
        norm (str or callable): either one of BN, GhostBN, FrozenBN, GN or SyncBN;
            or a callable that thakes a channel number and returns
            the normalization layer as a nn.Module
        out_channels: number of channels for normalization layer

    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "RN": DomainFreeze2d_share,
            'EN': EmulationNorm, 
            "BN": BatchNorm,
            "syncBN": SyncBatchNorm,
            "GhostBN": GhostBatchNorm,
            "FrozenBN": FrozenBatchNorm,
            "GN": lambda channels, **args: nn.GroupNorm(32, channels),
        }[norm]
    return norm(out_channels, **kwargs)
