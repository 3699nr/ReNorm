# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.layers import (
    IBN,
    SELayer,
    get_norm,
)
# from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fastreid.utils import comm


logger = logging.getLogger(__name__)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, in_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn11 = get_norm("RN", planes)

        self.in11 = get_norm("EN", planes)
        self.in12 = get_norm("EN", planes)
        self.in13 = get_norm("EN", planes)        

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        
        self.bn21 = get_norm("RN", planes)
        
        self.in21 = get_norm("EN", planes)
        self.in22 = get_norm("EN", planes)
        self.in23 = get_norm("EN", planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        
        self.bn31 = get_norm("RN", planes * self.expansion)
        
        self.in31 = get_norm("EN", planes * self.expansion)
        self.in32 = get_norm("EN", planes * self.expansion)
        self.in33 = get_norm("EN", planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, cur_idx, use_EN):
        identity = x

        if use_EN == False:
            out = self.conv1(x)
            out = self.bn11(out, cur_idx)            
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn21(out, cur_idx)                
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn31(out, cur_idx)

            if self.downsample is not None:

                identity = self.downsample[0](x)
                identity = self.downsample[1](identity, cur_idx)
                    

        else:
            out = self.conv1(x)
            if cur_idx == 0:
                out = self.in11(out, self.bn11.running_mean_1, self.bn11.running_var_1)

            elif cur_idx == 1:
                out = self.in12(out, self.bn11.running_mean_2, self.bn11.running_var_2)

            elif cur_idx == 2:
                out = self.in13(out, self.bn11.running_mean_3, self.bn11.running_var_3)
                
            out = self.relu(out)
            out = self.conv2(out)

            if cur_idx == 0:
                out = self.in21(out, self.bn21.running_mean_1, self.bn21.running_var_1)
                
            elif cur_idx == 1:
                out = self.in22(out, self.bn21.running_mean_2, self.bn21.running_var_3)
                
            elif cur_idx == 2:
                out = self.in23(out, self.bn21.running_mean_3, self.bn21.running_var_3)
                
            out = self.relu(out)
            out = self.conv3(out)

            if cur_idx == 0:
                out = self.in31(out, self.bn31.running_mean_1, self.bn31.running_var_1)
                
            elif cur_idx == 1:
                out = self.in32(out, self.bn31.running_mean_2, self.bn31.running_var_2)
                
            elif cur_idx == 2:
                out = self.in33(out, self.bn31.running_mean_3, self.bn31.running_var_3)
                
            if self.downsample is not None:         
                identity = self.downsample[0](x)

                if cur_idx == 0:
                    identity = self.downsample[2](identity, self.downsample[1].running_mean_1, self.downsample[1].running_var_1)
                    
                elif cur_idx == 1:
                    identity = self.downsample[3](identity, self.downsample[1].running_mean_2, self.downsample[1].running_var_2)
                    
                elif cur_idx == 2:
                    identity = self.downsample[4](identity, self.downsample[1].running_mean_3, self.downsample[1].running_var_3)

        out += identity
        out = self.relu(out)
        return out
    


class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block,  layers, non_layers):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se) 
        self.layer10 = layer1[0]
        self.layer11 = layer1[1]
        self.layer12 = layer1[2]

        layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer20 = layer2[0]
        self.layer21 = layer2[1]
        self.layer22 = layer2[2]
        self.layer23 = layer2[3]

        layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer30 = layer3[0]
        self.layer31 = layer3[1]
        self.layer32 = layer3[2]
        self.layer33 = layer3[3]
        self.layer34 = layer3[4]
        self.layer35 = layer3[5]

        layer4 = self._make_layer(block, 512, layers[3], 2, bn_norm, with_se=with_se)
        self.layer40 = layer4[0]
        self.layer41 = layer4[1]
        self.layer42 = layer4[2]


        self.random_init()



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, planes, blocks, stride=1, dn_norm="ReNorm", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm("RN", planes * block.expansion),

                get_norm("EN", planes * block.expansion),
                get_norm("EN", planes * block.expansion),
                get_norm("EN", planes * block.expansion),                
            )

        layers = []
        layers.append(block(self.inplanes, planes, dn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dn_norm, with_ibn, with_se))

        # return nn.Sequential(*layers)
        return layers

    

    def _forward_impl(self, x, cur_idx, use_EN):


        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        ## layer 1
        x = self.layer10(x, cur_idx, use_EN)
        x = self.layer11(x, cur_idx, use_EN)
        x = self.layer12(x, cur_idx, use_EN)
        ## layer 2
        x = self.layer20(x, cur_idx, use_EN)
        x = self.layer21(x, cur_idx, use_EN)
        x = self.layer22(x, cur_idx, use_EN)
        x = self.layer23(x, cur_idx, use_EN)
        ## layer 3
        x = self.layer30(x, cur_idx, use_EN)  
        x = self.layer31(x, cur_idx, use_EN)     
        x = self.layer32(x, cur_idx, use_EN)
        x = self.layer33(x, cur_idx, use_EN)
        x = self.layer34(x, cur_idx, use_EN)
        x = self.layer35(x, cur_idx, use_EN)
        ## layer 4
        x = self.layer40(x, cur_idx, use_EN)
        x = self.layer41(x, cur_idx, use_EN)
        x = self.layer42(x, cur_idx, use_EN)

        return x

    def forward(self, x, cur_idx, use_EN):
        return self._forward_impl(x, cur_idx, use_EN)
    

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
####### 
def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    # pretrain_dict = model_zoo.load_url(model_url)
    state_dict = torch.load(model_url) #, map_location='cuda:0')
    # state_dict = remove_module_key(state_dict)

    model_dict_org = model.state_dict()
    # pretrainedName = set()
    # modelName = set()
    # for k,v in state_dict.items():
    #     pretrainedName.update()
    pretrain_dict = {k: v for k, v in state_dict.items() if k in model_dict_org and model_dict_org[k].size() == v.size()}
    missing_dict = {k: v for k, v in model_dict_org.items() if( k not in state_dict) or (model_dict_org[k].size() != v.size())}

    res = dict()
    
    for k,vv in missing_dict.items():
        name_list = k.split('.')         
        if '.num_batches_tracked' in k or 'classifier' in k or 'bnneck' in k or 'mean_' in k or 'sim_f' in k or 'var_' in k or 'aff_' in k:
            res.update({k:vv})
            
        elif 'in11' in k or 'in12' in k or 'in13' in k or 'in21' in k or'in22' in k or 'in23' in k or 'in31' in k or 'in32' in k or 'in33' in k:
            res.update({k: state_dict[name_list[0][: -2]+ str(int(name_list[0][-2: -1]))+'.'+name_list[0][-1]+'.b'+name_list[1][1:3]+'.'+name_list[2]]})

        elif 'layer' in k:
            if 'bn11' in k or 'bn12' in k or 'bn13' in k or 'bn21' in k or'bn22' in k or 'bn23' in k or 'bn31' in k or 'bn32' in k or 'bn33' in k:
                res.update({k: state_dict[name_list[0][: -1]+'.'+name_list[0][-1]+'.'+name_list[1][:3]+'.'+name_list[2]]})
            elif 'bn1' in k or 'bn2' in k or 'bn3' in k:
                res.update({k: state_dict[name_list[0][: -1]+'.'+name_list[0][-1]+'.'+name_list[1]+'.'+name_list[2]]})
            elif 'downsample' in k:
                if name_list[2] == '0':
                    if '5' in k:
                        res.update({k: state_dict['layer3'+'.'+'0'+'.'+'downsample'+'.'+'0'+'.'+name_list[-1]]})
                    elif '6' in k:
                        res.update({k: state_dict['layer4'+'.'+'0'+'.'+'downsample'+'.'+'0'+'.'+name_list[-1]]})
                    else:
                        res.update({k: state_dict[name_list[0][: -1]+'.'+'0'+'.'+'downsample'+'.'+'0'+'.'+name_list[-1]]}) 
                else:
                    res.update({k : state_dict[name_list[0][: -1]+'.'+'0'+'.'+'downsample'+'.'+'1'+'.'+name_list[-1]]})
            else:
                if '5' in k:
                    res.update({k : state_dict['layer3'+'.'+name_list[0][-1]+'.'+name_list[1]+'.' + name_list[-1]]})
                elif '6' in k:
                    res.update({k : state_dict['layer4'+'.'+name_list[0][-1]+'.'+name_list[1]+'.' + name_list[-1]]})
                else:
                    res.update({k : state_dict[name_list[0][: -1]+'.'+name_list[0][-1]+'.'+name_list[1]+'.' + name_list[-1]]})

        elif 'bn1' in k:
            res.update({k : state_dict['bn1' + '.' + name_list[1]]})

    res.update(pretrain_dict)

    res_missing_dict = dict()
    for k,v in model_dict_org.items():
        if k not in res:
            res_missing_dict.update({k : v})
    model.load_state_dict(res)
    print('Initialized with pretrained weights from:{}'.format(model_url))
    # print('Missing pretrained weights:\n', res_missing_dict.keys())



@BACKBONE_REGISTRY.register()
def build_renorm_backbone(cfg):


    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]


    model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, Bottleneck,
                   num_blocks_per_stage, nl_layers_per_stage)
    
    if pretrain:

        init_pretrained_weights(model, pretrain_path)
    

    model.layer40.conv2.stride = (1,1)      
    model.layer40.downsample[0].stride = (1,1)

    return model
