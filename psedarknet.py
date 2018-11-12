import torch.nn as nn
from .se import *
from .pap import *
from .layers import *

class Darknet(nn.Module):
    "https://github.com/pjreddie/darknet"
    def make_group_layer(self, ch_in:int, num_blocks:int, stride:int=1):
        "starts with conv layer - `ch_in` channels in - then has `num_blocks` `ResLayer`"
        return [conv_layer(ch_in, ch_in*2,stride=stride)
               ] + [(PSEBasicBlock(ch_in*2)) for i in range(num_blocks)]

    def __init__(self, num_blocks:Collection[int], num_classes:int, nf=32):
        "create darknet with `nf` and `num_blocks` layers"
        super().__init__()
        layers = [conv_layer(3, nf, ks=3, stride=1)]
        for i,nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2-(i==1))
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nf, num_classes)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return self.layers(x)

def dark_small(): return Darknet([1,2,4,4,3], 28, 32)
def dark_53(): return Darknet([1,2,8,8,4], 28, 32)