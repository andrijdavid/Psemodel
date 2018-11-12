import torch.nn as nn
import torch.nn.functional as F
import torch

class PositionalAveragePooling(nn.Module):
    def __init__(self, kernel_size=1, stride=None, padding=0, ceil_mode=False,  count_include_pad=True):
        super(PositionalAveragePooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def extra_repr(self):
        return 'kernel_size={}, stride={}, padding={}'.format(
            self.kernel_size, self.stride, self.padding
        )

    def forward(self, x):
        dim = x.shape[1]
        x = F.avg_pool2d(x, self.kernel_size, self.stride,
                                  self.padding, self.ceil_mode, self.count_include_pad)
        x = torch.sum(x, 1, keepdim=True)/dim
        return x   