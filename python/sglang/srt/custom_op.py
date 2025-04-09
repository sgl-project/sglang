from typing import Optional

import torch
from torch import nn

from sglang.srt.utils import is_cuda, is_hip, is_hpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_hpu = is_hpu()


class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        if _is_cuda:
            return self.forward_cuda
        elif _is_hip:
            return self.forward_hip
        elif _is_hpu:
            return self.forward_hpu
        else:
            return self.forward_native
