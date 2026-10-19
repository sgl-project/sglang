# SPDX-License-Identifier: Apache-2.0
"""Native Ascend RMSNorm for the MiniMax H3 transformer."""

import torch
from torch import nn


class MiniMaxH3RMSNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.device.type == "npu"
            and x.dtype in (torch.float16, torch.bfloat16)
            and self.weight is not None
            and self.weight.device == x.device
            and self.weight.dtype == x.dtype
            and x.numel() > 0
            and hasattr(torch.ops.npu, "npu_rms_norm")
        ):
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            return torch.ops.npu.npu_rms_norm(x, self.weight, eps)[0]
        return super().forward(x)
