import torch
import torch_npu


def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
    out = torch_npu.npu_swiglu(x)
    return out
