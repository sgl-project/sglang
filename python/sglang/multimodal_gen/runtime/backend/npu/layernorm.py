from typing import Optional, Tuple, Union

import torch
import torch_npu


def rms_norm(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if residual is not None:
        out, _, residual_out = torch_npu.npu_add_rms_norm(
            residual, x, self.weight.data, self.variance_epsilon
        )
        return out, residual_out
    return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
