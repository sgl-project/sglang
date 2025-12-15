from typing import Any, Union

import torch

from sglang.srt.utils.common import is_cuda, is_npu

if is_cuda():
    from sgl_kernel import weak_ref_tensor
elif is_npu():
    from torch_npu._C import _weak_ref_tensor as weak_ref_tensor
else:
    raise NotImplementedError("weak_ref_tensor is implemented only for CUDA and NPU.")


def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]]
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")
