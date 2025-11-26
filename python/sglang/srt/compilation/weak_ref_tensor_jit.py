import os

import torch
from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))

load(
    name="weak_ref_tensor_ext",
    sources=[f"{_abs_path}/weak_ref_tensor.cpp"],
    extra_cflags=["-O3"],
)

x = torch.arange(12, device="cuda").reshape(3, 4)
y = torch.ops.jit_weak_ref_tensor.weak_ref_tensor(x)
print("alias:", x.data_ptr() == y.data_ptr())
