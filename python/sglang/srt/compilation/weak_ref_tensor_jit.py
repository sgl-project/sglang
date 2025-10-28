import os
import site

import torch
from torch.utils.cpp_extension import load

from sglang.srt.utils import is_npu

_abs_path = os.path.dirname(os.path.abspath(__file__))

if is_npu():
    site_packs_dir = site.getsitepackages()[0]
    load(
        name="weak_ref_tensor_npu_ext",
        sources=[f"{_abs_path}/weak_ref_tensor_npu.cpp"],
        extra_cflags=["-O3"],
        extra_ldflags=[
            f"-L/{os.path.join(site_packs_dir, 'torch_npu/lib/')}",
            "-ltorch_npu",
        ],
        extra_include_paths=[os.path.join(site_packs_dir, "torch_npu/include/")],
    )

    x = torch.arange(12, device="npu").reshape(3, 4)
    y = torch.ops.jit_weak_ref_tensor_npu.weak_ref_tensor(x)
else:
    load(
        name="weak_ref_tensor_ext",
        sources=[f"{_abs_path}/weak_ref_tensor.cpp"],
        extra_cflags=["-O3"],
    )
    x = torch.arange(12, device="cuda").reshape(3, 4)
    y = torch.ops.jit_weak_ref_tensor.weak_ref_tensor(x)

print("alias:", x.data_ptr() == y.data_ptr())
