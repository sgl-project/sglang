import os

from torch.utils.cpp_extension import load

_abs_path = os.path.dirname(os.path.abspath(__file__))

load(
    name="weak_ref_tensor_ext",
    sources=[f"{_abs_path}/weak_ref_tensor.cpp"],
    extra_cflags=["-O3"],
)
