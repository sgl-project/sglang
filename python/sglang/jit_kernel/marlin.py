from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

import torch
from sgl_kernel.scalar_type import ScalarType

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CPP_ENTRY = "gptq_marlin_gemm"
_PY_SYMBOL = "gptq_marlin_gemm"


@functools.cache
def _jit_gptq_marlin_module(
    *,
    a_dtype: torch.dtype,
    b_q_type_id: int,
    has_act_order: bool,
    has_zp: bool,
    is_zp_float: bool,
) -> Module:
    args = make_cpp_args(
        int(a_dtype == torch.bfloat16),
        b_q_type_id,
        int(has_act_order),
        int(has_zp),
        int(is_zp_float),
    )

    return load_jit(
        "gptq_marlin",
        *args,
        cuda_files=[
            "marlin/gptq_marlin_dispatcher.cuh",
            "marlin/gptq_marlin_kernel.cuh",
            "marlin/gptq_marlin.cuh",
        ],
        cuda_wrappers=[(_PY_SYMBOL, _CPP_ENTRY)],
    )


def gptq_marlin_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    if a.device.type != "cuda":
        raise ValueError("a must be CUDA tensor")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a must be fp16 or bf16")
    if c is None:
        c = torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)
    else:
        if c.shape != (size_m, size_n) or c.dtype != a.dtype or c.device != a.device:
            raise ValueError("c shape/dtype/device mismatch")

    # make tmp for GPTQ Quantization
    has_act_order = (g_idx is not None) and (perm is not None)
    has_zp = (b_zeros is not None) and (b_zeros.numel() > 0)

    if has_act_order:
        # It means has_act_order
        a_tmp = torch.empty((size_m, size_k), device=a.device, dtype=a.dtype)
    else:
        a_tmp = torch.empty((0), device=a.device, dtype=a.dtype)

    if use_fp32_reduce:
        max_m_block_size = min((size_m + 16 - 1) // 16 * 16, 64)
        sms = torch.cuda.get_device_properties(c.device).multi_processor_count
        # 256 for max_thread_n
        max_c_tmp_size = sms * max_m_block_size * 256
        c_tmp = torch.empty((max_c_tmp_size), device=c.device, dtype=torch.float32)
    else:
        c_tmp = torch.empty((0), device=c.device, dtype=torch.float32)

    # default set some optional params
    if global_scale is None:
        global_scale = torch.empty((0), device=a.device, dtype=a.dtype)
    if b_zeros is None:
        zp_dtype = a.dtype if is_zp_float else torch.int32
        b_zeros = torch.empty((0), device=a.device, dtype=zp_dtype)
    if g_idx is None:
        g_idx = torch.empty((0), device=a.device, dtype=torch.int32)
    if perm is None:
        perm = torch.empty((0), device=a.device, dtype=torch.int32)

    module = _jit_gptq_marlin_module(
        a_dtype=a.dtype,
        b_q_type_id=b_q_type.id,
        has_act_order=has_act_order,
        has_zp=has_zp,
        is_zp_float=is_zp_float,
    )

    module.gptq_marlin_gemm(
        a,
        a_tmp,
        c,
        c_tmp,
        b_q_weight,
        b_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        int(b_q_type.id),
        int(size_m),
        int(size_n),
        int(size_k),
        bool(is_k_full),
        bool(use_atomic_add),
        bool(use_fp32_reduce),
        bool(is_zp_float),
    )
    return c
