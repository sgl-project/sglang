"""Experimental FN H2560 router + NVFP4 quantization, based on Q35 R11.

Experimental default-off service binding. Native shared/draft computation stays
outside this candidate. Kernel micro parity passed; serving gates are pending.
"""
import ctypes
from functools import cache
from pathlib import Path
from typing import NamedTuple

import torch

from sglang.kernels.ops.moe import q38_router_jit as jit
from sglang.kernels.ops.moe.q38_moe_aot import q38_arch_name

_SOURCE = Path(__file__).resolve().parents[2] / "jit/csrc/moe/q38_router_front.cu"

HIDDEN, EXPERTS, THREADS, CTAS = 2560, 512, 256, 64
RESOURCE_RECEIPTS = {}


class Nvfp4RouterInput(NamedTuple):
    """Linear-layout NVFP4 input; original BF16 hidden remains with dispatch."""
    packed: torch.Tensor
    scales: torch.Tensor


class Params(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in (
        "x", "weight", "global_scale", "logits", "topk_weights", "topk_ids",
        "x_fp4", "x_sf", "tile_idx_to_expert_idx", "tile_idx_to_mn_limit",
        "expanded_idx_to_permuted_idx", "permuted_idx_to_expanded_idx",
        "total_num_padded_tokens", "num_non_exiting_tiles", "arrival", "trace",
    )] + [("num_tokens", ctypes.c_uint32), ("tile_log2", ctypes.c_uint32),
          ("mode", ctypes.c_uint32), ("sh_weight", ctypes.c_void_p),
          ("sh_in_scale", ctypes.c_void_p), ("sh_w_scale", ctypes.c_void_p),
          ("sh_out", ctypes.c_void_p), ("sh_n", ctypes.c_uint32)]


@cache
def kernel(device, capacity, ksplit):
    major, minor = torch.cuda.get_device_capability(device)
    arch = q38_arch_name((major, minor))
    if ksplit not in (2, 4, 8):
        raise ValueError("ksplit must be 2, 4 or 8")
    cubin = jit.compile_source(
        _SOURCE.read_text(),
        f"q38_router_h2560_m{capacity}_k{ksplit}", f"sm_{arch[2:]}",
        ("-std=c++17", "-default-device", f"-DQ35_ROUTER_MAX_M={capacity}",
         f"-DQ35_ROUTER_KSPLIT={ksplit}"))
    shared_bytes = (8 + capacity) * (HIDDEN * 2 + 16) + ksplit * 16 * 8 * 4 + 16
    loaded = jit.load(cubin, "q38_router_gemm_quant", (THREADS, 1, 1), shared_bytes)
    cu = jit.driver()
    attrs = cu.CUfunction_attribute
    static, = jit.check(cu.cuFuncGetAttribute(
        attrs.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, loaded.function), "static shared bytes")
    allowed, = jit.check(cu.cuFuncGetAttribute(
        attrs.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, loaded.function), "max dynamic shared bytes")
    # The H2560/M1 dynamic region is <48KiB, but together with static top-k
    # scratch it exceeds the default limit. R11's H4096 helper never hit this
    # case: its dynamic region alone required opt-in.
    if shared_bytes > allowed:
        jit.check(cu.cuFuncSetAttribute(loaded.function,
            attrs.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes),
            "FN dynamic shared opt-in")
    final, = jit.check(cu.cuFuncGetAttribute(
        attrs.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, loaded.function), "final dynamic shared bytes")
    RESOURCE_RECEIPTS[f"m{capacity}_k{ksplit}"] = {
        "static_shared_bytes": static, "requested_dynamic_bytes": shared_bytes,
        "allowed_dynamic_before": allowed, "allowed_dynamic_after": final}
    return loaded


def router_gemm_quant(x, weight, global_scale, ksplit=2):
    m = x.shape[0]
    if not 1 <= m <= 16:
        raise ValueError("row count must be in 1..16")
    for tensor, shape, dtype in (
        (x, (m, HIDDEN), torch.bfloat16),
        (weight, (EXPERTS, HIDDEN), torch.bfloat16),
        (global_scale, (1,), torch.float32),
    ):
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous() or tensor.device != x.device:
            raise ValueError(f"wrong input contract: {tensor.shape}/{tensor.dtype}/{tensor.device}")
    logits = torch.empty((m, EXPERTS), dtype=torch.bfloat16, device=x.device)
    quantized = torch.empty((m, HIDDEN // 2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((m, HIDDEN // 16), dtype=torch.uint8, device=x.device)
    params = Params(x=x.data_ptr(), weight=weight.data_ptr(), global_scale=global_scale.data_ptr(),
                    logits=logits.data_ptr(), x_fp4=quantized.data_ptr(), x_sf=scales.data_ptr(),
                    num_tokens=m, mode=1)
    capacity = 1 if m == 1 else 8 if m <= 8 else 16
    jit.launch(kernel(x.device.index, capacity, ksplit), (CTAS, 1, 1), [params],
               torch.cuda.current_stream().cuda_stream)
    return logits, quantized, scales
