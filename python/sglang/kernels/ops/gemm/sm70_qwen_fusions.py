"""Measured small-batch FP16 fusions for Qwen3.8 on Volta."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs
from sglang.kernels.ops.gemm.sm70_dense_gemv import supported as gemv_supported


def enabled() -> bool:
    return envs.SGLANG_SM70_QWEN_FUSIONS.get()


def gate_supported(x, weight, value, bias=None) -> bool:
    rows = x.shape[0] if x.ndim == 2 else 0
    return (
        enabled()
        and x.ndim == 2
        and (
            rows == 1
            or (
                envs.SGLANG_SM70_MTP_SHARED_GATE.get()
                and (
                    rows in (2, 4)
                    or (
                        envs.SGLANG_SM70_BATCHED_FUSIONS.get()
                        and rows in (8, 12, 16, 20, 24, 28, 32)
                    )
                )
            )
        )
        and x.is_contiguous()
        and tuple(weight.shape) == (1, 2560)
        and value.shape == x.shape
        and value.dtype == torch.float16
        and value.device == x.device
        and value.is_contiguous()
        and gemv_supported(x[:1], weight, bias)
    )


def gate_up_supported(x, weight) -> bool:
    return (
        enabled()
        and x.ndim == 2
        and x.shape[0] == 1
        and tuple(weight.shape) == (320, 2560)
        and gemv_supported(x, weight)
    )


def qkv_ba_supported(x, weight, tail) -> bool:
    return (
        enabled()
        and x.ndim == 2
        and x.shape[0] == 1
        and isinstance(weight, torch.Tensor)
        and isinstance(tail, torch.Tensor)
        and tuple(weight.shape) == (4096, 2560)
        and tuple(tail.shape) == (24, 2560)
        and tail.dtype == torch.float16
        and tail.device == x.device
        and tail.is_contiguous()
        and tail.data_ptr() % 16 == 0
        and gemv_supported(x, weight)
    )


def qkvzba_supported(x, weight, tail) -> bool:
    return (
        envs.SGLANG_SM70_MTP_QKVZBA.get()
        and x.ndim == 2
        and (
            x.shape[0] in (2, 4)
            or (envs.SGLANG_SM70_BATCHED_FUSIONS.get() and x.shape[0] == 8)
        )
        and x.is_contiguous()
        and qkv_ba_supported(x[:1], weight, tail)
    )


def reuse_qkv_prefix(projected, query, key, value):
    # This ratio-three GDN layout is already [Q512, K512, V1536, Z1536].
    # For one row its Q/K/V prefix is contiguous despite the parent row stride.
    if (
        enabled()
        and projected.is_cuda
        and projected.dtype == torch.float16
        and projected.shape == (1, 4096)
        and projected.is_contiguous()
        and query.shape == (1, 512)
        and key.shape == (1, 512)
        and value.shape == (1, 1536)
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
        and query.data_ptr() == projected.data_ptr()
        and key.data_ptr() == projected.data_ptr() + 1024
        and value.data_ptr() == projected.data_ptr() + 2048
        and torch.cuda.get_device_capability(projected.device) == (7, 0)
    ):
        return projected[:, :2560]
    return None


@cache_once
def _module():
    return load_jit(
        "sm70_qwen_fusions",
        cuda_files=["elementwise/sm70_qwen_fusions.cuh"],
        cuda_wrappers=[
            ("gate", "sglang::sm70_qwen_fusions::run<512>"),
            ("gate_up", "sglang::sm70_qwen_fusions::gate_up"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _projection_module():
    return load_jit(
        "sm70_qwen_qkv_ba",
        cuda_files=["elementwise/sm70_dense_gemv.cuh"],
        cuda_wrappers=[("qkv_ba", "sglang::sm70_dense_gemv::qkv_ba")],
    )


def gate(x, weight, value):
    output = torch.empty_like(value)
    _module().gate(x, weight, value, output)
    return output


def gate_up(x, weight):
    output = torch.empty((1, 160), dtype=x.dtype, device=x.device)
    _module().gate_up(x, weight, output)
    return output


def qkv_ba(x, weight, tail):
    output = torch.empty((1, 4120), dtype=x.dtype, device=x.device)
    _projection_module().qkv_ba(x, weight, tail, output)
    return output[:, :4096], output[:, 4096:]


@cache_once
def _gdn_module(rows):
    return load_jit(
        "sm70_qwen_gdn",
        rows,
        cuda_files=["elementwise/sm70_qwen_gdn.cuh"],
        cuda_wrappers=[("run", f"sglang::sm70_qwen_gdn::run<{rows},64>")],
    )


def qkvzba(x, weight, tail):
    rows = x.shape[0]
    qkv = torch.empty((rows, 2560), dtype=x.dtype, device=x.device)
    z = torch.empty((rows, 1536), dtype=x.dtype, device=x.device)
    b = torch.empty((rows, 12), dtype=x.dtype, device=x.device)
    a = torch.empty_like(b)
    _gdn_module(rows).run(x, weight, tail, qkv, z, b, a)
    return qkv, z.view(rows, 12, 128), b, a
