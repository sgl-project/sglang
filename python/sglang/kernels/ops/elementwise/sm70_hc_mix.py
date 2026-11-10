"""Native FP16 kernels for Qwen3.8's small-batch hyperconnection mix on SM70."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs


@cache_once
def _module():
    return load_jit(
        "sm70_hc_mix",
        cuda_files=["elementwise/sm70_hc_mix.cuh"],
        cuda_wrappers=[
            ("down", "sglang::sm70_hc::down"),
            ("up", "sglang::sm70_hc::up"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _batch_module(rows):
    return load_jit(
        "sm70_hc_batch",
        rows,
        cuda_files=["elementwise/sm70_hc_batch.cuh"],
        cuda_wrappers=[
            ("down", f"sglang::sm70_hc_batch::down_run<{rows}>"),
            ("up", f"sglang::sm70_hc_batch::up_run<{rows}>"),
        ],
        extra_cuda_cflags=["--fmad=false"],
    )


def hc_down(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    output = torch.empty((x.shape[0], 320), device=x.device, dtype=x.dtype)
    module = _module() if x.shape[0] == 1 else _batch_module(x.shape[0])
    module.down(x, weight, output)
    return output


def hc_up(
    activated: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    output = torch.empty((x.shape[0], 2560), device=x.device, dtype=x.dtype)
    module = _module() if x.shape[0] == 1 else _batch_module(x.shape[0])
    module.up(activated, x, weight, output)
    return output


def gate_supported(x, weight):
    # The caller has already checked the exact SM70 mix geometry and dtype.
    return (
        envs.SGLANG_SM70_MTP_HC_GATE.get()
        and x.shape[0] in (2, 4)
        and weight.shape == (4, 10240)
        and weight.device == x.device
        and weight.dtype == torch.float16
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


@cache_once
def _gate_module(rows):
    # Match hc_combine's default FMA contraction. FP16 products in the down
    # projection are exact in FP32, so its explicit rounding is preserved.
    return load_jit(
        "sm70_hc_gate",
        rows,
        cuda_files=["elementwise/sm70_hc_gate.cuh"],
        cuda_wrappers=[
            ("down", f"sglang::sm70_hc_gate::down<{rows}>"),
            ("apply", "sglang::sm70_hc_gate::apply"),
        ],
    )


def hc_down_with_gate(x, weight, inject):
    activated = torch.empty((x.shape[0], 320), device=x.device, dtype=x.dtype)
    partials = torch.empty((x.shape[0], 8, 4), device=x.device, dtype=torch.float32)
    _gate_module(x.shape[0]).down(x, weight, inject, activated, partials)
    return activated, partials


def hc_apply_gate(y, residual, partials):
    output = torch.empty_like(residual)
    _gate_module(y.shape[0]).apply(y, residual, partials, output)
    return output
