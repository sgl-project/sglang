"""HC Apply with caller-owned, pre-rounding FP32 square sums."""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

PARTIALS_PER_BRANCH = 2 * (160 // 32)


@cache_once
def _jit_module(c, h, dtype, write_partials=False):
    args = make_cpp_args(c, h, dtype, write_partials)
    wrappers = [("run", f"HcRawFp32StatKernel<{args}>::run")]
    if write_partials:
        wrappers.append(("reduce", "HcSumPartialsKernel::run"))
    return load_jit(
        "hc_apply_sum",
        *args,
        cuda_files=["elementwise/hc_apply_sum.cuh"],
        cuda_wrappers=wrappers,
    )


def _overlap(a, b):
    if not a.numel() or not b.numel():
        return False
    a_start, b_start = a.data_ptr(), b.data_ptr()
    return (
        a_start < b_start + b.numel() * b.element_size()
        and b_start < a_start + a.numel() * a.element_size()
    )


def hc_apply_sum(
    block_output,
    residual,
    alpha,
    *,
    out,
    sum_sq,
    hc_count=4,
    partial_sums=None,
):
    """Write updated residual and accumulate its pre-rounding FP32 square sums.

    Inputs: contiguous CUDA BF16/FP16 residual [T,C*H], block_output [T,H],
    and final FP32 alpha [T,C]. C=4, H is a positive multiple of 512
    no greater than 16384. C/H/dtype are shape-derived static JIT parameters.
    Dense value buffers must be 32-byte aligned; FP32 alpha/sums need 16 bytes.

    For each element, u = fma(alpha, block_output, residual) in FP32.
    out = cast_to_input_dtype(u); sum_sq += sum_hidden(u*u) in FP32.
    Statistics intentionally observe u BEFORE the BF16/FP16 output cast;
    they are neither inverse RMS nor the exact RMS of the stored residual.

    In atomic mode, the caller MUST zero sum_sq before each invocation,
    including every CUDA Graph replay. This function performs no allocation,
    reset, rsqrt, synchronization or data-dependent host validation. Writes
    must not overlap any input or each other. Inputs may alias each other.

    Two ordinary CTAs per token/branch, 160 threads per CTA; each warp issues
    one atomicAdd. No cluster or inter-CTA barrier is required. Atomic order
    may cause small FP32 rounding differences. Inference-only, no autograd.
    With 32B-aligned, caller-owned FP32 partial_sums [T,C,10], warps overwrite slots
    and a second kernel reduces them with a fixed shuffle tree, overwriting sum_sq.
    Neither partial_sums nor sum_sq needs clearing in this deterministic mode.
    Returns the supplied (out, sum_sq) buffers; T=0 launches nothing.
    """
    if (
        type(hc_count) is not int
        or hc_count != 4
        or residual.ndim != 2
        or not residual.is_cuda
        or residual.dtype not in (torch.bfloat16, torch.float16)
        or residual.shape[0] > 2**31 - 1
        or residual.shape[1] % hc_count
    ):
        raise ValueError("HC Apply requires CUDA BF16/FP16 [T,C*H], C=4, T<2**31")
    t, h = residual.shape[0], residual.shape[1] // hc_count
    if h <= 0 or h % 512 or h > 16384:
        raise ValueError("H must be a positive multiple of 512 <=16384")
    specs = (
        ("residual", residual, (t, hc_count * h), residual.dtype),
        ("block_output", block_output, (t, h), residual.dtype),
        ("alpha", alpha, (t, hc_count), torch.float32),
        ("out", out, (t, hc_count * h), residual.dtype),
        ("sum_sq", sum_sq, (t, hc_count), torch.float32),
    )
    if partial_sums is not None:
        specs += (
            (
                "partial_sums",
                partial_sums,
                (t, hc_count, PARTIALS_PER_BRANCH),
                torch.float32,
            ),
        )
    for name, tensor, shape, dtype in specs:
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != residual.device
            or not tensor.is_contiguous()
            or tensor.data_ptr() % (16 if name in ("alpha", "sum_sq") else 32)
        ):
            raise ValueError(
                f"{name}: invalid shape, dtype, device, stride or alignment"
            )
    destinations = (out, sum_sq)
    if partial_sums is not None:
        destinations += (partial_sums,)
    for i, destination in enumerate(destinations):
        if any(
            _overlap(destination, source) for source in (block_output, residual, alpha)
        ):
            raise ValueError("output buffers must not overlap inputs")
        if any(_overlap(destination, other) for other in destinations[:i]):
            raise ValueError("output buffers must not overlap each other")
    if t:
        module = _jit_module(hc_count, h, residual.dtype, partial_sums is not None)
        if partial_sums is None:
            module.run(block_output, residual, alpha, out, sum_sq)
        else:
            module.run(block_output, residual, alpha, out, partial_sums)
            module.reduce(partial_sums, sum_sq)
    return out, sum_sq
