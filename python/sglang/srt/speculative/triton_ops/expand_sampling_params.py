"""Fused repeat-interleave for sampling parameters in eagle_sample.

Replaces 3 separate ``torch.repeat_interleave`` calls (temperatures, top_ks,
top_ps) with a single Triton kernel launch, reducing eager-mode overhead.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_sampling_params_kernel(
    # Per-request inputs (length bs)
    temperatures_ptr,
    top_ks_ptr,
    top_ps_ptr,
    # Per-token outputs (length bs * draft_token_num)
    out_temperatures_ptr,
    out_top_ks_ptr,
    out_top_ps_ptr,
    # Scalar args
    draft_token_num: tl.constexpr,
):
    """One program per request: reads 3 scalars, writes draft_token_num copies."""
    pid = tl.program_id(axis=0)

    # Read per-request values
    temperature = tl.load(temperatures_ptr + pid)
    top_k = tl.load(top_ks_ptr + pid)
    top_p = tl.load(top_ps_ptr + pid)

    # Write to expanded outputs
    out_base = pid * draft_token_num
    offsets = tl.arange(0, draft_token_num)
    tl.store(out_temperatures_ptr + out_base + offsets, temperature)
    tl.store(out_top_ks_ptr + out_base + offsets, top_k)
    tl.store(out_top_ps_ptr + out_base + offsets, top_p)


def expand_sampling_params(
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    draft_token_num: int,
    out_temperatures: torch.Tensor | None = None,
    out_top_ks: torch.Tensor | None = None,
    out_top_ps: torch.Tensor | None = None,
):
    """Expand per-request sampling params to per-token in a single kernel.

    Args:
        temperatures: [bs, 1] float32 per-request temperatures.
        top_ks: [bs] int32 per-request top-k values.
        top_ps: [bs] float32 per-request top-p values.
        draft_token_num: Number of draft tokens per request.
        out_*: Optional pre-allocated output buffers of length ``bs * draft_token_num``.

    Returns:
        (expanded_temperatures, expanded_top_ks, expanded_top_ps) each of
        shape ``[bs * draft_token_num]`` (temperatures gets an extra dim 1).
    """
    bs = top_ks.shape[0]
    expanded_len = bs * draft_token_num

    # Flatten temperatures from [bs, 1] to [bs]
    temperatures_flat = temperatures.view(-1)

    if out_temperatures is None:
        out_temperatures = torch.empty(
            expanded_len, dtype=temperatures.dtype, device=temperatures.device
        )
    if out_top_ks is None:
        out_top_ks = torch.empty(expanded_len, dtype=top_ks.dtype, device=top_ks.device)
    if out_top_ps is None:
        out_top_ps = torch.empty(expanded_len, dtype=top_ps.dtype, device=top_ps.device)

    # Triton constexpr requires power-of-2 for tl.arange; round up.
    draft_token_num_po2 = triton.next_power_of_2(draft_token_num)

    # Use the non-constexpr wrapper to launch.  For small draft_token_num
    # values (typical: 4-64), the kernel with power-of-2 arange and masking
    # works fine.  For exact sizes we pass the power-of-2 and mask writes.
    if draft_token_num == draft_token_num_po2:
        _expand_sampling_params_kernel[(bs,)](
            temperatures_flat,
            top_ks,
            top_ps,
            out_temperatures,
            out_top_ks,
            out_top_ps,
            draft_token_num=draft_token_num,
        )
    else:
        # Non-power-of-2: use masked kernel
        _expand_sampling_params_kernel_masked[(bs,)](
            temperatures_flat,
            top_ks,
            top_ps,
            out_temperatures,
            out_top_ks,
            out_top_ps,
            draft_token_num,
            BLOCK_N=draft_token_num_po2,
        )

    # Reshape temperatures to [bs * draft_token_num, 1] to match original shape
    return out_temperatures.view(-1, 1), out_top_ks, out_top_ps


@triton.jit
def _expand_sampling_params_kernel_masked(
    # Per-request inputs
    temperatures_ptr,
    top_ks_ptr,
    top_ps_ptr,
    # Per-token outputs
    out_temperatures_ptr,
    out_top_ks_ptr,
    out_top_ps_ptr,
    # Scalar args
    draft_token_num,
    BLOCK_N: tl.constexpr,
):
    """Masked variant for non-power-of-2 draft_token_num."""
    pid = tl.program_id(axis=0)

    temperature = tl.load(temperatures_ptr + pid)
    top_k = tl.load(top_ks_ptr + pid)
    top_p = tl.load(top_ps_ptr + pid)

    out_base = pid * draft_token_num
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < draft_token_num
    tl.store(out_temperatures_ptr + out_base + offsets, temperature, mask=mask)
    tl.store(out_top_ks_ptr + out_base + offsets, top_k, mask=mask)
    tl.store(out_top_ps_ptr + out_base + offsets, top_p, mask=mask)
