"""DSpark's per-row draft sampling parameters in one launch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _draft_params_kernel(
    temperatures_ptr,
    top_ks_ptr,
    temperatures_out_ptr,
    greedy_out_ptr,
    step_out_ptr,
    n,
    min_temperature,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    t = tl.load(temperatures_ptr + i, mask=mask, other=1.0).to(tl.float32)
    # torch.clamp(min=...): a NaN stays NaN
    t = tl.where(t < min_temperature, min_temperature, t)
    greedy = tl.load(top_ks_ptr + i, mask=mask, other=0) <= 1
    tl.store(temperatures_out_ptr + i, t, mask=mask)
    tl.store(greedy_out_ptr + i, greedy, mask=mask)
    tl.store(step_out_ptr + i, tl.where(greedy, 0.0, t), mask=mask)


def fill_draft_params(
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    temperatures_out: torch.Tensor,
    greedy_out: torch.Tensor,
    step_out: torch.Tensor,
    min_temperature: float = 1e-5,
) -> None:
    """For the first ``len(temperatures_out)`` rows, in one launch:
    ``temperatures_out = temperatures.float().clamp(min=min_temperature)``,
    ``greedy_out = top_ks <= 1`` and ``step_out = where(greedy_out, 0, temperatures_out)``."""
    n = temperatures_out.shape[0]
    assert temperatures.shape[0] >= n and top_ks.shape[0] >= n
    assert temperatures_out.dtype == step_out.dtype == torch.float32
    assert (
        greedy_out.dtype == torch.bool and greedy_out.shape[0] == step_out.shape[0] == n
    )
    for t in (temperatures, top_ks, temperatures_out, greedy_out, step_out):
        assert t.dim() == 1 and t.stride(0) == 1
    if n:
        block = min(1024, triton.next_power_of_2(n))
        _draft_params_kernel[(triton.cdiv(n, block),)](
            temperatures,
            top_ks,
            temperatures_out,
            greedy_out,
            step_out,
            n,
            min_temperature,
            BLOCK=block,
            num_warps=4,
        )
