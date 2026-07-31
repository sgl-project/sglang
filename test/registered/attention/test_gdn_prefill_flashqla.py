"""Correctness test for the FlashQLA GDN prefill kernel backend.

Validates ``FlashQLAGDNKernel.extend`` (which wraps the external
``flash_qla`` TileLang chunked-prefill kernel) against the in-tree Triton
``TritonGDNKernel.extend`` reference, exercising the real state-pool
gather/scatter interface used by ``GDNKernelDispatcher``.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

# SM90+ single-GPU suite. Disabled in CI for the same reason as
# test_kda_prefill_flashkda.py: flash_qla is not in the public runner image, and
# the module-level pytest.skip aborts non-zero under `python3 file.py`. Drop
# `disabled=` once flash_qla ships in the runner image; still runs locally.
register_cuda_ci(
    est_time=90,
    stage="base-b",
    runner_config="1-gpu-large",
    disabled="flash_qla not in public CI runner image",
)

if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9):
    pytest.skip(
        "FlashQLA GDN prefill requires CUDA SM90+ (Hopper/Blackwell).",
        allow_module_level=True,
    )

try:
    import flash_qla  # noqa: F401
except ImportError:
    pytest.skip(
        "FlashQLA GDN prefill requires the optional flash-qla package.",
        allow_module_level=True,
    )

from sglang.srt.layers.attention.linear.kernels.gdn_flashqla import (  # noqa: E402
    _FLASHQLA_MIN_TOTAL_TOKENS,
    FlashQLAGDNKernel,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import (  # noqa: E402
    TritonGDNKernel,
)

NUM_K_HEADS = 4
NUM_V_HEADS = 8
HEAD_DIM = 128


def _make_gdn_inputs(seq_lens, state_dtype):
    """Build one varlen GDN prefill batch (packed B==1) plus a state pool."""
    cu_seqlens = torch.zeros(len(seq_lens) + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(seq_lens, device="cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())

    dtype = torch.bfloat16
    q = torch.randn(1, total_tokens, NUM_K_HEADS, HEAD_DIM, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(1, total_tokens, NUM_V_HEADS, HEAD_DIM, device="cuda", dtype=dtype)

    a = torch.randn(1, total_tokens, NUM_V_HEADS, device="cuda", dtype=dtype)
    b = torch.randn(1, total_tokens, NUM_V_HEADS, device="cuda", dtype=dtype)

    # Match the FLA GatedDeltaNet synthetic gating init (as in the CuteDSL test).
    A = torch.empty(NUM_V_HEADS, device="cuda", dtype=torch.float32).uniform_(0, 16)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(NUM_V_HEADS, device="cuda", dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, NUM_V_HEADS) * F.softplus(
        a.float() + dt_bias.view(1, 1, NUM_V_HEADS)
    )
    beta = torch.sigmoid(b.float())

    # sglang's GDN state pool is V-first: [num_slots, HV, V, K].
    num_slots = len(seq_lens) + 4
    pool = (
        torch.randn(
            num_slots, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, device="cuda", dtype=state_dtype
        )
        * 0.05
    )
    cache_indices = torch.arange(1, 1 + len(seq_lens), device="cuda", dtype=torch.int32)
    return q, k, v, g, beta, pool, cache_indices, cu_seqlens


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _extend_pair(kernel, seq_lens, state_dtype):
    """Run Triton and the given FlashQLA kernel over the same batch.

    Returns ``(ref_o, ref_state, out_o, out_state)``; each kernel gets its own
    clone of the state pool so the scatter-back is compared, not shared.
    """
    q, k, v, g, beta, pool, cache_indices, cu_seqlens = _make_gdn_inputs(
        seq_lens, state_dtype
    )
    args = (q, k, v, g, beta)
    kwargs = dict(cache_indices=cache_indices, query_start_loc=cu_seqlens)

    triton_pool = pool.clone()
    ref_o, _, _ = TritonGDNKernel().extend(*args, ssm_states=triton_pool, **kwargs)

    flashqla_pool = pool.clone()
    out_o, _, _ = kernel.extend(*args, ssm_states=flashqla_pool, **kwargs)
    torch.cuda.synchronize()

    return ref_o, triton_pool[cache_indices], out_o, flashqla_pool[cache_indices]


@pytest.mark.parametrize("seq_lens", [[512, 1024], [100, 777, 65, 8192], [4096]])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_flashqla_extend_matches_triton(seq_lens, state_dtype):
    """Ragged batches through the real gather/scatter path.

    ``min_total_tokens=0`` disables the dispatch threshold so these cheap
    shapes actually reach flash_qla -- at the production threshold (16K packed
    tokens) every case here would fall back and compare Triton against itself.
    Covers partial chunks and sub-chunk sequences (65 tokens).
    """
    torch.manual_seed(len(seq_lens))
    ref_o, ref_state, out_o, out_state = _extend_pair(
        FlashQLAGDNKernel(min_total_tokens=0), seq_lens, state_dtype
    )

    assert torch.isfinite(out_o).all(), "FlashQLA output has non-finite values"
    assert torch.isfinite(out_state).all(), "FlashQLA final state has non-finite values"
    # bf16 cross-implementation noise (TileLang vs Triton); measured cos
    # ~0.99999 on H200, so the sibling FlashKDA test's thresholds hold here.
    assert _cos(ref_o, out_o) > 0.999, f"output cos too low: {_cos(ref_o, out_o):.5f}"
    assert (
        _cos(ref_state, out_state) > 0.999
    ), f"state cos too low: {_cos(ref_state, out_state):.5f}"


def test_flashqla_extend_matches_triton_above_threshold():
    """Same check on a default-configured kernel past the dispatch threshold --
    the batch shape that actually reaches flash_qla in production."""
    torch.manual_seed(0)
    seq_lens = [12000, 4096, 777]
    assert sum(seq_lens) >= _FLASHQLA_MIN_TOTAL_TOKENS, "batch would hit the fallback"

    ref_o, ref_state, out_o, out_state = _extend_pair(
        FlashQLAGDNKernel(), seq_lens, torch.bfloat16
    )

    assert torch.isfinite(out_o).all()
    assert _cos(ref_o, out_o) > 0.999, f"output cos too low: {_cos(ref_o, out_o):.5f}"
    assert (
        _cos(ref_state, out_state) > 0.999
    ), f"state cos too low: {_cos(ref_state, out_state):.5f}"


def test_flashqla_falls_back_below_threshold():
    """extend() must route to Triton below the threshold, not merely report
    that it would.

    Asserting on ``_should_fall_back`` alone would still pass if the dispatch
    in ``extend`` were dropped, so compare outputs: the fallback re-runs the
    same Triton kernel and lands within float noise, while flash_qla's
    cross-implementation error is orders of magnitude larger.
    """
    torch.manual_seed(0)
    ref_o, ref_state, out_o, out_state = _extend_pair(
        FlashQLAGDNKernel(), [128, 256], torch.bfloat16
    )

    assert (out_o.float() - ref_o.float()).abs().max().item() < 1e-6
    assert (out_state.float() - ref_state.float()).abs().max().item() < 1e-6


def test_flashqla_falls_back_on_unsupported_dtype():
    """fp32 queries have no flash_qla kernel and must take the Triton path."""
    kernel = FlashQLAGDNKernel()
    q_big = torch.randn(
        1, 20000, NUM_K_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    assert kernel._should_fall_back(q_big) is False
    assert kernel._should_fall_back(q_big.float()) is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
