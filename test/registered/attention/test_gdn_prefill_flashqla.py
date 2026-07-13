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

# FlashQLA requires an SM90+ (Hopper) or SM100 (Blackwell) GPU.
register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

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


@pytest.mark.parametrize("seq_lens", [[512, 1024], [100, 777, 65, 8192], [4096]])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_flashqla_extend_matches_triton(seq_lens, state_dtype):
    q, k, v, g, beta, pool, cache_indices, cu_seqlens = _make_gdn_inputs(
        seq_lens, state_dtype
    )

    triton_pool = pool.clone()
    ref_o, _, _ = TritonGDNKernel().extend(
        q,
        k,
        v,
        g,
        beta,
        ssm_states=triton_pool,
        cache_indices=cache_indices,
        query_start_loc=cu_seqlens,
    )

    flashqla_pool = pool.clone()
    out_o, _, _ = FlashQLAGDNKernel().extend(
        q,
        k,
        v,
        g,
        beta,
        ssm_states=flashqla_pool,
        cache_indices=cache_indices,
        query_start_loc=cu_seqlens,
    )

    o_err = (out_o.float() - ref_o.float()).abs()
    state_err = (
        flashqla_pool[cache_indices].float() - triton_pool[cache_indices].float()
    ).abs()
    # bf16 chunked-prefill numerics: match the CuteDSL test's tolerances.
    assert o_err.max().item() < 2e-3
    assert o_err.mean().item() < 6e-5
    assert state_err.max().item() < 2e-2
    assert state_err.mean().item() < 6e-4


def test_flashqla_falls_back_below_threshold():
    """Batches below the token threshold must defer to the Triton path."""
    q, k, v, g, beta, pool, cache_indices, cu_seqlens = _make_gdn_inputs(
        [128, 256], torch.bfloat16
    )
    assert FlashQLAGDNKernel._should_fall_back(q) is True

    q_big = torch.randn(
        1, 20000, NUM_K_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    assert FlashQLAGDNKernel._should_fall_back(q_big) is False


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
