"""Cuda-graph padded rows of the GDN/KDA target-verify output must stay defined.

A padded request has an empty ``cu_seqlens`` segment, so no kernel program
writes its rows of the output. This test pins two properties that a precision
test against a reference cannot see. The padded rows are zero. The live rows
are bit-identical to the same batch run without padding.

The test frees a NaN tensor of the output's shape right before each call, so
the caching allocator hands that block back. Unwritten rows then come back NaN.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

try:
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

HV = 12
DIM = 128
POOL = 64


def _inputs(n_tok, device, is_kda, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=g, dtype=dtype, device=device)

    return dict(
        A_log=rnd(1, 1, HV, 1, dtype=torch.float32),
        dt_bias=rnd(HV * DIM if is_kda else HV),
        q=rnd(1, n_tok, HV, DIM),
        k=rnd(1, n_tok, HV, DIM),
        v=rnd(1, n_tok, HV, DIM),
        a=rnd(1, n_tok, HV * DIM if is_kda else HV),
        b=rnd(1, n_tok, HV),
    )


def _run(n_tok, cu_seqlens, state_indices, src, states, is_kda):
    # Hand the allocator a NaN block of exactly `o`'s shape to reuse.
    junk = torch.full(
        (1, n_tok, HV, DIM), float("nan"), dtype=src["q"].dtype, device=src["q"].device
    )
    del junk
    return fused_sigmoid_gating_delta_rule_update(
        A_log=src["A_log"],
        dt_bias=src["dt_bias"],
        q=src["q"][:, :n_tok],
        k=src["k"][:, :n_tok],
        v=src["v"][:, :n_tok],
        a=src["a"][:, :n_tok],
        b=src["b"][:, :n_tok],
        initial_state_source=states,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=is_kda,
    )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernel not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires a GPU")
@pytest.mark.parametrize("is_kda", [True, False])
@pytest.mark.parametrize("live_bs,pad_bs,q_len", [(19, 1, 4), (7, 3, 8), (1, 1, 1)])
def test_padded_rows_are_zero_and_live_rows_unchanged(
    is_kda: bool, live_bs: int, pad_bs: int, q_len: int
):
    device = "cuda"
    live_tok = live_bs * q_len
    pad_tok = (live_bs + pad_bs) * q_len
    src = _inputs(pad_tok, device, is_kda)
    # The padded rows carry leftover values, as the graph input buffers do.
    src["q"][:, live_tok:] *= 7.0
    src["k"][:, live_tok:] *= 7.0

    # Seeded explicitly: the global RNG is not reproducible across processes,
    # and these two arms are compared against each other.
    sg = torch.Generator(device=device).manual_seed(1234)
    base_states = torch.randn(
        POOL, HV, DIM, DIM, generator=sg, dtype=torch.float32, device=device
    )
    live_slots = torch.arange(1, live_bs + 1, dtype=torch.int32, device=device)

    # Arm A: no padding. The state update stays on, so the -1 sentinel on the
    # padded rows is exercised. The live rows write their slots in both arms,
    # so the two resulting pools must match each other. Both differ from the
    # pristine pool by design.
    cu_a = torch.arange(0, live_tok + 1, q_len, dtype=torch.int32, device=device)
    states_a = base_states.clone()
    out_a = _run(live_tok, cu_a, live_slots, src, states_a, is_kda)

    # Arm B: the same live batch plus padded requests. Each padded request has
    # an empty segment and the -1 state sentinel. This is what the cuda-graph
    # replay path builds.
    cu_b = torch.cat(
        [cu_a, torch.full((pad_bs,), live_tok, dtype=torch.int32, device=device)]
    )
    idx_b = torch.cat(
        [live_slots, torch.full((pad_bs,), -1, dtype=torch.int32, device=device)]
    )
    states_b = base_states.clone()
    out_b = _run(pad_tok, cu_b, idx_b, src, states_b, is_kda)

    live_a = out_a.reshape(-1, HV, DIM)
    live_b = out_b.reshape(-1, HV, DIM)[:live_tok]
    pad_b = out_b.reshape(-1, HV, DIM)[live_tok:]

    assert torch.equal(live_a, live_b), (
        "padding perturbed the live rows; they must be bit-identical to the "
        "unpadded run"
    )
    assert not torch.isnan(pad_b).any(), "padded rows hold uninitialized memory"
    assert bool((pad_b == 0).all().item()), "padded rows must be zero"
    assert torch.equal(states_a, states_b), (
        "padding changed the state pool; a padded request's slot index is the "
        "-1 sentinel and must be skipped"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
