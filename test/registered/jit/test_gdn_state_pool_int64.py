"""Regression test: GDN state-pool index loads must be widened to int64.

Both ``fused_recurrent_gated_delta_rule_update`` and the sibling
``fused_sigmoid_gating_delta_rule_update`` compute the h0 read/write-back and
intermediate-store offsets from caller-provided index tensors as
``idx * HV * K * V`` (or ``cache_idx * cache_steps * HV * K * V``). Loaded in
int32 and never widened, any pool slot whose flat offset exceeds ``2**31``
silently wraps negative and reads/writes the wrong state row (or faults).

Running identical physics at slot 0 and at a slot id past the wrap threshold
must produce identical output and identical written-back state.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

try:
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

H, HV, K, V = 16, 32, 128, 128
ROW = HV * K * V  # 524288
BIG_ID = (2**31) // ROW + 256  # 4352: BIG_ID * ROW > 2**31


def _make_inputs(N: int, T: int, device="cuda", seed=2025):
    torch.manual_seed(seed)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.bfloat16, device=device)
    a = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    b = torch.randn(1, N * T, HV, dtype=torch.bfloat16, device=device)
    q = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(1, N * T, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, N * T, HV, V, dtype=torch.bfloat16, device=device)
    cu_seqlens = torch.arange(0, N * T + 1, T, dtype=torch.int32, device=device)
    g, beta = fused_gdn_gating(A_log, a.view(-1, HV), b.view(-1, HV), dt_bias)
    g = g.view(a.shape)
    beta = beta.view(b.shape)
    return A_log, dt_bias, a, b, q, k, v, g, beta, cu_seqlens


def _run_recurrent(g, beta, q, k, v, state_src, indices, cu_seqlens):
    # fused_recurrent validates intermediate_state_indices shape against
    # cu_seqlens even when no intermediate buffer is used.
    N = len(cu_seqlens) - 1
    return fused_recurrent_gated_delta_rule_update(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state_source=state_src,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=False,
        intermediate_state_indices=torch.arange(N, dtype=torch.int32, device=q.device),
    )


def _run_sigmoid(A_log, dt_bias, a, b, q, k, v, state_src, indices, cu_seqlens):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=state_src,
        initial_state_indices=indices,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=False,
        disable_state_update=False,
    )


@pytest.mark.skipif(not KERNELS_AVAILABLE, reason="Kernels not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_large_state_pool_id_no_int32_overflow():
    N, T = 1, 4
    A_log, dt_bias, a, b, q, k, v, g, beta, cu_seqlens = _make_inputs(N, T)

    ref_idx = torch.zeros(N, dtype=torch.int32, device="cuda")
    big_idx = torch.full((N,), BIG_ID, dtype=torch.int32, device="cuda")

    def _fresh_states():
        # bf16 keeps the high-id buffer ~4.5 GB; kernels accumulate in fp32
        # internally, so storage dtype does not affect the overflow behaviour.
        ref_src = torch.randn(1, HV, K, V, dtype=torch.bfloat16, device="cuda")
        big_src = torch.zeros(BIG_ID + 1, HV, K, V, dtype=torch.bfloat16, device="cuda")
        big_src[BIG_ID].copy_(ref_src[0])
        return ref_src, big_src

    def _check(run):
        ref_src, big_src = _fresh_states()
        out_ref = run(ref_src, ref_idx)
        out_big = run(big_src, big_idx)
        torch.testing.assert_close(out_big, out_ref, rtol=1e-2, atol=1e-2)
        # Written-back final state must land in the correct high-id row.
        torch.testing.assert_close(big_src[BIG_ID], ref_src[0], rtol=1e-2, atol=1e-2)
        del big_src
        torch.cuda.empty_cache()

    # fused_recurrent update path (h0 read + intermediate store + write-back).
    _check(lambda src, idx: _run_recurrent(g, beta, q, k, v, src, idx, cu_seqlens))
    # fused_sigmoid_gating update path (sibling kernel, same offset expressions).
    _check(
        lambda src, idx: _run_sigmoid(
            A_log, dt_bias, a, b, q, k, v, src, idx, cu_seqlens
        )
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
