"""Parity: CACHE_RING ring-write under ragged (varlen) verify layouts.

Packed varlen verify (per-row verify_lens <= gamma): the fold of the first
acc entries of each row's ring must match the kernel's own per-step state,
as in the dense parity test. Covers partial commit (the compact commit shape)
and padding (-1) slots. GPU-only.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip(
        "KDA ReplaySSM ragged ring-write parity needs CUDA (triton).",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (  # noqa: E402
    commit_kda_replayssm_spec,
)

DEV = "cuda"

# (bs, gamma, HV, H, K, V)
SHAPES = [
    (16, 8, 4, 4, 64, 64),  # baseline square heads
    (8, 8, 32, 32, 128, 128),  # K3-like TP8 shape
    (8, 4, 8, 2, 128, 128),  # GQA: 4 v-heads per k-head
]
SHAPE_IDS = ["square", "k3-tp8", "gqa4"]


def _run_case(bs, gamma, HV, H, K, V, lens, acc, L, pad_last=False):
    scale = K**-0.5
    total = int(lens.sum())
    cu = torch.zeros(bs + 1, device=DEV, dtype=torch.int32)
    cu[1:] = torch.cumsum(lens, dim=0)

    q = torch.randn(1, total, H, K, device=DEV, dtype=torch.float32)
    k = torch.randn(1, total, H, K, device=DEV, dtype=torch.float32)
    v = torch.randn(1, total, HV, V, device=DEV, dtype=torch.float32)
    a = torch.randn(1, total, HV, K, device=DEV, dtype=torch.float32)
    b = torch.randn(1, total, HV, device=DEV, dtype=torch.float32)
    A_log = torch.randn(HV, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(HV, K, device=DEV, dtype=torch.float32)

    slots = torch.arange(1, bs + 1, device=DEV, dtype=torch.int32)
    if pad_last and bs > 1:
        slots[-1] = -1
    num_slots = bs + 1

    h0_src = torch.randn(num_slots, HV, V, K, device=DEV, dtype=torch.float32)
    max_len = int(lens.max())
    inter = torch.zeros(num_slots, max_len, HV, V, K, device=DEV, dtype=torch.float32)

    rawv = torch.zeros(num_slots, HV, L, V, device=DEV, dtype=torch.float32)
    rawk = torch.zeros(num_slots, H, L, K, device=DEV, dtype=torch.float32)
    gring = torch.zeros(num_slots, HV, L, K, device=DEV, dtype=torch.float32)
    betar = torch.zeros(num_slots, HV, L, device=DEV, dtype=torch.float32)

    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=h0_src,
        initial_state_indices=slots,
        cu_seqlens=cu,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
        disable_state_update=True,
        intermediate_states_buffer=inter,
        intermediate_state_indices=slots,
        cache_steps=max_len,
        cache_ring=True,
        replayssm_rawv=rawv,
        replayssm_rawk=rawk,
        replayssm_g=gring,
        replayssm_beta=betar,
    )

    ckpt = h0_src.clone()
    commit_kda_replayssm_spec(
        ckpt,
        rawv,
        rawk,
        gring,
        betar,
        slots,
        acc,
        max_cache_len=L,
        num_k_heads=H,
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
    )

    for j in range(bs):
        if slots[j] < 0 or int(acc[j]) <= 0:
            continue
        base = inter[slots[j], int(acc[j]) - 1]
        fold = ckpt[slots[j]]
        rel = ((fold - base).abs().max() / base.abs().max().clamp_min(1e-6)).item()
        assert (
            rel < 1e-3
        ), f"row={j} len={int(lens[j])} acc={int(acc[j])}: rel={rel:.3e}"


@pytest.mark.parametrize("bs,gamma,HV,H,K,V", SHAPES, ids=SHAPE_IDS)
def test_ragged_full_commit(bs, gamma, HV, H, K, V):
    torch.manual_seed(0)
    L = max(16, 2 * gamma)
    lens = torch.randint(1, gamma + 1, (bs,), device=DEV, dtype=torch.int32)
    _run_case(bs, gamma, HV, H, K, V, lens, acc=lens.clone(), L=L)


@pytest.mark.parametrize("bs,gamma,HV,H,K,V", SHAPES, ids=SHAPE_IDS)
def test_ragged_partial_commit(bs, gamma, HV, H, K, V):
    torch.manual_seed(1)
    L = max(16, 2 * gamma)
    lens = torch.randint(1, gamma + 1, (bs,), device=DEV, dtype=torch.int32)
    acc = (lens + 1) // 2
    _run_case(bs, gamma, HV, H, K, V, lens, acc=acc, L=L)


def test_ragged_pad_slot():
    torch.manual_seed(2)
    bs, gamma, HV, H, K, V = 8, 8, 4, 4, 64, 64
    L = 16
    lens = torch.randint(1, gamma + 1, (bs,), device=DEV, dtype=torch.int32)
    _run_case(bs, gamma, HV, H, K, V, lens, acc=lens.clone(), L=L, pad_last=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
