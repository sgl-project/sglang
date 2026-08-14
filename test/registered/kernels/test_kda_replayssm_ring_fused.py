"""Parity: CACHE_RING verify kernel's ring == the state the verify kernel commits.

This fuses the ReplaySSM ring-write into the recurrent verify kernel
(`fused_sigmoid_gating_delta_rule_update`, CACHE_RING=True): every draft step it
stores the pre-norm k / raw v / in-kernel gate / beta into the per-slot ring, in
place of the eager torch ring-write in kda_backend. Each case drives the fused
kernel end-to-end: run verify with CACHE_RING=True to fill the ring, fold the ring
back, and check the folded checkpoint matches the verify kernel's own per-step
state (intermediate_states_buffer). The ring's gate comes from the kernel's
tl.sigmoid/tl.exp (same as the state update), so the fold is bit-close on shape.

Shapes guard the fused store's head-index / tile / step offsets: GQA (H != HV),
non-pow2 K/V, T<gamma, single head, and padding (-1) slots. Both gate branches
(safe gate / softplus). A wrong index shows up as a shape-specific mismatch.
GPU-only.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip(
        "KDA ReplaySSM fused ring-write parity needs CUDA (triton).",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (  # noqa: E402
    commit_kda_replayssm_spec,
)

DEV = "cuda"

# (bs, T, HV, H, K, V) -- cover GQA (H<HV), non-pow2 K/V, small T, single head.
SHAPES = [
    (16, 7, 4, 4, 64, 64),  # baseline square heads
    (64, 7, 32, 32, 128, 128),  # K3-like TP8 shape
    (8, 4, 8, 2, 128, 128),  # GQA: 4 v-heads per k-head
    (4, 3, 6, 3, 96, 80),  # non-pow2 K/V, GQA 2:1
    (1, 1, 4, 4, 64, 64),  # single req, single step
]
SHAPE_IDS = ["square", "k3-tp8", "gqa4", "nonpow2", "single"]
GATES = [(-5.0, "safe"), (None, "softplus")]


@pytest.mark.parametrize("bs,T,HV,H,K,V", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize(
    "lower_bound", [g[0] for g in GATES], ids=[g[1] for g in GATES]
)
@pytest.mark.parametrize("pad", [False, True], ids=["nopad", "pad"])
def test_ring_fold_parity(bs, T, HV, H, K, V, lower_bound, pad):
    L = max(16, 2 * T)  # ring length; power-of-two backstop satisfied
    scale = K**-0.5
    torch.manual_seed(0)

    q = torch.randn(bs, T, H, K, device=DEV, dtype=torch.float32)
    k = torch.randn(bs, T, H, K, device=DEV, dtype=torch.float32)
    v = torch.randn(bs, T, HV, V, device=DEV, dtype=torch.float32)
    a = torch.randn(bs, T, HV, K, device=DEV, dtype=torch.float32)
    b = torch.randn(bs, T, HV, device=DEV, dtype=torch.float32)
    A_log = torch.randn(HV, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(HV, K, device=DEV, dtype=torch.float32)
    h0 = torch.randn(bs, HV, V, K, device=DEV, dtype=torch.float32)

    # slots 1..bs; optionally set one row to a -1 padding slot (must be skipped).
    slots = torch.arange(1, bs + 1, device=DEV, dtype=torch.int32)
    if pad and bs > 1:
        slots[-1] = -1
    num_slots = bs + 1

    h0_src = torch.zeros(num_slots, HV, V, K, device=DEV, dtype=torch.float32)
    for j in range(bs):
        if slots[j] >= 0:
            h0_src[slots[j]] = h0[j]
    inter = torch.zeros(num_slots, T, HV, V, K, device=DEV, dtype=torch.float32)

    # ring buffers filled by the fused kernel (CACHE_RING=True).
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
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=lower_bound,
        disable_state_update=True,
        intermediate_states_buffer=inter,
        intermediate_state_indices=slots,
        cache_steps=T,
        # fused ring-write (kwargs below are added by the fusion).
        cache_ring=True,
        replayssm_rawv=rawv,
        replayssm_rawk=rawk,
        replayssm_g=gring,
        replayssm_beta=betar,
    )

    acc = torch.full((bs,), T, device=DEV, dtype=torch.int32)
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
        if slots[j] < 0:
            continue  # padding row: ring not written, nothing to check
        base = inter[slots[j], T - 1]
        fold = ckpt[slots[j]]
        rel = ((fold - base).abs().max() / base.abs().max().clamp_min(1e-6)).item()
        assert rel < 1e-3, f"row={j}: rel={rel:.3e}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
