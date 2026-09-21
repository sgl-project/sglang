"""Parity: CuTe MTP verify kernel's ReplaySSM ring == its own state snapshots.

`fused_kda_decode_mtp_dspark(replayssm_*=...)` switches the CuTe DSpARK verify
kernel to CACHE_RING mode: per draft step it stores post-conv pre-l2norm k,
post-conv v, the log-decay gate gk, and sigmoid(beta) into the per-slot rings
(and skips the per-step intermediate_ssm snapshots). Each case runs the kernel
twice on identical inputs — baseline arm producing snapshots, ring arm
producing rings — folds the ring with `commit_kda_replayssm_spec`, and checks
the folded checkpoint against the baseline arm's last-step snapshot. A wrong
head/step/slot offset in the fused ring store shows up as a mismatch; the
output tensor must be untouched by the mode (bitwise equal).

Tolerance is bf16-bound: production rings store rawk/rawv in conv dtype
(bf16), so the fold re-quantizes k/v while the baseline snapshot keeps them
in fp32 registers.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# SM100-only, and hard: libNVVM refuses the generated device IR when the CuTe
# DSL kernel is compiled for sm_90a, so this is a build failure rather than a
# numeric one. The SM100 pool has no single-GPU runner_config, hence 4-gpu-b200
# for a one-GPU test.
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
    pytest.skip(
        "KDA CuTe MTP ReplaySSM ring parity needs SM100 (CuTe DSL kernel).",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (  # noqa: E402
    commit_kda_replayssm_spec,
)
from sglang.kernels.ops.kimi_k3.kda_decode_mtp import (  # noqa: E402
    fused_kda_decode_mtp_dspark,
)

DEV = "cuda"
K = 128
W = 4  # KERNEL_WIDTH


def _run(arm, *, N, H, num_spec, seed, onorm=False, pad_last=False):
    torch.manual_seed(seed)
    T = N * (1 + num_spec)
    num_slots = N + 2
    L = 16
    assert L >= 1 + num_spec

    x_q = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16)
    x_k = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16)
    x_v = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16)
    beta = torch.randn(1, T, H, device=DEV, dtype=torch.bfloat16)
    w_q = torch.randn(H * K, W, device=DEV, dtype=torch.float32) * 0.1
    w_k = torch.randn(H * K, W, device=DEV, dtype=torch.float32) * 0.1
    w_v = torch.randn(H * K, W, device=DEV, dtype=torch.float32) * 0.1
    cs_q = torch.randn(num_slots, H * K, W - 1, device=DEV, dtype=torch.bfloat16)
    cs_k = torch.randn(num_slots, H * K, W - 1, device=DEV, dtype=torch.bfloat16)
    cs_v = torch.randn(num_slots, H * K, W - 1, device=DEV, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(H * K, device=DEV, dtype=torch.float32)
    h0 = torch.randn(num_slots, H, K, K, device=DEV, dtype=torch.float32)

    slots = torch.arange(1, N + 1, device=DEV, dtype=torch.int32)
    if pad_last:
        slots[-1] = -1
    scratch = torch.arange(N, device=DEV, dtype=torch.int32)
    cu_seqlens = torch.arange(0, T + 1, 1 + num_spec, device=DEV, dtype=torch.int32)

    ic_q = torch.zeros(N, 1 + num_spec, H * K, W - 1, device=DEV, dtype=torch.bfloat16)
    ic_k = torch.zeros_like(ic_q)
    ic_v = torch.zeros_like(ic_q)

    kwargs = dict(
        x_q=x_q,
        x_k=x_k,
        x_v=x_v,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        cs_q=cs_q,
        cs_k=cs_k,
        cs_v=cs_v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        recurrent_state=h0,
        intermediate_state_indices=scratch,
        intermediate_conv_q=ic_q,
        intermediate_conv_k=ic_k,
        intermediate_conv_v=ic_v,
        ssm_state_indices=slots,
        cu_seqlens=cu_seqlens,
        lower_bound=-5.0,
    )
    norm = {}
    if onorm:
        norm = dict(
            gate=torch.randn(1, T, H, K, device=DEV, dtype=torch.bfloat16),
            weight=torch.randn(K, device=DEV, dtype=torch.float32),
            eps=1e-6,
        )
        kwargs.update(
            onorm_gate=norm["gate"],
            onorm_weight=norm["weight"],
            onorm_eps=norm["eps"],
        )
    if arm == "baseline":
        inter = torch.zeros(N, 1 + num_spec, H, K, K, device=DEV, dtype=torch.float32)
        out = fused_kda_decode_mtp_dspark(intermediate_ssm=inter, **kwargs)
        return out, dict(inter=inter, slots=slots, scratch=scratch, **norm)
    rawv = torch.zeros(num_slots, H, L, K, device=DEV, dtype=torch.bfloat16)
    rawk = torch.zeros_like(rawv)
    gring = torch.zeros(num_slots, H, L, K, device=DEV, dtype=torch.float32)
    betar = torch.zeros(num_slots, H, L, device=DEV, dtype=torch.float32)
    out = fused_kda_decode_mtp_dspark(
        intermediate_ssm=None,
        replayssm_rawv=rawv,
        replayssm_rawk=rawk,
        replayssm_g=gring,
        replayssm_beta=betar,
        **kwargs,
    )
    return out, dict(rawv=rawv, rawk=rawk, gring=gring, betar=betar, h0=h0, slots=slots)


@pytest.mark.parametrize(
    "N,H,num_spec", [(4, 2, 4), (1, 12, 5), (16, 2, 8)], ids=["small", "k3ish", "wide"]
)
def test_cutedsl_ring_fold_parity(N, H, num_spec):
    seed = 0
    out_base, base = _run("baseline", N=N, H=H, num_spec=num_spec, seed=seed)
    out_ring, ring = _run("ring", N=N, H=H, num_spec=num_spec, seed=seed)

    torch.testing.assert_close(out_ring, out_base, rtol=0, atol=0)

    T_req = 1 + num_spec
    acc = torch.full((N,), T_req, device=DEV, dtype=torch.int32)
    ckpt = ring["h0"].clone()
    commit_kda_replayssm_spec(
        ckpt,
        ring["rawv"],
        ring["rawk"],
        ring["gring"],
        ring["betar"],
        ring["slots"],
        acc,
        max_cache_len=ring["rawv"].shape[2],
        num_k_heads=H,
        use_qk_l2norm_in_kernel=True,
        null_block_id=-1,
    )
    for j in range(N):
        base_state = base["inter"][base["scratch"][j], T_req - 1]
        fold = ckpt[ring["slots"][j]]
        rel = (
            (fold - base_state).abs().max() / base_state.abs().max().clamp_min(1e-6)
        ).item()
        assert rel < 2e-2, f"req={j}: rel={rel:.3e}"


@pytest.mark.parametrize("N", [4, 32], ids=["small-grid", "large-grid"])
def test_cutedsl_fused_output_norm(N):
    H, num_spec, seed = 2, 2, 7
    raw, _ = _run("baseline", N=N, H=H, num_spec=num_spec, seed=seed)
    fused, norm = _run("baseline", N=N, H=H, num_spec=num_spec, seed=seed, onorm=True)

    ref = raw.float()
    ref = ref * torch.rsqrt(ref.square().mean(dim=-1, keepdim=True) + norm["eps"])
    ref = ref * norm["weight"] * torch.sigmoid(norm["gate"].float())
    torch.testing.assert_close(fused.float(), ref, rtol=2e-2, atol=3e-2)


@pytest.mark.parametrize("N", [4, 32], ids=["small-grid", "large-grid"])
def test_cutedsl_cuda_graph_padding_slot_is_safe(N):
    _, ring = _run("ring", N=N, H=2, num_spec=2, seed=11, pad_last=True)
    torch.cuda.synchronize()

    # The last logical request is graph padding. Its original physical slot N
    # is now unused and must remain untouched by all ReplaySSM ring stores.
    for name in ("rawv", "rawk", "gring", "betar"):
        assert torch.count_nonzero(ring[name][N]).item() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
