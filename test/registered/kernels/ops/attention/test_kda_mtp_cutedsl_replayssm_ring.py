"""Parity: CuTe MTP verify kernel's ReplaySSM ring == its own state snapshots.

`fused_kda_decode_mtp_dspark(replayssm_*=...)` switches the CuTe DSpARK verify
kernel to CACHE_RING mode: per draft step it stores post-conv pre-l2norm k,
post-conv v, the log-decay gate gk, and sigmoid(beta) into the rings
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

from types import SimpleNamespace

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

from sglang.kernels.ops.attention.kda_decode_mtp import (  # noqa: E402
    fused_kda_decode_mtp_dspark,
)
from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (  # noqa: E402
    commit_kda_replayssm_spec,
)
from sglang.srt.mem_cache.kv.decl import (  # noqa: E402
    StateGeometry,
    StateReplay,
    with_state_replay,
)
from sglang.srt.mem_cache.kv.layout.request import RequestSubPoolSpec  # noqa: E402

DEV = "cuda"
K = 128
W = 4  # KERNEL_WIDTH
L = 16  # ring length
# The kernel's ring arguments, by the names _run returns them under.
RING_PARTS = (
    ("rawv", "replayssm_rawv"),
    ("rawk", "replayssm_rawk"),
    ("gring", "replayssm_g"),
    ("betar", "replayssm_beta"),
)


def _run(
    arm,
    *,
    N,
    H,
    num_spec,
    seed,
    onorm=False,
    pad=0,
    ring_rows=None,
    rings=None,
    ring_count=None,
):
    """One verify step; the last ``pad`` requests are CUDA-graph padding (state
    slot -1). The ring arm writes the given rings, or new zeroed ones of
    ``ring_count`` rows."""
    torch.manual_seed(seed)
    T = N * (1 + num_spec)
    num_slots = N + 2
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
    if pad:
        slots[N - pad :] = -1
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
    if rings is None:
        # With ring_rows, the rings have rows of their own: ring_rows[j] for
        # request j.
        rows = ring_count or (
            num_slots if ring_rows is None else int(ring_rows.max()) + 1
        )
        rings = dict(
            rawv=torch.zeros(rows, H, L, K, device=DEV, dtype=torch.bfloat16),
            rawk=torch.zeros(rows, H, L, K, device=DEV, dtype=torch.bfloat16),
            gring=torch.zeros(rows, H, L, K, device=DEV, dtype=torch.float32),
            betar=torch.zeros(rows, H, L, device=DEV, dtype=torch.float32),
        )
    out = fused_kda_decode_mtp_dspark(
        intermediate_ssm=None,
        **{arg: rings[name] for name, arg in RING_PARTS},
        replayssm_ring_indices=ring_rows,
        **kwargs,
    )
    return out, dict(**rings, h0=h0, slots=slots)


def _request_sub_pool_rings(backing, *, rows, H, num_spec, layer):
    """The KDA speculative ring of one layer as the unified request sub-pool lays
    it out: each part a strided view inside per-unit entries that hold every ring
    part of every layer. The units start one entry into ``backing``, so a store
    to unit -1 stays in bounds and shows up as a changed byte."""
    geometry = with_state_replay(
        StateGeometry(
            conv_state_shapes=((3 * H * K, W - 1),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(H, K, K),
            temporal_dtype=torch.float32,
        ),
        StateReplay(
            decode=False, speculative=True, is_kda=True, cache_len=L, key_heads=H
        ),
        layer_ids=(0, 1),
        speculative_num_draft_tokens=1 + num_spec,
    )
    spec = RequestSubPoolSpec(
        name="request",
        layer_num=2,
        grow_direction="float",
        parts=geometry.request_parts,
    )
    entry = spec.entry_bytes()
    if backing is None:
        backing = torch.randint(
            0, 256, ((rows + 1) * entry,), device=DEV, dtype=torch.uint8
        )
    arena = SimpleNamespace(raw=backing[entry:], extent_entries=lambda name: rows)
    views = spec.part_views(arena)
    return backing, {name: views[part][layer] for name, part in RING_PARTS}


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
def test_cutedsl_ring_rows_in_request_sub_pool_entries_with_graph_padding(N):
    """The production arrangement under unified memory: the rings are strided
    views inside the request sub-pool's per-unit entries (another layer's and
    the decay rings' parts in between), indexed by each request's own unit, and
    a CUDA-graph replay pads the batch with state slot -1 and ring unit -1.
    Each live request's window lands in its own unit exactly as a slot-indexed
    ring records it at its slot, and no other byte of the entries, nor the
    entry before unit 0, changes."""
    H, num_spec, seed = 2, 3, 13
    pad = max(1, N // 4)
    live = N - pad
    rows = 2 * N + 4
    spare = rows - 1
    window = 1 + num_spec
    ring_rows = torch.arange(2 * N + 2, N + 2, -1, device=DEV, dtype=torch.int32)
    ring_rows[live:] = -1  # as the graph runner fills its padded rows
    backing, rings = _request_sub_pool_rings(
        None, rows=rows, H=H, num_spec=num_spec, layer=1
    )
    for name, _ in RING_PARTS:
        assert not rings[name].is_contiguous(), name
        assert rings[name].stride(0) * rings[name].element_size() > (
            rings[name][0].numel() * rings[name].element_size()
        ), name
    before = backing.clone()
    out, _ = _run(
        "ring",
        N=N,
        H=H,
        num_spec=num_spec,
        seed=seed,
        pad=pad,
        ring_rows=ring_rows,
        rings=rings,
    )
    # Reference: contiguous rings at the same rows; its padded rows point at a
    # spare row, which must stay zero.
    ref_rows = torch.where(ring_rows < 0, spare, ring_rows)
    out_ref, ref = _run(
        "ring",
        N=N,
        H=H,
        num_spec=num_spec,
        seed=seed,
        pad=pad,
        ring_rows=ref_rows,
        ring_count=rows,
    )
    _, by_slot = _run("ring", N=N, H=H, num_spec=num_spec, seed=seed, pad=pad)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    assert torch.count_nonzero(out[0, live * window :]).item() == 0

    expected = before.clone()
    _, expected_rings = _request_sub_pool_rings(
        expected, rows=rows, H=H, num_spec=num_spec, layer=1
    )
    live_rows = ring_rows[:live].long()
    live_slots = by_slot["slots"][:live].long()
    for name, _ in RING_PARTS:
        torch.testing.assert_close(
            ref[name][live_rows], by_slot[name][live_slots], rtol=0, atol=0
        )
        assert torch.count_nonzero(ref[name][spare]).item() == 0, name
        assert torch.count_nonzero(ref[name][:, :, window:]).item() == 0, name
        assert torch.count_nonzero(ref[name][live_rows, :, :window]).item() > 0, name
        expected_rings[name][live_rows, :, :window] = ref[name][live_rows, :, :window]
    changed = torch.nonzero(backing != expected).flatten()
    assert changed.numel() == 0, (
        f"{changed.numel()} bytes differ from the reference, first at byte "
        f"{changed[:4].tolist()}"
    )


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
    _, ring = _run("ring", N=N, H=2, num_spec=2, seed=11, pad=1)
    torch.cuda.synchronize()

    # The last logical request is graph padding. Its original physical slot N
    # is now unused and must remain untouched by all ReplaySSM ring stores.
    for name in ("rawv", "rawk", "gring", "betar"):
        assert torch.count_nonzero(ring[name][N]).item() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
