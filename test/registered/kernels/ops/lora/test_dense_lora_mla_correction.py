"""The absorbed-MLA kv_b_proj LoRA correction on the dense engine, against
the legacy per-request step kernels and a torch reference."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.lora.dense.kv_b_lora_absorbed import (  # noqa: E402
    step_a_q_fwd,
    step_a_v_fwd,
    step_b_q_fwd,
    step_b_v_fwd,
)
from sglang.srt.lora.dense import mla_correction  # noqa: E402
from sglang.srt.lora.dense.runner import DenseLoraRunner  # noqa: E402
from sglang.srt.lora.utils import LoRABatchInfo  # noqa: E402
from sglang.srt.lora.utils import Phase  # noqa: E402
from sglang.srt.lora.workspace import LoraWorkspace  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the correction kernels need CUDA"
)


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
SLOTS, RMAX = 4, 32
RANKS = [32, 16, 0, 8]
SCALINGS = [1.5, 0.5, 1.0, 2.0]
HEADS, QK_NOPE, V_HEAD, KV_RANK = 8, 128, 128, 512
FULL_K = QK_NOPE + V_HEAD


def _batch(decode: bool, tokens: int | None = None):
    seq_lens, slots = (
        ([1] * 13, [i % SLOTS for i in range(13)])
        if decode
        else ([5, 17, 1, 9], [1, 0, 2, 3])
    )
    if tokens is not None:
        seq_lens, slots = [1] * tokens, [i % SLOTS for i in range(tokens)]
    seg_lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    seg_indptr = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=DEVICE)
    seg_indptr[1:] = torch.cumsum(seg_lens, 0)
    info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=len(seq_lens),
        num_segments=len(seq_lens),
        seg_indptr=seg_indptr,
        weight_indices=torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        lora_ranks=torch.tensor(RANKS, dtype=torch.int64, device=DEVICE),
        scalings=torch.tensor(SCALINGS, dtype=torch.float32, device=DEVICE),
        max_len=max(seq_lens),
        seg_lens=seg_lens,
        permutation=None,
    )
    token_slots = [
        s if RANKS[s] > 0 else -1 for s, l in zip(slots, seq_lens) for _ in range(l)
    ]
    return info, token_slots


def _engine(info, token_slots, decode):
    engine = DenseLoraRunner(LoraWorkspace(), max_loras=SLOTS, device=DEVICE)
    engine.begin_batch(
        token_slots=torch.tensor(token_slots, dtype=torch.int32, device=DEVICE),
        lora_ranks=info.lora_ranks,
        scalings=info.scalings,
        num_tokens=int(info.seg_indptr[-1]),
        phase=Phase.DECODE if decode else Phase.PREFILL,
        graph_mode=False,
    )
    return engine


def _pool(seed, heads=HEADS):
    """kv_b_proj slots as the pool writes dense modules: stale past the rank."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    a = torch.randn(SLOTS, RMAX, KV_RANK, generator=g, device=DEVICE) * 100
    b = torch.randn(SLOTS, heads * FULL_K, RMAX, generator=g, device=DEVICE) * 100
    for slot, r in enumerate(RANKS):
        a[slot, :r] = torch.randn(r, KV_RANK, generator=g, device=DEVICE) * 0.05
        b[slot, :, :r] = (
            torch.randn(heads * FULL_K, r, generator=g, device=DEVICE) * 0.05
        )
    return a.to(DTYPE).contiguous(), b.to(DTYPE).contiguous()


def _attention(engine, info, a, b, heads=HEADS):
    return SimpleNamespace(
        num_local_heads=heads,
        qk_nope_head_dim=QK_NOPE,
        v_head_dim=V_HEAD,
        kv_b_proj=SimpleNamespace(
            set_lora=True,
            A_buffer=a,
            B_buffer=b,
            lora_backend=SimpleNamespace(
                name="triton_v2", runner=engine, batch_info=info
            ),
        ),
    )


@pytest.mark.parametrize("decode", [True, False])
def test_q_correction(decode):
    info, token_slots = _batch(decode)
    tokens = len(token_slots)
    a, b = _pool(5)
    g = torch.Generator(device=DEVICE).manual_seed(7)
    q_nope = torch.randn(
        tokens, HEADS, QK_NOPE, generator=g, device=DEVICE, dtype=DTYPE
    )
    base = torch.randn(tokens, HEADS, KV_RANK, generator=g, device=DEVICE, dtype=DTYPE)

    ours = mla_correction.apply_q_correction(
        _attention(_engine(info, token_slots, decode), info, a, b),
        q_nope,
        base.clone(),
    )
    legacy = step_b_q_fwd(step_a_q_fwd(q_nope, b, info, FULL_K), a, info, base.clone())
    torch.testing.assert_close(ours.float(), legacy.float(), rtol=2e-2, atol=2e-2)

    expected = base.float().clone()
    for t, s in enumerate(token_slots):
        if s < 0:
            continue
        r = RANKS[s]
        for h in range(HEADS):
            bridge = (
                (
                    q_nope[t, h].float()
                    @ b[s, h * FULL_K : h * FULL_K + QK_NOPE, :r].float()
                )
                .to(DTYPE)
                .float()
            )
            expected[t, h] += SCALINGS[s] * (bridge @ a[s, :r].float())
    torch.testing.assert_close(ours.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("decode", [True, False])
def test_v_correction(decode):
    info, token_slots = _batch(decode)
    tokens = len(token_slots)
    a, b = _pool(9)
    g = torch.Generator(device=DEVICE).manual_seed(13)
    attn = torch.randn(tokens, HEADS, KV_RANK, generator=g, device=DEVICE, dtype=DTYPE)
    base = torch.randn(tokens, HEADS, V_HEAD, generator=g, device=DEVICE, dtype=DTYPE)

    ours = mla_correction.apply_v_correction(
        _attention(_engine(info, token_slots, decode), info, a, b),
        attn,
        base.clone().flatten(1, 2),
    ).view_as(base)
    legacy_base = base.clone()
    step_b_v_fwd(step_a_v_fwd(attn, a, info), b, info, legacy_base, QK_NOPE, V_HEAD)
    torch.testing.assert_close(ours.float(), legacy_base.float(), rtol=2e-2, atol=2e-2)

    expected = base.float().clone()
    for t, s in enumerate(token_slots):
        if s < 0:
            continue
        r = RANKS[s]
        for h in range(HEADS):
            bridge = (attn[t, h].float() @ a[s, :r].float().T).to(DTYPE).float()
            rows = slice(h * FULL_K + QK_NOPE, h * FULL_K + QK_NOPE + V_HEAD)
            expected[t, h] += SCALINGS[s] * (bridge @ b[s, rows, :r].float().T)
    torch.testing.assert_close(ours.float(), expected, rtol=2e-2, atol=2e-2)


def test_pending_q_and_v_keep_separate_scratch():
    info, slots = _batch(True)
    a, b = _pool(17)
    attention = _attention(_engine(info, slots, True), info, a, b)
    q = torch.randn(len(slots), HEADS, QK_NOPE, device=DEVICE, dtype=DTYPE)
    v = torch.randn(len(slots), HEADS, KV_RANK, device=DEVICE, dtype=DTYPE)
    q_base = torch.randn(len(slots), HEADS, KV_RANK, device=DEVICE, dtype=DTYPE)
    v_base = torch.randn(len(slots), HEADS, V_HEAD, device=DEVICE, dtype=DTYPE)
    q_expected = mla_correction.apply_q_correction(attention, q, q_base.clone())
    v_expected = mla_correction.apply_v_correction(attention, v, v_base.clone())
    q_prepared = mla_correction.prepare_q_correction(attention, q)
    v_prepared = mla_correction.prepare_v_correction(attention, v)
    assert q_prepared.bridge.data_ptr() != v_prepared.bridge.data_ptr()
    q_actual = mla_correction.apply_q_correction(attention, q, q_base, q_prepared)
    v_actual = mla_correction.apply_v_correction(attention, v, v_base, v_prepared)
    torch.testing.assert_close(
        q_actual.float(), q_expected.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        v_actual.float(), v_expected.float(), rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("mode", ["eager", "decode_graph", "prefill_graph"])
@pytest.mark.parametrize("heads", [HEADS, 32])
def test_pair_route_stream_reuse_and_replay(monkeypatch, mode, heads):
    """Decode modes run the shrink on the side stream (overlap a), prefill in line."""
    decode = mode != "prefill_graph"
    info, slots = _batch(decode, tokens=16 if heads == 32 else None)
    tokens = len(slots)
    engine = _engine(info, slots, decode)
    engine.begin_batch(
        token_slots=engine.token_slots,
        lora_ranks=info.lora_ranks,
        scalings=info.scalings,
        num_tokens=tokens,
        phase=Phase.DECODE if decode else Phase.PREFILL,
        graph_mode=mode != "eager",
        is_prefill_graph=mode == "prefill_graph",
    )
    layers = [
        _attention(engine, info, *_pool(seed, heads), heads=heads) for seed in (31, 37)
    ]
    # This layout makes flattening Q copy, before the fork's ready event.
    q = torch.randn(heads, tokens, QK_NOPE, device=DEVICE, dtype=DTYPE).transpose(0, 1)
    v = torch.randn(tokens, heads, KV_RANK, device=DEVICE, dtype=DTYPE)
    bases = [
        torch.randn(tokens, heads, width, device=DEVICE, dtype=DTYPE)
        for width in (KV_RANK, V_HEAD)
    ]
    outputs = [[torch.empty_like(base) for base in bases] for _ in layers]
    caller = torch.cuda.Stream()
    route_stream = caller  # the pair route is always built before the fork
    builds = []
    build = engine._build_pair_route

    def checked_build(*args):
        assert torch.cuda.current_stream() == route_stream
        route = build(*args)
        builds.append(route)
        return route

    monkeypatch.setattr(engine, "_build_pair_route", checked_build)

    def forward():
        engine.reset_routes()
        routes = []
        for layer, (q_out, v_out) in zip(layers, outputs):
            q_prepared = mla_correction.prepare_q_correction(layer, q)
            q_out.copy_(bases[0])
            mla_correction.apply_q_correction(layer, q, q_out, q_prepared)
            v_prepared = mla_correction.prepare_v_correction(layer, v)
            v_out.copy_(bases[1])
            mla_correction.apply_v_correction(layer, v, v_out, v_prepared)
            routes.extend((q_prepared.route, v_prepared.route))
        assert all(route is routes[0] for route in routes)

    caller.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(caller):
        forward()
        forward()
    torch.cuda.synchronize()
    assert len(builds) == 2
    graph = None
    if mode != "eager":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=caller):
            forward()
        assert len(builds) == 3

    for step in range(3):
        current_slots = [(slot + step) % SLOTS if slot >= 0 else -1 for slot in slots]
        current_slots = [
            slot if slot >= 0 and RANKS[slot] else -1 for slot in current_slots
        ]
        scales = [scale * (step + 1) for scale in SCALINGS]
        engine.token_slots.copy_(torch.tensor(current_slots, device=DEVICE))
        info.scalings.copy_(torch.tensor(scales, device=DEVICE))
        q.add_(0.01)
        v.sub_(0.01)
        if graph is None:
            caller.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(caller):
                forward()
            assert len(builds) == 3 + step
        else:
            graph.replay()
            assert len(builds) == 3  # Replay executes kernels, not the Python cache.
        torch.cuda.synchronize()
        for layer, (q_out, v_out) in zip(layers, outputs):
            a, b = layer.kv_b_proj.A_buffer, layer.kv_b_proj.B_buffer
            q_expected, v_expected = [base.float().clone() for base in bases]
            for token, slot in enumerate(current_slots):
                if slot < 0:
                    continue
                rank = RANKS[slot]
                b_heads = b[slot, :, :rank].reshape(heads, FULL_K, rank).float()
                q_bridge = (
                    torch.einsum("hk,hkr->hr", q[token].float(), b_heads[:, :QK_NOPE])
                    .to(DTYPE)
                    .float()
                )
                q_expected[token] += scales[slot] * (q_bridge @ a[slot, :rank].float())
                v_bridge = (
                    (v[token].float() @ a[slot, :rank].float().T).to(DTYPE).float()
                )
                v_expected[token] += scales[slot] * torch.einsum(
                    "hr,hkr->hk", v_bridge, b_heads[:, QK_NOPE:]
                )
            torch.testing.assert_close(q_out.float(), q_expected, rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(v_out.float(), v_expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
