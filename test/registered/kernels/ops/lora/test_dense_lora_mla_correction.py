"""Absorbed MLA correction: legacy parity, rounded reference, and graph replay."""

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
from sglang.srt.lora.backend.base_backend import _compute_moe_lora_info  # noqa: E402
from sglang.srt.lora.dense import mla_correction  # noqa: E402
from sglang.srt.lora.dense.runner import DenseLoraRunner  # noqa: E402
from sglang.srt.lora.utils import LoRABatchInfo, Phase  # noqa: E402
from sglang.srt.lora.workspace import LoraWorkspace  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the correction kernels need CUDA"
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
SLOTS = 4
RANKS = [32, 16, 0, 8]
SCALINGS = [1.5, 0.5, 1.0, 2.0]
HEADS, QK_NOPE, V_HEAD, KV_RANK = 8, 128, 128, 512
FULL_K = QK_NOPE + V_HEAD


def _batch(decode, tokens=None, *, ranks=RANKS):
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
        lora_ranks=torch.tensor(ranks, dtype=torch.int64, device=DEVICE),
        scalings=torch.tensor(SCALINGS, dtype=torch.float32, device=DEVICE),
        max_len=max(seq_lens),
        seg_lens=seg_lens,
        permutation=None,
    )
    token_slots = [
        s if ranks[s] > 0 else -1
        for s, length in zip(slots, seq_lens)
        for _ in range(length)
    ]
    return info, token_slots


def _engine(info, token_slots, decode, *, graph_mode=False):
    engine = DenseLoraRunner(LoraWorkspace(), max_loras=SLOTS, device=DEVICE)
    engine.begin_batch(
        token_slots=torch.tensor(token_slots, dtype=torch.int32, device=DEVICE),
        lora_ranks=info.lora_ranks,
        scalings=info.scalings,
        num_tokens=len(token_slots),
        phase=Phase.DECODE if decode else Phase.PREFILL,
        graph_mode=graph_mode,
        is_prefill_graph=graph_mode and not decode,
    )
    return engine


def _pool(seed, heads=HEADS, *, qk_nope=QK_NOPE, v_dim=V_HEAD, kv=KV_RANK, ranks=RANKS):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    full_k, rank_max = qk_nope + v_dim, max(ranks)
    # Inactive rank tails deliberately contain stale pool contents.
    a = torch.randn(SLOTS, rank_max, kv, generator=g, device=DEVICE) * 100
    b = torch.randn(SLOTS, heads * full_k, rank_max, generator=g, device=DEVICE) * 100
    for slot, rank in enumerate(ranks):
        a[slot, :rank] = torch.randn(rank, kv, generator=g, device=DEVICE) * 0.05
        b[slot, :, :rank] = (
            torch.randn(heads * full_k, rank, generator=g, device=DEVICE) * 0.05
        )
    return a.to(DTYPE).contiguous(), b.to(DTYPE).contiguous()


def _attention(engine, info, a, b, heads=HEADS, *, qk_nope=QK_NOPE, v_dim=V_HEAD):
    return SimpleNamespace(
        num_local_heads=heads,
        qk_nope_head_dim=qk_nope,
        v_head_dim=v_dim,
        kv_b_proj=SimpleNamespace(
            set_lora=True,
            A_buffer=a,
            B_buffer=b,
            lora_backend=SimpleNamespace(
                name="triton_v2", runner=engine, batch_info=info
            ),
        ),
    )


def _reference(x, a, b, slots, side, *, ranks=RANKS, scales=SCALINGS, qk_nope=QK_NOPE):
    tokens, heads, _ = x.shape
    full_k = b.shape[1] // heads
    width = a.shape[-1] if side == "q" else full_k - qk_nope
    delta = torch.zeros(tokens, heads, width, dtype=torch.float32, device=x.device)
    for token, slot in enumerate(slots):
        if slot < 0 or ranks[slot] == 0:
            continue
        rank = ranks[slot]
        b_heads = b[slot, :, :rank].reshape(heads, full_k, rank).float()
        if side == "q":
            bridge = torch.einsum("hk,hkr->hr", x[token].float(), b_heads[:, :qk_nope])
            bridge = bridge.to(DTYPE).float()  # The shrink materializes a BF16 bridge.
            value = bridge @ a[slot, :rank].float()
        else:
            bridge = (x[token].float() @ a[slot, :rank].float().T).to(DTYPE).float()
            value = torch.einsum("hr,hkr->hk", bridge, b_heads[:, qk_nope:])
        delta[token] = value * scales[slot]
    return delta


@pytest.mark.parametrize("decode", [True, False])
@pytest.mark.parametrize("side", ["q", "v"])
def test_correction_matches_legacy_and_reference(decode, side):
    info, slots = _batch(decode)
    a, b = _pool(5)
    attention = _attention(_engine(info, slots, decode), info, a, b)
    g = torch.Generator(device=DEVICE).manual_seed(7)
    input_dim, output_dim = (QK_NOPE, KV_RANK) if side == "q" else (KV_RANK, V_HEAD)
    x = torch.randn(
        len(slots), HEADS, input_dim, generator=g, device=DEVICE, dtype=DTYPE
    )
    base = torch.randn(
        len(slots), HEADS, output_dim, generator=g, device=DEVICE, dtype=DTYPE
    )
    prepare = (
        mla_correction.prepare_q_correction
        if side == "q"
        else mla_correction.prepare_v_correction
    )
    prepared = prepare(attention, x)
    assert (prepared.done is not None) == decode
    destination = base.clone()
    ours = mla_correction.apply_correction(
        prepared, destination if side == "q" else destination.flatten(1, 2)
    ).view_as(base)
    if side == "q":
        legacy = step_b_q_fwd(step_a_q_fwd(x, b, info, FULL_K), a, info, base.clone())
    else:
        legacy = base.clone()
        step_b_v_fwd(step_a_v_fwd(x, a, info), b, info, legacy, QK_NOPE, V_HEAD)
    torch.testing.assert_close(ours.float(), legacy.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        ours.float(),
        base.float() + _reference(x, a, b, slots, side),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("mode", ["eager", "decode_graph", "prefill_graph"])
def test_pair_route_stream_reuse_and_replay(monkeypatch, mode):
    decode = mode != "prefill_graph"
    info, slots = _batch(decode)
    tokens = len(slots)
    engine = _engine(info, slots, decode, graph_mode=mode != "eager")
    layers = [_attention(engine, info, *_pool(seed)) for seed in (31, 37)]
    # Flattening this Q copies; the copy must precede the side-stream ready event.
    q = torch.randn(HEADS, tokens, QK_NOPE, device=DEVICE, dtype=DTYPE).transpose(0, 1)
    v = torch.randn(tokens, HEADS, KV_RANK, device=DEVICE, dtype=DTYPE)
    bases = [
        torch.randn(tokens, HEADS, width, device=DEVICE, dtype=DTYPE)
        for width in (KV_RANK, V_HEAD)
    ]
    outputs = [[torch.empty_like(base) for base in bases] for _ in layers]
    caller = torch.cuda.Stream()
    builds = []
    build = engine._build_pair_route

    def checked_build(*args):
        assert torch.cuda.current_stream() == caller
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
            mla_correction.apply_correction(q_prepared, q_out)
            v_prepared = mla_correction.prepare_v_correction(layer, v)
            v_out.copy_(bases[1])
            mla_correction.apply_correction(v_prepared, v_out)
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
            assert len(builds) == 3  # Replay cannot rebuild a Python route cache.
        torch.cuda.synchronize()
        for layer, (q_out, v_out) in zip(layers, outputs):
            a, b = layer.kv_b_proj.A_buffer, layer.kv_b_proj.B_buffer
            for side, x, out, base in zip(("q", "v"), (q, v), (q_out, v_out), bases):
                expected = base.float() + _reference(
                    x, a, b, current_slots, side, scales=scales
                )
                torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("phase", [Phase.DECODE, Phase.PREFILL])
def test_mla_pair_route_is_rebuilt_by_the_replayed_graph(phase):
    ranks = [64, 32, 0, 16]
    info, _ = _batch(True, tokens=37, ranks=ranks)
    tokens, heads, qk_nope, v_dim, kv = 37, 4, 64, 32, 96
    a, b = _pool(23, heads, qk_nope=qk_nope, v_dim=v_dim, kv=kv, ranks=ranks)
    generator = torch.Generator(device=DEVICE).manual_seed(23)

    def random(*shape):
        return torch.randn(*shape, generator=generator, device=DEVICE, dtype=DTYPE)

    w_kc, w_vc = random(heads, qk_nope, kv) * 0.1, random(heads, kv, v_dim) * 0.1
    q, v = random(tokens, heads, qk_nope), random(tokens, heads, kv)
    q_out, v_out = (
        torch.empty_like(v),
        torch.empty(tokens, heads, v_dim, device=DEVICE, dtype=DTYPE),
    )
    engine = DenseLoraRunner(LoraWorkspace(), max_loras=SLOTS, device=DEVICE)
    attention = _attention(engine, info, a, b, heads, qk_nope=qk_nope, v_dim=v_dim)
    static_slots = torch.full((tokens + 27,), -1, dtype=torch.int32, device=DEVICE)
    block = mla_correction._mla_plan(phase).block_size

    def begin(live=tokens):
        _compute_moe_lora_info(
            live,
            info.seg_indptr[: live + 1],
            info.lora_ranks,
            info.weight_indices[:live],
            None,
            static_slots,
            max_len=info.max_len,
        )
        engine.begin_batch(
            token_slots=static_slots,
            lora_ranks=info.lora_ranks,
            scalings=info.scalings,
            num_tokens=live,
            phase=phase,
            graph_mode=True,
        )

    def forward():
        engine.reset_routes()
        for side, x, weight, out, prepare in (
            ("q", q, w_kc, q_out, mla_correction.prepare_q_correction),
            ("v", v, w_vc, v_out, mla_correction.prepare_v_correction),
        ):
            prepared = prepare(attention, x)
            assert (prepared.done is not None) == (phase is Phase.DECODE)
            destination = torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)
            assert not destination.is_contiguous()
            out.copy_(mla_correction.apply_correction(prepared, destination))

    def check():
        slots = static_slots[:tokens].tolist()
        for side, x, weight, out in (("q", q, w_kc, q_out), ("v", v, w_vc, v_out)):
            expected = torch.bmm(x.transpose(0, 1), weight).transpose(0, 1).float()
            expected += _reference(x, a, b, slots, side, ranks=ranks, qk_nope=qk_nope)
            torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        begin()
        assert not engine.workspace.routes
        forward()
        route = engine.pair_route(heads, block, num_tokens=tokens)
        pair_ptr, group_ptr = (
            route.sorted_pair_ids.data_ptr(),
            route.group_ids.data_ptr(),
        )
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    begin()
    assert not engine.workspace.routes
    with torch.cuda.graph(graph, stream=stream):
        forward()
        captured_route = engine.pair_route(heads, block, num_tokens=tokens)
    assert captured_route.sorted_pair_ids.data_ptr() == pair_ptr
    assert captured_route.group_ids.data_ptr() == group_ptr

    for live in (tokens, tokens - 13):
        q.copy_(random(tokens, heads, qk_nope))
        v.copy_(random(tokens, heads, kv))
        info.weight_indices.copy_(torch.roll(info.weight_indices, 1))
        begin(live)
        assert torch.all(static_slots[live:] == -1)
        assert not engine.workspace.routes
        assert captured_route.group_ids.data_ptr() == group_ptr
        captured_route.group_ids.fill_(-7)
        captured_route.sorted_pair_ids.fill_(-7)
        captured_route.block_bucket_ids.fill_(-7)
        captured_route.num_pairs_post_padded.zero_()
        graph.replay()
        torch.cuda.synchronize()
        check()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
