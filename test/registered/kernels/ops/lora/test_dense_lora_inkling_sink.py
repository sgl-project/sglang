"""Inkling sink graph replay across adapter switches, slot swaps and loader reloads.
The wider matrix lives in test_dense_lora_inkling_sink_matrix.py.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.srt.model_executor.runner_utils.capture_mode import (  # noqa: E402
    capture_lora_variant,
    model_capture_mode,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.kits.inkling_sink_lora_kit import (
    DEVICE,
    DTYPE,
    RANKS,
    SLOTS,
    TOL,
    V2,
    _base_w13,
    _batch,
    _fill_slot,
    _forward,
    _inputs,
    _layer,
    _load_w2,
    _load_w13,
    _pool,
    _prepare,
    _real_sink,
    _reference,
    _v2_backend,
)

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the sink kernels need CUDA"
)


def _loaded_v2(backend, bufs, w13_lin, w2_lin):
    """Load and post-process base weights before attaching the LoRA wrapper."""
    layer = _real_sink(V2)
    _load_w13(layer, w13_lin)
    _load_w2(layer, w2_lin)
    layer.process_weights_after_loading()
    layer.initialize_lora(backend)
    layer.set_lora_info(*bufs)
    return layer


def _capture(backend, fn):
    """Warm up, then capture one forward and return its graph and static output."""

    def forward():
        backend.reset_routing_cache()
        return fn()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with model_capture_mode(), capture_lora_variant(None):
        with torch.cuda.stream(stream):
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = forward()
    return graph, out


@pytest.mark.parametrize(
    "prefill", [False, True], ids=["decode_graph", "prefill_graph"]
)
@pytest.mark.parametrize("shared_outer", [True, False], ids=["shared", "per_expert"])
def test_v2_graph_replay_follows_adapters_and_slot_swaps(shared_outer, prefill):
    bufs = _pool(shared_outer, seed=9)
    fb, slots, token_slots = _batch(not prefill, slots=[2] * (4 if prefill else 13))
    t = fb.extend_num_tokens
    x, gammas, w13_lin, w2_lin = _inputs(t)
    backend = _v2_backend()
    if prefill:
        backend.init_prefill_cuda_graph_batch_info(t)
        _prepare(backend, fb, slots, prefill_graph=True)
    else:
        backend.init_decode_cuda_graph_batch_info(fb.batch_size, 1)
        _prepare(backend, fb, slots, graph=True)
    assert not backend.batch_info.has_active_lora
    layer = _layer(V2, backend, bufs)
    w13_base = _base_w13(w13_lin)
    graph, out = _capture(backend, lambda: _forward(layer, x, gammas, w13_base, w2_lin))

    def check(token_slots):
        graph.replay()
        expected = _reference(
            x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer
        )
        torch.testing.assert_close(out.float(), expected, **TOL)

    # Captured on an empty route: every bucket is -1, so the replay is the base.
    check(token_slots)
    fb, slots, token_slots = _batch(not prefill)
    _prepare(backend, fb, slots, graph=not prefill, prefill_graph=prefill)
    check(token_slots)
    # The next batch routes its requests to other adapters (and one to none).
    new_slots = [3, 2, 1, 0] if prefill else [(i + 1) % SLOTS for i in range(13)]
    fb2, _, token_slots2 = _batch(not prefill, slots=new_slots)
    _prepare(backend, fb2, new_slots, graph=not prefill, prefill_graph=prefill)
    check(token_slots2)
    # A pool slot is replaced (the empty slot 2 takes a rank-8 adapter, slot 0
    # a new rank-32 one) after capture: the replay reads the new weights.
    g = torch.Generator(device=DEVICE).manual_seed(99)
    ranks = list(RANKS)
    for slot, rank in ((2, 8), (0, 32)):
        _fill_slot(bufs, slot, rank, g)
        ranks[slot] = rank
        layer.on_lora_slots_updated({slot})
    fb3, _, token_slots3 = _batch(not prefill, slots=new_slots, ranks=ranks)
    _prepare(backend, fb3, new_slots, ranks, graph=not prefill, prefill_graph=prefill)
    check(token_slots3)


@pytest.mark.parametrize(
    "prefill,shared_outer",
    [(False, True), (True, False)],
    ids=["decode_graph-shared", "prefill_graph-per_expert"],
)
def test_w13_reload_through_the_loader_reaches_the_captured_graph(
    prefill, shared_outer
):
    """Reloads reach the next replay without post-load processing or an eager repair."""
    bufs = _pool(shared_outer, seed=21)
    fb, slots, token_slots = _batch(not prefill)
    t = fb.extend_num_tokens
    x, gammas, w13_lin, w2_lin = _inputs(t)
    backend = _v2_backend()
    if prefill:
        backend.init_prefill_cuda_graph_batch_info(t)
        _prepare(backend, fb, slots, prefill_graph=True)
    else:
        backend.init_decode_cuda_graph_batch_info(fb.batch_size, 1)
        _prepare(backend, fb, slots, graph=True)
    layer = _loaded_v2(backend, bufs, w13_lin, w2_lin)
    assert torch.equal(layer.w13_weight.data.view_as(w13_lin), _base_w13(w13_lin))
    graph, out = _capture(
        backend, lambda: layer.forward(x, gammas, use_reduce_scatter=True)
    )
    addresses = (
        layer.w13_weight.data_ptr(),
        layer.w2_weight.data_ptr(),
        layer._w2_lin.data_ptr(),
    )

    def check(w13_now, w2_now):
        # The loaded rows are converted exactly once (twice would not be
        # [gate | up] either), then the existing graph replays on them.
        assert torch.equal(layer.w13_weight.data.view_as(w13_now), _base_w13(w13_now))
        graph.replay()
        expected = _reference(
            x, gammas, w13_now, w2_now, bufs, token_slots, shared_outer
        )
        torch.testing.assert_close(out.float(), expected, **TOL)
        assert (
            layer.w13_weight.data_ptr(),
            layer.w2_weight.data_ptr(),
            layer._w2_lin.data_ptr(),
        ) == addresses

    check(w13_lin, w2_lin)
    # From here on nothing may repair the weights but the loader itself.
    layer.forward = layer.get_bf16_linearized_weights = lambda *a, **k: pytest.fail(
        "a reload must not need an eager forward or the linearized getter"
    )
    g = torch.Generator(device=DEVICE).manual_seed(31)

    def fresh(like):
        return (torch.randn(like.shape, generator=g, device=DEVICE) * 0.02).to(DTYPE)

    w13_a = fresh(w13_lin)
    _load_w13(layer, w13_a)  # W13 alone
    check(w13_a, w2_lin)
    w2_b, w13_b = fresh(w2_lin), fresh(w13_lin)
    _load_w2(layer, w2_b)  # W2, then W13
    _load_w13(layer, w13_b)
    check(w13_b, w2_b)
    w13_c, w2_c = fresh(w13_lin), fresh(w2_lin)
    _load_w13(layer, w13_c)  # W13, then W2
    _load_w2(layer, w2_c)
    check(w13_c, w2_c)
    w13_d = fresh(w13_lin)
    _load_w13(layer, w13_d)  # and again
    check(w13_d, w2_c)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
