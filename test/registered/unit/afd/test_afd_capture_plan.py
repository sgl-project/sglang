"""Native size planning, bounded A/F mapping, and admission without observation slots."""

import pytest

from sglang.srt.afd.cache import AFDShapeCache, make_shape
from sglang.srt.afd.config import AFDConfig
from sglang.srt.afd.contracts import AFDReason
from sglang.srt.afd.integration import _capture_sizes
from sglang.srt.model_executor.cuda_graph_config import (
    CudaGraphConfig,
    PhaseConfig,
    filter_capture_sizes,
    pad_to_capture_size,
)
from sglang.srt.model_executor.runner.base_cuda_graph_runner import BaseCudaGraphRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "alignment,width,expected",
    [(1, 1, [1, 2, 4, 6]), (2, 1, [2, 4, 6]), (4, 2, [2, 4, 6])],
)
def test_native_capacity_filter_and_matching(alignment, width, expected):
    sizes = filter_capture_sizes(
        [8, 4, 2, 1, 4], max_size=6, alignment=alignment, request_width=width
    )
    assert sizes == expected
    assert BaseCudaGraphRunner._pad_to_bucket is pad_to_capture_size
    assert pad_to_capture_size(3, sizes) == 4
    with pytest.raises(AssertionError, match="exceeds max"):
        pad_to_capture_size(7, sizes)


@pytest.mark.parametrize(
    "native_sizes,rows,expected",
    [([2, 4, 30], (15, 14), 15), ([2, 4, 32], (15, 15), 16)],
)
def test_best_fixed_points_keep_the_same_stage_payload(native_sizes, rows, expected):
    graph = CudaGraphConfig(
        decode=PhaseConfig(backend="full", bs=native_sizes, max_bs=max(native_sizes)),
        prefill=PhaseConfig(backend="disabled"),
    )
    sizes = _capture_sizes(graph)
    cfg = AFDConfig(lanes=2, attention_lanes=8)
    shapes = [
        make_shape(
            lane=lane,
            lane_rows=(rows,) * 8,
            hidden_size=64,
            dtype="bfloat16",
            config=cfg,
            capture_sizes=sizes,
        )
        for lane in range(8)
    ]
    assert {shape.digest for shape in shapes} == {shapes[0].digest}
    assert all(shape.bucket_rows == (expected, expected) for shape in shapes)
    assert shapes[0].group_total_bucket_rows(ffn_ordinal=0) == (expected * 4,) * 2
    graph.decode.backend = "disabled"
    assert _capture_sizes(graph) == ()


def test_unplanned_shapes_do_not_consume_planned_capacity():
    cfg = AFDConfig()
    sizes = (16,)
    cache = AFDShapeCache(config=cfg, capture_sizes=sizes)
    for step, rows in enumerate(range(17, 26)):
        shape = make_shape(
            lane=0,
            lane_rows=((rows, rows),),
            hidden_size=64,
            dtype="bfloat16",
            config=cfg,
            capture_sizes=sizes,
        )
        result = cache.select(shape=shape, estimated_hbm_bytes=128)
        assert result.reason == AFDReason.BUCKET_LIMIT
        assert cache.buckets == ()
        assert cache.retained_hbm_bytes == 0
    hot = make_shape(
        lane=0,
        lane_rows=((15, 14),),
        hidden_size=64,
        dtype="bfloat16",
        config=cfg,
        capture_sizes=sizes,
    )
    selection = cache.select(shape=hot, estimated_hbm_bytes=128, capture=True)
    assert selection.arming
    assert len(cache.buckets) == 1
    with pytest.raises(RuntimeError, match="CONCURRENT_ARM"):
        cache.select(shape=hot, estimated_hbm_bytes=128, capture=True)
    assert cache.retained_hbm_bytes == 128


def test_only_startup_installs_graphs_and_ready_requires_the_whole_plan():
    from types import SimpleNamespace

    from sglang.srt.afd.contracts import AFDError

    cfg = AFDConfig()
    cache = AFDShapeCache(config=cfg, capture_sizes=(2, 4))
    shapes = [
        make_shape(
            lane=0,
            lane_rows=((width, width),),
            hidden_size=8,
            dtype="bfloat16",
            config=cfg,
            capture_sizes=(2, 4),
        )
        for width in (4, 2)
    ]
    # Native eager/autotune forwards before capture must not install a graph.
    assert (
        cache.select(shape=shapes[0], estimated_hbm_bytes=0).reason
        == AFDReason.FORWARD_MODE
    )
    assert not cache.buckets
    for index, shape in enumerate(shapes):
        with pytest.raises(AFDError, match="STARTUP_INCOMPLETE"):
            cache.finish_capture()
        selection = cache.select(shape=shape, estimated_hbm_bytes=128, capture=True)
        selection.bucket.program = SimpleNamespace(close=lambda: None)
        cache.install(bucket=selection.bucket)
    cache.finish_capture()
    assert all(cache.select(shape=s, estimated_hbm_bytes=0).replay for s in shapes)
    with pytest.raises(AFDError, match="RUNTIME_CAPTURE_FORBIDDEN"):
        cache.select(shape=shapes[0], estimated_hbm_bytes=128, capture=True)
    with pytest.raises(AFDError, match="STARTUP_ALREADY_FINISHED"):
        cache.finish_capture()


@pytest.mark.parametrize(
    "sizes,fail", [((1, 2, 4), False), ((), False), ((1, 2, 4), True)]
)
def test_startup_reuses_native_dummy_inputs_in_descending_order(
    monkeypatch, sizes, fail
):
    from types import SimpleNamespace

    import torch

    from sglang.srt.afd.contracts import AFDRole
    from sglang.srt.afd.pipeline import AFDAttentionPipeline
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    events = []
    graph = SimpleNamespace(
        capture_sizes=sizes, finish_capture=lambda: events.append("seal")
    )
    pipeline = AFDAttentionPipeline(
        adapter=SimpleNamespace(role=AFDRole.ATTENTION),
        config=AFDConfig(),
        connector=SimpleNamespace(
            graph_strategy=graph,
            transport=SimpleNamespace(capture_ready=lambda: events.append("ready_ack")),
        ),
        shape_factory=make_shape,
    )
    buffers = object()

    def allocate(**kwargs):
        assert kwargs == dict(max_bs=2 * sizes[-1], allocate_logits_buffer=False)
        events.append("allocate")
        return buffers

    def run(bs, *, forward_mode_override, buffers):
        assert forward_mode_override == ForwardMode.DECODE
        assert pipeline._startup_capture
        events.append(bs)
        if fail:
            raise RuntimeError("capture failed")

    runner = SimpleNamespace(
        _alloc_dummy_decode_buffers=allocate,
        _dummy_run=run,
        model_runner=SimpleNamespace(
            tp_group=SimpleNamespace(barrier=lambda: events.append("barrier"))
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    if fail:
        with pytest.raises(RuntimeError, match="capture failed"):
            pipeline.capture_startup(runner)
        assert events == ["allocate", 8]
    else:
        pipeline.capture_startup(runner)
        assert events == (["allocate", 8, 4, 2, "sync"] if sizes else []) + [
            "seal",
            "ready_ack",
            "barrier",
        ]
    assert not pipeline._startup_capture


@pytest.mark.parametrize("fail", [False, True])
def test_ffn_acknowledges_ready_only_after_complete_capture(monkeypatch, fail):
    from types import SimpleNamespace

    from sglang.srt.afd.contracts import AFDRole, AFDStepDescriptor
    from sglang.srt.afd.pipeline import AFDFFNPipeline

    events = []

    def finish():
        events.append("seal")
        if fail:
            raise RuntimeError("incomplete capture")

    descriptor = AFDStepDescriptor(
        kind="READY",
        step_id=-1,
        lane_stage_rows=(),
        hidden_size=0,
        dtype="",
        num_layers=0,
        graph_eligible=False,
    )
    pipeline = AFDFFNPipeline(
        adapter=SimpleNamespace(
            role=AFDRole.FFN, validate_merge_reduce_scatter=lambda: None
        ),
        config=AFDConfig(),
        device="cpu",
        dtype="bfloat16",
        shape_factory=make_shape,
        connector=SimpleNamespace(
            graph_strategy=SimpleNamespace(finish_capture=finish),
            transport=SimpleNamespace(
                begin_step=lambda value: descriptor,
                capture_ready=lambda: events.append("ack"),
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline, "_run_descriptor", lambda value: pytest.fail("READY ran FFN")
    )
    if fail:
        with pytest.raises(RuntimeError, match="incomplete"):
            pipeline.run_once()
        assert events == ["seal"]
    else:
        assert pipeline.run_once()
        assert events == ["seal", "ack"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
