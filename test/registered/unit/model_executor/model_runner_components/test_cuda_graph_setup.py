import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    _align_pipeline_layers,
    capture_decode_graph,
    has_standard_gqa_for_all_local_layers,
    index_attention_layers_by_global_id,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _publish_graph_config(request, *, prefill_backend, decode_backend):
    # The worker's ServerArgs record only carries the raw flag value; the
    # resolved config lives in the exec bag, which is what the prewarm reads.
    override = get_context().override_server_args(
        cuda_graph_config=SimpleNamespace(
            prefill=SimpleNamespace(backend=prefill_backend),
            decode=SimpleNamespace(backend=decode_backend),
        ),
    )
    override.install()
    request.addfinalizer(override.restore)


def test_standard_gqa_gate_uses_pipeline_local_layer_range():
    # PP rank owns layers [23, 46), while the full model has 92 layers.
    assert has_standard_gqa_for_all_local_layers(
        attention_layer_count=23, start_layer=23, end_layer=46
    )
    assert not has_standard_gqa_for_all_local_layers(
        attention_layer_count=22, start_layer=23, end_layer=46
    )


def test_standard_gqa_gate_is_unchanged_without_pipeline_parallelism():
    assert has_standard_gqa_for_all_local_layers(
        attention_layer_count=92, start_layer=0, end_layer=92
    )


def test_pipeline_attention_metadata_is_indexed_by_global_layer_id():
    layer23 = SimpleNamespace(layer_id=23)
    layer24 = SimpleNamespace(layer_id=24)
    companion24 = object()

    attention, companions = index_attention_layers_by_global_id(
        [layer23, layer24], [None, companion24]
    )

    assert len(attention) == 25
    assert all(layer is None for layer in attention[:23])
    assert attention[23] is layer23
    assert attention[24] is layer24
    assert companions[23] is None
    assert companions[24] is companion24


def test_reuse_tables_pass_through_but_distinct_duplicates_raise():
    looped = SimpleNamespace(layer_id=1)
    companion = object()
    attention_in = [SimpleNamespace(layer_id=0), looped, looped]
    companions_in = [None, companion, companion]

    attention, companions = index_attention_layers_by_global_id(
        attention_in, companions_in
    )

    assert attention is attention_in
    assert companions is companions_in

    with pytest.raises(ValueError, match="duplicate attention layer_id: 2"):
        index_attention_layers_by_global_id(
            [SimpleNamespace(layer_id=2), SimpleNamespace(layer_id=2)], [None, None]
        )


def test_model_runner_can_override_decode_graph_runner(monkeypatch):
    # The capture decision reads the graph configuration and the MoE backends
    # out of the bags.
    override = get_context().override_server_args(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="default")),
    )
    override.install()

    class CustomGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = False
        spec_algorithm = SimpleNamespace(is_speculative=lambda: False)
        server_args = SimpleNamespace(model_impl="auto")

        def _decode_cuda_graph_runner_cls(self):
            return CustomGraphRunner

    model_runner = TestModelRunner()
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )

    try:
        capture = capture_decode_graph(model_runner=model_runner)

        assert isinstance(capture.runner, CustomGraphRunner)
        assert capture.runner.model_runner is model_runner
    finally:
        override.restore()


def test_align_pipeline_layers_uses_absolute_indices():
    class PipelineStage:
        start_layer = 3
        end_layer = 5
        layers = [object()] * 8

    local_layers = ["layer-3", "layer-4"]
    assert _align_pipeline_layers(local_layers, PipelineStage()) == [
        None,
        None,
        None,
        "layer-3",
        "layer-4",
        None,
        None,
        None,
    ]
    full_model = SimpleNamespace(layers=local_layers)
    assert _align_pipeline_layers(local_layers, full_model) == local_layers
    with pytest.raises(AssertionError, match="together"):
        _align_pipeline_layers(
            local_layers, SimpleNamespace(start_layer=0, layers=local_layers)
        )


def test_cuda_graph_prewarm_delegates_to_the_language_model(monkeypatch, request):
    _publish_graph_config(request, prefill_backend="full", decode_backend="piecewise")
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    language_model = SimpleNamespace(prewarm_cuda_graphs=prewarm)
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
    )
    monkeypatch.setattr(
        cuda_graph_setup, "resolve_language_model", lambda _: language_model
    )
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: True, raising=False
    )

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_called_once_with(runner, capture_decode_cuda_graph=True)


def test_cuda_graph_prewarm_is_required_for_ple_offload(monkeypatch, request):
    _publish_graph_config(request, prefill_backend="full", decode_backend="piecewise")
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_offload_embedding=True)
        ),
    )
    monkeypatch.setattr(
        cuda_graph_setup,
        "resolve_language_model",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: True, raising=False
    )

    with pytest.raises(RuntimeError, match="PLE offload.*prewarm_cuda_graphs"):
        cuda_graph_setup._prewarm_model_cuda_graphs(
            runner, capture_decode_cuda_graph=True
        )


def test_cuda_graph_prewarm_does_not_reach_non_sm120_models(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    runner = SimpleNamespace(
        device="cuda",
        model=SimpleNamespace(prewarm_cuda_graphs=prewarm),
    )
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", lambda model: model)
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: False, raising=False
    )

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_not_called()


def test_cuda_graph_prewarm_does_not_reach_sm121_models(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    runner = SimpleNamespace(
        device="cuda",
        model=SimpleNamespace(prewarm_cuda_graphs=prewarm),
    )
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", lambda model: model)
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: True, raising=False
    )
    monkeypatch.setattr(cuda_graph_setup, "is_sm121", lambda: True, raising=False)

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_not_called()


def test_capture_cuda_graphs_prewarms_before_prefill_capture(monkeypatch):
    # The capture path reads the forward-hooks and symm-mem decisions out of
    # the bags, so publish them instead of faking them on the runner.
    override = get_context().override_server_args(
        forward_hooks=None,
        enable_symm_mem=False,
    )
    override.install()

    runner = SimpleNamespace(
        device="cpu",
        model=object(),
        model_config=SimpleNamespace(quantization=None),
        is_draft_worker=False,
        server_args=SimpleNamespace(
            moe_runner_backend="cutlass",
            moe_a2a_backend="none",
        ),
        forward_stream=None,
        canary_manager=None,
    )
    eager_runner = object()
    calls = MagicMock()
    prewarm = calls.prewarm
    capture_prefill = calls.capture_prefill
    prefill = cuda_graph_setup.GraphCapture(
        runner=eager_runner,
        memory_phase="prefill",
        memory_usage_gb=0,
        capture_time=0,
    )
    monkeypatch.setattr(
        cuda_graph_setup.GraphSharedOutput,
        "create_for_model_runner",
        lambda _: object(),
    )
    monkeypatch.setattr(cuda_graph_setup, "EagerRunner", lambda _: eager_runner)
    monkeypatch.setattr(cuda_graph_setup, "_prewarm_model_cuda_graphs", prewarm)
    capture_prefill.return_value = prefill
    monkeypatch.setattr(cuda_graph_setup, "capture_prefill_graph", capture_prefill)
    monkeypatch.setattr(
        cuda_graph_setup, "prealloc_symmetric_memory_pool", lambda **_: None
    )

    try:
        cuda_graph_setup.capture_cuda_graphs(
            model_runner=runner, capture_decode_cuda_graph=False
        )

        assert calls.mock_calls[:2] == [
            call.prewarm(runner, capture_decode_cuda_graph=False),
            call.capture_prefill(model_runner=runner, eager_runner=eager_runner),
        ]
    finally:
        override.restore()


def test_cuda_graph_prewarm_skips_when_both_phases_are_disabled(monkeypatch, request):
    _publish_graph_config(
        request, prefill_backend="disabled", decode_backend="disabled"
    )
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    runner = SimpleNamespace(
        device="cuda",
        model=SimpleNamespace(prewarm_cuda_graphs=prewarm),
    )
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", lambda model: model)
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: True, raising=False
    )

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
