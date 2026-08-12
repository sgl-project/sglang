import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.cuda_graph_config import Phase
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    has_standard_gqa_for_all_local_layers,
    index_attention_layers_by_global_id,
    capture_decode_graph,
    should_skip_auto_prefill_cuda_graph_for_memory,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_auto_prefill_cuda_graph_memory_gate():
    assert should_skip_auto_prefill_cuda_graph_for_memory(3.99, set())
    assert not should_skip_auto_prefill_cuda_graph_for_memory(4.0, set())


def test_explicit_prefill_backend_bypasses_memory_gate():
    assert not should_skip_auto_prefill_cuda_graph_for_memory(
        0.0, {(Phase.PREFILL, "backend")}
    )


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


def test_model_runner_can_override_decode_graph_runner(monkeypatch):
    class CustomGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = False
        spec_algorithm = SimpleNamespace(is_speculative=lambda: False)
        server_args = SimpleNamespace(
            model_impl="auto",
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="default")
            ),
        )

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

    capture = capture_decode_graph(model_runner=model_runner)

    assert isinstance(capture.runner, CustomGraphRunner)
    assert capture.runner.model_runner is model_runner


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
