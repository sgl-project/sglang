import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    _align_pipeline_layers,
    capture_decode_graph,
    count_attention_free_layers,
    has_standard_gqa_for_all_local_layers,
    index_attention_layers_by_global_id,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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


def test_standard_gqa_gate_counts_attention_free_hybrid_layers():
    # Nemotron-H style stack: attention, Mamba, MoE, MLP. The MoE and MLP
    # stages keep a None attention slot but need no attention metadata.
    attention = SimpleNamespace()
    mamba = SimpleNamespace()
    moe = SimpleNamespace(is_attention_free=True)
    mlp = SimpleNamespace(is_attention_free=True)
    layer_model = SimpleNamespace(layers=[attention, mamba, moe, mlp])
    attention_layers = [attention, mamba, None, None]

    attention_free = count_attention_free_layers(layer_model)

    assert attention_free == 2
    assert has_standard_gqa_for_all_local_layers(
        attention_layer_count=sum(layer is not None for layer in attention_layers)
        + attention_free,
        start_layer=0,
        end_layer=4,
    )


def test_standard_gqa_gate_still_rejects_unrecognized_attention():
    # The None slot belongs to an attention layer the gate cannot see.
    attention = SimpleNamespace()
    unrecognized = SimpleNamespace()
    moe = SimpleNamespace(is_attention_free=True)
    layer_model = SimpleNamespace(layers=[attention, unrecognized, moe])
    attention_layers = [attention, None, None]

    assert not has_standard_gqa_for_all_local_layers(
        attention_layer_count=sum(layer is not None for layer in attention_layers)
        + count_attention_free_layers(layer_model),
        start_layer=0,
        end_layer=3,
    )


def test_nemotron_h_ffn_layers_are_attention_free():
    from sglang.srt.models.nemotron_h import (
        NemotronHAttentionDecoderLayer,
        NemotronHMambaDecoderLayer,
        NemotronHMLPDecoderLayer,
        NemotronHMoEDecoderLayer,
    )

    assert NemotronHMLPDecoderLayer.is_attention_free
    assert NemotronHMoEDecoderLayer.is_attention_free
    assert not getattr(NemotronHAttentionDecoderLayer, "is_attention_free", False)
    assert not getattr(NemotronHMambaDecoderLayer, "is_attention_free", False)


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
    from sglang.srt.runtime_context import get_context

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
