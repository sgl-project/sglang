import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    _align_pipeline_layers,
    capture_decode_graph,
    has_standard_gqa_for_all_local_layers,
    index_attention_layers_by_global_id,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("width", [1, 6])
@pytest.mark.parametrize("gathered_buffer_required", [False, True])
@pytest.mark.parametrize(
    "cp_size,dispatch,overrides,model_overrides,expected",
    [
        (1, False, {}, {}, {1: [1, 2, 4, 8], 6: [1, 2, 4, 8]}),
        (8, False, {}, {}, {1: [8], 6: [4, 8]}),
        (8, True, {}, {}, {1: [1, 2, 4, 8], 6: [1, 2, 4, 8]}),
        (4, True, {}, {}, {1: [4, 8], 6: [2, 4, 8]}),
        (8, True, {"enable_lora": True}, {}, {1: [8], 6: [4, 8]}),
        (8, True, {"enable_layernorm_sp": True}, {}, {1: [8], 6: [4, 8]}),
        (8, True, {"enable_quant_communications": True}, {}, {1: [8], 6: [4, 8]}),
        (8, True, {}, {"q_lora_rank": None}, {1: [8], 6: [4, 8]}),
        (8, True, {}, {"index_topk": None}, {1: [8], 6: [4, 8]}),
    ],
)
def test_cp_decode_dispatch_capture_buckets(
    width,
    gathered_buffer_required,
    cp_size,
    dispatch,
    overrides,
    model_overrides,
    expected,
):
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        get_batch_sizes_to_capture,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.utils.common import get_cuda_graph_batch_size_alignment

    hf_config = SimpleNamespace(
        **{
            "architectures": ["GlmMoeDsaForCausalLM"],
            "index_topk": 2048,
            "q_lora_rank": 2048,
            **model_overrides,
        }
    )
    model_runner = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        req_to_token_pool=SimpleNamespace(size=8),
        server_args=SimpleNamespace(dllm_algorithm=None),
    )
    with get_context().override_server_args(
        tp_size=8,
        attn_cp_size=cp_size,
        enable_prefill_cp=cp_size > 1,
        enable_cp_decode_attn_tp=dispatch,
        cp_strategy="interleave",
        device="cuda",
        enable_torch_compile=False,
        enable_two_batch_overlap=False,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="default", bs=[1, 2, 4, 8])
        ),
        **overrides,
    ):
        capture_bs, compile_bs = get_batch_sizes_to_capture(
            model_runner, width, gathered_buffer_required=gathered_buffer_required
        )
        # A gather has its own row-alignment contract. Dispatch must not
        # relax it, even when attention alone could run an unpadded graph.
        assert capture_bs == (
            {1: [8], 6: [4, 8]}[width] if gathered_buffer_required else expected[width]
        )
        assert compile_bs == []
        # Only the decode bucket builder opts out. Other users, including
        # startup capacity alignment, retain the original CP requirement.
        assert (
            get_cuda_graph_batch_size_alignment(gathered_buffer_required=False)
            == cp_size
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


def test_draft_startup_preserves_runner_local_capture_sizes(monkeypatch, caplog):
    """Startup must not reject local draft buckets using global TP alignment."""
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        get_batch_sizes_to_capture,
    )
    from sglang.srt.runtime_context import get_context
    from sglang.srt.utils import common

    capture_bs = [1, 2, 3, 4, 8]
    # The capture decision reads the graph configuration and the MoE backends
    # out of the bags.
    override = get_context().override_server_args(
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(backend="default", bs=capture_bs)
        ),
        enable_torch_compile=False,
        enable_two_batch_overlap=False,
    )
    override.install()

    class CustomGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner
            self.capture_bs, _ = get_batch_sizes_to_capture(
                model_runner, 7, gathered_buffer_required=False
            )

    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = True
        spec_algorithm = SimpleNamespace(is_speculative=lambda: True)
        server_args = SimpleNamespace(model_impl="auto", dllm_algorithm=None)
        req_to_token_pool = SimpleNamespace(size=32)

        def decode_num_tokens_per_req(self):
            return 7

        def _decode_cuda_graph_runner_cls(self):
            return CustomGraphRunner

    model_runner = TestModelRunner()
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(common, "require_gathered_buffer", lambda: True)
    monkeypatch.setattr(
        common, "get_parallel", lambda: SimpleNamespace(attn_tp_size=16, attn_cp_size=1)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )

    try:
        with caplog.at_level("INFO", logger=cuda_graph_setup.__name__):
            capture = capture_decode_graph(model_runner=model_runner)

        assert isinstance(capture.runner, CustomGraphRunner)
        assert capture.runner.model_runner is model_runner
        assert capture.runner.capture_bs == capture_bs
        assert f"bs={capture_bs}" in caplog.text
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
