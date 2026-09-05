import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    _align_pipeline_layers,
    capture_decode_graph,
    has_standard_gqa_for_all_local_layers,
    index_attention_layers_by_global_id,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


class TestDeepGemmLayoutBudgetCapture(CustomTestCase):
    def test_final_refresh_skips_uninitialized_non_cuda_and_forced_layouts(self):
        from sglang.srt.layers.moe.moe_runner import deep_gemm

        real_import = __import__

        def forbid_deep_gemm_import(name, *args, **kwargs):
            if name == "sglang.srt.layers.moe.moe_runner.deep_gemm":
                raise AssertionError("inactive budget must not import DeepGEMM")
            return real_import(name, *args, **kwargs)

        for device, layout, budget in (
            ("cuda", "auto", None),
            ("cpu", "auto", 4 * (1 << 30)),
            ("cuda", "masked", 4 * (1 << 30)),
            ("cuda", "compact", 4 * (1 << 30)),
        ):
            with (
                self.subTest(device=device, layout=layout, budget=budget),
                (
                    patch.object(
                        deep_gemm, "_masked_standard_layout_memory_budget_bytes", budget
                    )
                ),
                patch.object(
                    cuda_graph_setup,
                    "_deep_gemm_layout_memory_budget_initialized",
                    budget is not None,
                ),
                patch("builtins.__import__", side_effect=forbid_deep_gemm_import),
                deep_gemm.envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override(layout),
                (
                    patch.object(
                        cuda_graph_setup,
                        "get_world_group",
                        side_effect=AssertionError(
                            "no final collective for an inactive auto budget"
                        ),
                    )
                ),
                patch.object(
                    cuda_graph_setup,
                    "get_available_gpu_memory",
                    side_effect=AssertionError(
                        "inactive budget must not query GPU memory"
                    ),
                ),
            ):
                cuda_graph_setup.refresh_deep_gemm_layout_memory_budget(
                    SimpleNamespace(device=device), only_if_initialized=True
                )
                self.assertEqual(
                    deep_gemm._masked_standard_layout_memory_budget_bytes, budget
                )

    def test_precapture_budget_backend_and_layout_guards(self):
        from sglang.srt.layers.moe.moe_runner import deep_gemm

        cases = (
            # device, target backend, draft backend, is draft, layout, query
            ("cuda", "deep_gemm", None, False, "auto", True),
            ("cuda", "triton", "deep_gemm", True, "auto", True),
            ("cuda", "deep_gemm", "triton", True, "auto", False),
            ("cuda", "triton", None, False, "auto", False),
            ("cpu", "deep_gemm", None, False, "auto", False),
            ("cuda", "deep_gemm", None, False, "masked", False),
            ("cuda", "deep_gemm", None, False, "compact", False),
        )
        for device, target, draft, is_draft, layout, should_query in cases:
            with (
                self.subTest(device=device, target=target, draft=draft, layout=layout),
                ExitStack() as stack,
            ):
                stack.enter_context(
                    patch.object(
                        deep_gemm, "_masked_standard_layout_memory_budget_bytes", None
                    )
                )
                stack.enter_context(
                    patch.object(
                        cuda_graph_setup,
                        "_deep_gemm_layout_memory_budget_initialized",
                        False,
                    )
                )
                stack.enter_context(
                    deep_gemm.envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.override(layout)
                )
                stack.enter_context(
                    deep_gemm.envs.SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION.override(
                        0.25
                    )
                )
                stack.enter_context(
                    patch.object(
                        cuda_graph_setup,
                        "get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(
                                moe_runner_backend=target, moe_a2a_backend="none"
                            ),
                            comm=SimpleNamespace(enable_symm_mem=False),
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        cuda_graph_setup,
                        "get_spec",
                        return_value=SimpleNamespace(
                            speculative_moe_runner_backend=draft,
                            speculative_moe_a2a_backend=None,
                        ),
                    )
                )
                world = SimpleNamespace(world_size=8, cpu_group=object())
                stack.enter_context(
                    patch.object(
                        cuda_graph_setup, "get_world_group", return_value=world
                    )
                )
                queries = []

                def free_memory(device, gpu_id, *, distributed, cpu_group):
                    queries.append((device, gpu_id, distributed, cpu_group))
                    return 16.0

                stack.enter_context(
                    patch.object(
                        cuda_graph_setup,
                        "get_available_gpu_memory",
                        side_effect=free_memory,
                    )
                )
                for name in ("EagerRunner", "prealloc_symmetric_memory_pool"):
                    stack.enter_context(patch.object(cuda_graph_setup, name))
                stack.enter_context(
                    patch.object(
                        cuda_graph_setup.GraphSharedOutput, "create_for_model_runner"
                    )
                )
                stack.enter_context(
                    patch.object(cuda_graph_setup, "capture_prefill_graph")
                )
                runner = SimpleNamespace(
                    device=device,
                    gpu_id=0,
                    is_draft_worker=is_draft,
                    model_config=SimpleNamespace(quantization="fp8"),
                    server_args=SimpleNamespace(forward_hooks=None),
                    forward_stream=object(),
                    canary_manager=None,
                )
                if not should_query:
                    real_import = __import__

                    def forbid_deep_gemm_import(name, *args, **kwargs):
                        if name == "sglang.srt.layers.moe.moe_runner.deep_gemm":
                            raise AssertionError(
                                "ineligible backend must not import DeepGEMM"
                            )
                        return real_import(name, *args, **kwargs)

                    stack.enter_context(
                        patch(
                            "builtins.__import__", side_effect=forbid_deep_gemm_import
                        )
                    )
                cuda_graph_setup.capture_cuda_graphs(
                    model_runner=runner, capture_decode_cuda_graph=False
                )

                self.assertEqual(
                    queries,
                    [("cuda", 0, True, world.cpu_group)] if should_query else [],
                )
                self.assertEqual(
                    deep_gemm._masked_standard_layout_memory_budget_bytes,
                    4 * (1 << 30) if should_query else None,
                )
                self.assertEqual(
                    cuda_graph_setup._deep_gemm_layout_memory_budget_initialized,
                    should_query,
                )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
