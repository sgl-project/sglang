"""CPU contracts for LoRA entry points, graph metadata, and routing-cache reset."""

from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, sentinel

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _lora_method(module, class_name, name):
    # Load the exact method without importing the CUDA backend dependency graph.
    path = Path(__file__).resolve().parents[4] / "python/sglang/srt/lora" / module
    backend = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in backend.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    scope = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        scope,
    )
    return scope[name]


def _base_backend_method(name):
    return _lora_method("backend/base_backend.py", "BaseLoRABackend", name)


@pytest.fixture
def forward_with_base():
    return _base_backend_method("forward_with_base")


@pytest.mark.parametrize("kwargs", [{}, {"all_reduce": None}], ids=["column", "tp1"])
def test_no_reduction_keeps_layer_serial_path(forward_with_base, kwargs):
    calls = Mock()
    calls.base.return_value = sentinel.base_output
    calls.apply_lora.return_value = sentinel.output
    layer = SimpleNamespace(apply_lora=calls.apply_lora)

    output = forward_with_base(
        SimpleNamespace(),
        layer,
        sentinel.x,
        calls.base,
        sentinel.lora_a,
        sentinel.lora_b,
        sentinel.output_offset,
        sentinel.offsets,
        **kwargs,
    )

    assert output is sentinel.output
    assert calls.method_calls == [
        ("base", (), {}),
        ("apply_lora", (sentinel.base_output, sentinel.x), {}),
    ]


def test_row_reduction_keeps_legacy_two_collective_order(forward_with_base):
    calls = Mock()
    calls.run_a.return_value = sentinel.local_a
    calls.base.return_value = sentinel.local_base
    calls.reduce.side_effect = [sentinel.reduced_base, sentinel.reduced_a]
    calls.run_b.return_value = sentinel.output
    backend = SimpleNamespace(
        run_lora_a_sgemm=calls.run_a,
        run_lora_b_sgemm=calls.run_b,
    )
    layer = SimpleNamespace(output_offset_cpu=sentinel.output_offset_cpu)

    output = forward_with_base(
        backend,
        layer,
        sentinel.x,
        calls.base,
        sentinel.lora_a,
        sentinel.lora_b,
        sentinel.output_offset,
        sentinel.offsets,
        all_reduce=calls.reduce,
    )

    assert output is sentinel.output
    assert calls.method_calls == [
        ("base", (), {}),
        ("run_a", (sentinel.x, sentinel.lora_a), {}),
        ("reduce", (sentinel.local_base,), {}),
        ("reduce", (sentinel.local_a,), {}),
        (
            "run_b",
            (),
            {
                "x": sentinel.reduced_a,
                "weights": sentinel.lora_b,
                "output_offset": sentinel.output_offset,
                "output_offset_cpu": sentinel.output_offset_cpu,
                "base_output": sentinel.reduced_base,
            },
        ),
    ]


@pytest.mark.parametrize("has_workspace", [False, True])
def test_routing_cache_reset_preserves_batch_metadata_and_storage(has_workspace):
    workspace = SimpleNamespace(
        routes={"route": sentinel.route},
        _graph_storage={"buffer": sentinel.graph_buffer},
        _eager_buffers={"buffer": sentinel.eager_buffer},
    )
    metadata = {
        "batch_info": sentinel.batch,
        "lm_head_batch_info": sentinel.lm_head,
        "lm_head_pass_batch_infos": sentinel.lm_head_passes,
        "_lm_head_pass_idx": sentinel.pass_index,
        "cuda_graph_batch_info": sentinel.decode_metadata,
        "decode_cuda_graph_batch_info": sentinel.decode_metadata,
        "prefill_cuda_graph_batch_info": sentinel.prefill_metadata,
    }
    backend = SimpleNamespace(
        **metadata, lora_workspace=workspace if has_workspace else None
    )
    routes = workspace.routes

    _base_backend_method("reset_routing_cache")(backend)

    for name, value in metadata.items():
        assert getattr(backend, name) is value
    if has_workspace:
        assert backend.lora_workspace is workspace
        assert workspace.routes is routes and not routes
        assert workspace._graph_storage == {"buffer": sentinel.graph_buffer}
        assert workspace._eager_buffers == {"buffer": sentinel.eager_buffer}


@pytest.mark.parametrize("legacy_signature", [False, True], ids=["v2", "legacy"])
@pytest.mark.parametrize("graph_enabled", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("mode", ["decode", "prefill", "target_verify"])
def test_manager_preserves_graph_ownership_across_backend_signatures(
    legacy_signature, graph_enabled, mode
):
    metadata = {
        family: SimpleNamespace(use_cuda_graph=family != "eager")
        for family in ("eager", "decode", "prefill")
    }
    backend = SimpleNamespace(skip_inactive_lora_batches=False)
    calls = Mock()

    def prepare(batch, indices, ranks, scalings, decode, use_prefill_cuda_graph=False):
        calls(batch, indices, ranks, scalings, decode, use_prefill_cuda_graph)
        family = (
            "decode" if decode else "prefill" if use_prefill_cuda_graph else "eager"
        )
        backend.batch_info = metadata[family]

    def legacy(
        batch, indices, ranks, scalings, use_cuda_graph, use_prefill_cuda_graph=False
    ):
        prepare(batch, indices, ranks, scalings, use_cuda_graph, use_prefill_cuda_graph)

    def renamed(
        batch,
        indices,
        ranks,
        scalings,
        use_decode_cuda_graph,
        use_prefill_cuda_graph=False,
    ):
        prepare(
            batch,
            indices,
            ranks,
            scalings,
            use_decode_cuda_graph,
            use_prefill_cuda_graph,
        )

    backend.prepare_lora_batch = legacy if legacy_signature else renamed
    slots = {None: 0, "adapter": 1}
    manager = SimpleNamespace(
        lora_backend=backend,
        attn_dp_enabled=False,
        max_loras_per_batch=2,
        memory_pool=SimpleNamespace(
            uid_to_buffer_id=slots, get_buffer_id=slots.__getitem__
        ),
        loras={"adapter": SimpleNamespace(config=SimpleNamespace(r=8), scaling=0.5)},
        can_use_prefill_cuda_graph=lambda batch: graph_enabled and mode == "prefill",
    )
    if graph_enabled:
        manager.max_bs_in_decode_cuda_graph = 2
    manager._use_cuda_graph_batch = MethodType(
        _lora_method("lora_manager.py", "LoRAManager", "_use_cuda_graph_batch"), manager
    )
    batch = SimpleNamespace(
        batch_size=2,
        lora_ids=["adapter", None],
        forward_mode=SimpleNamespace(
            is_cuda_graph=lambda: mode != "prefill",
            is_extend=lambda: mode != "decode",
        ),
    )

    _lora_method("lora_manager.py", "LoRAManager", "prepare_lora_batch")(manager, batch)

    decode_graph = graph_enabled and mode != "prefill"
    prefill_graph = graph_enabled and mode == "prefill"
    calls.assert_called_once_with(
        batch, [1, 0], [0, 8], [0, 0.5], decode_graph, prefill_graph
    )
    family = "decode" if decode_graph else "prefill" if prefill_graph else "eager"
    assert backend.batch_info is metadata[family]
    assert backend.batch_info.use_cuda_graph is graph_enabled
    assert backend.batch_info.has_active_lora
    # MoE keeps target verification on its decode plan even on eager fallback.
    assert backend.batch_info.is_prefill is (mode == "prefill")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
