"""CPU contracts for LoRA entry points, graph metadata, and routing-cache reset."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
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


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_moe_core_calls_bound_runner_with_batch_metadata(
    monkeypatch, is_prefill, use_cuda_graph
):
    module = ModuleType("sglang.srt.lora.moe.runner")
    module.MoeLoraBatch = SimpleNamespace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    layout = SimpleNamespace(TP_GLOBAL=sentinel.global_layout)
    metadata = SimpleNamespace(
        token_slots=[-1, 1, -1],
        num_tokens=2,
        is_prefill=is_prefill,
        use_cuda_graph=use_cuda_graph,
    )
    calls = Mock()
    calls.get_batch_info.return_value = metadata
    calls.run.return_value = sentinel.combine_input
    layer = SimpleNamespace(
        lora_backend=SimpleNamespace(get_batch_info=calls.get_batch_info),
        _moe_lora_runner=SimpleNamespace(run=calls.run),
        gate_up_lora_a_weights=sentinel.gate_up_a,
        gate_up_lora_b_weights=sentinel.gate_up_b,
        down_lora_a_weights=sentinel.down_a,
        down_lora_b_weights=sentinel.down_b,
    )
    for name in ("_get_moe_lora_batch", "_run_moe_core_with_lora"):
        method = _lora_method("layers.py", "FusedMoEWithLoRA", name)
        method.__globals__["LoRABatchLayout"] = layout
        setattr(layer, name, MethodType(method, layer))

    assert layer._run_moe_core_with_lora(sentinel.dispatch) is sentinel.combine_input
    calls.run.assert_called_once()
    dispatch, batch = calls.run.call_args.args
    assert dispatch is sentinel.dispatch
    assert batch.gate_up_lora_a is sentinel.gate_up_a
    assert batch.gate_up_lora_b is sentinel.gate_up_b
    assert batch.down_lora_a is sentinel.down_a
    assert batch.down_lora_b is sentinel.down_b
    assert batch.token_lora_mapping == [-1, 1]
    assert batch.is_prefill is is_prefill
    assert batch.use_cuda_graph is use_cuda_graph
    assert all(
        call.args == (sentinel.global_layout,)
        for call in calls.get_batch_info.call_args_list
    )


def test_moe_core_without_batch_preserves_existing_base_callback():
    method = _lora_method("layers.py", "FusedMoEWithLoRA", "_run_moe_core_with_lora")
    method.__globals__["LoRABatchLayout"] = SimpleNamespace(
        TP_GLOBAL=sentinel.global_layout
    )
    base = Mock(return_value=sentinel.combine_input)
    layer = SimpleNamespace(
        lora_backend=SimpleNamespace(get_batch_info=lambda layout: None),
        _base_run_moe_core=base,
    )
    assert method(layer, sentinel.dispatch) is sentinel.combine_input
    base.assert_called_once_with(dispatch_output=sentinel.dispatch)


def test_moe_initialization_only_binds_the_owned_runner(monkeypatch):
    calls = Mock()
    calls.from_layer.return_value = sentinel.runner
    runner_module = ModuleType("sglang.srt.lora.moe.runner")
    runner_module.MoeLoraRunner = SimpleNamespace(from_layer=calls.from_layer)
    monkeypatch.setitem(sys.modules, runner_module.__name__, runner_module)
    base = SimpleNamespace(run_moe_core=sentinel.base_callback)
    layer = SimpleNamespace(
        lora_backend=SimpleNamespace(lora_workspace=sentinel.workspace),
        experts_shared_outer_loras=True,
        _max_lora_rank=32,
        _run_moe_core_with_lora=sentinel.callback,
    )
    _lora_method("layers.py", "FusedMoEWithLoRA", "_initialize_moe_lora_execution")(
        layer, base
    )
    calls.from_layer.assert_called_once_with(
        base, workspace=sentinel.workspace, is_shared_outer=True, physical_rank=32
    )
    assert layer._moe_lora_runner is sentinel.runner
    assert layer._base_run_moe_core is sentinel.base_callback
    assert base.run_moe_core is sentinel.callback


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
