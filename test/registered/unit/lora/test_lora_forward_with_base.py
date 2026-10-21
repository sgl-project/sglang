"""CPU contracts for LoRA MoE batch metadata and manager graph-mode plumbing."""

from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.parametrize("graph_enabled", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("mode", ["decode", "prefill", "target_verify"])
def test_manager_preserves_graph_flags_and_phase(graph_enabled, mode):
    metadata = {
        family: SimpleNamespace(use_cuda_graph=family != "eager")
        for family in ("eager", "decode", "prefill")
    }
    backend = SimpleNamespace(
        skip_inactive_lora_batches=False,
        init_decode_cuda_graph_batch_info=lambda **_: None,
        resize_cuda_graph_moe_buffers=lambda _: False,
    )
    calls = Mock()

    def prepare(
        batch, indices, ranks, scalings, use_cuda_graph, use_prefill_cuda_graph=False
    ):
        calls(batch, indices, ranks, scalings, use_cuda_graph, use_prefill_cuda_graph)
        family = (
            "decode"
            if use_cuda_graph
            else "prefill"
            if use_prefill_cuda_graph
            else "eager"
        )
        backend.batch_info = metadata[family]

    backend.prepare_lora_batch = prepare
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
        # The real init sets the attribute _use_cuda_graph_batch probes by name,
        # so a one-sided rename of either side fails the decode-graph cases.
        init_decode_graph = _lora_method(
            "lora_manager.py", "LoRAManager", "init_decode_cuda_graph_batch_info"
        )
        init_decode_graph.__globals__["_SGLANG_EXPERIMENTAL_LORA_OPTI"] = False
        init_decode_graph(manager, max_bs_in_cuda_graph=2, num_tokens_per_req=1)
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
