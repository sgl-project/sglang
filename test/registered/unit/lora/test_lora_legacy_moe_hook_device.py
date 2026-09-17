"""Legacy graph buffers use the base experts' device, including Marlin w13_qweight.
The new MoE runner uses the base layer's device without legacy buffers or quant info.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_BACKEND = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/lora/backend/base_backend.py"
)


def _init_cuda_graph_moe_buffers():
    # The exact method, without importing the CUDA backend dependency graph.
    cls = next(
        node
        for node in ast.parse(_BACKEND.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "BaseLoRABackend"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "init_cuda_graph_moe_buffers"
    )
    scope = {"torch": torch}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(_BACKEND), "exec"),
        scope,
    )
    return scope["init_cuda_graph_moe_buffers"]


def _layer(*, own_runner: bool, quant_info):
    return SimpleNamespace(
        _lora_runner_backend=SimpleNamespace(is_lora=lambda: own_runner),
        _quant_info=quant_info,
        base_layer=SimpleNamespace(
            top_k=2, num_experts=4, w13_weight=torch.empty(1, device="meta")
        ),
    )


def _buffers(layer, *, prefill=False):
    backend = SimpleNamespace(moe_cg_buffers=None, prefill_moe_cg_buffers=None)
    _init_cuda_graph_moe_buffers()(
        backend, 8, 2, torch.bfloat16, layer, prefill=prefill
    )
    return backend.prefill_moe_cg_buffers if prefill else backend.moe_cg_buffers


def test_marlin_quant_info_packs_the_experts_as_w13_qweight():
    layer = _layer(
        own_runner=False,
        quant_info=SimpleNamespace(w13_qweight=torch.empty(1, device="meta")),
    )
    buffers = _buffers(layer)
    assert buffers["adapter_enabled"].device.type == "meta"
    assert buffers["sorted_token_ids_lora"].device.type == "meta"


def test_w13_weight_is_the_device_source_when_the_runner_keeps_it():
    layer = _layer(
        own_runner=False,
        quant_info=SimpleNamespace(
            w13_weight=torch.empty(1), w13_qweight=torch.empty(1, device="meta")
        ),
    )
    assert _buffers(layer)["token_lora_mapping"].device.type == "cpu"


def test_own_runner_takes_the_base_layer_and_allocates_no_legacy_buffers():
    layer = _layer(own_runner=True, quant_info=None)
    buffers = _buffers(layer, prefill=True)
    assert buffers["adapter_enabled"].device.type == "meta"
    assert set(buffers) == {"adapter_enabled", "token_lora_mapping"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
