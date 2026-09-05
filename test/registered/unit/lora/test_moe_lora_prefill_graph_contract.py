"""Hermetic contracts for MoE LoRA prefill CUDA-graph metadata.

The production modules pull in the full scheduler, Triton, and CUDA import
graph.  These tests compile the exact small method bodies under test from the
source tree and provide only their narrow CPU dependencies.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

_LORA_RUNNER = SimpleNamespace(is_lora=lambda: True)
_OTHER_RUNNER = SimpleNamespace(is_lora=lambda: False)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[4]
LORA_MANAGER_PATH = REPO_ROOT / "python/sglang/srt/lora/lora_manager.py"
BASE_BACKEND_PATH = REPO_ROOT / "python/sglang/srt/lora/backend/base_backend.py"


def _load_class_with_members(
    path: Path,
    source_class: str,
    *member_names: str,
    namespace: dict[str, object] | None = None,
):
    tree = ast.parse(path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == source_class
    )
    members = [
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in member_names
    ]
    assert {node.name for node in members} == set(member_names)
    test_class = ast.ClassDef(
        name=f"_{source_class}UnderTest",
        bases=[],
        keywords=[],
        body=members,
        decorator_list=[],
    )
    scope = dict(namespace or {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope[test_class.name]


def _load_function(
    path: Path,
    function_name: str,
    *,
    namespace: dict[str, object] | None = None,
):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    scope = dict(namespace or {})
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope[function_name]


def test_moe_lora_is_the_only_moe_engine_eligible_for_prefill_graph() -> None:
    manager_type = _load_class_with_members(
        LORA_MANAGER_PATH,
        "LoRAManager",
        "supports_prefill_cuda_graph",
    )
    manager = manager_type()
    manager.lora_backend = SimpleNamespace(
        supports_prefill_cuda_graph=True,
        is_moe_lora=True,
    )
    manager.enable_dp_attention = False
    manager.prefill_cuda_graph_backend = "breakable"

    manager.moe_lora_runner_backend = _LORA_RUNNER
    assert manager.supports_prefill_cuda_graph

    manager.moe_lora_runner_backend = _OTHER_RUNNER
    assert not manager.supports_prefill_cuda_graph

    manager.moe_lora_runner_backend = _LORA_RUNNER
    manager.enable_dp_attention = True
    assert not manager.supports_prefill_cuda_graph

    manager.enable_dp_attention = False
    manager.lora_backend.supports_prefill_cuda_graph = False
    assert not manager.supports_prefill_cuda_graph

    manager.lora_backend.supports_prefill_cuda_graph = True
    # Only "breakable" hosts LoRA prefill graphs: "full" is decode-only and
    # tc_piecewise breaks Dynamo guards on per-batch LoRABatchInfo rebinds.
    for backend in ("tc_piecewise", "full", "disabled", None):
        manager.prefill_cuda_graph_backend = backend
        assert not manager.supports_prefill_cuda_graph

    manager.prefill_cuda_graph_backend = "breakable"
    assert manager.supports_prefill_cuda_graph


def test_prefill_and_decode_graphs_have_independent_mapping_capacities() -> None:
    backend_type = _load_class_with_members(
        BASE_BACKEND_PATH,
        "BaseLoRABackend",
        "init_cuda_graph_moe_buffers",
        "init_prefill_cuda_graph_moe_buffers",
        namespace={"torch": torch},
    )
    backend = backend_type()
    backend.max_loras_per_batch = 4
    backend.device = torch.device("cpu")
    backend.is_moe_lora = True
    moe_layer = SimpleNamespace(
        base_layer=SimpleNamespace(w13_weight=torch.empty(1)),
    )

    backend.init_cuda_graph_moe_buffers(
        max_bs=3,
        max_loras=4,
        compute_dtype=torch.float32,
        moe_layer=moe_layer,
        include_legacy_kernel_buffers=False,
    )
    backend.init_prefill_cuda_graph_moe_buffers(max_num_tokens=37)

    decode_mapping = backend.moe_cg_buffers["token_lora_mapping"]
    prefill_mapping = backend.prefill_moe_cg_buffers["token_lora_mapping"]
    assert decode_mapping.shape == (3,)
    assert prefill_mapping.shape == (37,)
    assert decode_mapping.data_ptr() != prefill_mapping.data_ptr()
    assert torch.all(decode_mapping == -1)
    assert torch.all(prefill_mapping == -1)


@dataclass
class _MoeInfo:
    seg_indptr: torch.Tensor
    req_to_lora: torch.Tensor
    adapter_enabled: torch.Tensor
    token_lora_mapping: torch.Tensor


def test_graph_metadata_selects_prefill_buffer_by_batch_identity() -> None:
    selected_buffers: list[torch.Tensor] = []

    def _capture_compute(
        _num_tokens,
        _seg_indptr,
        _lora_ranks,
        _weight_indices,
        adapter_enabled,
        token_lora_mapping,
        *,
        max_len,
    ):
        del max_len
        selected_buffers.append(token_lora_mapping)
        return adapter_enabled, token_lora_mapping

    backend_type = _load_class_with_members(
        BASE_BACKEND_PATH,
        "BaseLoRABackend",
        "_add_moe_lora_info",
        namespace={
            "ForwardBatch": object,
            "LoRABatchInfo": object,
            "MoELoRABatchInfo": _MoeInfo,
            "_compute_moe_lora_info": _capture_compute,
            "get_batch_token_counts": lambda fb: (
                (fb.extend_num_tokens, max(fb.extend_seq_lens_cpu))
                if fb.forward_mode.is_extend()
                else (fb.batch_size, 1)
            ),
        },
    )
    backend = backend_type()
    backend.is_moe_lora = True
    backend.moe_cg_buffers = {
        "adapter_enabled": torch.zeros(2, dtype=torch.int32),
        "token_lora_mapping": torch.full((2,), -1, dtype=torch.int32),
    }
    backend.prefill_moe_cg_buffers = {
        "adapter_enabled": torch.zeros(2, dtype=torch.int32),
        "token_lora_mapping": torch.full((8,), -1, dtype=torch.int32),
    }
    prefill_batch = SimpleNamespace(
        use_cuda_graph=True,
        bs=2,
        num_segments=2,
        seg_indptr=torch.tensor([0, 2, 5], dtype=torch.int32),
        weight_indices=torch.tensor([0, 1], dtype=torch.int32),
        lora_ranks=torch.tensor([16, 16], dtype=torch.int32),
        req_seg_indptr=None,
        req_weight_indices=None,
    )
    backend.prefill_cuda_graph_batch_info = prefill_batch
    extend_mode = SimpleNamespace(
        is_extend=lambda: True,
        is_decode=lambda: False,
        is_target_verify=lambda: False,
    )
    forward_batch = SimpleNamespace(
        forward_mode=extend_mode,
        extend_seq_lens_cpu=[2, 3],
        extend_num_tokens=5,
        batch_size=2,
    )

    backend._add_moe_lora_info(forward_batch, prefill_batch)
    assert selected_buffers[-1] is backend.prefill_moe_cg_buffers["token_lora_mapping"]

    decode_batch = SimpleNamespace(**vars(prefill_batch))
    backend._add_moe_lora_info(forward_batch, decode_batch)
    assert selected_buffers[-1] is backend.moe_cg_buffers["token_lora_mapping"]


def test_smaller_replay_resets_unused_mapping_tail() -> None:
    compute = _load_function(
        BASE_BACKEND_PATH,
        "_compute_moe_lora_info",
        namespace={
            "torch": torch,
            # The CPU path does not invoke either dependency, but names remain
            # present so an accidental branch change fails clearly.
            "triton": SimpleNamespace(),
            "_compute_moe_lora_info_kernel": None,
        },
    )
    stable_mapping = torch.full((8,), 7, dtype=torch.int32)
    adapter_enabled = torch.ones(2, dtype=torch.int32)

    enabled, current_mapping = compute(
        3,
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([16, 0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        adapter_enabled,
        stable_mapping,
        max_len=3,
    )

    assert current_mapping.data_ptr() == stable_mapping.data_ptr()
    assert current_mapping.tolist() == [0, 0, 0]
    assert stable_mapping[3:].tolist() == [-1, -1, -1, -1, -1]
    assert enabled.tolist() == [1, 0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
