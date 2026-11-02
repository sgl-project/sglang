"""CPU contracts for graph metadata and scratch using exact production method bodies.
Narrow dependencies avoid importing the scheduler/CUDA stack.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_BACKEND_PATH = REPO_ROOT / "python/sglang/srt/lora/backend/base_backend.py"
V2_BACKEND_PATH = REPO_ROOT / "python/sglang/srt/lora/backend/triton_v2_backend.py"
LAYERS_PATH = REPO_ROOT / "python/sglang/srt/lora/layers.py"
MOE_RUNNER_PATH = REPO_ROOT / "python/sglang/srt/lora/moe/runner.py"
SHRINK_PATH = REPO_ROOT / "python/sglang/kernels/ops/lora/common/lora_a.py"
WORKSPACE_PATH = REPO_ROOT / "python/sglang/srt/lora/workspace.py"


def _load_class_with_members(
    path: Path,
    source_class: str,
    *member_names: str,
    include_fields: bool = False,
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
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in member_names
        )
        or (include_fields and isinstance(node, ast.AnnAssign))
    ]
    assert {
        node.name
        for node in members
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    } == set(member_names)
    test_class = ast.ClassDef(
        name=f"_{source_class}UnderTest",
        bases=[],
        keywords=[],
        body=members,
        decorator_list=class_node.decorator_list if include_fields else [],
    )
    scope = {"__name__": __name__, **(namespace or {})}
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


@pytest.fixture
def grouped_a_runner():
    # Keep allocation and grid arithmetic real; only record the GPU call.
    triton = SimpleNamespace(cdiv=lambda a, b: -(-a // b))
    shrink_tiles = _load_function(
        SHRINK_PATH, "shrink_column_tiles", namespace={"triton": triton}
    )
    workspace_type = _load_class_with_members(
        WORKSPACE_PATH,
        "LoraWorkspace",
        "__init__",
        "tensor",
        "_capturing",
        namespace={"torch": torch},
    )
    workspace = workspace_type()
    allocations, launches = [], []

    def tensor(name, shape, **kwargs):
        value = workspace.tensor(name, shape, **kwargs)
        allocations.append((name, tuple(shape), kwargs, value))
        return value

    site = SimpleNamespace(GATE_UP=object(), DOWN=object())
    runner_type = _load_class_with_members(
        MOE_RUNNER_PATH,
        "MoeLoraRunner",
        "_run_a",
        namespace={
            "torch": torch,
            "triton": triton,
            "shrink_column_tiles": shrink_tiles,
            "AFamily": SimpleNamespace(
                TOKEN_GROUPED=object(), PER_ROW=object(), TOKEN_DENSE=object()
            ),
            "BridgeLayout": SimpleNamespace(TOKEN_MAJOR=object()),
            "Site": site,
            "grouped_lora_a": lambda *args, **kwargs: launches.append((args, kwargs)),
        },
    )
    runner = runner_type()
    runner.workspace = SimpleNamespace(tensor=tensor)
    runner.lora_delta_dtype = torch.bfloat16
    return runner, site, allocations, launches


@pytest.mark.parametrize(
    (
        "split_k",
        "down",
        "tokens",
        "rows",
        "capacity",
        "block_m",
        "columns",
        "block_n",
        "tiles",
    ),
    [
        (4, False, 7, 21, 84, 16, 35, 32, 12),
        (8, True, 3, 9, 40, 16, 16, 64, 3),
    ],
)
def test_grouped_a_splitk_scratch_and_forwarded_operands(
    grouped_a_runner,
    split_k,
    down,
    tokens,
    rows,
    capacity,
    block_m,
    columns,
    block_n,
    tiles,
):
    runner, site, allocations, launches = grouped_a_runner
    spec = SimpleNamespace(
        site=site.DOWN if down else site.GATE_UP,
        family=SimpleNamespace(value="grouped"),
        is_shared_outer=False,
        output_layout=object(),
    )
    route = SimpleNamespace(
        num_tokens=tokens,
        num_rows=rows,
        sorted_pair_ids=torch.empty(capacity, dtype=torch.int32),
        block_size=block_m,
    )
    config = {"SPLIT_K": split_k, "BLOCK_SIZE_N": block_n}
    requested_sites = []

    def for_a(selected_site):
        requested_sites.append(selected_site)
        return config

    input = torch.empty((rows if down else tokens, 128), dtype=torch.bfloat16)
    weight = torch.empty((4, columns, 128), dtype=torch.bfloat16)
    pair_to_row = torch.arange(rows, dtype=torch.int32) if down else None
    output = runner._run_a(
        SimpleNamespace(for_a=for_a),
        spec,
        input,
        weight,
        SimpleNamespace(aligned=lambda shared: route),
        "down_a" if down else "gate_up_a",
        pair_to_row=pair_to_row,
    )
    name = "down_a" if down else "gate_up_a"
    assert requested_sites == [spec.site]
    assert output.shape == (rows, columns)  # Neither token count nor padded capacity.
    assert output.dtype == torch.bfloat16
    assert allocations[0][:2] == (f"{name}:output", (rows, columns))
    assert all(kwargs["device"] == input.device for _, _, kwargs, _ in allocations)
    assert len(launches) == 1
    args, kwargs = launches[0]
    assert len(args) == 4
    assert all(
        actual is expected
        for actual, expected in zip(args, (input, weight, output, route))
    )
    assert kwargs["config"] is config
    assert kwargs["pair_input"] is down
    assert kwargs["pair_to_row"] is pair_to_row
    assert len(allocations) == 3
    partial, counters = kwargs["partial"], kwargs["counters"]
    assert partial is allocations[1][3]
    assert counters is allocations[2][3]
    assert allocations[1][:2] == (f"{name}:partial", (split_k, rows, columns))
    assert allocations[2][:2] == (f"{name}:splitk_counters", (tiles,))
    assert partial.shape == (split_k, rows, columns)
    assert counters.shape == (tiles,)
    assert partial.dtype == torch.float32
    assert counters.dtype == torch.int32
    assert allocations[2][2]["zero_on_first_allocation"] is True
    assert counters.count_nonzero().item() == 0


@pytest.fixture
def mapping_functions():
    return _load_function(
        BASE_BACKEND_PATH,
        "_compute_token_lora_mapping",
        namespace={
            "torch": torch,
            "triton": SimpleNamespace(cdiv=lambda a, b: -(-a // b)),
        },
    )


@pytest.fixture
def v2_backend(mapping_functions):
    def unpinned_empty(*args, **kwargs):
        # CPU-only builds have no pinned allocator; tensor operations stay real.
        kwargs.pop("pin_memory", None)
        return torch.empty(*args, **kwargs)

    cpu_torch = SimpleNamespace(
        empty=unpinned_empty,
        zeros=torch.zeros,
        arange=torch.arange,
        as_tensor=torch.as_tensor,
        int32=torch.int32,
        float32=torch.float32,
    )
    batch_type = _load_class_with_members(
        V2_BACKEND_PATH,
        "_BatchInfo",
        include_fields=True,
        namespace={"torch": torch, "dataclass": dataclass},
    )
    get_counts = _load_function(
        REPO_ROOT / "python/sglang/srt/lora/utils.py", "get_batch_token_counts"
    )
    backend_type = _load_class_with_members(
        V2_BACKEND_PATH,
        "TritonV2LoRABackend",
        "_new_batch",
        "_fill_batch",
        "prepare_lora_batch",
        "init_decode_cuda_graph_batch_info",
        "init_prefill_cuda_graph_batch_info",
        namespace={
            "torch": cpu_torch,
            "_BatchInfo": batch_type,
            "_compute_token_lora_mapping": mapping_functions,
            "get_batch_token_counts": get_counts,
            "Phase": SimpleNamespace(PREFILL="prefill", DECODE="decode"),
        },
    )
    backend = backend_type()
    backend.name = "triton_v2"
    backend.max_loras_per_batch = 3
    backend.device = torch.device("cpu")
    backend.get_batch_info = MethodType(
        _load_class_with_members(
            BASE_BACKEND_PATH, "BaseLoRABackend", "get_batch_info"
        ).get_batch_info,
        backend,
    )
    backend.lm_head_runner = None
    bindings = []
    backend.runner = SimpleNamespace(
        begin_batch=lambda **kwargs: bindings.append(kwargs)
    )
    backend.bindings = bindings
    return backend


@pytest.fixture
def moe_payload_builder(monkeypatch):
    module_name = "sglang.srt.lora.moe.runner"
    module = ModuleType(module_name)
    module.MoeLoraBatch = SimpleNamespace
    monkeypatch.setitem(sys.modules, module_name, module)
    layer_type = _load_class_with_members(
        LAYERS_PATH,
        "FusedMoEWithLoRA",
        "_run_moe_core_with_lora",
        namespace={"LoRABatchLayout": SimpleNamespace(TP_GLOBAL=object())},
    )
    layer = layer_type()
    layer.gate_up_lora_a_weights = torch.empty(1)
    layer.gate_up_lora_b_weights = torch.empty(1)
    layer.down_lora_a_weights = torch.empty(1)
    layer.down_lora_b_weights = torch.empty(1)

    def build(backend):
        layer.lora_backend = backend
        layer._moe_lora_runner = SimpleNamespace(run=lambda dispatch, batch: batch)
        payload = layer._run_moe_core_with_lora(object())
        assert payload.gate_up_lora_a is layer.gate_up_lora_a_weights
        assert payload.gate_up_lora_b is layer.gate_up_lora_b_weights
        assert payload.down_lora_a is layer.down_lora_a_weights
        assert payload.down_lora_b is layer.down_lora_b_weights
        return payload

    return build


@pytest.mark.parametrize("mode", ["eager", "prefill_graph", "decode_graph"])
def test_v2_live_moe_views_preserve_dense_graph_capacity(
    v2_backend, moe_payload_builder, mode
):
    decode = mode == "decode_graph"
    graph = mode != "eager"
    if decode:
        v2_backend.init_decode_cuda_graph_batch_info(6, 1)
        cases = [([1, 1, 1], [1, 2, 0]), ([1], [2]), ([], [])]
    else:
        if graph:
            v2_backend.init_prefill_cuda_graph_batch_info(12)
        cases = [([2, 0, 3, 1], [1, 2, 0, 1]), ([1, 2], [2, 0]), ([0, 0], [1, 2])]
    addresses = None
    for lengths, slots in cases:
        forward_batch = SimpleNamespace(
            batch_size=len(lengths),
            extend_num_tokens=sum(lengths),
            extend_seq_lens_cpu=lengths,
            forward_mode=SimpleNamespace(
                is_decode=lambda: decode,
                is_target_verify=lambda: False,
                is_idle=lambda: False,
                is_extend=lambda: not decode,
                is_extend_without_speculative=lambda: not decode,
            ),
        )
        ranks = [0, 8, 0]
        v2_backend.prepare_lora_batch(
            forward_batch,
            slots,
            ranks,
            [0.0, 0.25, 0.0],
            use_decode_cuda_graph=decode,
            use_prefill_cuda_graph=mode == "prefill_graph",
        )
        batch = v2_backend.batch_info
        # The manager selects the MoE phase after backend preparation.
        batch.is_prefill = not decode
        expected = [
            slot if ranks[slot] else -1
            for length, slot in zip(lengths, slots)
            for _ in range(length)
        ]
        assert batch.num_tokens == len(expected)
        capacity = (6 if decode else 12) if graph else len(expected)
        request_capacity = capacity if graph else len(lengths)
        assert batch.token_slots.tolist() == expected + [-1] * (
            capacity - len(expected)
        )
        assert batch.weight_indices.numel() == request_capacity
        assert batch.seg_indptr.numel() == request_capacity + 1
        assert batch.weight_indices[: len(slots)].tolist() == slots
        bound = v2_backend.bindings[-1]
        assert bound["token_slots"] is batch.token_slots
        assert bound["token_slots"].shape == (capacity,)
        assert bound["num_tokens"] == len(expected)
        assert bound["graph_mode"] is graph
        assert bound["is_prefill_graph"] is (mode == "prefill_graph")

        payload = moe_payload_builder(v2_backend)
        assert payload.token_lora_mapping.tolist() == expected
        assert payload.token_lora_mapping.shape == (batch.num_tokens,)
        assert payload.token_lora_mapping.untyped_storage().data_ptr() == (
            batch.token_slots.untyped_storage().data_ptr()
        )
        assert payload.use_cuda_graph is graph
        assert payload.is_prefill is (not decode)
        if graph:
            current = batch.packed.data_ptr(), batch.token_slots.data_ptr()
            if addresses is not None:
                assert current == addresses
            addresses = current


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
