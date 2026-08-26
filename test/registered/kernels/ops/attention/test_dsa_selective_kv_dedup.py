import importlib.util
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_ROOT = Path(__file__).resolve().parents[5]
_MODULE_PATH = _ROOT / "python/sglang/kernels/ops/attention/dsa/selective_kv_dequant.py"
_RUNTIME_MODULE_PATH = (
    _ROOT / "python/sglang/srt/layers/attention/dsa/selective_kv_dequant.py"
)


def _load_kernel_module():
    assert _MODULE_PATH.exists(), "dense epoch dedup kernel module is not implemented"
    spec = importlib.util.spec_from_file_location(
        "dsa_selective_kv_dequant_kernel", _MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "deduplicate_kv_slots_dense_epoch")
    return module


def _load_runtime_module():
    spec = importlib.util.spec_from_file_location(
        "dsa_selective_kv_dequant_runtime", _RUNTIME_MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "prepare_dense_epoch_selection")
    return module


def _cuda_available():
    return torch.cuda.is_available()


def _run_case(page_table_values, topk_values, pool_rows=None, state=None):
    module = _load_kernel_module()
    device = torch.device("cuda")
    page_table = torch.tensor(page_table_values, dtype=torch.int32, device=device)
    topk = torch.tensor(topk_values, dtype=torch.int32, device=device)
    if pool_rows is None:
        pool_rows = max(page_table_values) + 1
    selection_capacity = min(page_table.numel(), topk.numel())

    if state is None:
        state = {
            "slot_epoch": torch.full(
                (pool_rows,), -1, dtype=torch.int64, device=device
            ),
            "slot_to_compact": torch.empty(pool_rows, dtype=torch.int32, device=device),
            "selected": torch.empty(
                selection_capacity, dtype=torch.int32, device=device
            ),
            "remapped": torch.empty_like(topk),
            "block_offsets": torch.empty(
                (pool_rows + 255) // 256,
                dtype=torch.int32,
                device=device,
            ),
            "epoch": torch.zeros(1, dtype=torch.int64, device=device),
            "num_unique": torch.zeros(1, dtype=torch.int32, device=device),
        }

    selected, remapped, num_unique = module.deduplicate_kv_slots_dense_epoch(
        page_table,
        topk,
        slot_epoch=state["slot_epoch"],
        slot_to_compact=state["slot_to_compact"],
        selected_physical_slots=state["selected"],
        remapped_topk=state["remapped"],
        block_offsets=state["block_offsets"],
        epoch=state["epoch"],
        num_unique=state["num_unique"],
        num_pool_rows=pool_rows,
        selection_capacity=selection_capacity,
    )
    torch.cuda.synchronize()
    return state, selected, remapped, num_unique


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is required")
def test_dense_epoch_dedup_reconstructs_physical_slots_and_padding():
    _, selected, remapped, num_unique = _run_case(
        [7, 3, 7, 9, 5],
        [[0, 1, -1], [2, 3, 1]],
        pool_rows=16,
    )

    unique_count = int(num_unique.item())
    assert unique_count == 3
    torch.testing.assert_close(
        selected[:unique_count].cpu(),
        torch.tensor([3, 7, 9], dtype=torch.int32),
    )
    torch.testing.assert_close(
        remapped.cpu()[0, 2], torch.tensor(-1, dtype=torch.int32)
    )
    valid = remapped >= 0
    reconstructed = selected[:unique_count][remapped[valid].long()]
    expected = torch.tensor([7, 3, 7, 9, 3], dtype=torch.int32, device="cuda")
    torch.testing.assert_close(reconstructed, expected)
    assert torch.unique(selected[:unique_count]).numel() == unique_count


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is required")
def test_dense_epoch_dedup_reuses_state_without_stale_slot_leakage():
    state, _, _, first_count = _run_case([2, 4, 6, 8], [[0, 1, 2, 3]], pool_rows=16)
    first_epoch = int(state["epoch"].item())
    assert int(first_count.item()) == 4

    # Reuse every persistent tensor but select only slots 4 and 8.  Epoch tags
    # from physical slots 2 and 6 must not enter the new compact set.
    state["selected"] = torch.empty(3, dtype=torch.int32, device="cuda")
    state["remapped"] = torch.empty((1, 3), dtype=torch.int32, device="cuda")
    state, selected, remapped, second_count = _run_case(
        [2, 4, 6, 8], [[1, 3, 1]], pool_rows=16, state=state
    )

    assert int(state["epoch"].item()) == first_epoch + 1
    assert int(second_count.item()) == 2
    assert set(selected[:2].cpu().tolist()) == {4, 8}
    reconstructed = selected[:2][remapped.reshape(-1).long()]
    torch.testing.assert_close(
        reconstructed,
        torch.tensor([4, 8, 4], dtype=torch.int32, device="cuda"),
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is required")
def test_dense_epoch_dedup_prefix_scan_crosses_256_slot_boundaries():
    physical = [0, 255, 256, 257, 511, 512, 599]
    _, selected, remapped, num_unique = _run_case(
        physical,
        [[6, 1, 2, 4, 0, 5, 3, 2, 6]],
        pool_rows=600,
    )

    count = int(num_unique.item())
    assert count == len(physical)
    torch.testing.assert_close(
        selected[:count].cpu(), torch.tensor(physical, dtype=torch.int32)
    )
    reconstructed = selected[:count][remapped.reshape(-1).long()]
    torch.testing.assert_close(
        reconstructed,
        torch.tensor(
            [599, 255, 256, 511, 0, 512, 257, 256, 599],
            dtype=torch.int32,
            device="cuda",
        ),
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is required")
def test_dense_epoch_dedup_cuda_graph_replays_with_new_indices():
    module = _load_kernel_module()
    device = torch.device("cuda")
    page_table = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device=device)
    topk = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32, device=device)
    slot_epoch = torch.full((8,), -1, dtype=torch.int64, device=device)
    slot_to_compact = torch.empty(8, dtype=torch.int32, device=device)
    selected = torch.empty(4, dtype=torch.int32, device=device)
    remapped = torch.empty_like(topk)
    block_offsets = torch.empty(1, dtype=torch.int32, device=device)
    epoch = torch.zeros(1, dtype=torch.int64, device=device)
    num_unique = torch.zeros(1, dtype=torch.int32, device=device)

    def launch():
        return module.deduplicate_kv_slots_dense_epoch(
            page_table,
            topk,
            slot_epoch=slot_epoch,
            slot_to_compact=slot_to_compact,
            selected_physical_slots=selected,
            remapped_topk=remapped,
            block_offsets=block_offsets,
            epoch=epoch,
            num_unique=num_unique,
            num_pool_rows=8,
            selection_capacity=4,
        )

    launch()  # JIT before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    topk.copy_(torch.tensor([[2, 3, 2, 3]], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()

    count = int(num_unique.item())
    assert count == 2
    reconstructed = selected[:count][remapped.reshape(-1).long()]
    torch.testing.assert_close(
        reconstructed,
        torch.tensor([5, 7, 5, 7], dtype=torch.int32, device=device),
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is required")
def test_no_dedup_preallocated_metadata_cuda_graph_replay():
    runtime = _load_runtime_module()
    device = torch.device("cuda")
    page_table = torch.tensor([11, 13, 17, 19], dtype=torch.int32, device=device)
    topk = torch.tensor([[0, -1, 2, 0]], dtype=torch.int32, device=device)
    workspace = runtime.SelectiveKVWorkspace(device)
    physical, remapped = workspace.get_occurrence_metadata(topk.numel())
    remapped = remapped.view_as(topk)

    def launch():
        return runtime.build_selective_kv_no_dedup(
            page_table,
            topk,
            physical_slots_out=physical,
            remapped_topk_out=remapped,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    topk.copy_(torch.tensor([[3, 1, -1, 2]], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        physical,
        torch.tensor([19, 13, 11, 17], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        remapped,
        torch.tensor([[0, 1, -1, 3]], dtype=torch.int32, device=device),
    )


@pytest.mark.skipif(not _cuda_available(), reason="CUDA is required")
def test_runtime_selection_plan_uses_workspace_capacity_without_host_compaction():
    runtime = _load_runtime_module()
    workspace = runtime.SelectiveKVWorkspace(torch.device("cuda"))
    page_table = torch.tensor([7, 3, 7, 9, 5], dtype=torch.int32, device="cuda")
    topk = torch.tensor([[0, 1, -1], [2, 3, 1]], dtype=torch.int32, device="cuda")

    plan = runtime.prepare_dense_epoch_selection(
        page_table,
        topk,
        num_pool_rows=16,
        workspace=workspace,
        deduplicate_fn=_load_kernel_module().deduplicate_kv_slots_dense_epoch,
    )
    torch.cuda.synchronize()

    assert plan.capacity == 5
    assert plan.selected_physical_slots.shape == (5,)
    assert plan.remapped_topk.shape == topk.shape
    assert plan.num_unique.shape == (1,)
    count = int(plan.num_unique.item())
    assert count == 3
    reconstructed = plan.selected_physical_slots[:count][
        plan.remapped_topk[plan.remapped_topk >= 0].long()
    ]
    torch.testing.assert_close(
        reconstructed,
        torch.tensor([7, 3, 7, 9, 3], dtype=torch.int32, device="cuda"),
    )
