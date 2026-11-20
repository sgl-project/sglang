"""A5 C8 owner-plan metadata reuse contract."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


def _load_c8_classes(prepare_metadata):
    path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/"
        "c8_manager.py"
    )
    package = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload"
    dependencies = {
        "sgl_kernel_npu.sparsity_driven_kv_offload": Mock(),
        "sglang.kernels.ops.kvcache.npu_sparse_kv": Mock(),
        "sglang.srt.runtime_context": Mock(),
        f"{package}.fused_budget": SimpleNamespace(C8_ROW_BYTES=656),
        f"{package}.fused_lim": SimpleNamespace(
            LIMDecodeAdapter=object, sparse_query_width=Mock()
        ),
        f"{package}.fused_manager": SimpleNamespace(FusedSparseKVCacheManager=object),
        f"{package}.memfabric_pool": SimpleNamespace(MemFabricKVPool=object),
        f"{package}.npu_lim_copy_sfa_metadata": SimpleNamespace(
            prepare_copy_sfa_metadata=prepare_metadata
        ),
    }
    spec = importlib.util.spec_from_file_location("a5_c8_manager_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module.C8DecodeAdapter, module.C8SparseKVCacheManager


def test_a5_copy_sfa_metadata_runs_once_per_lim_owner_plan():
    metadata_calls = []

    def prepare_metadata(**kwargs):
        metadata_calls.append(kwargs)
        valid = kwargs["valid_rows"]
        sparse = kwargs["sparse_rows"]
        width = kwargs["query_width"]
        capacity = kwargs["capacity"]
        source_ids = kwargs["source_ids"].view(valid.numel(), width, -1)
        source_ids[~valid] = -1
        lengths = kwargs["actual_seq_lengths_kv"]
        lengths.copy_(
            torch.where(
                valid,
                torch.where(sparse, capacity + width - 1, lengths),
                0,
            )
        )
        table = kwargs["hbm_block_table"]
        pages = table.shape[1]
        rows = torch.where(valid, kwargs["request_slots"] - kwargs["hbm_row_offset"], 0)
        table.copy_(rows[:, None] * pages + torch.arange(pages))

    adapter_type, manager_type = _load_c8_classes(prepare_metadata)
    adapter = object.__new__(adapter_type)
    batch, width, topk, capacity = 2, 4, 3, 8
    rows = batch * width
    adapter.capacity = capacity
    adapter.prepare = lambda *args: (batch, width)
    adapter.operator = Mock()
    adapter.source_ids = torch.full((rows, topk), 99, dtype=torch.int32)
    adapter.device_slots = torch.zeros((rows, topk), dtype=torch.int32)
    adapter.topk_misses = torch.zeros(rows, dtype=torch.int32)
    adapter.miss_sources = torch.zeros((batch, topk), dtype=torch.int32)
    adapter.miss_slots = torch.zeros((batch, topk), dtype=torch.int32)
    adapter.miss_counts = torch.zeros(batch, dtype=torch.int32)
    adapter.block_table = torch.zeros((batch, 2), dtype=torch.int32)
    adapter.query_ends = torch.arange(1, width + 1)[:, None].repeat(1, batch)
    adapter._pre_lengths = torch.tensor([9000, 100], dtype=torch.int32)
    adapter._pre_cache_sizes = torch.tensor([capacity, 0], dtype=torch.int32)
    adapter._pre_states = torch.zeros(batch, dtype=torch.int32)
    adapter._pre_sparse = torch.tensor([True, False])
    adapter._pre_valid = torch.tensor([True, False])
    adapter.request_safe_slots = torch.tensor([7, 0], dtype=torch.int32)
    adapter.resident_safe_slots = torch.tensor([2, 1], dtype=torch.int32)
    adapter.finish = lambda layer, count, query_width: SimpleNamespace(
        source_ids=adapter.source_ids[: count * query_width],
        device_slots=adapter.device_slots[: count * query_width],
        topk_misses=adapter.topk_misses[: count * query_width],
        miss_sources=adapter.miss_sources[:count],
        miss_slots=adapter.miss_slots[:count],
        miss_counts=adapter.miss_counts[:count],
        sparse_rows=adapter._pre_sparse[:count],
        valid_rows=adapter._pre_valid[:count],
        request_safe_slots=adapter.request_safe_slots[:count],
        resident_safe_slots=adapter.resident_safe_slots[:count],
        query_width=query_width,
        owner_layer=layer,
    )
    manager = object.__new__(manager_type)
    manager.start_layer = 0
    manager.planner_hbm_block_table = torch.full((batch, 2), -1, dtype=torch.int32)
    manager.device_slot_map = [torch.zeros((3, capacity), dtype=torch.int32)]

    def select():
        return adapter.select(
            manager,
            query=torch.zeros((rows, 1, 1)),
            weights=torch.zeros((rows, 1)),
            key=torch.zeros(1),
            block_table=torch.zeros((batch, 2), dtype=torch.int32),
            forward_batch=object(),
            layer_id=0,
            query_scale=torch.zeros((rows, 1)),
            key_scale=torch.zeros(1),
        )

    plan = select()
    assert len(metadata_calls) == 1
    assert adapter._pre_lengths.tolist() == [capacity + width - 1, 0]
    assert plan.source_ids[width:].eq(-1).all()
    assert manager.planner_hbm_block_table.tolist() == [[2, 3], [0, 1]]

    manager.budget = SimpleNamespace(copy_sfa_query_heads=1)
    manager.page_size = 1
    manager.device_kv_buffer = [torch.zeros((3 * capacity, 1, 656))]
    manager.host_kv_buffer = [torch.zeros((1, 1, 1, 656))]
    manager.dev_ptr_list = [0]
    manager.lim_adapter = adapter
    manager.actual_request_rows = 3
    manager.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros((3, 16), dtype=torch.int32)
    )
    manager.resident_end = torch.zeros((1, 3), dtype=torch.int32)
    manager._copy_sfa_operator = Mock(
        side_effect=lambda *args, **kwargs: args[17].fill_(1)
    )
    current = (
        torch.zeros((rows, 656), dtype=torch.uint8),
        torch.arange(rows),
        torch.ones(rows, dtype=torch.bool),
    )
    layer = SimpleNamespace(layer_id=0, scaling=0.1)
    for _ in range(2):
        manager.copy_sfa_attention(
            layer,
            torch.zeros((rows, 1, 512)),
            torch.zeros((rows, 1, 64)),
            plan,
            current,
        )
    assert manager._copy_sfa_operator.call_count == 2
    assert len(metadata_calls) == 1

    adapter.source_ids.fill_(99)
    adapter._pre_lengths.copy_(torch.tensor([9001, 101], dtype=torch.int32))
    select()
    assert len(metadata_calls) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
