"""A5 C8 owner-plan metadata reuse contract."""

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


@contextmanager
def _load_c8_classes(prepare_metadata):
    path = (
        Path(__file__).parents[4]
        / "python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload"
    )
    package = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload"
    dependencies = {
        "sgl_kernel_npu.sparsity_driven_kv_offload": Mock(),
        "sglang.kernels.ops.attention.npu_copy_sfa": Mock(),
        "sglang.kernels.ops.attention.npu_lim": Mock(),
        "sglang.srt.mem_cache.memory_pool": Mock(),
        "sglang.srt.model_executor.forward_batch_info": Mock(),
        "sglang.srt.runtime_context": Mock(),
        f"{package}.config": SimpleNamespace(MLA_KV_HEAD_NUM=1),
        f"{package}.fused_budget": SimpleNamespace(C8_ROW_BYTES=656),
        f"{package}.host_pool": SimpleNamespace(HostKVPool=object),
        f"{package}.prefill_kv_plan": Mock(),
        f"{package}.npu_lim_copy_sfa_metadata": SimpleNamespace(
            prepare_copy_sfa_metadata=prepare_metadata
        ),
    }
    with patch.dict(sys.modules, dependencies):
        for name in ("fused_lim", "fused_manager", "c8_manager"):
            full_name = f"{package}.{name}"
            spec = importlib.util.spec_from_file_location(
                full_name, path / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        yield module.C8DecodeAdapter, module.C8SparseKVCacheManager


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
                torch.where(sparse, capacity + width - 1, kwargs["seq_lens"]),
                0,
            )
        )
        table = kwargs["hbm_block_table"]
        pages = table.shape[1]
        rows = torch.where(valid, kwargs["request_slots"] - kwargs["hbm_row_offset"], 0)
        table.copy_(rows[:, None] * pages + torch.arange(pages))

    with _load_c8_classes(prepare_metadata) as (adapter_type, manager_type):
        adapter = object.__new__(adapter_type)
        batch, width, topk, capacity = 2, 4, 3, 8
        rows = batch * width
        adapter.capacity = capacity
        adapter.rows = 8
        adapter._prepared_batch = None
        adapter._fused_prepare_lim_metadata = Mock()
        adapter.operator = Mock()
        adapter.source_ids = torch.full((rows, topk), 99, dtype=torch.int32)
        adapter.device_slots = torch.zeros((rows, topk), dtype=torch.int32)
        adapter.topk_misses = torch.zeros(rows, dtype=torch.int32)
        adapter.miss_sources = torch.zeros((batch, topk), dtype=torch.int32)
        adapter.miss_slots = torch.zeros((batch, topk), dtype=torch.int32)
        adapter.miss_counts = torch.zeros(batch, dtype=torch.int32)
        adapter.block_table = torch.zeros((batch, 2), dtype=torch.int32)
        adapter.query_ends = torch.arange(1, width + 1)[:, None].repeat(1, batch)
        adapter.last_prefix = torch.full((4,), -1, dtype=torch.int32)
        adapter._pre_lengths = torch.tensor([9000, 100], dtype=torch.int32)
        adapter._copy_lengths = torch.empty_like(adapter._pre_lengths)
        adapter._pre_next_prefix = torch.tensor([9000, -1], dtype=torch.int32)
        adapter._pre_cache_sizes = torch.tensor([capacity, 0], dtype=torch.int32)
        adapter._pre_states = torch.zeros(batch, dtype=torch.int32)
        adapter._pre_sparse = torch.tensor([True, False])
        adapter._pre_valid = torch.tensor([True, False])
        adapter.request_safe_slots = torch.tensor([7, 0], dtype=torch.int32)
        adapter.resident_safe_slots = torch.tensor([2, 3], dtype=torch.int32)

        manager = object.__new__(manager_type)
        manager.start_layer = 0
        manager.budget = SimpleNamespace(copy_sfa_query_heads=1, resident_rows=3)
        manager.lim_adapter = adapter
        manager.request_to_resident = None
        manager.planner_hbm_block_table = torch.full((batch, 2), -1, dtype=torch.int32)
        manager.device_slot_map = [
            torch.zeros((4, capacity), dtype=torch.int32) for _ in range(2)
        ]
        forward_batch = SimpleNamespace(
            batch_size=batch,
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            spec_info=SimpleNamespace(num_tokens_per_req=width),
            seq_lens=torch.tensor([9000, 0], dtype=torch.int32),
            positions=torch.arange(rows),
            req_pool_indices=torch.tensor([7, 0]),
        )

        def select(layer_id, *, reuse_metadata=False):
            return adapter.select(
                manager,
                query=torch.zeros((rows, 1, 1)),
                weights=torch.zeros((rows, 1)),
                key=torch.zeros(1),
                block_table=torch.zeros((batch, 2), dtype=torch.int32),
                forward_batch=forward_batch,
                layer_id=layer_id,
                query_scale=torch.zeros((rows, 1)),
                key_scale=torch.zeros(1),
                reuse_metadata=reuse_metadata,
            )

        plan = select(0)
        assert len(metadata_calls) == 1
        assert adapter._pre_lengths.tolist() == [9000, 100]
        assert adapter._copy_lengths.tolist() == [capacity + width - 1, 0]
        assert plan.source_ids[width:].eq(-1).all()
        assert manager.planner_hbm_block_table.tolist() == [[2, 3], [0, 1]]

        manager.page_size = 1
        manager.device_k_buffer = [torch.zeros((2 * capacity, 1, 656))]
        manager.host_kv_buffer = [torch.zeros((1, 1, 1, 656))]
        manager.dev_ptr_list = [0]
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
        assert manager._copy_sfa_operator.call_args.args[11].tolist() == [
            capacity + width - 1,
            0,
        ]
        assert len(metadata_calls) == 1

        select(1, reuse_metadata=True)
        adapter._fused_prepare_lim_metadata.assert_called_once()
        assert len(metadata_calls) == 2
        assert adapter.operator.call_args.kwargs["lengths"].tolist() == [9000, 100]
        assert (
            adapter.operator.call_args.kwargs["slot_pool"] is manager.device_slot_map[1]
        )
        assert adapter.last_prefix.tolist() == [-1, -1, -1, -1]

        adapter.commit_forward(batch)
        assert adapter.last_prefix.tolist() == [-1, -1, 9000, -1]
        select(0)
        assert adapter._fused_prepare_lim_metadata.call_count == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
