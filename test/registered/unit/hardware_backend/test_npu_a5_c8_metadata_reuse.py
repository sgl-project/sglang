"""CPU contracts for A5 C8 LIM scale and owner-plan reuse."""

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
        "sgl_kernel_npu.sparsity_driven_kv_offload": SimpleNamespace(
            unidex_copy_inplace=Mock()
        ),
        "sglang.kernels.ops.attention.npu_copy_sfa": SimpleNamespace(
            fused_scatter_copy_sparse_flash_attention=Mock(),
            load_copy_sfa_operator=Mock(),
        ),
        "sglang.kernels.ops.attention.npu_lim": SimpleNamespace(
            fused_lightning_indexer_manage=Mock(),
            load_fused_lim_operator=Mock(),
        ),
        "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget": SimpleNamespace(
            C8_ROW_BYTES=656
        ),
        "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.host_pool": SimpleNamespace(
            HostKVPool=object
        ),
        "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.prefill_kv_plan": Mock(),
        "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.resident_pool": SimpleNamespace(
            ResidentKVPool=object
        ),
        "sglang.srt.layers.dp_attention": SimpleNamespace(
            get_eager_max_batch_size=Mock()
        ),
        "sglang.srt.mem_cache.memory_pool": SimpleNamespace(
            MLATokenToKVPool=object, ReqToTokenPool=object
        ),
        "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(
            ForwardBatch=object
        ),
        "sglang.srt.runtime_context": SimpleNamespace(get_parallel=Mock()),
        "sgl_kernel_npu.operators.attention.prepare_copy_sfa_metadata": SimpleNamespace(
            prepare_copy_sfa_metadata=prepare_metadata
        ),
    }
    loaded = []
    with patch.dict(sys.modules, dependencies):
        try:
            for name in ("fused_lim", "fused_manager", "c8_manager"):
                full_name = f"{package}.{name}"
                spec = importlib.util.spec_from_file_location(
                    full_name, path / f"{name}.py"
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = module
                loaded.append(full_name)
                spec.loader.exec_module(module)
            yield module.C8DecodeAdapter, module.C8SparseKVCacheManager
        finally:
            for name in reversed(loaded):
                sys.modules.pop(name, None)


def _adapter(adapter_type, batch, width, topk, capacity):
    adapter = object.__new__(adapter_type)
    rows = batch * width
    adapter.capacity = capacity
    adapter._prepared_batch = None
    adapter._prepare_forward = Mock(
        side_effect=lambda _table, forward_batch, _width: setattr(
            adapter, "_prepared_batch", forward_batch
        )
    )
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
    adapter.actual_seq_lengths_kv = torch.empty_like(adapter._pre_lengths)
    adapter._pre_cache_sizes = torch.tensor([capacity, 0], dtype=torch.int32)
    adapter._pre_states = torch.zeros(batch, dtype=torch.int32)
    adapter._pre_sparse = torch.tensor([True, False])
    adapter._pre_valid = torch.tensor([True, False])
    adapter.request_safe_slots = torch.tensor([7, 0], dtype=torch.int32)
    adapter.resident_safe_slots = torch.tensor([2, 3], dtype=torch.int32)
    return adapter


def test_c8_lim_reuses_base_metadata_but_consumes_per_layer_scales():
    metadata_calls = []

    def prepare_metadata(**kwargs):
        metadata_calls.append(kwargs)
        kwargs["actual_seq_lengths_kv"].copy_(torch.tensor([11, 0]))

    with _load_c8_classes(prepare_metadata) as (adapter_type, manager_type):
        batch, width, topk, capacity = 2, 4, 3, 8
        rows = batch * width
        adapter = _adapter(adapter_type, batch, width, topk, capacity)
        forward_batch = SimpleNamespace(
            batch_size=batch,
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            spec_info=SimpleNamespace(num_tokens_per_req=width),
        )
        query_scale = torch.arange(rows, dtype=torch.float16).view(rows, 1)
        key_scale = torch.ones(3, dtype=torch.float16)
        slot_pools = [torch.zeros((4, capacity), dtype=torch.int32) for _ in range(2)]

        def select(slot_pool, *, reuse_metadata=False):
            return adapter.select(
                query=torch.zeros((rows, 1, 128), dtype=torch.int8),
                weights=torch.zeros((rows, 1), dtype=torch.bfloat16),
                key=torch.zeros(1, dtype=torch.int8),
                block_table=torch.zeros((batch, 2), dtype=torch.int32),
                forward_batch=forward_batch,
                slot_pool=slot_pool,
                query_scale=query_scale,
                key_scale=key_scale,
                reuse_metadata=reuse_metadata,
            )

        manager = object.__new__(manager_type)
        manager.lim_adapter = adapter
        manager.planner_hbm_block_table = torch.full((batch, 2), -1, dtype=torch.int32)

        first = select(slot_pools[0])
        manager.prepare_copy_sfa_plan(first)
        second = select(slot_pools[1], reuse_metadata=True)
        manager.prepare_copy_sfa_plan(second)

        adapter._prepare_forward.assert_called_once()
        assert adapter.operator.call_count == 2
        assert adapter.operator.call_args.kwargs["slot_pool"] is slot_pools[1]
        assert (
            adapter.operator.call_args.kwargs["query_scale"].data_ptr()
            == query_scale.data_ptr()
        )
        assert adapter.operator.call_args.kwargs["key_scale"] is key_scale
        assert len(metadata_calls) == 2
        assert adapter.actual_seq_lengths_kv.tolist() == [11, 0]


def test_c8_copy_sfa_consumes_the_explicitly_prepared_plan():
    with _load_c8_classes(Mock()) as (_, manager_type):
        batch, width, capacity = 2, 4, 8
        rows = batch * width
        manager = object.__new__(manager_type)
        manager.start_layer = 0
        manager.page_size = 4
        manager.kv_lora_rank = 512
        manager.actual_request_rows = 3
        manager.budget = SimpleNamespace(copy_sfa_query_heads=1)
        manager.lim_adapter = SimpleNamespace(
            block_table=torch.zeros((batch, 2), dtype=torch.int32),
            query_ends=torch.arange(1, width + 1)[:, None].repeat(1, batch),
            actual_seq_lengths_kv=torch.tensor([11, 0], dtype=torch.int32),
            _pre_cache_sizes=torch.tensor([capacity, 0], dtype=torch.int32),
        )
        manager.planner_hbm_block_table = torch.tensor([[0, 1], [2, 3]])
        manager.resident_pool = SimpleNamespace(
            device_k_buffer=[torch.zeros((2, capacity, 1, 656), dtype=torch.int8)],
            short_resident_end=torch.zeros((1, 3), dtype=torch.int32),
        )
        manager.host_kv_buffer = [torch.zeros((1, 4, 1, 656), dtype=torch.int8)]
        manager.dev_ptr_list = [0]
        manager.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((3, 16), dtype=torch.int32)
        )
        manager._copy_sfa_operator = Mock(
            side_effect=lambda *args, **kwargs: args[17].fill_(1)
        )
        plan = SimpleNamespace(
            valid_rows=torch.tensor([True, False]),
            query_width=width,
            device_slots=torch.zeros((rows, 3), dtype=torch.int32),
            source_ids=torch.zeros((rows, 3), dtype=torch.int32),
            topk_misses=torch.zeros(rows, dtype=torch.int32),
            miss_sources=torch.zeros((batch, 3), dtype=torch.int32),
            miss_slots=torch.zeros((batch, 3), dtype=torch.int32),
            miss_counts=torch.zeros(batch, dtype=torch.int32),
            request_safe_slots=torch.tensor([1, 0], dtype=torch.int32),
        )
        current = (
            torch.zeros((rows, 656), dtype=torch.int8),
            torch.arange(rows),
            torch.ones(rows, dtype=torch.bool),
        )

        output = manager.copy_sfa_attention(
            SimpleNamespace(layer_id=0, scaling=0.1),
            torch.zeros((rows, 1, 512)),
            torch.zeros((rows, 1, 64)),
            plan,
            current,
        )

        assert output.shape == (rows, 512)
        args = manager._copy_sfa_operator.call_args.args
        assert args[5].max().item() == args[2].shape[0] - 1
        assert (
            args[11].data_ptr() == manager.lim_adapter.actual_seq_lengths_kv.data_ptr()
        )
        assert manager._copy_sfa_operator.call_args.kwargs["native_lim_plan"] is True


def test_c8_mixed_prefill_invalidates_identity_lim_state_on_the_last_layer():
    with _load_c8_classes(Mock()) as (_, manager_type):
        manager = object.__new__(manager_type)
        manager.start_layer = 0
        manager.layer_num = 2
        manager.kv_lora_rank = 512
        manager.qk_rope_head_dim = 64
        manager.host_pool = SimpleNamespace(rank=1, publish=Mock())
        manager.resident_pool = SimpleNamespace(
            request_to_resident=None,
            short_resident_end=torch.full((2, 4), 7, dtype=torch.int32),
            last_prefix=torch.full((4,), 9, dtype=torch.int32),
        )
        pack = Mock(return_value=torch.zeros((2, 656), dtype=torch.int8))
        modules = {
            "sglang.srt.hardware_backend.npu.kv_capability": SimpleNamespace(
                DSA_KV_QUANT_BLOCK_SIZE=128
            ),
            "sglang.srt.hardware_backend.npu.memory_pool_npu": SimpleNamespace(
                pack_dsa_quant_kv_cache=pack
            ),
        }
        batch = SimpleNamespace(
            batch_size=2,
            req_pool_indices=torch.tensor([1, 2, -1]),
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        )
        rows = (
            torch.arange(2),
            torch.arange(2),
            torch.arange(2),
            torch.ones(2, dtype=torch.bool),
        )
        with patch.dict(sys.modules, modules):
            manager.offload_v2(
                torch.zeros((2, 1, 512)),
                torch.zeros((2, 1, 64)),
                SimpleNamespace(layer_id=1),
                batch,
                offload_rows=rows,
            )

        assert manager.resident_pool.short_resident_end[1].tolist() == [7, -1, -1, 7]
        assert manager.resident_pool.last_prefix.tolist() == [9, -1, -1, 9]
        manager.host_pool.publish.assert_called_once()
        assert pack.call_args.kwargs["quant_block_size"] == 128
        assert pack.call_args.kwargs["dtype"] == torch.float8_e4m3fn


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
