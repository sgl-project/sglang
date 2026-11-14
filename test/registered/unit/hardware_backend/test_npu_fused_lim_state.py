"""CPU contracts for native-only sparse offload and actual query geometry."""

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget import (
    resolve_sparse_kv_budget,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_lim import (
    LIMDecodeAdapter,
)
from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
    mtp_row_geometry,
    sparse_query_width,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class TestLIMState(CustomTestCase):
    def test_fused_attention_rejects_missing_host_publication_before_side_effects(self):
        module_name = (
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_attention"
        )
        manager = SimpleNamespace(offload_v2=Mock())
        backend = SimpleNamespace(sparse_kv_manager=manager)
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            attention = importlib.import_module(module_name)
            with patch.object(torch, "cat") as cat:
                with self.assertRaisesRegex(
                    ValueError, "publishing current KV to host"
                ):
                    attention.forward_fused_sparse_kv_offload(
                        backend, None, None, None, None, None, save_kv_cache=False
                    )
            cat.assert_not_called()
        manager.offload_v2.assert_not_called()

    def options(self, **kwargs):
        return dict(
            model_config=SimpleNamespace(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
                hf_config=SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLM"],
                    index_head_dim=128,
                    index_n_heads=64,
                    index_topk=2048,
                ),
            ),
            page_size=128,
            copy_sfa_query_heads=128,
            request_rows=3,
            graph_batch_capacity=8,
            context_len=32768,
            num_layers=2,
            top_k=2048,
            kv_bytes_per_token=1152,
            **kwargs,
        )

    def test_model_index_heads_do_not_change_lim_scale_free_budget(self):
        options = self.options()
        options["model_config"].hf_config.index_n_heads = 32
        small = resolve_sparse_kv_budget(**options)
        options["model_config"].hf_config.index_n_heads = 64
        large = resolve_sparse_kv_budget(**options)
        self.assertNotIn("query_scale", small.adapter_buffers)
        self.assertNotIn("query_scale", large.adapter_buffers)
        self.assertEqual(large.managed_hbm_bytes, small.managed_hbm_bytes)

    def test_native_source_addressing_limit(self):
        options = self.options()
        options["context_len"] = 2**21
        self.assertEqual(resolve_sparse_kv_budget(**options).context_len, 2**21)
        options["context_len"] += 1
        with self.assertRaisesRegex(ValueError, "context length"):
            resolve_sparse_kv_budget(**options)

    def test_query_geometry_uses_actual_verify_width(self):
        for width in (1, 2, 3, 4, 5, 8, 14):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(num_tokens_per_req=width),
            )
            self.assertEqual(sparse_query_width(batch), width)
            batch.forward_mode = ForwardMode.DECODE
            self.assertEqual(sparse_query_width(batch), 1)

    def test_missing_verify_geometry_is_rejected(self):
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY, spec_info=None)
        with self.assertRaisesRegex(ValueError, "actual chain"):
            sparse_query_width(batch)

    def test_budget_defaults_to_configured_union_capacity(self):
        for width in (1, 2, 3, 4, 5, 8, 14):
            budget = resolve_sparse_kv_budget(**self.options(max_query_width=width))
            self.assertEqual(budget.cache_tokens, width * 2048)

    def test_optional_prefill_workspace_budget_and_fallback(self):
        base = resolve_sparse_kv_budget(**self.options())
        plan = resolve_sparse_kv_budget(**self.options(prefill_workspace_tokens=129))
        self.assertEqual(base.prefill_workspace_tokens, 0)
        self.assertEqual(plan.prefill_workspace_tokens, 129)
        self.assertNotIn("prefill_packed", plan.manager_buffers)
        for name, width in (
            ("prefill_nope", 512),
            ("prefill_rope", 64),
        ):
            spec = plan.manager_buffers[name]
            self.assertEqual(spec.shape, (129, 1, width))
            self.assertEqual(spec.dtype, torch.bfloat16)
        self.assertEqual(plan.managed_hbm_bytes - base.managed_hbm_bytes, 129 * 576 * 2)
        self.assertEqual(plan.prefill_workspace_bytes, 129 * 576 * 2)
        self.assertEqual(plan.copy_sfa_bytes, base.copy_sfa_bytes)
        fitting = resolve_sparse_kv_budget(
            **self.options(
                prefill_workspace_tokens=129, hbm_budget_bytes=plan.managed_hbm_bytes
            )
        )
        self.assertEqual(fitting.prefill_workspace_tokens, 129)
        with self.assertLogs(
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_budget",
            level="WARNING",
        ) as logs:
            fallback = resolve_sparse_kv_budget(
                **self.options(
                    prefill_workspace_tokens=129,
                    hbm_budget_bytes=base.managed_hbm_bytes,
                )
            )
        self.assertIn("129 tokens", logs.output[0])
        self.assertEqual(fallback.prefill_workspace_tokens, 0)
        self.assertEqual(fallback.managed_hbm_bytes, base.managed_hbm_bytes)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            resolve_sparse_kv_budget(
                **self.options(
                    prefill_workspace_tokens=129,
                    hbm_budget_bytes=base.managed_hbm_bytes - 1,
                )
            )

    def test_budget_rejects_insufficient_union_capacity_and_width(self):
        for width in (2, 3, 5, 14):
            with self.assertRaisesRegex(ValueError, "query"):
                resolve_sparse_kv_budget(
                    **self.options(max_query_width=width, cache_tokens=2048)
                )
        for width in (0, 15):
            with self.assertRaisesRegex(ValueError, "query"):
                resolve_sparse_kv_budget(**self.options(max_query_width=width))

    def test_pool_sizing_reserves_native_capacity_before_workspace(self):
        from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator

        for fits in (True, False):
            plan = resolve_sparse_kv_budget(
                **self.options(prefill_workspace_tokens=129)
            )
            configurator = DefaultPoolConfigurator.__new__(DefaultPoolConfigurator)
            configurator._sparse_budget = plan
            configurator._sparse_min_tokens = 129
            configurator._cell_size = 8
            configurator._fixed_overhead_bytes = plan.managed_hbm_bytes
            required = plan.managed_hbm_bytes + 256 * 8
            result = configurator.calculate_pool_sizes(required - int(not fits), 128)
            self.assertEqual(plan.prefill_workspace_tokens, 129 if fits else 0)
            self.assertGreaterEqual(result.max_total_num_tokens, 256)
            # Later recalculation cannot alter an already selected plan.
            with self.assertRaisesRegex(RuntimeError, "Not enough memory"):
                configurator.calculate_pool_sizes(0, 128)
            self.assertEqual(plan.prefill_workspace_tokens, 129 if fits else 0)

    def test_graph_padding_grows_scratch_not_request_residency(self):
        small = resolve_sparse_kv_budget(
            **{**self.options(), "graph_batch_capacity": 3}
        )
        padded = resolve_sparse_kv_budget(**self.options())
        self.assertEqual(small.device_kv_bytes, padded.device_kv_bytes)
        self.assertEqual(small.slot_map_bytes, padded.slot_map_bytes)
        self.assertGreater(padded.copy_sfa_bytes, small.copy_sfa_bytes)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            resolve_sparse_kv_budget(
                **self.options(hbm_budget_bytes=small.managed_hbm_bytes)
            )

    def test_adapter_budget_matches_actual_dynamic_query_allocations(self):
        prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload."
        for width, indexer_heads in (
            (1, 32),
            (4, 32),
            (1, 64),
            (3, 64),
            (4, 64),
            (8, 64),
            (14, 64),
        ):
            options = self.options(max_query_width=width)
            options["model_config"].hf_config.index_n_heads = indexer_heads
            manager = SimpleNamespace(
                budget=resolve_sparse_kv_budget(**options),
                cache_capacity=width * 2048,
                size=3,
                batch_capacity=8,
                max_query_width=width,
                sparse_topk=2048,
                max_context_len=32768,
                layer_num=2,
                device="cpu",
                store_dtype=torch.bfloat16,
                start_layer=0,
                paged_kv_cache=SimpleNamespace(
                    get_index_k_buffer=lambda _: torch.empty(
                        (256, 128, 1, 128), dtype=torch.bfloat16
                    )
                ),
            )
            pool = manager.paged_kv_cache
            pool.size, pool.page_size, pool.layer_num, pool.start_layer = (
                32640,
                128,
                2,
                0,
            )
            pool.store_dtype, pool.kv_lora_rank, pool.qk_rope_head_dim = (
                torch.bfloat16,
                512,
                64,
            )
            manager.budget.bind_native_pool(pool)
            pre, post = ModuleType("pre"), ModuleType("post")
            pre.fused_prepare_lim_metadata = Mock()
            post.prepare_copy_sfa_metadata = Mock()
            with (
                patch(prefix + "fused_lim.load_fused_lim_operator", return_value=None),
                patch.dict(
                    "sys.modules",
                    {
                        prefix + "npu_lim_input_metadata": pre,
                        prefix + "npu_lim_copy_sfa_metadata": post,
                    },
                ),
            ):
                adapter = LIMDecodeAdapter(manager)
            allocated = sum(
                t.numel() * t.element_size()
                for t in vars(adapter).values()
                if isinstance(t, torch.Tensor)
            )
            self.assertEqual(manager.budget.adapter_bytes, allocated)
            manager.budget.validate_buffers(adapter, manager.budget.adapter_buffers)
            self.assertEqual(
                adapter.query_ends.tolist(),
                [[i * j for j in range(1, 9)] for i in range(1, width + 1)],
            )
            self.assertFalse(hasattr(adapter, "query_scale"))
            self.assertFalse(hasattr(adapter, "key_scale"))
            for indices in (adapter.source_ids, adapter.device_slots):
                self.assertEqual(indices.shape, (8 * width, 2048))

    def test_plan_alignment_and_limits(self):
        for width in (1, 4):
            for batch in (3, 8):
                with self.subTest(width=width, batch=batch):
                    options = {
                        **self.options(),
                        "context_len": 4097,
                        "graph_batch_capacity": batch,
                        "max_query_width": width,
                    }
                    plan = resolve_sparse_kv_budget(**options)
                    self.assertEqual(plan.context_len, 4224)
                    self.assertEqual(plan.batch_capacity, batch)

                    self.assertEqual(
                        resolve_sparse_kv_budget(
                            **options, hbm_budget_bytes=plan.managed_hbm_bytes
                        ).managed_hbm_bytes,
                        plan.managed_hbm_bytes,
                    )
                    with self.assertRaisesRegex(ValueError, "exceeding"):
                        resolve_sparse_kv_budget(
                            **options, hbm_budget_bytes=plan.managed_hbm_bytes - 1
                        )
        with self.assertRaises(ValueError):
            resolve_sparse_kv_budget(**self.options(kv_dtype=torch.float32))

    def test_manager_and_adapter_use_same_bound_plan(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool

        native_package = ModuleType("sgl_kernel_npu")
        native_package.__path__ = []
        native_api = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload")
        native_fused = ModuleType("sgl_kernel_npu.sparsity_driven_kv_offload.fused_shm")
        native_fused.create_fused_shm_tensor = Mock()
        native_api.unidex_copy_inplace = Mock()
        module_name = (
            "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager"
        )
        with patch.dict(
            sys.modules,
            {
                "sgl_kernel_npu": native_package,
                "sgl_kernel_npu.sparsity_driven_kv_offload": native_api,
                native_fused.__name__: native_fused,
            },
        ):
            fused_manager = importlib.import_module(module_name)

        prefix = "sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload."
        plan = resolve_sparse_kv_budget(
            model_config=self.options()["model_config"],
            page_size=128,
            copy_sfa_query_heads=128,
            request_rows=4,
            context_len=4097,
            num_layers=2,
            top_k=2048,
            kv_bytes_per_token=1152,
            max_query_width=4,
            cache_tokens=8192,
            graph_batch_capacity=8,
        )
        pool = object.__new__(MLATokenToKVPool)
        pool.size, pool.page_size, pool.layer_num, pool.start_layer = 4096, 128, 2, 0
        pool.store_dtype, pool.kv_lora_rank, pool.qk_rope_head_dim = (
            torch.bfloat16,
            512,
            64,
        )
        pool.get_index_k_buffer = lambda _: torch.empty(
            (33, 128, 1, 128), dtype=torch.bfloat16
        )
        pool.sparse_kv_budget_plan = plan
        pool.indexer_layer_ids = (1,)
        pool.get_index_k_buffer = Mock(return_value=pool.get_index_k_buffer(0))
        plan.bind_native_pool(pool)
        req_pool = ReqToTokenPool(2, 4097, "cpu", False)
        req_pool.req_to_token[2, 4096] = 73
        pre, post = ModuleType("pre"), ModuleType("post")
        pre.fused_prepare_lim_metadata = Mock()
        post.prepare_copy_sfa_metadata = Mock()

        def shm(shape, dtype, device_id, name):
            tensor = torch.empty(shape, dtype=dtype)
            return tensor, tensor.data_ptr(), tensor.data_ptr()

        with (
            patch.object(
                fused_manager,
                "load_copy_sfa_operator",
                return_value=SimpleNamespace(
                    _schema=SimpleNamespace(
                        arguments=[SimpleNamespace(name="dram_rope_dev_ptr")]
                    )
                ),
            ),
            patch.object(fused_manager, "create_shm_tensor", side_effect=shm),
            patch.object(
                torch, "npu", SimpleNamespace(current_device=lambda: 0), create=True
            ),
            patch(prefix + "fused_lim.load_fused_lim_operator", return_value=None),
            patch.dict(
                sys.modules,
                {
                    prefix + "npu_lim_input_metadata": pre,
                    prefix + "npu_lim_copy_sfa_metadata": post,
                },
            ),
        ):
            manager = fused_manager.FusedSparseKVCacheManager(
                req_pool, None, 2048, paged_kv_cache=pool
            )
        self.assertIs(manager.budget, plan)
        plan.validate_buffers(manager, plan.manager_buffers)
        plan.validate_buffers(manager.lim_adapter, plan.adapter_buffers)
        self.assertEqual(
            [tuple(component[0].shape) for component in manager.host_pool.buffers],
            [
                (1, plan.host_token_capacity, 1, 512),
                (1, plan.host_token_capacity, 1, 64),
            ],
        )
        self.assertEqual(
            sum(t.nbytes for component in manager.host_pool.buffers for t in component),
            plan.host_kv_bytes,
        )
        self.assertIs(pool.sparse_kv_budget_plan, plan)
        pool.get_index_k_buffer.assert_called_with(1)

        def actual_bytes(owner, names):
            tensors = [getattr(owner, name) for name in names]
            return sum(
                t.numel() * t.element_size()
                for value in tensors
                for t in (value if isinstance(value, list) else [value])
            )

        self.assertEqual(
            actual_bytes(manager, plan.manager_buffers)
            + actual_bytes(manager.lim_adapter, plan.adapter_buffers),
            plan.managed_hbm_bytes,
        )
        self.assertFalse(hasattr(manager.lim_adapter, "key_scale"))

        self.assertEqual(
            manager.host_source_indices(
                torch.tensor([3, 99]), torch.tensor([4096, 9999])
            ).tolist(),
            [73, 73],
        )
        original = manager.lim_adapter.source_ids
        manager.lim_adapter.source_ids = original[:1]
        with self.assertRaises(ValueError):
            plan.validate_buffers(manager.lim_adapter, plan.adapter_buffers)
        manager.lim_adapter.source_ids = original
        # Exercise both native boundaries: selecting column zero would silently
        # truncate the plan, while a framework unsqueeze would break CopySFA's ABI.
        rows = 2 * manager.max_query_width
        source_values = torch.arange(rows * 2048, dtype=torch.int32).view(rows, 2048)
        slot_values = source_values.remainder(manager.cache_capacity)

        def native_lim(*tensors, full_visible=False):
            self.assertEqual(tensors[11].shape, source_values.shape)
            self.assertEqual(tensors[12].shape, slot_values.shape)
            tensors[11].copy_(source_values)
            tensors[12].copy_(slot_values)

        adapter = manager.lim_adapter
        adapter.operator = native_lim
        adapter.request_safe_slots[:2].copy_(torch.tensor([1, 2]))
        adapter._pre_next_prefix[:2].zero_()
        adapter._pre_valid[:2].fill_(True)
        adapter._pre_cache_sizes[:2].copy_(torch.tensor([8192, 0]))
        adapter.block_table[:2].copy_(torch.arange(66).view(2, 33))
        selection = adapter.select(
            manager,
            query=torch.zeros(rows, 64, 128, dtype=torch.bfloat16),
            weights=torch.ones(rows, 64, dtype=torch.bfloat16),
            key=pool.get_index_k_buffer(1),
            block_table=torch.zeros(2, 33, dtype=torch.int32),
            forward_batch=SimpleNamespace(
                batch_size=2,
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(num_tokens_per_req=manager.max_query_width),
                seq_lens=torch.zeros(2, dtype=torch.int32),
                positions=torch.arange(rows, dtype=torch.int32),
                req_pool_indices=torch.tensor([1, 2]),
            ),
            layer_id=0,
        )
        torch.testing.assert_close(selection.source_ids, source_values)
        torch.testing.assert_close(selection.device_slots, slot_values)
        self.assertEqual(selection.source_ids.data_ptr(), adapter.source_ids.data_ptr())
        self.assertEqual(
            selection.device_slots.data_ptr(), adapter.device_slots.data_ptr()
        )

        def native_copy_sfa(*tensors, **kwargs):
            self.assertEqual(tensors[2].shape[-1], 512)
            self.assertEqual(tensors[3].shape[-1], 64)
            torch.testing.assert_close(
                tensors[4],
                torch.tensor([manager.max_query_width, rows], dtype=torch.int32),
            )
            self.assertEqual(
                tensors[4].data_ptr(),
                adapter.query_ends[manager.max_query_width - 1].data_ptr(),
            )
            self.assertEqual(tensors[6].data_ptr(), adapter._pre_cache_sizes.data_ptr())
            torch.testing.assert_close(
                tensors[6], torch.tensor([8192, 0], dtype=torch.int32)
            )
            self.assertEqual(tensors[14].data_ptr(), adapter.block_table.data_ptr())
            torch.testing.assert_close(
                tensors[14], torch.arange(66, dtype=torch.int32).view(2, 33)
            )
            self.assertEqual(tensors[7].ndim, 2)
            self.assertEqual(tensors[8].ndim, 2)
            torch.testing.assert_close(tensors[7], source_values)
            torch.testing.assert_close(tensors[8], slot_values)
            tensors[16].fill_(3)

        manager._copy_sfa_operator = native_copy_sfa
        result = manager.copy_sfa_attention(
            SimpleNamespace(layer_id=0, scaling=0.1),
            torch.zeros(rows, 8, 512, dtype=torch.bfloat16),
            torch.zeros(rows, 8, 64, dtype=torch.bfloat16),
            selection,
            (
                torch.empty(rows, 1, 576, dtype=torch.bfloat16),
                torch.arange(rows, dtype=torch.int32),
                torch.ones(rows, dtype=torch.bool),
            ),
        )
        torch.testing.assert_close(
            result, torch.full((rows, 8 * 512), 3, dtype=result.dtype)
        )
        # Use the allocated index pages, including their dummy page, rather
        # than reconstructing the physical capacity from a logical size.
        pool.size = 3968

        self.assertEqual(plan.host_token_capacity, 4096 + 128)
        self.assertEqual(plan.host_kv_bytes, (4096 + 128) * 576 * 2 * 2)

    def test_host_bytes_are_unavailable_before_physical_pool_binding(self):
        plan = resolve_sparse_kv_budget(**self.options())
        with self.assertRaisesRegex(RuntimeError, "Bind the native KV pool"):
            _ = plan.host_kv_bytes

    def test_tp_suffix_does_not_extend_actual_query_windows(self):
        for width in (2, 3, 5, 8, 14):
            count = 2 * width + 3
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(num_tokens_per_req=width),
                req_pool_indices=torch.tensor([1, 0]),
                seq_lens=torch.tensor([100, 0]),
                positions=torch.tensor(
                    list(range(100, 100 + width)) + [0] * (count - width)
                ),
                out_cache_loc=torch.tensor(list(range(width)) + [-1] * (count - width)),
            )
            self.assertEqual(mtp_row_geometry(batch, count), (2, width, 2 * width))


if __name__ == "__main__":
    unittest.main()
