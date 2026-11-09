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
    prepare_mtp_rows,
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
                    attention.forward_sparsity_driven_kv_offload(
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

    def test_model_index_heads_control_query_scale_budget(self):
        options = self.options()
        options["model_config"].hf_config.index_n_heads = 32
        small = resolve_sparse_kv_budget(**options)
        options["model_config"].hf_config.index_n_heads = 64
        large = resolve_sparse_kv_budget(**options)
        self.assertEqual(small.adapter_buffers["query_scale"].shape[-1], 32)
        self.assertEqual(large.adapter_buffers["query_scale"].shape[-1], 64)
        self.assertEqual(
            large.managed_hbm_bytes - small.managed_hbm_bytes,
            small.batch_capacity * small.max_query_width * (64 - 32) * 4,
        )

    def test_unsupported_model_geometry_is_rejected(self):
        for field, value in (
            ("kv_lora_rank", 256),
            ("qk_rope_head_dim", 32),
            ("index_head_dim", 96),
        ):
            options = self.options()
            setattr(options["model_config"], field, value)
            with self.assertRaises(ValueError):
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
        for name, width in (
            ("prefill_packed", 576),
            ("prefill_nope", 512),
            ("prefill_rope", 64),
        ):
            spec = plan.manager_buffers[name]
            self.assertEqual(spec.shape, (129, 1, width))
            self.assertEqual(spec.dtype, torch.bfloat16)
        self.assertEqual(
            plan.managed_hbm_bytes - base.managed_hbm_bytes, 129 * 1152 * 2
        )
        self.assertEqual(plan.prefill_workspace_bytes, 129 * 1152 * 2)
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
        with self.assertRaises(ValueError):
            resolve_sparse_kv_budget(**self.options(prefill_workspace_tokens=-1))

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
                sparse_context_len=2048,
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
            post.fused_lim_copy_sfa_metadata = Mock()
            with (
                patch(prefix + "fused_lim.load_fused_lim_operator", return_value=None),
                patch.object(
                    torch.ops.npu,
                    "lim_full_visible_max_query_width",
                    SimpleNamespace(default=lambda: 14),
                    create=True,
                ),
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
                if isinstance(t, torch.Tensor) and t is not adapter.key_scale
            )
            self.assertEqual(manager.budget.adapter_bytes, allocated)
            self.assertEqual(adapter.query_scale.shape[-1], indexer_heads)

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
        req_pool = ReqToTokenPool(2, 4097, "cpu", False)
        req_pool.req_to_token[2, 4096] = 73
        pre, post = ModuleType("pre"), ModuleType("post")
        pre.fused_prepare_lim_metadata = Mock()
        post.fused_lim_copy_sfa_metadata = Mock()

        def shm(shape, dtype, device_id, name):
            tensor = torch.empty(shape, dtype=dtype)
            return tensor, tensor.data_ptr(), tensor.data_ptr()

        with (
            patch.object(fused_manager, "load_copy_sfa_operator", return_value=None),
            patch.object(fused_manager, "create_shm_tensor", side_effect=shm),
            patch.object(
                torch, "npu", SimpleNamespace(current_device=lambda: 0), create=True
            ),
            patch(prefix + "fused_lim.load_fused_lim_operator", return_value=None),
            patch.object(
                torch.ops.npu,
                "lim_full_visible_max_query_width",
                SimpleNamespace(default=lambda: 14),
                create=True,
            ),
            patch.dict(
                sys.modules,
                {
                    prefix + "npu_lim_input_metadata": pre,
                    prefix + "npu_lim_copy_sfa_metadata": post,
                },
            ),
        ):
            manager = fused_manager.SparseKVCacheManager(
                req_pool, None, 2048, graph_batch_capacity=8, paged_kv_cache=pool
            )
        self.assertIs(manager.budget, plan)
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
        self.assertEqual(
            manager.lim_adapter.key_scale.numel()
            * manager.lim_adapter.key_scale.element_size(),
            (4096 + 128) * 4,
        )
        self.assertIs(plan.bind_native_pool(pool), plan)
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
        pool.kv_lora_rank = 256
        with self.assertRaises(ValueError):
            plan.bind_native_pool(pool)
        pool.kv_lora_rank = 512
        pool.size = 3968
        with self.assertRaises(ValueError):
            plan.bind_native_pool(pool)

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
            req, pos, valid = prepare_mtp_rows(batch, count, 3, 1024)
            self.assertEqual(pos.shape, (2, width))
            self.assertEqual(pos[valid].tolist(), list(range(100, 100 + width)))
            self.assertEqual(req[valid].tolist(), [1] * width)


if __name__ == "__main__":
    unittest.main()
