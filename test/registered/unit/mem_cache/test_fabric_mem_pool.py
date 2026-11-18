"""Unit tests for native CUDA VMM CU_MEM_HANDLE_TYPE_FABRIC custom memory pool."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.common import staging_handler
from sglang.srt.disaggregation.mooncake.utils import init_mooncake_custom_mem_pool
from sglang.srt.environ import envs
from sglang.srt.mem_cache import fabric_mem_pool
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.runtime_context import get_context


class TestFabricMemPool(unittest.TestCase):
    def setUp(self):
        fabric_mem_pool._FABRIC_MEM_POOLS.clear()

    def tearDown(self):
        fabric_mem_pool._FABRIC_MEM_POOLS.clear()

    def test_cpp_pluggable_allocator_stub_compiles_and_tracks_all_sizes(self):
        stub = fabric_mem_pool.CudaVmmFabricAllocatorStub(
            build_pluggable_allocator=False
        )
        gran = 2 * 1024 * 1024  # 2 MiB VMM recommended granularity
        stub.set_mock_mode(True, granularity=gran)
        self.addCleanup(stub.set_mock_mode, False, gran)

        # Both multi-GiB target KV buffers and smaller speculative EAGLE draft
        # KV buffers must be backed without any minimum size cutoff.
        target_req_bytes = 8 * 1024 * 1024 * 1024 + 4096
        draft_req_bytes = 64 * 1024 * 1024 + 512

        ptr_target = stub._fn_malloc(target_req_bytes, 0, None)
        ptr_draft = stub._fn_malloc(draft_req_bytes, 0, None)
        self.assertIsNotNone(ptr_target)
        self.assertIsNotNone(ptr_draft)
        self.assertNotEqual(ptr_target, ptr_draft)
        self.assertEqual(ptr_target % gran, 0)
        self.assertEqual(ptr_draft % gran, 0)
        self.assertEqual(stub.active_allocations, 2)

        expected_target = ((target_req_bytes + gran - 1) // gran) * gran
        expected_draft = ((draft_req_bytes + gran - 1) // gran) * gran
        self.assertEqual(stub.mapped_bytes, expected_target + expected_draft)
        self.assertEqual(stub.last_error, 0)

        stub._fn_free(ptr_draft, draft_req_bytes, 0, None)
        self.assertEqual(stub.active_allocations, 1)
        self.assertEqual(stub.mapped_bytes, expected_target)

        stub._fn_free(ptr_target, target_req_bytes, 0, None)
        self.assertEqual(stub.active_allocations, 0)
        self.assertEqual(stub.mapped_bytes, 0)

    def test_per_device_singleton_mem_pool_reused_across_target_and_draft_pools(self):
        entered_devices = []

        @contextmanager
        def fake_cuda_device(dev):
            entered_devices.append(dev)
            yield

        fake_stub = SimpleNamespace(allocator=object())
        created_pools = []

        def fake_mem_pool_ctor(allocator):
            pool = SimpleNamespace(allocator=allocator, idx=len(created_pools))
            created_pools.append(pool)
            return pool

        with (
            patch.object(
                fabric_mem_pool, "_get_allocator_stub", return_value=fake_stub
            ),
            patch("torch.cuda.device", side_effect=fake_cuda_device),
            patch("torch.cuda.MemPool", side_effect=fake_mem_pool_ctor),
        ):
            target_pool_d2 = fabric_mem_pool.get_cuda_vmm_fabric_mem_pool("cuda:2")
            draft_pool_d2 = fabric_mem_pool.get_cuda_vmm_fabric_mem_pool(
                torch.device("cuda", 2)
            )
            target_pool_d0 = fabric_mem_pool.get_cuda_vmm_fabric_mem_pool("cuda:0")

        self.assertIs(target_pool_d2, draft_pool_d2)
        self.assertIsNot(target_pool_d2, target_pool_d0)
        self.assertEqual(len(created_pools), 2)
        self.assertEqual(
            entered_devices,
            [torch.device("cuda", 2), torch.device("cuda", 0)],
        )

    def test_explicit_fabric_and_nvlink_importerror_fallback(self):
        sentinel_pool = object()

        with (
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override("FABRIC"),
            patch.object(
                fabric_mem_pool,
                "get_cuda_vmm_fabric_mem_pool",
                return_value=sentinel_pool,
            ),
        ):
            enabled, pool, pool_type = maybe_init_custom_mem_pool("cuda:1")
            self.assertTrue(enabled)
            self.assertIs(pool, sentinel_pool)
            self.assertEqual(pool_type, "FABRIC")

        # When SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK (or True) is set in a
        # NIXL-only image without mooncake.allocator, fabric-ready GPUs fall
        # back to the native CUDA VMM FABRIC pool instead of cudaMalloc.
        with (
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override("True"),
            patch.dict("sys.modules", {"mooncake": None, "mooncake.allocator": None}),
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "sglang.srt.utils.cuda_vmm_utils.is_gpu_fabric_ready",
                return_value=True,
            ),
            patch.object(
                fabric_mem_pool,
                "get_cuda_vmm_fabric_mem_pool",
                return_value=sentinel_pool,
            ),
        ):
            enabled, pool, pool_type = init_mooncake_custom_mem_pool("cuda:0")
            self.assertTrue(enabled)
            self.assertIs(pool, sentinel_pool)
            self.assertEqual(pool_type, "FABRIC")

        # Non-fabric GPUs keep the existing default-pool fallback on ImportError.
        with (
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override("NVLINK"),
            patch.dict("sys.modules", {"mooncake": None, "mooncake.allocator": None}),
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "sglang.srt.utils.cuda_vmm_utils.is_gpu_fabric_ready",
                return_value=False,
            ),
        ):
            enabled, pool, pool_type = init_mooncake_custom_mem_pool("cuda:0")
            self.assertFalse(enabled)
            self.assertIsNone(pool)
            self.assertIsNone(pool_type)

    def test_auto_nixl_fabric_activation_and_opt_out(self):
        sentinel_pool = object()
        override = get_context().override_server_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            enable_memory_saver=False,
        )
        override.install()
        self.addCleanup(override.restore)

        with (
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(None),
            envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(True),
            patch("torch.cuda.is_available", return_value=True),
            patch.object(fabric_mem_pool, "is_gpu_fabric_ready", return_value=True),
            patch.object(
                fabric_mem_pool,
                "get_cuda_vmm_fabric_mem_pool",
                return_value=sentinel_pool,
            ),
        ):
            # Target pool, speculative draft pool, and staging buffer all
            # receive the same FABRIC MemPool automatically.
            enabled_t, pool_t, type_t = maybe_init_custom_mem_pool("cuda:0")
            enabled_d, pool_d, type_d = maybe_init_custom_mem_pool("cuda:0")
            staging_pool, staging_type = staging_handler._get_custom_mem_pool("cuda:0")
            self.assertTrue(enabled_t)
            self.assertTrue(enabled_d)
            self.assertIs(pool_t, sentinel_pool)
            self.assertIs(pool_d, sentinel_pool)
            self.assertIs(staging_pool, sentinel_pool)
            self.assertEqual(type_t, "FABRIC")
            self.assertEqual(type_d, "FABRIC")
            self.assertEqual(staging_type, "FABRIC")

            # Opt-out env var disables automatic activation cleanly.
            with envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(False):
                enabled_off, pool_off, type_off = maybe_init_custom_mem_pool("cuda:0")
                self.assertFalse(enabled_off)
                self.assertIsNone(pool_off)
                self.assertIsNone(type_off)

        # Non-fabric hosts (e.g. RoCE/InfiniBand without MNNVL clique) stay off.
        with (
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(None),
            envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(True),
            patch("torch.cuda.is_available", return_value=True),
            patch.object(fabric_mem_pool, "is_gpu_fabric_ready", return_value=False),
        ):
            enabled_ib, pool_ib, type_ib = maybe_init_custom_mem_pool("cuda:0")
            self.assertFalse(enabled_ib)
            self.assertIsNone(pool_ib)
            self.assertIsNone(type_ib)

    def test_unpublished_runtime_context_and_memory_saver_return_false(self):
        with (
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(None),
            envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(True),
            patch.object(
                fabric_mem_pool,
                "get_disagg",
                side_effect=ValueError("disagg not published"),
            ),
        ):
            self.assertEqual(maybe_init_custom_mem_pool("cuda:0"), (False, None, None))

        override = get_context().override_server_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
            enable_memory_saver=True,
        )
        override.install()
        self.addCleanup(override.restore)

        with (
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(None),
            envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(True),
            patch("torch.cuda.is_available", return_value=True),
            patch.object(fabric_mem_pool, "is_gpu_fabric_ready", return_value=True),
        ):
            self.assertEqual(maybe_init_custom_mem_pool("cuda:0"), (False, None, None))

    def test_nixl_fabric_custom_mem_pool_planned_predicate(self):
        cfg = MagicMock(
            device="cuda",
            enable_memory_saver=False,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
        )
        with (
            envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch.object(fabric_mem_pool, "is_gpu_fabric_ready", return_value=True),
        ):
            self.assertTrue(fabric_mem_pool.nixl_fabric_custom_mem_pool_planned(cfg))
            with envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(False):
                self.assertFalse(
                    fabric_mem_pool.nixl_fabric_custom_mem_pool_planned(cfg)
                )
            cfg.disaggregation_transfer_backend = "mooncake"
            self.assertFalse(fabric_mem_pool.nixl_fabric_custom_mem_pool_planned(cfg))

    def test_post_capture_kv_sizing_skips_when_nixl_fabric_pool_planned(self):
        from sglang.srt.arg_groups.overrides import post_capture_kv_sizing_planned
        from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
        from sglang.srt.server_args import ServerArgs

        prefill_graph = SimpleNamespace(backend=Backend.BREAKABLE, bs=[4096])
        args = ServerArgs(
            model_path="dummy",
            enable_kv_cache_sharding=False,
            device="cuda",
            dcp_size=1,
            kv_cache_dtype="auto",
            prefill_only_disable_kv_cache=False,
            enable_memory_saver=False,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
            cuda_graph_config=CudaGraphConfig(prefill=prefill_graph),
            chunked_prefill_size=4096,
        )
        with (
            patch(
                "sglang.srt.arg_groups.overrides.use_mla_backend", return_value=False
            ),
            patch(
                "sglang.srt.arg_groups.overrides.max_prefill_buffer_tokens",
                return_value=4096,
            ),
            patch(
                "sglang.srt.arg_groups.overrides.model_config_of",
                return_value=SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"])
                ),
            ),
            envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.override(True),
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(None),
            envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(True),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch.object(fabric_mem_pool, "is_gpu_fabric_ready", return_value=True),
        ):
            self.assertFalse(post_capture_kv_sizing_planned(args))
            with envs.SGLANG_ENABLE_NIXL_FABRIC_MEM_POOL.override(False):
                self.assertTrue(post_capture_kv_sizing_planned(args))


if __name__ == "__main__":
    unittest.main()
