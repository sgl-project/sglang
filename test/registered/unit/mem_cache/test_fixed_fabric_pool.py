"""CPU contracts for the fixed GLM DSA FABRIC pool; these do not prove CUDA/RDMA."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import fixed_fabric_pool as ffp
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
from sglang.srt.utils import cuda_vmm_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

# The only geometry validate_glm_geometry accepts, at the smallest pool size.
GLM_POOL = {
    "size": 64,
    "page_size": 64,
    "kv_lora_rank": 512,
    "dtype": torch.float8_e4m3fn,
    "qk_rope_head_dim": 64,
    "layer_num": 78,
    "device": "cpu",
    "index_head_dim": 128,
    "enable_memory_saver": False,
    "kv_cache_dim": 576,
    "index_buf_size": 64,
    "fixed_fabric_pool": True,
}


class FixedBufferOwnerTests(CustomTestCase):
    def test_buffers_are_256_byte_aligned_views_of_one_allocation(self):
        # A CPU root stands in for the FABRIC mapping; 0xA5 marks bytes never zeroed.
        roots = []

        def as_tensor(allocation, **kwargs):
            roots.append(torch.full((allocation.nbytes,), 0xA5, dtype=torch.uint8))
            return roots[-1]

        with (
            patch.object(
                ffp, "_FabricAllocation", lambda nbytes, device_id: NS(nbytes=nbytes)
            ),
            patch.object(ffp.torch, "as_tensor", new=as_tensor),
        ):
            owner = ffp.FixedFabricBufferOwner(
                device="cuda:0", dtype=torch.float32, shapes=[(3,), (257,), (2,)]
            )
        (root,) = roots
        self.assertEqual(owner.allocation.nbytes, 1544)
        self.assertEqual(
            [tuple(s) for s in owner.buffer_slices], [(0, 12), (256, 1028), (1536, 8)]
        )
        for tensor, (offset, _), shape in zip(
            owner.tensors, owner.buffer_slices, [(3,), (257,), (2,)], strict=True
        ):
            self.assertEqual((tensor.shape, tensor.dtype), (shape, torch.float32))
            self.assertEqual(tensor.data_ptr(), root.data_ptr() + offset)
            self.assertTrue(torch.all(tensor == 0))
        self.assertTrue(torch.all(root[12:256] == 0xA5))
        self.assertTrue(torch.all(root[1284:1536] == 0xA5))


class FabricAllocationTests(CustomTestCase):
    """_FabricAllocation against a recording CUDA driver/VMM boundary."""

    def setUp(self):
        self.calls = calls = []

        class Reservation:
            def __init__(self, size, prop, device, alignment):
                calls.append(("reserve", size, device, alignment))
                self.base = 4 << 20

            def map(self, offset, size, retain_handle):
                calls.append(("map", offset, size, retain_handle))
                return 123

        def prop(device, **kwargs):
            calls.append(("prop", device, kwargs))
            return kwargs

        self.driver = NS(
            CUmemAllocationHandleType=NS(CU_MEM_HANDLE_TYPE_FABRIC=8),
            cuMemExportToShareableHandle=lambda *args: (0, b"fabric"),
        )
        for patcher in (
            patch.multiple(
                cuda_vmm_utils,
                _get_cuda_driver=lambda: self.driver,
                make_device_allocation_prop=prop,
                get_allocation_granularity=lambda prop: 2 << 20,
                VmmReservation=Reservation,
                check_drv=lambda result, label: result[1],
            ),
            patch.object(torch.cuda, "device", new=lambda device: nullcontext()),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fabric_and_rdma_without_fallback(self):
        allocation = ffp._FabricAllocation(3 << 20, 2)
        self.assertEqual(
            self.calls,
            [
                ("prop", 2, {"handle_types": 8, "gpu_direct_rdma": True}),
                ("reserve", 4 << 20, 2, 2 << 20),
                ("map", 0, 4 << 20, True),
            ],
        )
        self.assertEqual(allocation.__cuda_array_interface__["shape"], (3 << 20,))

        # An unsupported device fails at startup instead of trying other handle types.
        def unsupported(*args):
            raise RuntimeError("FABRIC unavailable")

        self.calls.clear()
        self.driver.cuMemExportToShareableHandle = unsupported
        with self.assertRaisesRegex(RuntimeError, "FABRIC unavailable"):
            ffp._FabricAllocation(8448, 0)
        self.assertEqual([call[0] for call in self.calls], ["prop", "reserve", "map"])


class GlmPoolStartupTests(CustomTestCase):
    """The real DSA pool constructor with only the FABRIC owner replaced."""

    def setUp(self):
        self.owners = owners = []

        class RecordingOwner:
            def __init__(self, *, device, dtype, shapes):
                self.shapes, self.dtype = list(shapes), dtype
                self.tensors = [torch.empty(shape, dtype=dtype) for shape in shapes]
                owners.append(self)

        patcher = patch.object(ffp, "FixedFabricBufferOwner", RecordingOwner)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_kv_and_index_buffers_come_from_fixed_owners(self):
        pool = DSATokenToKVPool(**GLM_POOL)
        kv, index = self.owners
        self.assertEqual((kv.shapes, kv.dtype), ([(128, 1, 576)] * 78, torch.uint8))
        self.assertEqual((index.shapes, index.dtype), ([(2, 8448)] * 78, torch.uint8))
        self.assertTrue(
            all(a is b for a, b in zip(pool.kv_buffer, kv.tensors, strict=True))
        )
        self.assertTrue(
            all(
                a is b
                for a, b in zip(pool.index_key_cache.buffer, index.tensors, strict=True)
            )
        )
        # The pointer table is built from the fixed buffers, after they exist.
        self.assertEqual(
            pool.data_ptrs.view(torch.int64).tolist(),
            [t.data_ptr() for t in kv.tensors],
        )

    def test_unsupported_geometry_fails_before_allocating(self):
        # The FP4 indexer packs 68 index bytes per token instead of 132.
        for change in ({"layer_num": 77}, {"use_fp4_indexer": True}):
            with self.subTest(**change), self.assertRaises(ValueError):
                DSATokenToKVPool(**(GLM_POOL | change))
        self.assertEqual(self.owners, [])

    def test_fixed_backing_is_refused_outside_the_dsa_pool(self):
        """Direct MLA or subclass construction would skip the GLM layout check."""
        dsa_only = ("index_head_dim", "index_buf_size", "kv_cache_dim")
        mla = {k: v for k, v in GLM_POOL.items() if k not in dsa_only}
        for build in (
            lambda: MLATokenToKVPool(**mla),
            lambda: type("SubPool", (DSATokenToKVPool,), {})(**GLM_POOL),
        ):
            with self.assertRaisesRegex(ValueError, "built-in DSA"):
                build()
        self.assertEqual(self.owners, [])


def _configurator(**attrs):
    configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
    for name, value in attrs.items():
        setattr(configurator, name, value)
    return configurator


class ConfiguratorTests(CustomTestCase):
    def test_dsa_builder_passes_fabric_flag_only_when_opted_in(self):
        class Pool:
            def __init__(self, size, **kwargs):
                self.kwargs = kwargs

        configurator = _configurator(
            kv_shard_rank=None,
            kv_shard_spec=None,
            kv_shard_group=None,
            is_draft_worker=False,
            kv_cache_dtype=torch.float8_e4m3fn,
            device="cuda:0",
            model_config=NS(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                hf_config=NS(architectures=["GlmMoeDsaForCausalLM"]),
            ),
            layer_info=NS(start_layer=0, end_layer=78, num_effective_layers=78),
        )

        def build(*, fixed):
            with (
                envs.SGLANG_GLM_DSA_FIXED_FABRIC_POOL.override(fixed),
                patch(
                    "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                    return_value=(None, 1),
                ),
                patch.multiple(
                    kcc,
                    get_schedule=lambda: NS(page_size=64),
                    get_memory=lambda: NS(enable_hisparse=False),
                    get_exec=lambda: NS(
                        features=NS(enable_memory_saver=False),
                        kernel=NS(enable_fp4_indexer=False),
                    ),
                    _should_elide_dsa_index_k=lambda **kw: False,
                    calculate_mla_kv_cache_dim=lambda **kw: 576,
                    get_dsa_index_head_dim=lambda config: 128,
                    max_speculative_num_draft_tokens=lambda: 0,
                ),
            ):
                return configurator._build_dsa_kv_pool(
                    max_total_num_tokens=64,
                    max_running_requests=2,
                    dsa_pool_class=Pool,
                ).kwargs

        self.assertNotIn("fixed_fabric_pool", build(fixed=False))
        self.assertIs(build(fixed=True)["fixed_fabric_pool"], True)

    def test_fixed_pool_launch_accepts_only_the_supported_mode(self):
        configurator = _configurator(
            model_config=NS(hf_config=NS(architectures=["GlmMoeDsaForCausalLM"])),
            use_mla_backend=True,
            spec_algorithm=kcc.SpeculativeAlgorithm.NONE,
            is_draft_worker=False,
            is_hybrid_swa=False,
            kv_shard_rank=None,
        )
        memory = NS(
            enable_unified_memory=False,
            enable_page_major_kv_layout=False,
            enable_hisparse=False,
        )

        def validate(
            *, attn_cp_size=1, kv_shard_rank=None, cuda_graph_persistence=None
        ):
            configurator.kv_shard_rank = kv_shard_rank
            parallel = NS(
                attn_cp_size=attn_cp_size, attn_dcp_size=1, attn_dp_size=1, pp_size=1
            )
            exec_config = NS(
                features=NS(enable_memory_saver=False),
                graph=NS(cuda_graph_persistence=cuda_graph_persistence),
            )
            with (
                patch.multiple(
                    kcc,
                    get_parallel=lambda: parallel,
                    get_memory=lambda: memory,
                    get_exec=lambda: exec_config,
                ),
                envs.SGLANG_ENABLE_POST_CAPTURE_KV_SIZING.override(False),
            ):
                configurator._validate_fixed_fabric_launch()

        # The default tree core backend (Rust) is accepted.
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
            validate()
        # CP and KV sharding swap in another pool class; Foundry would place the
        # allocation itself and replay graphs that address the pool.
        for changes, name in (
            ({"attn_cp_size": 2}, "context parallelism"),
            ({"kv_shard_rank": 0}, "KV cache sharding"),
            ({"cuda_graph_persistence": "load"}, "CUDA graph persistence"),
        ):
            with self.subTest(**changes), self.assertRaisesRegex(ValueError, name):
                validate(**changes)


if __name__ == "__main__":
    unittest.main()
