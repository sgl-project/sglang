"""CPU contracts for the fixed GLM DSA FABRIC pool; these do not prove CUDA/RDMA."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace as NS
from typing import ClassVar
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


class FakeAllocation:
    """Stands in for _FabricAllocation; ``live`` holds unreleased bases."""

    live: ClassVar[set] = set()
    count = 0
    fail_at = None

    def __init__(self, nbytes, device_id):
        type(self).count += 1
        self.base = None
        if self.count == self.fail_at:
            raise RuntimeError("allocation failed")
        self.nbytes = nbytes
        self.base = self.count << 40
        self.live.add(self.base)

    def __del__(self):
        self.live.discard(self.base)


class FakeTensor:
    def __init__(self, allocation):
        self.allocation = allocation
        self.shape = None

    def narrow(self, dim, start, length):
        return FakeTensor(self.allocation)

    def view(self, dtype):
        return self

    def reshape(self, shape):
        self.shape = shape
        return self

    def zero_(self):
        return self


class FixedBufferOwnerTests(CustomTestCase):
    def setUp(self):
        FakeAllocation.count, FakeAllocation.fail_at = 0, None
        for patcher in (
            patch.object(ffp, "_FabricAllocation", FakeAllocation),
            patch.object(ffp.torch, "as_tensor", new=lambda x, **kw: FakeTensor(x)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _owner(self, *, dtype=torch.uint8, shapes=((128, 1, 576), (2, 8448)), **kw):
        return ffp.FixedFabricBufferOwner(
            device="cuda:0", dtype=dtype, shapes=shapes, **kw
        )

    def test_buffers_share_one_256_byte_aligned_allocation(self):
        for dtype, slices, total in (
            (torch.uint8, [(0, 3), (256, 257), (768, 2)], 770),
            (torch.float32, [(0, 12), (256, 1028), (1536, 8)], 1544),
        ):
            with self.subTest(dtype=dtype):
                owner = self._owner(dtype=dtype, shapes=[(3,), (257,), (2,)])
                self.assertEqual(owner.allocation.nbytes, total)
                self.assertEqual([tuple(s) for s in owner.buffer_slices], slices)
                self.assertEqual([t.shape for t in owner.tensors], [(3,), (257,), (2,)])
                owner.close()
        self.assertFalse(FakeAllocation.live)

    def test_int64_overflow_is_rejected_before_allocating(self):
        for shapes in (
            [(2**63,)],
            [(2**62, 4)],
            # The second buffer's 256-byte aligned offset is 2**63.
            [(2**63 - 1,), (1,)],
        ):
            with self.subTest(shapes=shapes), self.assertRaises(ValueError):
                self._owner(shapes=shapes)
        self.assertEqual(FakeAllocation.count, 0)

    def test_cpu_views_preserve_shape_stride_dtype_and_gaps(self):
        # Actual CPU torch views validate reinterpretation, not CAI/CUDA lifetime.
        root = torch.full((1284,), 0xA5, dtype=torch.uint8)
        with patch.object(ffp.torch, "as_tensor", new=lambda *a, **kw: root):
            owner = self._owner(dtype=torch.float32, shapes=[(3,), (257,)])
        self.assertEqual([t.shape for t in owner.tensors], [(3,), (257,)])
        for t in owner.tensors:
            self.assertEqual(
                (t.dtype, t.stride(), t.is_contiguous()), (torch.float32, (1,), True)
            )
            self.assertTrue(torch.all(t == 0))
        self.assertEqual(owner.tensors[1].data_ptr(), root.data_ptr() + 256)
        self.assertTrue(torch.all(root[12:256] == 0xA5))
        owner.close()

    def test_construction_failure_releases_allocations_without_fallback(self):
        def fail(*args, **kwargs):
            raise RuntimeError("injected")

        for stage, fail_at, failing in (
            ("allocation", 1, None),
            ("as_tensor", None, (ffp.torch, "as_tensor")),
            ("zero_", None, (FakeTensor, "zero_")),
        ):
            with self.subTest(stage=stage):
                FakeAllocation.count, FakeAllocation.fail_at = 0, fail_at
                with (
                    patch.object(*failing, new=fail) if failing else nullcontext(),
                    self.assertRaises(RuntimeError),
                ):
                    self._owner()
                self.assertEqual(FakeAllocation.count, 1)
                self.assertFalse(FakeAllocation.live)


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

            def close(self):
                calls.append(("close",))

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
            patch.object(
                torch.cuda, "synchronize", new=lambda d: calls.append(("sync", d))
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_explicit_fabric_and_rdma_no_fallback(self):
        allocation = ffp._FabricAllocation(3 << 20, 2)
        self.assertEqual(
            self.calls[:3],
            [
                ("prop", 2, {"handle_types": 8, "gpu_direct_rdma": True}),
                ("reserve", 4 << 20, 2, 2 << 20),
                ("map", 0, 4 << 20, True),
            ],
        )
        self.assertEqual(allocation.__cuda_array_interface__["shape"], (3 << 20,))
        del allocation
        self.assertEqual(self.calls[-2:], [("sync", 2), ("close",)])

    def test_export_failure_cleans_allocation_without_fallback(self):
        def unsupported(*args):
            raise RuntimeError("FABRIC unavailable")

        self.driver.cuMemExportToShareableHandle = unsupported
        with self.assertRaisesRegex(RuntimeError, "FABRIC unavailable"):
            ffp._FabricAllocation(8448, 0)
        self.assertEqual([call[0] for call in self.calls].count("prop"), 1)
        self.assertEqual(self.calls[-2:], [("sync", 0), ("close",)])

    def test_interpreter_exit_skips_teardown_without_quarantine(self):
        allocation = ffp._FabricAllocation(8448, 0)
        del self.calls[:]
        with patch.object(ffp.sys, "is_finalizing", return_value=True):
            del allocation
        self.assertEqual(self.calls, [])
        self.assertEqual(ffp._QUARANTINE, [])


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
        for change in (
            {"layer_num": 77},
            {"use_fp4_indexer": True},
            {"kv_scale_ue8m0": True},
            # Equal to the size, but a float would only fail in the index owner.
            {"index_buf_size": 64.0},
        ):
            with self.subTest(**change), self.assertRaises(ValueError):
                DSATokenToKVPool(**(GLM_POOL | change))
        self.assertEqual(self.owners, [])

    def test_fixed_backing_is_refused_outside_the_dsa_pool(self):
        """Direct MLA or subclass construction used to skip the GLM layout check."""
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
    def test_dsa_builder_passes_fabric_flags_only_when_opted_in(self):
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

        def build(*, fixed, pool_class=Pool):
            with (
                envs.SGLANG_GLM_DSA_FIXED_FABRIC_POOL.override(fixed),
                patch(
                    "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                    return_value=(None, 1),
                ),
                patch.multiple(
                    kcc,
                    DSATokenToKVPool=Pool,
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
                    dsa_pool_class=pool_class,
                ).kwargs

        self.assertNotIn("fixed_fabric_pool", build(fixed=False))
        self.assertIs(build(fixed=True)["fixed_fabric_pool"], True)
        with self.assertRaisesRegex(ValueError, "built-in DSA"):
            build(fixed=True, pool_class=type("Custom", (Pool,), {}))
        # KV sharding swaps in the page-interleave pool after the class is passed in.
        configurator.kv_shard_rank = 0
        with self.assertRaisesRegex(ValueError, "built-in DSA"):
            build(fixed=True)

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
        # CP and KV sharding are rejected only here; Foundry would place the
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
