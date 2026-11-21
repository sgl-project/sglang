"""CPU checks for EGM host-pool configuration and allocation rollback."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host import base, common, egm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEgmConfiguration(CustomTestCase):
    def test_remote_peers_cover_each_owner_once(self):
        for localities in (
            [("host-a", 0), ("host-a", 1)],
            [("host-a", 0), ("host-a", 0), ("host-b", 0), ("host-b", 0)],
            [("host-a", 0), ("host-b", 0), ("host-c", 0)],
            [("host-b", 1), ("host-a", 0), ("host-b", 0), ("host-a", 1)],
        ):
            with self.subTest(localities=localities):
                peers = egm._remote_peers(localities)
                self.assertEqual(sorted(peers), list(range(len(localities))))
                for rank, peer in enumerate(peers):
                    self.assertNotEqual(localities[rank], localities[peer])
                if len({host for host, _ in localities}) == 2 and all(
                    sum(host == h for h, _ in localities) == len(localities) // 2
                    for host, _ in localities
                ):
                    for rank, peer in enumerate(peers):
                        self.assertNotEqual(localities[rank][0], localities[peer][0])

        for localities in (
            [("host-a", 0)],
            [("host-a", 0), ("host-a", 0)],
            [("host-a", 0), ("host-a", 0), ("host-a", 0), ("host-b", 0)],
        ):
            with self.subTest(localities=localities), self.assertRaises(ValueError):
                egm._remote_peers(localities)

    def test_locality_uses_kernel_identity_across_containers(self):
        group = SimpleNamespace(world_size=2, rank_in_group=0, cpu_group=object())
        with (
            patch(
                "sglang.srt.distributed.parallel_state.get_tp_group", return_value=group
            ),
            patch("torch.cuda.current_device", return_value=0),
            patch.object(egm, "_allocation_properties", return_value=(None, 0, 1024)),
            patch.object(egm.Path, "read_text", return_value="same-kernel\n"),
            patch.object(
                egm.EgmRemoteTensorAllocator,
                "_exchange",
                side_effect=lambda local: [local, (("same-kernel", 0), None)],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "distinct hosts or host NUMA"):
                egm.EgmRemoteTensorAllocator()
            for peer in (("same-kernel", 1), ("other-kernel", 0)):
                with (
                    self.subTest(peer=peer),
                    patch.object(
                        egm.EgmRemoteTensorAllocator,
                        "_exchange",
                        side_effect=lambda local: [local, (peer, None)],
                    ),
                ):
                    self.assertIsInstance(
                        egm.EgmRemoteTensorAllocator(), egm.EgmRemoteTensorAllocator
                    )

    def test_allocator_selection_and_storage_conflict(self):
        with (
            envs.SGLANG_HICACHE_EGM.override(True),
            patch.object(common, "is_cuda", return_value=True),
            patch.object(
                common,
                "get_memory",
                return_value=SimpleNamespace(
                    hicache_storage_backend=None,
                    hicache_mem_layout="layer_first",
                    hicache_io_backend="kernel",
                ),
            ),
        ):
            self.assertEqual(common.get_allocator_type(), "egm_remote")
            self.assertIs(
                common.get_allocator_class_from_storage("egm_remote"),
                egm.EgmRemoteTensorAllocator,
            )
            with patch.object(
                common,
                "get_memory",
                return_value=SimpleNamespace(hicache_storage_backend="mooncake"),
            ):
                with self.assertRaisesRegex(ValueError, "storage backends"):
                    common.get_allocator_type()
            for layout, io_backend in (
                ("page_first", "kernel"),
                ("layer_first", "direct"),
            ):
                with (
                    self.subTest(layout=layout, io_backend=io_backend),
                    patch.object(
                        common,
                        "get_memory",
                        return_value=SimpleNamespace(
                            hicache_storage_backend=None,
                            hicache_mem_layout=layout,
                            hicache_io_backend=io_backend,
                        ),
                    ),
                ):
                    with self.assertRaisesRegex(
                        ValueError, "layer_first layout and kernel"
                    ):
                        common.get_allocator_type()

        with (
            envs.SGLANG_HICACHE_EGM.override(False),
            patch.object(
                common,
                "get_memory",
                return_value=SimpleNamespace(hicache_storage_backend=None),
            ),
        ):
            self.assertEqual(common.get_allocator_type(), "default")
        with (
            envs.SGLANG_HICACHE_EGM.override(True),
            patch.object(common, "is_cuda", return_value=False),
            patch.object(
                common,
                "get_memory",
                return_value=SimpleNamespace(hicache_storage_backend=None),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                common.get_allocator_type()

    def test_vmm_granularity_is_booked_per_mapping(self):
        with (
            patch.object(egm, "_allocation_properties", return_value=(None, 0, 1024)),
            patch("torch.cuda.current_device", return_value=0),
            base.host_memory_budget_scope(4096),
        ):
            first = base.host_memory_preflight(
                [1, 1025], egm.EgmRemoteTensorAllocator, "cuda"
            )
            second = base.host_memory_preflight(
                [1025], egm.EgmRemoteTensorAllocator, "cuda"
            )
        self.assertEqual(first, (3072, 4096))
        self.assertEqual(second, (2048, 1024))

    def test_preflight_ignores_dynamic_allocator_attributes(self):
        allocator = Mock(supports_hugetlb=Mock(return_value=False))
        with base.host_memory_budget_scope(4096):
            result = base.host_memory_preflight([1], allocator, "cuda")
        self.assertEqual(result, (1, 4096))

    def test_preflight_rejects_mismatched_cuda_device(self):
        with patch("torch.cuda.current_device", return_value=0):
            with self.assertRaisesRegex(ValueError, "differs from current CUDA device"):
                base.host_memory_preflight([1], egm.EgmRemoteTensorAllocator, "cuda:1")

    def test_failed_vmm_allocation_releases_acquired_resources(self):
        class Driver(types.ModuleType):
            CUresult = SimpleNamespace(CUDA_SUCCESS=0)
            CUmemLocationType = SimpleNamespace(
                CU_MEM_LOCATION_TYPE_HOST_NUMA=1, CU_MEM_LOCATION_TYPE_DEVICE=2
            )
            CUmemAccess_flags = SimpleNamespace(CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3)

            def __init__(self, fail_at):
                super().__init__("cuda.bindings.driver")
                self.fail_at = fail_at
                self.handle = self.address = self.mapped = False

            def CUmemAccessDesc(self):
                return SimpleNamespace(location=SimpleNamespace(), flags=None)

            def cuMemCreate(self, *args):
                self.handle = True
                return (0, 7)

            def cuMemAddressReserve(self, *args):
                if self.fail_at == "reserve":
                    return (1,)
                self.address = True
                return (0, 4096)

            def cuMemMap(self, *args):
                if self.fail_at == "map":
                    return (1,)
                self.mapped = True
                return (0,)

            def cuMemSetAccess(self, *args):
                return (1,) if self.fail_at == "access" else (0,)

            def cuMemUnmap(self, *args):
                self.mapped = False
                return (0,)

            def cuMemAddressFree(self, *args):
                self.address = False
                return (0,)

            def cuMemRelease(self, *args):
                self.handle = False
                return (0,)

        for fail_at in ("reserve", "map", "access"):
            with self.subTest(fail_at=fail_at):
                driver = Driver(fail_at)
                package = types.ModuleType("cuda")
                bindings = types.ModuleType("cuda.bindings")
                bindings.driver = driver
                package.bindings = bindings
                with (
                    patch.dict(
                        sys.modules, {"cuda": package, "cuda.bindings": bindings}
                    ),
                    patch.object(
                        egm, "_allocation_properties", return_value=(None, 0, 1024)
                    ),
                ):
                    with self.assertRaises(RuntimeError):
                        egm.EgmAllocation(100, 0)
                self.assertFalse(driver.handle)
                self.assertFalse(driver.address)
                self.assertFalse(driver.mapped)


if __name__ == "__main__":
    unittest.main()
