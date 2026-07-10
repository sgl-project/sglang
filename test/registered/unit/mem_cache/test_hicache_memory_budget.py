import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv_cache_builder import (
    _get_hicache_component_bytes_per_token,
    _get_hicache_local_process_count,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import (
    _discover_cgroup_memory_files,
    build_hicache_memory_plan,
    get_effective_hicache_host_layer_num,
    get_hicache_available_memory,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiCacheAvailableMemory(unittest.TestCase):
    @staticmethod
    def _write_v2_memory_values(path: Path, limit: str, current: int):
        path.mkdir(parents=True, exist_ok=True)
        (path / "memory.max").write_text(f"{limit}\n")
        (path / "memory.current").write_text(f"{current}\n")

    @staticmethod
    def _write_v1_memory_values(path: Path, limit: int, usage: int):
        path.mkdir(parents=True, exist_ok=True)
        (path / "memory.limit_in_bytes").write_text(f"{limit}\n")
        (path / "memory.usage_in_bytes").write_text(f"{usage}\n")

    def test_cgroup_v2_remaining_memory_caps_host_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            limit = root / "memory.max"
            current = root / "memory.current"
            limit.write_text("1000\n")
            current.write_text("350\n")

            self.assertEqual(
                get_hicache_available_memory(
                    host_available_bytes=10_000,
                    cgroup_memory_files=((limit, current, "cgroup v2"),),
                ),
                (650, "cgroup v2"),
            )

    def test_unlimited_cgroup_falls_back_to_host_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            limit = root / "memory.max"
            current = root / "memory.current"
            limit.write_text("max\n")
            current.write_text("350\n")

            self.assertEqual(
                get_hicache_available_memory(
                    host_available_bytes=10_000,
                    cgroup_memory_files=((limit, current, "cgroup v2"),),
                ),
                (10_000, "host"),
            )

    def test_host_memory_remains_the_tighter_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            limit = root / "memory.limit_in_bytes"
            usage = root / "memory.usage_in_bytes"
            limit.write_text("10000\n")
            usage.write_text("1000\n")

            self.assertEqual(
                get_hicache_available_memory(
                    host_available_bytes=500,
                    cgroup_memory_files=((limit, usage, "cgroup v1"),),
                ),
                (500, "host"),
            )

    def test_nested_v2_membership_is_capped_by_ancestor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mount_point = root / "cgroup2"
            unrelated_mount = root / "unrelated-cgroup2-bind"
            current = mount_point / "workload.slice" / "job.scope" / "task"
            unrelated_current = (
                unrelated_mount / "workload.slice" / "job.scope" / "task"
            )
            self._write_v2_memory_values(current, "max", 100)
            self._write_v2_memory_values(current.parent, "3000", 1000)
            self._write_v2_memory_values(current.parent.parent, "10000", 5000)
            self._write_v2_memory_values(mount_point, "max", 6000)
            self._write_v2_memory_values(unrelated_current, "100", 0)

            proc_cgroup = root / "cgroup"
            proc_mountinfo = root / "mountinfo"
            proc_cgroup.write_text("0::/workload.slice/job.scope/task\n")
            proc_mountinfo.write_text(
                f"29 23 0:26 / {mount_point} rw - cgroup2 cgroup rw\n"
                f"30 23 0:26 /other.slice {unrelated_mount} rw "
                "- cgroup2 cgroup rw\n"
            )

            memory_files = _discover_cgroup_memory_files(
                proc_cgroup_path=proc_cgroup,
                proc_mountinfo_path=proc_mountinfo,
            )
            self.assertEqual(
                [limit.parent for limit, _, _ in memory_files],
                [current, current.parent, current.parent.parent, mount_point],
            )
            self.assertEqual(
                get_hicache_available_memory(
                    host_available_bytes=20_000,
                    cgroup_memory_files=memory_files,
                ),
                (2000, "cgroup v2"),
            )

    def test_v2_namespace_relative_membership_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mount_point = root / "cgroup2"
            current = mount_point / "child"
            self._write_v2_memory_values(current, "5000", 1000)
            self._write_v2_memory_values(mount_point, "max", 2000)

            proc_cgroup = root / "cgroup"
            proc_mountinfo = root / "mountinfo"
            proc_cgroup.write_text("0::/child\n")
            proc_mountinfo.write_text(
                f"29 23 0:26 /docker/abc {mount_point} rw " "- cgroup2 cgroup rw\n"
            )

            memory_files = _discover_cgroup_memory_files(
                proc_cgroup_path=proc_cgroup,
                proc_mountinfo_path=proc_mountinfo,
            )
            self.assertEqual(
                [limit.parent for limit, _, _ in memory_files],
                [current, mount_point],
            )
            self.assertEqual(
                get_hicache_available_memory(
                    host_available_bytes=20_000,
                    cgroup_memory_files=memory_files,
                ),
                (4000, "cgroup v2"),
            )

    def test_nested_v1_membership_respects_non_root_mount(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mount_point = root / "memory"
            current = mount_point / "task"
            self._write_v1_memory_values(current, 5000, 1000)
            self._write_v1_memory_values(mount_point, 3000, 1000)

            proc_cgroup = root / "cgroup"
            proc_mountinfo = root / "mountinfo"
            proc_cgroup.write_text("5:cpu,memory:/docker/abc/task\n")
            proc_mountinfo.write_text(
                f"36 29 0:32 /docker/abc {mount_point} rw "
                "- cgroup cgroup rw,memory\n"
            )

            memory_files = _discover_cgroup_memory_files(
                proc_cgroup_path=proc_cgroup,
                proc_mountinfo_path=proc_mountinfo,
            )
            self.assertEqual(
                [limit.parent for limit, _, _ in memory_files],
                [current, mount_point],
            )
            self.assertEqual(
                get_hicache_available_memory(
                    host_available_bytes=20_000,
                    cgroup_memory_files=memory_files,
                ),
                (2000, "cgroup v1"),
            )


class TestHiCacheMemoryPlan(unittest.TestCase):
    def test_total_budget_derives_one_shared_page_count(self):
        plan = build_hicache_memory_plan(
            host_size_gb=1,
            page_size=10,
            component_bytes_per_token={
                "target KV": 3,
                "DSA indexer": 5,
                "draft KV": 2,
            },
            available_memory=(2_000_000_000, "test"),
            reserve_bytes=0,
            local_process_count=1,
        )

        self.assertEqual(plan.page_num, 10_000_000)
        self.assertEqual(
            dict(plan.component_bytes),
            {
                "target KV": 300_000_000,
                "DSA indexer": 500_000_000,
                "draft KV": 200_000_000,
            },
        )
        self.assertEqual(plan.pool_bytes, 1_000_000_000)
        self.assertGreater((plan.page_num + 1) * 10 * (3 + 5 + 2), plan.budget_bytes)

    def test_validation_reports_all_pools_reserve_and_cgroup_source(self):
        with self.assertRaisesRegex(ValueError, "target KV=0.60 GB") as context:
            build_hicache_memory_plan(
                host_size_gb=1,
                page_size=10,
                component_bytes_per_token={
                    "target KV": 6,
                    "DSA indexer": 3,
                    "draft KV": 1,
                },
                available_memory=(1_050_000_000, "cgroup v2"),
                reserve_bytes=100_000_000,
                local_process_count=1,
            )

        message = str(context.exception)
        self.assertIn("DSA indexer=0.30 GB", message)
        self.assertIn("draft KV=0.10 GB", message)
        self.assertIn("0.10 GB reserve", message)
        self.assertIn("cgroup v2", message)

    def test_validation_accounts_for_all_local_ranks(self):
        with self.assertRaisesRegex(ValueError, "4 local ranks"):
            build_hicache_memory_plan(
                host_size_gb=1,
                page_size=10,
                component_bytes_per_token={"target KV": 10},
                available_memory=(3_500_000_000, "cgroup v2"),
                reserve_bytes=0,
                local_process_count=4,
            )

    def test_ratio_page_count_still_validates_combined_pools(self):
        plan = build_hicache_memory_plan(
            page_num=100,
            page_size=10,
            component_bytes_per_token={
                "target KV": 6,
                "DSA indexer": 3,
                "draft KV": 1,
            },
            available_memory=(20_000, "test"),
            reserve_bytes=0,
            local_process_count=1,
        )

        self.assertIsNone(plan.budget_bytes)
        self.assertEqual(plan.page_num, 100)
        self.assertEqual(plan.pool_bytes, 10_000)

    def test_dsa_and_draft_component_sizes_are_combined(self):
        target = DSATokenToKVPool.__new__(DSATokenToKVPool)
        target.kv_cache_dim = 576
        target.layer_num = 80
        target.store_dtype = torch.uint8
        target.index_head_dim = 128

        draft = MHATokenToKVPool.__new__(MHATokenToKVPool)
        draft.head_dim = 128
        draft.v_head_dim = 128
        draft.head_num = 8
        draft.layer_num = 3
        draft.store_dtype = torch.bfloat16

        self.assertEqual(
            _get_hicache_component_bytes_per_token(
                target_kv_pool=target,
                draft_kv_pool=draft,
            ),
            {
                "target KV": 576 * 80,
                "DSA indexer": 132 * 80,
                "draft KV": 256 * 8 * 3 * 2,
            },
        )

    def test_uneven_layer_shards_use_host_allocation_layer_count(self):
        for layer_num, shard_size, expected in ((10, 4, 3), (2, 4, 1), (12, 4, 3)):
            with self.subTest(layer_num=layer_num, shard_size=shard_size):
                pool = SimpleNamespace(
                    layer_num=layer_num,
                    layer_shard_enabled=True,
                    layer_shard_size=shard_size,
                )
                self.assertEqual(get_effective_hicache_host_layer_num(pool), expected)

        target = DSATokenToKVPool.__new__(DSATokenToKVPool)
        target.kv_cache_dim = 576
        target.layer_num = 10
        target.layer_shard_enabled = True
        target.layer_shard_size = 4
        target.store_dtype = torch.uint8
        target.index_head_dim = 128

        draft = MHATokenToKVPool.__new__(MHATokenToKVPool)
        draft.head_dim = 128
        draft.v_head_dim = 128
        draft.head_num = 8
        draft.layer_num = 7
        draft.layer_shard_enabled = True
        draft.layer_shard_size = 3
        draft.store_dtype = torch.bfloat16

        self.assertEqual(
            _get_hicache_component_bytes_per_token(
                target_kv_pool=target,
                draft_kv_pool=draft,
            ),
            {
                "target KV": 576 * 3,
                "DSA indexer": 132 * 3,
                "draft KV": 256 * 8 * 3 * 2,
            },
        )

    def test_local_process_count_includes_non_attention_dp_groups(self):
        args = SimpleNamespace(
            tp_size=4,
            pp_size=2,
            dp_size=3,
            nnodes=2,
            enable_dp_attention=False,
        )
        self.assertEqual(_get_hicache_local_process_count(args), 12)

        args.enable_dp_attention = True
        self.assertEqual(_get_hicache_local_process_count(args), 4)


if __name__ == "__main__":
    unittest.main()
