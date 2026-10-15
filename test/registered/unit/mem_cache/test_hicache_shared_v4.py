"""Real shared V4 host regions, joint failure cleanup, and hybrid budgets."""

import json
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache import memory_pool_host as host_pools
from sglang.srt.mem_cache.mla_host_shared import SharedCPUHostDedupContext
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


def _make_pool(kind, name, group, *, layout="page_first_direct"):
    common = dict(
        num_host_pages=8,
        layout=layout,
        pin_memory=False,
        shared_memory_group=group,
    )
    if kind == "paged":
        return host_pools.DeepSeekV4PagedHostPool(
            name,
            [torch.empty((8, 16), dtype=torch.uint8) for _ in range(2)],
            item_bytes=16,
            slot_page_size=2,
            **common,
        )
    return host_pools.DeepSeekV4StateHostPool(
        name,
        [
            SimpleNamespace(
                ring_size=2,
                kv_score_buffer=SimpleNamespace(
                    kv_score=torch.empty((16, 3), dtype=torch.float32)
                ),
            )
            for _ in range(2)
        ],
        swa_page_size=2,
        **common,
    )


def _expect_failure(function, text):
    try:
        function()
    except (ValueError, RuntimeError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f"Expected a joint failure containing {text!r}")


def _shared_v4_rank(rank, rendezvous, output, scenario):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    pools, allocators = [], []
    real_factory = host_pools.get_allocator_from_storage

    def factory(*args, **kwargs):
        allocator = real_factory(*args, **kwargs)
        allocators.append(allocator)
        return allocator

    try:
        with mock.patch.object(
            host_pools, "get_allocator_from_storage", side_effect=factory
        ):
            if scenario == "mapping":
                for kind, name in (
                    ("paged", "c2_kv"),
                    ("paged", "c4_kv"),
                    ("paged", "indexer_scale"),
                    ("state", "compress_state"),
                    ("state", "indexer_state"),
                ):
                    with host_memory_budget_scope(2**20):
                        pool = _make_pool(kind, name, dist.group.WORLD)
                    pools.append(pool)
                    assert pool.allocator.is_owner == (rank == 0)
                    if kind == "paged":
                        assert pool.available_size() == 16
                    # Every layer/region has distinct bytes in the one shared copy.
                    expected = torch.arange(pool.kv_buffer.numel(), dtype=torch.int64)
                    expected = (
                        (expected + len(pools))
                        .to(torch.uint8)
                        .reshape_as(pool.kv_buffer)
                    )
                    if rank == 0:
                        pool.kv_buffer.copy_(expected)
                    dist.barrier()
                    actual = pool.kv_buffer.clone()
                    assert torch.equal(actual, expected), {
                        "pool": name,
                        "rank": rank,
                        "mismatches": (actual != expected).nonzero().tolist(),
                    }
                    # Both readers must finish comparing before rank 1 starts
                    # the next mutation; the preceding barrier only published
                    # rank 0's initial write.
                    dist.barrier()
                    if rank == 1:
                        pool.kv_buffer[-1, -1, 0, -1] = 123
                    dist.barrier()
                    assert pool.kv_buffer[-1, -1, 0, -1].item() == 123

                context = SharedCPUHostDedupContext(dist.group.WORLD)

                def transfer(pool, rows, *, private=False):
                    if private:
                        pool = SimpleNamespace(
                            allocator=SimpleNamespace(is_shared=False)
                        )
                    return SimpleNamespace(
                        host_pool=pool,
                        host_indices=torch.tensor(rows, dtype=torch.int64),
                        # Device rows are rank-local and must not enter the digest.
                        device_indices=torch.tensor([10 + rank, 20 + rank]),
                    )

                ordered = [transfer(pools[0], [4, 5]), transfer(pools[1], [8, 9])]
                context.validate_host_rows(
                    ordered + [transfer(None, [rank * 3], private=True)]
                )
                context.validate_host_rows([transfer(None, [rank], private=True)])
                _expect_failure(
                    lambda: context.validate_host_rows(ordered[:: -1 if rank else 1]),
                    "row order diverged",
                )
                same_rows = [transfer(pools[0], [4, 5]), transfer(pools[1], [4, 5])]
                _expect_failure(
                    lambda: context.validate_host_rows(same_rows[:: -1 if rank else 1]),
                    "row order diverged",
                )
                _expect_failure(
                    lambda: context.validate_host_rows(
                        [transfer(pools[0], [5, 4] if rank else [4, 5])]
                    ),
                    "row order diverged",
                )
                _expect_failure(
                    lambda: context.validate_host_rows(
                        ordered, failure="private preparation failed" if rank else None
                    ),
                    "rank 1: private preparation failed",
                )
            else:
                for kind in ("paged", "state"):
                    pool_type = (
                        host_pools.DeepSeekV4PagedHostPool
                        if kind == "paged"
                        else host_pools.DeepSeekV4StateHostPool
                    )
                    real_metadata = pool_type._init_write_back_staging_buffers

                    def metadata(self):
                        real_metadata(self)
                        if scenario == "metadata" and rank == 1:
                            raise RuntimeError("injected peer pointer metadata failure")

                    expected = {
                        "metadata": "injected peer pointer metadata failure",
                        "admission": "Not enough host memory",
                        "layout": "non-dummy page_first_direct",
                    }[scenario]
                    with (
                        host_memory_budget_scope(
                            32 if scenario == "admission" and rank else 2**20
                        ),
                        mock.patch.object(
                            pool_type,
                            "_init_write_back_staging_buffers",
                            new=metadata,
                        ),
                    ):
                        _expect_failure(
                            lambda: _make_pool(
                                kind,
                                f"{kind}_{scenario}",
                                dist.group.WORLD,
                                layout="layer_first"
                                if scenario == "layout" and rank
                                else "page_first_direct",
                            ),
                            expected,
                        )
                assert len(allocators) == 2
                assert all(value._closed for value in allocators)
                assert all(not value._mappings for value in allocators)

        Path(output, f"{rank}.json").write_text(
            json.dumps({"rank": rank, "scenario": scenario, "passed": True})
        )
    finally:
        for pool in reversed(pools):
            pool.destroy()
            pool.destroy()
            assert pool.allocator._closed and not pool.allocator._mappings
        dist.destroy_process_group()


class TestSharedV4HostRegions(CustomTestCase):
    def _run(self, scenario):
        with tempfile.TemporaryDirectory(prefix="shared-v4-") as directory:
            context = mp.spawn(
                _shared_v4_rank,
                args=(str(Path(directory, "rdzv")), directory, scenario),
                nprocs=2,
                join=False,
            )
            try:
                deadline = time.monotonic() + 70
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail(f"Shared V4 {scenario} exceeded 70 seconds")
                for rank in range(2):
                    result = json.loads(Path(directory, f"{rank}.json").read_text())
                    self.assertTrue(result["passed"])
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)

    def test_shared_regions_and_ordered_pool_identity(self):
        self._run("mapping")

    def test_peer_metadata_failure_releases_all_mappings(self):
        self._run("metadata")

    def test_peer_budget_failure_reaches_every_rank(self):
        self._run("admission")

    def test_peer_layout_rejection_reaches_every_rank(self):
        self._run("layout")


class TestSharedHybridAdmission(CustomTestCase):
    def test_asymmetric_private_state_is_charged_per_rank(self):
        # One 100-byte shared region, plus private state of 20 and 40 bytes.
        # Rank 1 maps 140 bytes but physically owns only its 40-byte state.
        for allowances in ((2000, 2000), (2000, 1600)):
            records = [[0, allowances[0], 120], [0, allowances[1], 40]]
            host_budget = min(allowances)
            budgets = []
            for rank, mapped in enumerate((120, 140)):

                def gather(outputs, local, *, group):
                    self.assertEqual(local.tolist(), records[rank])
                    for output, record in zip(outputs, records):
                        output.copy_(torch.tensor(record))

                with (
                    self.subTest(allowances=allowances, rank=rank),
                    mock.patch.object(
                        sizing,
                        "get_parallel",
                        return_value=SimpleNamespace(
                            node_rank=0,
                            world_group=SimpleNamespace(cpu_group="world"),
                        ),
                    ),
                    mock.patch.object(sizing, "ranks_per_host", return_value=2),
                    mock.patch.object(dist, "is_initialized", return_value=True),
                    mock.patch.object(dist, "get_world_size", return_value=2),
                    mock.patch.object(dist, "all_gather", side_effect=gather),
                ):
                    ratio, budget = sizing._dedup_host_budget(
                        records[rank][2], allowances[rank] // 2, mapped_bytes=mapped
                    )
                self.assertEqual(ratio, host_budget / 160)
                self.assertEqual(budget, host_budget * mapped // 160)
                budgets.append(budget)
            # Mapped admission is larger than physical use: shared bytes occur
            # in both mappings but only once in the node's actual allocation.
            self.assertGreater(sum(budgets), host_budget)


if __name__ == "__main__":
    unittest.main()
