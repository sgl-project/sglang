"""V4 shared construction must fail together before the next pool collective."""

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

from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.mla_host_shared import SharedCPUHostDedupContext
from sglang.srt.mem_cache.pool_host.base import host_memory_budget_scope
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _ordinary(value):
    if value is None or isinstance(value, (str, int, bool)):
        return True
    return isinstance(value, (tuple, list)) and all(_ordinary(x) for x in value)


def _rank(rank, rendezvous, output, scenario):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    pools, created, destroyed = [], [], []
    real_pool = assembler.DeepSeekV4PagedHostPool
    real_regions = assembler._dsv4_indexer_regions
    real_gather = dist.all_gather_object
    signatures = []

    def tracked_pool(**kwargs):
        # Real CPU shared mappings and allocator collectives. CUDA pinning is
        # intentionally disabled in this CPU construction/lifetime test.
        kwargs["pin_memory"] = False
        with host_memory_budget_scope(2**20):
            pool = real_pool(**kwargs)
        pools.append(pool)
        created.append(pool.pool_name)
        original_destroy = pool.destroy

        def destroy():
            if not getattr(pool, "_destroyed", False):
                destroyed.append(pool.pool_name)
            original_destroy()

        pool.destroy = destroy
        return pool

    def index_regions(cache, page_size):
        if scenario == "metadata" and rank == 1:
            raise RuntimeError("injected next-pool metadata failure after C4 entry")
        regions = real_regions(cache, page_size)
        if scenario == "plan" and rank == 1:
            regions[0] = assembler._IndexerRegion(
                name=assembler.PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
                device_buffers=regions[0].device_buffers,
                item_bytes=regions[0].item_bytes,
                slot_page_size=regions[0].slot_page_size,
                page_aligned_only=regions[0].page_aligned_only,
            )
        return regions

    def controller(*args, **kwargs):
        group = args[1]
        # HostPoolGroup is real: all placeholders must already be replaced.
        assert all(
            not isinstance(entry.host_pool, getattr(assembler, "_V4HostPoolPlan", ()))
            for entry in group.entries
        )
        if scenario == "controller" and rank == 1:
            raise RuntimeError("injected rank-local controller initialization failure")
        return SimpleNamespace(host_pool=group)

    def gather(result, value, *args, **kwargs):
        # The assembler plan must not serialize device addresses/tensors.
        if isinstance(value, list) and value and isinstance(value[0], tuple):
            assert _ordinary(value), value
            signatures.append(value)
        return real_gather(result, value, *args, **kwargs)

    memory = SimpleNamespace(
        enable_mla_hicache_host_dedup=True,
        hicache_ratio=1,
        hicache_size=0,
        hicache_mem_layout="page_first_direct",
        hicache_write_policy="write_through",
        hicache_io_backend="direct",
        hicache_host_memory_mode="default",
    )
    buffers = [torch.zeros((8, 16), dtype=torch.uint8) for _ in range(2)]
    cache = SimpleNamespace(
        _unified_kv=True,
        size=16,
        swa_size=16,
        swa_page_size=2,
        swa_kv_pool=None,
        c4_kv_pool=SimpleNamespace(),
        c4_indexer_kv_pool=SimpleNamespace(index_k_with_scale_buffer=buffers),
        sources_by_ratio={},
        unified_region_buffers=lambda ratio: (buffers, 16),
        unified_rope_region_buffers=lambda ratio: None,
    )
    allocator = SimpleNamespace(size_full=16)
    params = SimpleNamespace(
        page_size=2,
        mtp_draft_device_pools=(),
        token_to_kv_pool_allocator=allocator,
        tp_cache_group=dist.group.WORLD,
        attn_cp_cache_group=None,
        attn_tp_cache_group=None,
        pp_cache_group=None,
    )
    mappings = assembler._DeepSeekV4LayerMappings(
        2,
        {0: 0, 1: 1},
        {},
        {0: 0, 1: 1},
        {},
        {},
        [],
    )
    context = SharedCPUHostDedupContext(dist.group.WORLD)
    try:
        with (
            mock.patch.object(assembler, "get_memory", return_value=memory),
            mock.patch.object(assembler, "_get_allocator_type", return_value="default"),
            mock.patch.object(assembler, "DeepSeekV4PagedHostPool", new=tracked_pool),
            mock.patch.object(assembler, "_dsv4_indexer_regions", new=index_regions),
            mock.patch.object(assembler, "HybridCacheController", new=controller),
        ):
            if scenario == "legacy":
                for enabled in (False, True):
                    memory.enable_mla_hicache_host_dedup = enabled
                    legacy = (
                        SimpleNamespace(is_dummy_rank=bool(rank)) if enabled else None
                    )
                    start = len(created)
                    with (
                        mock.patch.object(
                            assembler, "_create_mla_dedup_context", return_value=legacy
                        ),
                        mock.patch.object(
                            dist,
                            "all_reduce",
                            side_effect=AssertionError("unexpected shared all_reduce"),
                        ),
                        mock.patch.object(
                            dist,
                            "all_gather_object",
                            side_effect=AssertionError("unexpected shared plan gather"),
                        ),
                    ):
                        group, _ = assembler.build_deepseek_v4_hicache_stack(
                            params=params,
                            kvcache=cache,
                            load_cache_event=None,
                            storage_backend=None,
                            layer_mappings=mappings,
                        )
                    assert len(created) - start == 2
                    for entry in group.entries:
                        entry.host_pool.destroy()
            else:
                expected = {
                    "metadata": "rank 1: RuntimeError: injected next-pool metadata failure",
                    "controller": "rank 1: RuntimeError: injected rank-local controller initialization failure",
                    "plan": "host pool plan differs",
                }[scenario]
                with (
                    mock.patch.object(
                        assembler, "_create_mla_dedup_context", return_value=context
                    ),
                    mock.patch.object(dist, "all_gather_object", side_effect=gather),
                ):
                    try:
                        assembler.build_deepseek_v4_hicache_stack(
                            params=params,
                            kvcache=cache,
                            load_cache_event=None,
                            storage_backend=None,
                            layer_mappings=mappings,
                        )
                    except (RuntimeError, ValueError) as error:
                        assert expected in str(error), str(error)
                    else:
                        raise AssertionError(
                            "expected all ranks to reject construction"
                        )
                if scenario == "controller":
                    assert len(created) == 2
                    assert destroyed == list(reversed(created)), (created, destroyed)
                    assert signatures
                else:
                    assert not created, created
                assert all(pool._destroyed and pool.kv_buffer is None for pool in pools)
                assert all(
                    pool.allocator._closed and not pool.allocator._mappings
                    for pool in pools
                )
        Path(output, f"{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "scenario": scenario,
                    "created": created,
                    "destroyed": destroyed,
                    "plan_payloads_are_ordinary": True,
                    "passed": True,
                }
            )
        )
    finally:
        for pool in reversed(pools):
            pool.destroy()
        dist.destroy_process_group()


class TestSharedV4Construction(CustomTestCase):
    def _run(self, scenario):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processes = mp.spawn(
                _rank,
                args=(str(root / "gloo"), directory, scenario),
                nprocs=2,
                join=False,
            )
            try:
                deadline = time.monotonic() + 90
                while not processes.join(timeout=1):
                    if time.monotonic() >= deadline:
                        self.fail(f"Shared V4 {scenario} exceeded 90 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
            for rank in range(2):
                self.assertTrue(
                    json.loads((root / f"{rank}.json").read_text())["passed"]
                )

    def test_metadata_failure_agreed_before_pool_allocation(self):
        self._run("metadata")

    def test_controller_failure_releases_all_pools_in_reverse(self):
        self._run("controller")

    def test_pool_plan_mismatch_rejected_before_allocation(self):
        self._run("plan")

    def test_disabled_and_broadcast_keep_immediate_construction(self):
        self._run("legacy")


if __name__ == "__main__":
    unittest.main()
