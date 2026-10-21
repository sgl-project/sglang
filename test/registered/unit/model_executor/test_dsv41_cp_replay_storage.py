"""CPU/Gloo tests of CP cache namespaces and sparse restore endpoints."""

import datetime
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_dsv41_cp_replay_layout import ROOT, definitions


def contract():
    spec = importlib.util.spec_from_file_location(
        "storage_contract",
        ROOT / "python/sglang/srt/mem_cache/dsv41_replay_contract.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pool(**changes):
    values = dict(
        kv_layout=NS(value="v4"),
        compressed_kv_layout=lambda ratio: NS(value="v4"),
        page_size=256,
        compression_ratios=[0, 1, 2, 4, 128],
        kv_source_layers=[0, 1, 2],
        num_slots=8,
    )
    return NS(**(values | changes))


def test_cp_storage_identity_separates_topology_and_recipe():
    module = contract()
    producer = dict(
        cp=8, attention_tp=1, strategy="interleave", gemm="triton", dtype="bf16"
    )
    first = module.replay_storage_tag(pool(), producer=producer)
    assert first.startswith("dsv41-cp-replay-v1-")
    assert first == module.replay_storage_tag(pool(num_slots=128), producer=producer)
    for changes in (
        {"cp": 1, "attention_tp": 8},
        {"gemm": "deep_gemm"},
        {"dtype": "fp16"},
    ):
        assert first != module.replay_storage_tag(pool(), producer=producer | changes)
    assert first != module.replay_storage_tag(pool(page_size=128), producer=producer)


def test_storage_tag_preserves_user_namespace_without_mutating_config():
    config = {"extra_backend_tag": "user", "master_server_address": "store:50051"}
    result = contract().with_storage_layout_tag(config, "cp-replay")
    assert result["extra_backend_tag"] == "user_cp-replay"
    assert config["extra_backend_tag"] == "user"


def test_cp1_prefetch_sync_retains_legacy_scalar_collective():
    calls = []

    def reduce(value, op, groups):
        calls.append((value.shape, groups))

    functions = definitions(
        "srt/managers/cache_controller.py",
        ["_sync_prefetch_hit_query"],
        klass="HiCacheController",
        strip_imports=True,
    )
    controller = NS(
        cp_encoder_replay=False,
        _all_reduce=reduce,
        prefetch_hits_sync_groups="prefetch-only",
    )
    assert functions["_sync_prefetch_hit_query"](controller, object(), 512) == 512
    assert calls == [(torch.Size([]), "prefetch-only")]


@pytest.mark.parametrize("query_failure", [False, True])
def test_scheduler_probe_uses_scheduler_groups_even_when_query_fails(query_failure):
    class Key:
        def __init__(self, tokens, **kwargs):
            self.tokens = tokens

        def __len__(self):
            return len(self.tokens)

        def page_aligned(self, page_size):
            return self

    operation = NS(restorable_prefix_pages=[1, 3], query_pool_hit_pages={})
    reductions = []

    def reduce(value, op):
        reductions.append(value.numel())

    query = (
        Mock(side_effect=RuntimeError("query failed"))
        if query_failure
        else Mock(return_value=(["a", "b", "c"], 6))
    )
    cache = NS(
        enable_storage=True,
        cache_controller=NS(
            prefetch_rate_limited=lambda: False, _storage_hit_query=query
        ),
        cp_encoder_replay=True,
        tree_core=NS(is_eagle=False),
        page_size=2,
        prefetch_threshold=2,
        _all_reduce_attn_groups=reduce,
    )
    functions = definitions(
        "srt/mem_cache/unified_radix_cache.py",
        ["query_storage_hit_length"],
        {
            "sync_restorable_prefix": contract().sync_restorable_prefix,
            "PoolName": ["kv"],
            "RadixKey": Key,
            "PrefetchOperation": lambda *args: operation,
            "CacheRequestHandle": lambda *args: None,
            "logger": Mock(),
        },
        klass="UnifiedRadixCache",
        strip_imports=True,
    )
    result = functions["query_storage_hit_length"](cache, 0, list(range(6)))
    assert result == (0 if query_failure else 6)
    assert reductions == ([3] if query_failure else [3, 4])


def _endpoint_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        module = contract()
        background = dist.new_group([0, 1], backend="gloo")
        cases = [
            ([1, 3] if rank == 0 else [1, 2], 6 if rank == 0 else 4, 2),
            ([2] if rank == 0 else [1], 4 if rank == 0 else 2, 0),
            (None if rank == 0 else [1, 3], 8 if rank == 0 else 6, 6),
            ([] if rank == 0 else None, 8, 0),
            ([1, 3] if rank == 0 else None, 6 if rank == 0 else 0, 0),
            (None, 2 * (rank + 1), 2),
        ]
        for group in (dist.group.WORLD, background):
            for candidates, count, expected in cases:
                operation = NS(
                    restorable_prefix_pages=candidates,
                    query_pool_hit_pages={"kv": count // 2}
                    | ({"state": 3} if rank == 0 else {}),
                )
                sizes = []

                def reduce_min(tensor):
                    sizes.append(tensor.numel())
                    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)

                actual = module.sync_restorable_prefix(
                    operation,
                    count,
                    page_size=2,
                    pools=["kv", "state"],
                    reduce_min=reduce_min,
                )
                assert actual == expected, (rank, candidates, actual, expected)
                assert "state" not in operation.query_pool_hit_pages
                if actual and candidates is not None:
                    assert actual // 2 in candidates
                peer_sizes = [None, None]
                dist.all_gather_object(peer_sizes, sizes, group=group)
                assert peer_sizes[0] == peer_sizes[1]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU Gloo is unavailable")
def test_sparse_endpoint_intersection_and_failure_collective_order():
    with tempfile.TemporaryDirectory(prefix="v41-cp-replay-gloo-") as directory:
        mp.spawn(
            _endpoint_worker,
            args=(str(Path(directory) / "rendezvous"),),
            nprocs=2,
            join=True,
        )
