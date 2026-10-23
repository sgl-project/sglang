"""CPU/Gloo regressions for CP cache identity, restore endpoints and ownership."""

import datetime
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_dsv41_cp_replay_layout import contract, definitions

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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


@pytest.mark.parametrize("backend", ["file", "mori"])
def test_dynamic_cp_storage_attach_does_not_bypass_mooncake_envelope(backend):
    attach = Mock()
    functions = definitions(
        "srt/mem_cache/hybrid_cache/hybrid_cache_controller.py",
        ["attach_storage_backend"],
        {
            "UnifiedPageEnvelopeHostPool": type("Envelope", (), {}),
            "super": lambda: NS(attach_storage_backend=attach),
        },
        klass="HybridCacheController",
        strip_imports=True,
    )
    controller = NS(
        cp_encoder_replay=True,
        storage_host_pool=object(),
        mem_pool_host=NS(),
        host_memory_mode="cache",
        pp_group=None,
        pp_prefetch_command_group=None,
        storage_backend=NS(on_host_pools_registered=Mock()),
    )
    with pytest.raises(ValueError, match="CP encoder replay L3.*Mooncake"):
        functions["attach_storage_backend"](controller, backend)
    attach.assert_not_called()


@pytest.mark.parametrize("cp_enabled,replayed", [(False, True), (True, False)])
def test_legacy_cache_does_not_match_or_reduce_for_cp_replay(cp_enabled, replayed):
    functions = definitions(
        "srt/mem_cache/unified_radix_cache.py",
        ["_replay_overlaps_unadopted_cache"],
        klass="UnifiedRadixCache",
    )
    cache = NS(
        cp_encoder_replay=cp_enabled,
        tree_core=NS(match_prefix=Mock()),
        _all_reduce=Mock(),
    )
    assert not functions["_replay_overlaps_unadopted_cache"](
        cache, NS(kv=NS(encoder_swa_replayed=replayed)), object()
    )
    cache.tree_core.match_prefix.assert_not_called()
    cache._all_reduce.assert_not_called()


@pytest.mark.parametrize(
    "local_hit,peer_overlap,expected",
    [(4, False, False), (8, False, True), (4, True, True)],
)
def test_unadopted_snapshot_overlap_is_a_group_decision(
    local_hit, peer_overlap, expected
):
    functions = definitions(
        "srt/mem_cache/unified_radix_cache.py",
        ["_replay_overlaps_unadopted_cache"],
        {"MatchPrefixParams": NS},
        klass="UnifiedRadixCache",
    )
    actions = object()

    def reduce(value, op):
        assert op == torch.distributed.ReduceOp.MAX
        if peer_overlap:
            value.fill_(1)

    cache = NS(
        cp_encoder_replay=True,
        tree_core=NS(
            match_prefix=Mock(
                return_value=NS(full_kv_hit_length=local_hit, cache_actions=actions)
            )
        ),
        _apply_cache_actions=Mock(),
        _all_reduce=reduce,
    )
    req = NS(kv=NS(encoder_swa_replayed=True, cache_protected_len=4))
    assert (
        functions["_replay_overlaps_unadopted_cache"](cache, req, object()) is expected
    )
    cache._apply_cache_actions.assert_called_once_with(actions)


@pytest.mark.parametrize("finished", [False, True])
def test_declined_cache_adoption_keeps_request_pages_lock_and_insert_cursor(finished):
    class Key:
        def __init__(self, tokens, *args, **kwargs):
            self.tokens = tokens

        def __len__(self):
            return len(self.tokens)

        def page_aligned(self, size):
            return self

    functions = definitions(
        "srt/mem_cache/unified_radix_cache.py",
        ["checkpoint_into_tree"],
        {
            "InsertParams": NS,
            "RadixKey": Key,
            "envs": NS(
                SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=NS(get=lambda: False)
            ),
        },
        klass="UnifiedRadixCache",
    )
    table = torch.arange(24).reshape(3, 8)
    before = table.clone()
    component = NS(
        prepare_for_caching_req=Mock(return_value=None),
        floor_cache_len=lambda n: n,
        cleanup_after_caching_req=Mock(),
    )
    lock = object()
    kv = NS(req_pool_idx=2, cache_protected_len=2, cache_inserted_len=2)
    req = NS(
        finished=lambda: finished,
        full_untruncated_fill_ids=list(range(8)),
        kv=kv,
        priority=None,
        session_id=None,
        kv_rotation_base=None,
        extra_key=None,
        cache_salt=None,
        lock=lock,
    )
    cache = NS(
        disable=False,
        req_to_token_pool=NS(req_to_token=table),
        _components_tuple=[component],
        tree_core=NS(is_eagle=False),
        page_size=2,
        _replay_overlaps_unadopted_cache=Mock(return_value=True),
        insert=Mock(),
    )
    functions["checkpoint_into_tree"](cache, req, up_to=8)
    cache.insert.assert_not_called()
    component.cleanup_after_caching_req.assert_called_once()
    assert (
        component.cleanup_after_caching_req.call_args.kwargs["is_finished"] is finished
    )
    assert torch.equal(table, before)
    assert req.lock is lock
    assert kv.cache_inserted_len == kv.cache_protected_len == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
