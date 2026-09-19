"""Byte-level split-load tests, with CPU tensors and concurrent fake ranks."""

import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.umbp import umbp_direct_linker as module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Mode(Enum):
    Local = 0
    StandaloneProcess = 1


class _Storage:
    """Exercise the real range descriptors, including object offsets."""

    def __init__(self):
        self.client = self
        self._disable_zero_copy_register = False
        self.objects = {}
        self.reads = []
        self.fail_read = False

    def get_deployment_mode(self):
        return _Mode.StandaloneProcess

    def get_backend_mode(self):
        return _Mode.Local

    def supports_ranged_io(self):
        return True

    def register_memory(self, *args):
        return True

    def deregister_memory(self, *args):
        pass

    def close(self):
        pass

    def _get_hybrid_page_component_keys(self, keys, transfer):
        entry = self.registered_pools[transfer.name]
        count = 1 if entry.packed else len(entry.components)
        return [
            f"{key}:{transfer.name}:{component}"
            for key in keys
            for component in range(count)
        ], count

    def batch_exists(self, keys):
        return [key in self.objects for key in keys]

    def batch_put_ranges_from_ptr(self, keys, lengths, pointers, sizes, offsets):
        for key, length, ptrs, counts, starts in zip(
            keys, lengths, pointers, sizes, offsets
        ):
            data = bytearray(length)
            for ptr, count, start in zip(ptrs, counts, starts):
                data[start : start + count] = ctypes.string_at(ptr, count)
            self.objects[key] = bytes(data)
        return [True] * len(keys)

    def batch_get_ranges_into_ptr(self, keys, pointers, sizes, offsets):
        if self.fail_read:
            return [False] * len(keys)
        for key, ptrs, counts, starts in zip(keys, pointers, sizes, offsets):
            for ptr, count, start in zip(ptrs, counts, starts):
                ctypes.memmove(ptr, self.objects[key][start : start + count], count)
                self.reads.append((key, start, count))
        return [True] * len(keys)


class _Collectives:
    def __init__(self, world):
        self.world = world
        self.rank = threading.local()
        self.barrier = threading.Barrier(world, timeout=10)
        self.values = {}
        self.calls = []

    def _collect(self, kind, tensor, output=None, group=None):
        self.values[self.rank.value] = (
            kind,
            tensor.clone(),
            output if output is not None else tensor,
            group,
        )
        leader = self.barrier.wait() == 0
        if leader:
            values = [self.values[rank] for rank in range(self.world)]
            assert len({(value[0], value[3]) for value in values}) == 1
            data = [value[1] for value in values]
            result = torch.stack(data).amin(0) if kind == "reduce" else torch.cat(data)
            for _, _, destination, _ in values:
                destination.copy_(result)
            self.calls.append((kind, group, tensor.numel()))
        self.barrier.wait()

    def all_reduce(self, tensor, op=None, group=None):
        self._collect("reduce", tensor, group=group)

    def all_gather_into_tensor(self, output, tensor, group=None):
        # clone() above makes this also check the in-place send/receive form.
        self._collect("gather", tensor, output, group)

    def run(self, linkers, function):
        def worker(rank):
            self.rank.value = rank
            return function(linkers[rank])

        with ThreadPoolExecutor(max_workers=self.world) as executor:
            return list(executor.map(worker, range(self.world)))


@pytest.fixture
def factory(monkeypatch):
    created = []
    monkeypatch.setattr(module, "freeze_gc", Mock())
    monkeypatch.setattr(
        module.device_module,
        "Event",
        lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None),
    )
    monkeypatch.setattr(
        module,
        "get_memory",
        lambda: SimpleNamespace(hicache_storage_backend_extra_config=None),
    )
    monkeypatch.setattr(
        module, "get_model", lambda: SimpleNamespace(model_path="split-test")
    )
    monkeypatch.setattr(
        module, "get_disagg", lambda: SimpleNamespace(disaggregation_mode="null")
    )
    monkeypatch.setattr(module, "_parse_storage_extra_config", lambda value: {})
    monkeypatch.setenv("UMBP_LOAD_SPLIT_MIN_PAGES", "1")
    monkeypatch.setenv("UMBP_LOAD_SPLIT_STAGING_MIB", "1")
    monkeypatch.setenv("UMBP_LAYER_GROUP", "2")

    def make(
        world=3, sparse=False, enabled=True, replicated=True, invalid_geometry=False
    ):
        collectives = _Collectives(world)
        monkeypatch.setattr(torch.distributed, "all_reduce", collectives.all_reduce)
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_into_tensor",
            collectives.all_gather_into_tensor,
        )
        monkeypatch.setattr(
            module,
            "get_parallel",
            lambda: SimpleNamespace(
                tp_size=world, pp_size=1, attn_cp_size=1, enable_dp_attention=False
            ),
        )
        monkeypatch.setenv("UMBP_LOAD_SPLIT", str(int(enabled)))
        linkers = []
        for rank in range(world):
            entries = []
            for name, source, span, components in (
                (PoolName.KV, PoolName.KV, 2, 2),
                (PoolName.INDEXER, PoolName.KV, 1, 1),
                (PoolName.SWA, PoolName.SWA, 1, 1),
            ):
                mapping = (
                    {2: 0, 0: 1}
                    if sparse and name == PoolName.INDEXER
                    else {i: i for i in range(3)}
                )
                buffers = [
                    [
                        torch.full((32 * span, 3 + component), 231, dtype=torch.uint8)
                        for _ in mapping
                    ]
                    for component in range(components)
                ]
                entries.append(
                    DevicePoolEntry(
                        name=name,
                        indices_from_pool=source,
                        device_pool=None,
                        components=buffers,
                        layer_mapping=mapping,
                        page_size=2,
                        rows_are_pages=span == 1,
                        packed=not sparse,
                    )
                )
            if invalid_geometry:
                pointer, stride, size = entries[0].buffer_meta[0][0]
                entries[0].buffer_meta[0][0] = pointer, stride, size + 1
            pools = DevicePoolGroup(entries, 3, 2, rank_replicated=replicated)
            params = SimpleNamespace(
                page_size=2,
                token_to_kv_pool_allocator=Mock(),
                req_to_token_pool=Mock(),
                tp_cache_group="cpu",
                attn_tp_cache_group="cpu",
                pp_rank=0,
                pp_size=1,
                attn_cp_rank=0,
                attn_cp_size=1,
            )
            with (
                patch.object(torch.distributed, "is_initialized", return_value=True),
                patch.object(torch.distributed, "get_rank", return_value=rank),
                patch.object(torch.distributed, "get_world_size", return_value=world),
                patch.object(
                    module, "resolve_hybrid_device_pool_group", return_value=pools
                ),
                patch(
                    "sglang.srt.distributed.parallel_state.get_attn_tp_group",
                    return_value=SimpleNamespace(device_group="device"),
                ),
            ):
                linker = module.UMBPDirectLinker(
                    SimpleNamespace(), params, components=set(), _storage=_Storage()
                )
            linker.test_rank = rank
            created.append(linker)
            linkers.append(linker)
        return linkers, collectives

    yield make
    for linker in created:
        linker.close()


def _transfers(linker, pages, side=()):
    def transfer(name, selected):
        selected = list(selected)
        # Each rank uses different, non-contiguous physical rows for the same keys.
        slots = torch.tensor([(page * 3 + linker.test_rank) % 32 for page in selected])
        indices = (slots[:, None] * 2 + torch.arange(2)).reshape(-1).to(torch.int64)
        return PoolTransfer(
            name=name,
            keys=[f"p{page}" for page in selected],
            device_indices=indices,
            hit_policy=(
                PoolHitPolicy.ALL_PAGES
                if name == PoolName.KV
                else PoolHitPolicy.TRAILING_PAGES
            ),
        )

    result = [transfer(PoolName.KV, pages)] if len(pages) else []
    if len(side):
        result.append(transfer(PoolName.SWA, side))
    return result


def _seed(linkers, pages=12):
    snapshots = []
    for linker in linkers:
        for pool, entry in enumerate(linker.pools.values()):
            for component, buffers in enumerate(entry.components):
                for layer, buffer in enumerate(buffers):
                    span = 2 if entry.name == PoolName.KV else 1
                    for page in range(pages):
                        row = ((page * 3 + linker.test_rank) % 32) * span
                        data = torch.arange(
                            span * buffer.shape[1], dtype=torch.uint8
                        ).reshape(span, -1)
                        buffer[row : row + span] = (
                            data + page * 13 + pool * 5 + component * 3 + layer * 17
                        )
        snapshots.append(
            {
                name: [
                    [buffer.clone() for buffer in component]
                    for component in entry.components
                ]
                for name, entry in linker.pools.items()
            }
        )
        assert linker.offload(_transfers(linker, range(pages), range(pages)))
        linker._offload_queue.join()
        assert linker.pop_completed_offload()
        for entry in linker.pools.values():
            for buffer in entry.get_hybrid_pool_buffer():
                buffer.fill_(231)
    return snapshots


def _queue(linker, pages, rid="request", lookup_pages=12, side=()):
    linker.lookup(rid, _transfers(linker, range(lookup_pages), [lookup_pages - 1]))
    transfers = _transfers(linker, pages, side)
    if transfers:
        assert linker.load(rid, transfers)


def _finish(linker):
    index = linker.start_layer_wise_loading()
    linker.layer_done_counter.set_consumer(index)
    for layer in range(linker.num_layers):
        linker.layer_done_counter.wait_until(layer)
    linker._load_queue.join()
    return index


def test_windows_cover_pages_once():
    for pages in (0, 1, 7, 8, 17, 64):
        for world in (2, 3, 8):
            windows = module._split_windows(pages, world)
            assert [
                page for start, end in windows for page in range(start, end)
            ] == list(range(pages))
            assert all(end - start <= -(-pages // world) for start, end in windows)


@pytest.mark.parametrize("sparse", [False, True])
def test_split_restores_every_byte_and_keeps_remainders_local(factory, sparse):
    linkers, collectives = factory(sparse=sparse)
    snapshots = _seed(linkers)
    needs = [{0, 1, 2, 3, 4, 7, 9}, {0, 2, 3, 4, 7, 9, 10}, {0, 1, 3, 4, 6, 7, 9, 11}]
    common = sorted(set.intersection(*needs))
    for rank, linker in enumerate(linkers):
        if sparse:
            # The same common window may cross request boundaries.
            _queue(linker, sorted(needs[rank] & set(range(6))), rid="first")
            _queue(
                linker,
                sorted(needs[rank] - set(range(6))),
                rid="second",
                side=[11] if rank else [],
            )
        else:
            _queue(linker, sorted(needs[rank]), side=[11] if rank else [])
    staging = [linker._split_recv.data_ptr() for linker in linkers]
    collectives.run(linkers, _finish)
    for rank, linker in enumerate(linkers):
        start, end = module._split_windows(len(common), 3)[rank]
        read_pages = (needs[rank] - set(common)) | set(common[start:end])
        for name, entry in linker.pools.items():
            wanted = ({11} if rank else set()) if name == PoolName.SWA else needs[rank]
            actual_reads = {
                int(key.split(":")[0][1:])
                for key, _, _ in linker.storage.reads
                if f":{name}:" in key
            }
            assert actual_reads == (wanted if name == PoolName.SWA else read_pages)
            span = 2 if name == PoolName.KV else 1
            for c, component in enumerate(entry.components):
                for layer, buffer in enumerate(component):
                    expected = torch.full_like(buffer, 231)
                    for page in wanted:
                        row = ((page * 3 + rank) % 32) * span
                        expected[row : row + span] = snapshots[rank][name][c][layer][
                            row : row + span
                        ]
                    assert torch.equal(buffer, expected), (rank, name, layer)
        assert linker._split_recv.data_ptr() == staging[rank]
        assert linker._stats["split_batches"] == 1
        assert linker._stats["split_local_pages"] == len(needs[rank]) - len(common)
        assert not linker._split_state
        # Offload was complete on every rank even though split was enabled.
        assert len(linker.storage.objects) == 12 * sum(
            1 if e.packed else len(e.components) for e in linker.pools.values()
        )
    assert sum(kind == "gather" for kind, _, _ in collectives.calls) == 2


@pytest.mark.parametrize(
    "reason",
    ["unknown", "missing_cache", "overflow", "page_limit", "empty", "rid", "length"],
)
def test_one_rank_veto_or_mismatch_falls_back_before_page_reduce(factory, reason):
    linkers, collectives = factory()
    for linker in linkers:
        _queue(linker, [0, 1])
    victim = linkers[1]
    if reason == "unknown":
        victim._pending_pages["request"][0] = "different"
    elif reason == "missing_cache":
        victim._pending_pages["request"] = None
    elif reason == "overflow":
        for index in range(victim._split_max_rids):
            _queue(victim, [0, 1], rid=f"extra{index}")
    elif reason == "page_limit":
        victim._pending_pages["request"] = [
            f"p{i}" for i in range(module.SPLIT_MAX_PAGES + 1)
        ]
    elif reason == "empty":
        victim._pending.clear()
    elif reason == "rid":
        victim._pending["different"] = victim._pending.pop("request")
        victim._pending_pages["different"] = victim._pending_pages.pop("request")
    else:
        victim._pending_pages["request"].append("extra")
    assert (
        collectives.run(linkers, lambda linker: linker._prepare_split_share())
        == [None] * 3
    )
    assert collectives.calls == [("reduce", "cpu", 1 + 4 * linkers[0]._split_max_rids)]


@pytest.mark.parametrize("failure", ["loader", "indices"])
def test_failure_reaches_every_rank_before_allgather(factory, monkeypatch, failure):
    linkers, collectives = factory()
    _seed(linkers)
    for linker in linkers:
        _queue(linker, list(range(9)))
    if failure == "loader":
        linkers[1].storage.fail_read = True
    else:
        monkeypatch.setattr(
            linkers[1],
            "_split_rows",
            Mock(side_effect=RuntimeError("index allocation failed")),
        )

    def fail(linker):
        with pytest.raises(RuntimeError, match="layer-wise KV load failed"):
            _finish(linker)

    collectives.run(linkers, fail)
    assert not any(kind == "gather" for kind, _, _ in collectives.calls)


def test_capacity_fallback_and_reset(factory):
    linkers, collectives = factory()
    for linker in linkers:
        _queue(linker, list(range(9)))
        linker._split_recv = linker._split_recv[:1]
    assert (
        collectives.run(linkers, lambda linker: linker._prepare_split_share())
        == [None] * 3
    )
    assert len(collectives.calls) == 2
    for linker in linkers:
        linker.reset()
        assert (
            not linker._pending_pages
            and not linker._lookup_pages
            and not linker._split_state
        )


def test_disabled_path_has_no_collectives(factory):
    linkers, collectives = factory(enabled=False)
    _seed(linkers)
    for linker in linkers:
        _queue(linker, [0, 1])
        _finish(linker)
    assert not collectives.calls
    assert all(not linker._lookup_pages for linker in linkers)


def test_non_replicated_pool_is_rejected(factory):
    with pytest.raises(ValueError, match="rank-replicated KV"):
        factory(replicated=False)


@pytest.mark.parametrize("hip", [None, "7.2"])
def test_any_replicated_pool_group_is_admitted(monkeypatch, hip):
    """rank_replicated is the whole replication gate, on every platform.

    A DSv4-shaped group is admitted; only its KV-source pools are split.
    """
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(
        module,
        "get_parallel",
        lambda: SimpleNamespace(
            attn_cp_size=1, enable_dp_attention=False, pp_size=1, tp_size=1
        ),
    )
    monkeypatch.setattr(
        module, "get_disagg", lambda: SimpleNamespace(disaggregation_mode="null")
    )
    linker = module.UMBPDirectLinker.__new__(module.UMBPDirectLinker)
    linker.pool_group = SimpleNamespace(rank_replicated=True)
    linker.pools = {
        PoolName.DEEPSEEK_V4_C4: SimpleNamespace(indices_from_pool=PoolName.KV),
        PoolName.DEEPSEEK_V4_C4_STATE: SimpleNamespace(indices_from_pool=PoolName.SWA),
    }
    with (
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(torch.distributed, "get_world_size", return_value=1),
        patch(
            "sglang.srt.distributed.parallel_state.get_attn_tp_group",
            return_value=SimpleNamespace(device_group=object()),
        ),
    ):
        # world_size 1 returns before staging is built.
        linker._init_split(None, 0)
    assert linker._split_pools == [PoolName.DEEPSEEK_V4_C4]


def test_invalid_split_geometry_is_rejected_at_initialization(factory):
    with pytest.raises(ValueError, match="invalid row geometry"):
        factory(invalid_geometry=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("attn_cp_size", 2),
        ("enable_dp_attention", True),
        ("pp_size", 2),
        ("disaggregation_mode", "decode"),
    ],
)
def test_unsupported_topology_is_rejected_before_group_creation(
    monkeypatch, field, value
):
    config = SimpleNamespace(
        attn_cp_size=1, enable_dp_attention=False, pp_size=1, disaggregation_mode="null"
    )
    setattr(config, field, value)
    monkeypatch.setattr(module, "get_parallel", lambda: config)
    monkeypatch.setattr(module, "get_disagg", lambda: config)
    linker = module.UMBPDirectLinker.__new__(module.UMBPDirectLinker)
    linker.pool_group = SimpleNamespace(rank_replicated=True)
    with pytest.raises(ValueError, match="requires rank-replicated"):
        linker._init_split(None, 0)
