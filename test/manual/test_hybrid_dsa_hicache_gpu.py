#!/usr/bin/env python3
"""Exact-image GPU byte oracle. Run only in an isolated qualification Job.

Exercises the real index sidecar builder and unmodified CPU/GPU copy paths,
including noncontiguous physical pages, incremental backup, draft layers,
and restoration onto different physical pages after poisoning device memory.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _build_hybrid_dsa_index_entry,
    build_hybrid_mamba_stack,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import HybridCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry


def check_derived_slot_lifecycle():
    """The INDEXER borrows KV addresses; it must never allocate/free its own."""
    pools = {name: MagicMock() for name in (PoolName.KV, PoolName.MAMBA, PoolName.INDEXER)}
    group = HostPoolGroup([
        PoolEntry(name=name, host_pool=pool, device_pool=None,
                  layer_mapper=lambda layer: layer,
                  is_primary_index_anchor=name == PoolName.KV)
        for name, pool in pools.items()
    ])
    controller = object.__new__(HybridCacheController)
    controller.mem_pool_host = group
    controller.write_queue = []
    controller.start_writing = MagicMock()
    for iteration in range(100):
        # Reuse a bounded host range after each prior operation was released.
        size = 512 if iteration % 2 == 0 else 256
        host = torch.arange(size, dtype=torch.int64)
        device = host + 1024
        pools[PoolName.KV].alloc.return_value = host
        pools[PoolName.MAMBA].alloc.return_value = torch.tensor([iteration % 4])
        pools[PoolName.MAMBA].free.return_value = 1
        result = controller.write(device, node_id=iteration, extra_pools=[
            PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV),
            PoolTransfer(PoolName.MAMBA, device_indices=torch.tensor([3])),
        ])
        assert result is host
        operation = controller.write_queue.pop()
        derived = next(t for t in operation.pool_transfers if t.name == PoolName.INDEXER)
        assert derived.host_indices is host
        assert derived.device_indices is device
        assert group.release_transfers(operation.pool_transfers) == 1
        # A published node may split only at a whole compression group.
        for child in host.split(256):
            group.free(child)
    pools[PoolName.INDEXER].alloc.assert_not_called()
    pools[PoolName.INDEXER].free.assert_not_called()
    assert pools[PoolName.MAMBA].free.call_count == 100
    assert pools[PoolName.KV].free.call_count == 150
    print("Derived INDEXER controller lifecycle: 100/100 passed", flush=True)


def check_ratio_mode():
    prefix = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
    target = make_pool((True, True))
    target.kv_cache_dim = 656
    memory = SimpleNamespace(hicache_size=0, hicache_ratio=2.5,
                             hicache_mem_layout="page_first",
                             hicache_write_policy="write_through",
                             hicache_io_backend="kernel", hicache_host_memory_mode=None)
    params = SimpleNamespace(page_size=64, mtp_draft_device_pools=(),
                             req_to_token_pool=SimpleNamespace(mamba_allocator=MagicMock()),
                             token_to_kv_pool_allocator=MagicMock(), tp_cache_group=None,
                             attn_cp_cache_group=None, attn_tp_cache_group=None, pp_cache_group=None)
    with (
        patch(prefix + "get_memory", return_value=memory),
        patch(prefix + "_get_allocator_type", return_value="default"),
        patch(prefix + "_split_hicache_size") as split,
        patch(prefix + "build_kv_host_pool") as build_kv,
        patch(prefix + "MambaPoolHost"),
        patch(prefix + "DeepSeekV4PagedHostPool") as index_host,
        patch(prefix + "HybridCacheController"),
    ):
        build_kv.return_value.page_num = 20
        group, _ = build_hybrid_mamba_stack(
            params=params, kv_pool=target, mamba_pool=MagicMock(),
            full_layer_mapping={0: 0, 3: 1}, mamba_layer_mapping={1: 0, 2: 1},
            load_cache_event=None, storage_backend=None, use_mla=True,
        )
        split.assert_not_called()
        assert build_kv.call_args.kwargs["host_size"] is None
        assert index_host.call_args.kwargs["num_host_pages"] == 20
        assert PoolName.INDEXER in group.entry_map
    print("Hybrid DSA ratio-mode assembly: passed", flush=True)


def make_pool(live_layers):
    pool = object.__new__(DSATokenToKVPool)
    pool.page_size = 64
    pool.index_kpool = 4
    pool.kpool_use_compress = True
    pool.layer_num = len(live_layers)
    pool.index_key_cache = SimpleNamespace(buffer=[
        torch.zeros((8 if live else 0, 64 * 132), dtype=torch.uint8, device="cuda")
        for live in live_layers
    ])
    return pool


def tokens(pages):
    return (pages[:, None] * 64 + torch.arange(64, device=pages.device)).flatten()


def run_case(layout, backend, trial):
    target = make_pool((True, False, True))
    draft = make_pool((True,))
    prefix = "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
    with (
        patch(prefix + "get_memory", return_value=SimpleNamespace(hicache_mem_layout=layout)),
        patch(prefix + "_get_allocator_type", return_value="default"),
    ):
        entry = _build_hybrid_dsa_index_entry(
            kv_pool=target, kv_host_pool=SimpleNamespace(page_num=16),
            layer_mapping={0: 0, 4: 1, 8: 2, 45: 3},
            transfer_layer_num=46, draft_pools=(draft,),
        )
    assert entry.layer_mapper(4) is None
    assert entry.layer_mapper(45) == 2
    host = entry.host_pool
    assert len(host.device_buffers) == 3
    assert host.item_bytes == 8448
    generator = torch.Generator(device="cpu").manual_seed(37625 + trial)
    originals = []
    for buffer in host.device_buffers:
        original = torch.randint(0, 256, buffer.shape, dtype=torch.uint8, generator=generator)
        buffer.copy_(original)
        originals.append(original)
    source_pages = torch.randperm(8, generator=generator)
    host_pages = torch.randperm(16, generator=generator)[:8]
    dest_pages = torch.randperm(8, generator=generator)
    index_device = "cuda" if backend == "kernel" else "cpu"
    src = tokens(source_pages.to(index_device))
    dst = tokens(dest_pages.to(index_device))
    cached = tokens(host_pages.to(index_device))
    # The staged page-first D2H path consumes CPU destination indices,
    # exactly as HybridCacheController._move_write_operation supplies them.
    backup_cached = cached.cpu() if host.can_use_write_back_jit else cached
    # Explicit dependencies model asynchronous transfers without relying on
    # the default stream's implicit ordering.
    backup_stream = torch.cuda.Stream()
    restore_stream = torch.cuda.Stream()
    backup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(backup_stream):
        for start in (0, 256):
            host.backup_from_device_all_layer(
                target, backup_cached[start:start + 256], src[start:start + 256], backend,
            )
        backed_up = backup_stream.record_event()
    torch.cuda.current_stream().wait_event(backed_up)
    for buffer in host.device_buffers:
        buffer.fill_(173)
    restore_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(restore_stream):
        for layer in range(3):
            host.load_to_device_per_layer(target, cached, dst, layer, backend)
        restored = restore_stream.record_event()
    restored.synchronize()
    for layer, buffer in enumerate(host.device_buffers):
        actual = buffer.cpu()[dest_pages]
        expected = originals[layer][source_pages]
        assert torch.equal(actual, expected), (layout, backend, trial, layer)
    print(json.dumps({"layout": layout, "backend": backend, "trial": trial,
                      "layers": 3, "pages": 8, "bytes_compared": 3 * 8 * 8448,
                      "passed": True}), flush=True)


def main():
    assert torch.cuda.is_available(), "requires an isolated CUDA Job"
    print(json.dumps({"gpu": torch.cuda.get_device_name(),
                      "capability": torch.cuda.get_device_capability()}), flush=True)
    with torch.inference_mode():
        check_derived_slot_lifecycle()
        check_ratio_mode()
        for layout, backend in (
            ("page_first", "kernel"), ("layer_first", "kernel"),
            ("layer_first", "direct"), ("page_first_direct", "direct"),
        ):
            for trial in range(5):
                run_case(layout, backend, trial)
    print("Hybrid DSA HiCache GPU byte oracle: 20/20 passed", flush=True)


if __name__ == "__main__":
    main()
