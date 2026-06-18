#!/usr/bin/env python3
"""Test 2: Verify LocalEngramStore works correctly through EngramStoreManager.

Tests the full lifecycle: create manager -> get_or_create_store -> put_sharded
-> get_one -> get_many -> close.

Usage:
    python test/srt/engram/test_local_store_manager.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sglang.srt.mem_cache.engram import EngramStoreConfig, EngramStoreManager
from sglang.srt.mem_cache.engram.local_engram_store import LocalEngramStore


def main() -> None:
    print("=" * 60)
    print("Test 2: LocalEngramStore via EngramStoreManager")
    print("=" * 60)

    vocab_size = 2048
    embedding_dim = 64
    layer_ids = [1, 15]

    cfg = EngramStoreConfig(store_backend="local", layer_ids=layer_ids)
    mgr = EngramStoreManager(cfg)
    print(f"  Created EngramStoreManager with backend='local', layers={layer_ids}")

    # --- create stores for both layers ---
    stores = {}
    for lid in layer_ids:
        s = mgr.get_or_create_store(
            layer_id=lid, vocab_size=vocab_size, embedding_dim=embedding_dim
        )
        assert isinstance(
            s, LocalEngramStore
        ), f"Expected LocalEngramStore, got {type(s).__name__}"
        stores[lid] = s
    print("  [PASS] get_or_create_store returns LocalEngramStore for each layer")

    # --- idempotency ---
    for lid in layer_ids:
        s2 = mgr.get_or_create_store(
            layer_id=lid, vocab_size=vocab_size, embedding_dim=embedding_dim
        )
        assert s2 is stores[lid], "get_or_create_store must return the same object"
    print("  [PASS] get_or_create_store is idempotent (same object)")

    # --- put_sharded + get_many + get_one ---
    vocab_table = torch.arange(vocab_size * embedding_dim, dtype=torch.float16).view(
        vocab_size, embedding_dim
    )
    device = torch.device("cpu")

    for lid in layer_ids:
        store = stores[lid]
        store.put_sharded(vocab_table)

        indices = torch.tensor([0, 1, 42, 100, 999, vocab_size - 1], dtype=torch.long)
        result = store.get_many(indices, layer_id=lid, device=device)
        expected = vocab_table[indices]
        assert torch.equal(result, expected), f"get_many mismatch on layer {lid}"

        for idx in [0, 42, vocab_size - 1]:
            single = store.get_one(idx, layer_id=lid, device=device)
            assert torch.equal(
                single, vocab_table[idx]
            ), f"get_one mismatch on layer {lid}, index {idx}"

    print(f"  [PASS] put_sharded / get_many / get_one correct for all layers")

    # --- get_one out-of-range returns zeros ---
    zero_vec = stores[1].get_one(-1, layer_id=1, device=device)
    assert torch.equal(
        zero_vec, torch.zeros(embedding_dim, dtype=torch.float16)
    ), "OOB get_one should return zeros"
    zero_vec = stores[1].get_one(vocab_size + 100, layer_id=1, device=device)
    assert torch.equal(
        zero_vec, torch.zeros(embedding_dim, dtype=torch.float16)
    ), "OOB get_one should return zeros"
    print("  [PASS] get_one out-of-range returns zero vector")

    # --- global accessor ---
    from sglang.srt.mem_cache.engram import (
        close_global_engram_store_manager,
        get_global_engram_store_manager,
        set_global_engram_store_manager,
    )

    assert get_global_engram_store_manager() is None, "Global should start as None"
    set_global_engram_store_manager(mgr)
    assert get_global_engram_store_manager() is mgr, "Global should be our manager"
    close_global_engram_store_manager()
    assert (
        get_global_engram_store_manager() is None
    ), "Global should be None after close"
    print("  [PASS] global accessor set/get/close lifecycle correct")

    print("\n" + "=" * 60)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
