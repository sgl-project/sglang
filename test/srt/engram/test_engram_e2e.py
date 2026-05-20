#!/usr/bin/env python3
"""Test 3: End-to-end Engram module + EngramStoreManager integration.

Verifies:
  - Engram module accepts a store_manager and uses it to create stores
  - The store instance in MultiHeadEmbedding matches the manager's store
  - Forward pass produces correct output shapes
  - Backward compatibility: Engram without store_manager still works

Usage:
    python test/srt/engram/test_engram_e2e.py
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
from sglang.srt.models.engram import engram as engram_mod
from sglang.srt.models.engram.engram import BackBoneConfig, Engram, EngramConfig

LAYER_IDS = [1]
HIDDEN = 1024
HC_MULT = 4
BATCH = 2
SEQ = 32


def _setup_configs():
    engram_mod.engram_cfg = EngramConfig(
        tokenizer_name_or_path="deepseek-ai/DeepSeek-V3",
        engram_vocab_size=[1024, 1024],
        max_ngram_size=3,
        n_embed_per_ngram=512,
        n_head_per_ngram=8,
        layer_ids=LAYER_IDS,
        pad_id=2,
        seed=0,
        kernel_size=4,
        store_backend="local",
        enable_prefetch=False,
    )
    engram_mod.backbone_config = BackBoneConfig(
        hidden_size=HIDDEN,
        hc_mult=HC_MULT,
        vocab_size=129280,
        num_layers=2,
    )


def test_with_store_manager():
    print("\n--- Test: Engram with EngramStoreManager ---")
    _setup_configs()

    store_cfg = EngramStoreConfig(store_backend="local", layer_ids=LAYER_IDS)
    mgr = EngramStoreManager(store_cfg)

    model = Engram(layer_id=1, store_manager=mgr)
    print(f"  Store type: {type(model.multi_head_embedding.store).__name__}")
    assert isinstance(
        model.multi_head_embedding.store, LocalEngramStore
    ), f"Expected LocalEngramStore, got {type(model.multi_head_embedding.store).__name__}"
    print("  [PASS] Store is LocalEngramStore from manager")

    assert mgr.has_layer(1), "Layer 1 should be registered"
    assert (
        mgr.get_store(1) is model.multi_head_embedding.store
    ), "Store identity mismatch"
    print("  [PASS] Store identity matches manager's registry")

    input_ids = torch.randint(0, 129280, (BATCH, SEQ))
    hidden = torch.randn(BATCH, SEQ, HC_MULT, HIDDEN)
    with torch.no_grad():
        out = model(hidden_states=hidden, input_ids=input_ids)
    assert out.shape == (BATCH, SEQ, HC_MULT, HIDDEN), f"Unexpected shape: {out.shape}"
    print(f"  [PASS] Forward output shape: {out.shape}")

    mgr.close()
    print("  [PASS] Manager closed without error")


def test_without_store_manager():
    print("\n--- Test: Engram without store_manager (backward compat) ---")
    _setup_configs()

    model = Engram(layer_id=1, store_manager=None)
    print(f"  Store type: {type(model.multi_head_embedding.store).__name__}")
    assert isinstance(model.multi_head_embedding.store, LocalEngramStore)
    print("  [PASS] Fallback creates LocalEngramStore directly")

    input_ids = torch.randint(0, 129280, (BATCH, SEQ))
    hidden = torch.randn(BATCH, SEQ, HC_MULT, HIDDEN)
    with torch.no_grad():
        out = model(hidden_states=hidden, input_ids=input_ids)
    assert out.shape == (BATCH, SEQ, HC_MULT, HIDDEN), f"Unexpected shape: {out.shape}"
    print(f"  [PASS] Forward output shape: {out.shape}")


def test_multi_layer():
    print("\n--- Test: Multiple Engram layers sharing one manager ---")
    layer_ids = [1, 15]
    engram_mod.engram_cfg = EngramConfig(
        tokenizer_name_or_path="deepseek-ai/DeepSeek-V3",
        engram_vocab_size=[1024, 1024],
        max_ngram_size=3,
        n_embed_per_ngram=512,
        n_head_per_ngram=8,
        layer_ids=layer_ids,
        pad_id=2,
        seed=0,
        kernel_size=4,
        store_backend="local",
        enable_prefetch=False,
    )
    engram_mod.backbone_config = BackBoneConfig(
        hidden_size=HIDDEN,
        hc_mult=HC_MULT,
        vocab_size=129280,
        num_layers=30,
    )

    store_cfg = EngramStoreConfig(store_backend="local", layer_ids=layer_ids)
    mgr = EngramStoreManager(store_cfg)

    models = {}
    for lid in layer_ids:
        models[lid] = Engram(layer_id=lid, store_manager=mgr)

    assert mgr.has_layer(1) and mgr.has_layer(15), "Both layers should be registered"
    assert mgr.get_store(1) is not mgr.get_store(
        15
    ), "Each layer should have its own store"
    print("  [PASS] Two layers have independent stores in one manager")

    for lid in layer_ids:
        input_ids = torch.randint(0, 129280, (BATCH, SEQ))
        hidden = torch.randn(BATCH, SEQ, HC_MULT, HIDDEN)
        with torch.no_grad():
            out = models[lid](hidden_states=hidden, input_ids=input_ids)
        assert out.shape == (BATCH, SEQ, HC_MULT, HIDDEN)
    print(f"  [PASS] Forward pass correct for both layers")

    mgr.close()
    print("  [PASS] Manager closed without error")


def main() -> None:
    print("=" * 60)
    print("Test 3: End-to-end Engram + EngramStoreManager")
    print("=" * 60)

    test_with_store_manager()
    test_without_store_manager()
    test_multi_layer()

    print("\n" + "=" * 60)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
