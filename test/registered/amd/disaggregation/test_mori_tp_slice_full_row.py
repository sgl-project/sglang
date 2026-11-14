"""Mori KV transfer between prefill and decode with different attention TP sizes.

When every rank stores whole, replicated KV heads (fewer KV heads than the larger
attention TP), the TP-mismatch head slice is the whole token row and the transfer
must use the contiguous page plan rather than one RDMA write per token.
"""

import importlib
import sys
import types
import unittest
from enum import IntEnum
from types import SimpleNamespace

import numpy as np


def _install_mori_stubs() -> None:
    try:
        importlib.import_module("mori.cpp")
        importlib.import_module("mori.io")
        return
    except (ImportError, OSError):
        pass

    mori = types.ModuleType("mori")
    mori.__path__ = []
    mori_cpp = types.ModuleType("mori.cpp")
    mori_io = types.ModuleType("mori.io")

    class FakeStatusCode(IntEnum):
        SUCCESS = 0
        IN_PROGRESS = 1
        FAILED = 2

    mori_cpp.TransferStatus = object
    for name in (
        "BackendType",
        "EngineDesc",
        "IOEngine",
        "IOEngineConfig",
        "MemoryDesc",
        "MemoryLocationType",
        "PollCqMode",
        "RdmaBackendConfig",
    ):
        setattr(mori_io, name, object)
    mori_io.StatusCode = FakeStatusCode
    mori.cpp = mori_cpp
    mori.io = mori_io
    sys.modules["mori"] = mori
    sys.modules["mori.cpp"] = mori_cpp
    sys.modules["mori.io"] = mori_io


_install_mori_stubs()

from sglang.srt.disaggregation.mori.conn import (
    BatchTransferPlan,
    GroupedIndexPlan,
    MoriKVManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 16
HEAD_BYTES = 256  # head_dim 256, fp8


def _manager(prefill_tp: int, prefill_rank: int, total_kv_heads: int) -> MoriKVManager:
    heads = max(1, total_kv_heads // prefill_tp)
    manager = object.__new__(MoriKVManager)
    manager.attn_tp_size = prefill_tp
    manager.is_mla_backend = False
    manager.is_hybrid_mla_backend = False
    manager.kv_args = SimpleNamespace(
        page_size=PAGE_SIZE,
        kv_item_lens=[PAGE_SIZE * heads * HEAD_BYTES],
        total_kv_head_num=total_kv_heads,
        kv_head_num=heads,
        engine_rank=prefill_rank,
    )
    return manager


def _peer(decode_tp: int, decode_rank: int, total_kv_heads: int):
    heads = max(1, total_kv_heads // decode_tp)
    return SimpleNamespace(
        decode_tp_size=decode_tp,
        decode_tp_rank=decode_rank,
        dst_kv_item_len=PAGE_SIZE * heads * HEAD_BYTES,
        dst_kv_mem_descs=[],
        dst_kv_layer_ids=[],
    )


def _byte_ranges(plan: BatchTransferPlan):
    """Merged (local_start, remote_start, length) runs covered by a plan."""
    runs = sorted(zip(plan.local_offsets, plan.remote_offsets, plan.sizes))
    merged = []
    for local, remote, size in runs:
        if merged:
            l0, r0, s0 = merged[-1]
            if l0 + s0 == local and r0 + s0 == remote:
                merged[-1] = (l0, r0, s0 + size)
                continue
        merged.append((local, remote, size))
    return merged


class TestTPSliceFullRow(CustomTestCase):
    def _is_full_row(self, prefill_tp, decode_tp, total_kv_heads):
        """Full-row verdicts for every (prefill rank, decode rank) pair."""
        verdicts = set()
        for p_rank in range(prefill_tp):
            for d_rank in range(decode_tp):
                manager = _manager(prefill_tp, p_rank, total_kv_heads)
                cfg = manager._build_tp_slice_config(
                    _peer(decode_tp, d_rank, total_kv_heads)
                )
                verdicts.add(MoriKVManager._tp_slice_is_full_row(cfg))
        self.assertEqual(len(verdicts), 1, "verdict must not depend on the rank pair")
        return verdicts.pop()

    def test_replicated_heads_are_full_rows(self):
        # 2 KV heads (e.g. Qwen3.5-397B): one whole head per rank on both sides.
        for prefill_tp, decode_tp in ((2, 4), (2, 8), (4, 8), (4, 2), (8, 2)):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                self.assertTrue(self._is_full_row(prefill_tp, decode_tp, 2))

    def test_split_heads_still_slice(self):
        # A side holding several heads must still send per-head slices.
        for prefill_tp, decode_tp, heads in (
            (1, 2, 2),
            (1, 4, 2),
            (2, 1, 2),
            (2, 4, 8),
            (4, 2, 8),
        ):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp, heads=heads):
                self.assertFalse(self._is_full_row(prefill_tp, decode_tp, heads))

    def test_full_row_slice_plan_matches_contiguous_plan(self):
        rng = np.random.default_rng(0)
        manager = _manager(2, 1, 2)
        cfg = manager._build_tp_slice_config(_peer(4, 3, 2))
        self.assertTrue(MoriKVManager._tp_slice_is_full_row(cfg))
        # Mix contiguous runs and scattered pages, as the allocator produces them.
        src = np.concatenate(
            [np.arange(100, 140), rng.choice(5000, 30, replace=False) + 200]
        )
        dst = np.concatenate(
            [np.arange(900, 940), rng.choice(5000, 30, replace=False) + 6000]
        )
        src, dst = src.astype(np.int32), dst.astype(np.int32)

        slice_plan = manager._build_tp_slice_transfer_plan(src, dst, cfg)
        grouped = GroupedIndexPlan.from_indices(
            src.astype(np.int64), dst.astype(np.int64)
        )
        page_plan = grouped.materialize(manager.kv_args.kv_item_lens[0])

        self.assertEqual(_byte_ranges(slice_plan), _byte_ranges(page_plan))
        self.assertEqual(len(slice_plan.sizes), len(src) * PAGE_SIZE)
        self.assertLess(len(page_plan.sizes), len(src))

    def _submitted_plans(self, manager, peer, src, dst):
        plans = []
        num_layers = 3
        manager._get_mha_mem_desc_slices = lambda descs, layer_ids: (
            [f"k{i}" for i in range(num_layers)],
            [f"v{i}" for i in range(num_layers)],
            [f"dk{i}" for i in range(num_layers)],
            [f"dv{i}" for i in range(num_layers)],
            num_layers,
        )
        manager._submit_batch_transfer_plan = lambda s, d, plan: (
            plans.append(plan) or []
        )
        manager.send_kvcache(peer, src, dst)
        self.assertEqual(len(plans), 2 * num_layers)
        return plans

    def test_send_kvcache_uses_page_plan_for_replicated_heads(self):
        src = np.arange(10, 74, dtype=np.int32)
        dst = np.arange(500, 564, dtype=np.int32)
        plans = self._submitted_plans(_manager(2, 0, 2), _peer(4, 1, 2), src, dst)
        for plan in plans:
            self.assertEqual(plan.sizes, [len(src) * PAGE_SIZE * HEAD_BYTES])

    def test_send_kvcache_keeps_token_slices_for_split_heads(self):
        src = np.arange(10, 74, dtype=np.int32)
        dst = np.arange(500, 564, dtype=np.int32)
        plans = self._submitted_plans(_manager(1, 0, 2), _peer(2, 1, 2), src, dst)
        for plan in plans:
            self.assertEqual(len(plan.sizes), len(src) * PAGE_SIZE)
            self.assertEqual(set(plan.sizes), {HEAD_BYTES})


if __name__ == "__main__":
    unittest.main()
