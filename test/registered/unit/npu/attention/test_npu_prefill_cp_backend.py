"""
Unit tests for the NPU prefill context parallel paths in
sglang.srt.hardware_backend.npu.attention.ascend_backend.

These guard the two defects left behind when the CP v1 runtime was removed:

* the KV all-gather for prefill CP must go through the CP v2 strategy, not
  through the deleted ``cp_all_gather_rerange_kv_cache`` shim;
* ``do_cp_attn_fia`` must hand FIA a query whose row count matches the
  cumulative ``actual_seq_lengths`` it advertises -- the model runner pads the
  CP shard to ``per_rank_actual_token[0]`` rows, which previously made FIA
  tiling fail with error 561002.

Everything runs on CPU tensors, so the checks are hardware independent.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(_, unittest.mock.MagicMock())

from sglang.srt.hardware_backend.npu.attention import ascend_backend
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    _cp_allgather_and_save_kv_npu,
)


class _RecordingPool:
    def __init__(self):
        self.calls = []

    def set_kv_buffer(self, layer, loc, k, v, *args, **kwargs):
        self.calls.append((tuple(k.shape), tuple(v.shape)))


class _RecordingStrategy:
    def __init__(self, cp_size=2):
        self.cp_size = cp_size
        self.kv_calls = []

    def materialize_full_kv(self, forward_batch, layer, k, v, swa_loc=None):
        self.kv_calls.append((tuple(k.shape), tuple(v.shape)))


def _cp_batch(out_cache_loc):
    return SimpleNamespace(
        out_cache_loc=out_cache_loc, encoder_out_cache_loc=None
    )


def _attn_layer():
    return SimpleNamespace(is_cross_attention=False, k_scale=None, v_scale=None)


def _fia_layer():
    return SimpleNamespace(
        tp_q_head_num=16,
        tp_k_head_num=8,
        tp_v_head_num=8,
        qk_head_dim=128,
        v_head_dim=128,
        scaling=0.08838834764831845,
    )


class TestCPKVWrite(unittest.TestCase):
    """`_cp_allgather_and_save_kv_npu` must use the CP v2 strategy API."""

    def setUp(self):
        self.k = torch.zeros(3, 8, 128, dtype=torch.bfloat16)
        self.v = torch.zeros(3, 8, 128, dtype=torch.bfloat16)
        self.batch = _cp_batch(torch.arange(6))
        self.layer = _attn_layer()

    def test_uses_cp_v2_strategy(self):
        strategy = _RecordingStrategy(cp_size=2)
        pool = _RecordingPool()
        with patch.object(ascend_backend, "get_cp_strategy", return_value=strategy):
            _cp_allgather_and_save_kv_npu(
                self.batch, self.layer, self.k, self.v, 2, pool, swa_loc=None
            )
        self.assertEqual(len(strategy.kv_calls), 1)
        self.assertEqual(strategy.kv_calls[0], ((3, 8, 128), (3, 8, 128)))
        # the strategy owns the pool write; the helper must not bypass it
        self.assertEqual(pool.calls, [])

    def test_falls_back_to_plain_write_without_cp(self):
        strategy = _RecordingStrategy(cp_size=1)
        pool = _RecordingPool()
        with patch.object(ascend_backend, "get_cp_strategy", return_value=strategy):
            _cp_allgather_and_save_kv_npu(
                self.batch, self.layer, self.k, self.v, 1, pool, swa_loc=None
            )
        self.assertEqual(strategy.kv_calls, [])
        self.assertEqual(pool.calls, [((3, 8, 128), (3, 8, 128))])

    def test_deleted_cp_v1_helper_is_not_referenced(self):
        with open(ascend_backend.__file__) as handle:
            source = handle.read()
        self.assertNotIn("cp_all_gather_rerange_kv_cache", source)


class TestDoCpAttnFia(unittest.TestCase):
    """`do_cp_attn_fia` must trim and re-append the physical pad rows."""

    PHYSICAL_ROWS = 4

    def setUp(self):
        self.backend = AscendAttnBackend.__new__(AscendAttnBackend)
        self.backend.page_size = 128
        self.backend.device = torch.device("cpu")
        self.backend.fia_mask = torch.zeros(2048, 2048, dtype=torch.bool)
        self.backend.forward_metadata = SimpleNamespace(
            block_tables=torch.zeros(1, 1, dtype=torch.int32)
        )
        self.layer = _fia_layer()
        self.q = torch.randn(self.PHYSICAL_ROWS, 16 * 128, dtype=torch.bfloat16)
        self.k_cache = torch.zeros(2163 * 128, 8 * 128, dtype=torch.bfloat16)
        self.v_cache = torch.zeros(2163 * 128, 8 * 128, dtype=torch.bfloat16)
        self.calls = []

    def _fia(self, query, key, value, **kwargs):
        record = dict(kwargs)
        record["query"] = query
        self.calls.append(record)
        return (
            torch.zeros(query.shape[0], query.shape[1], 128, dtype=query.dtype),
            None,
        )

    def _run(self, total_q_prev, total_q_next, prev_len, next_len):
        cp_meta = SimpleNamespace(
            total_q_prev_tokens=total_q_prev,
            total_q_next_tokens=total_q_next,
            actual_seq_q_prev_list=[prev_len],
            actual_seq_q_next_list=[next_len],
            kv_len_prev_list=[prev_len],
            kv_len_next_list=[total_q_prev + total_q_next],
        )
        forward_batch = SimpleNamespace(attn_cp_metadata=cp_meta)
        fake_ops = SimpleNamespace(
            npu=SimpleNamespace(npu_fused_infer_attention_score=self._fia)
        )
        with patch.object(torch, "ops", fake_ops):
            return self.backend.do_cp_attn_fia(
                self.q, self.k_cache, self.v_cache, self.layer, forward_batch
            )

    def test_pad_rows_are_trimmed_before_fia(self):
        # logical tokens are 2 + 1 = 3 of the 4 physical rows
        self._run(total_q_prev=2, total_q_next=1, prev_len=2, next_len=1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0]["query"].shape[0], 2)
        self.assertEqual(self.calls[1]["query"].shape[0], 1)

    def test_query_rows_match_cumulative_actual_seq_lengths(self):
        self._run(total_q_prev=2, total_q_next=1, prev_len=2, next_len=1)
        for call in self.calls:
            self.assertEqual(
                call["query"].shape[0], sum(call["actual_seq_lengths"])
            )

    def test_kv_lengths_are_forwarded(self):
        self._run(total_q_prev=2, total_q_next=1, prev_len=2, next_len=1)
        self.assertEqual(self.calls[0]["actual_seq_lengths_kv"], [2])
        self.assertEqual(self.calls[1]["actual_seq_lengths_kv"], [3])

    def test_output_keeps_physical_row_count(self):
        out = self._run(total_q_prev=2, total_q_next=1, prev_len=2, next_len=1)
        self.assertEqual(out.shape[0], self.PHYSICAL_ROWS)
        self.assertEqual(out.shape[1], 16 * 128)

    def test_no_padding_keeps_all_rows(self):
        # 2 + 2 == physical rows, so nothing is trimmed
        self._run(total_q_prev=2, total_q_next=2, prev_len=2, next_len=2)
        self.assertEqual(self.calls[0]["query"].shape[0], 2)
        self.assertEqual(self.calls[1]["query"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
