"""Unit tests for Double Sparsity TP-aware calibration slicing."""

import unittest
from pathlib import Path

import torch

from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
    channel_indices_for_runtime,
    parse_calibration_dict,
    parse_calibration_file,
    slice_for_tp,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


FIXTURE_PATH = Path(__file__).parent / "_fixtures" / "tiny_ds_calibration.json"


def _make_calib_with_n_kv_heads(num_kv_heads: int):
    """Build an in-memory calibration with the requested global head count."""
    head_dim = 32
    heavy = 8
    num_layers = 2
    raw = {
        "schema_version": 1,
        "model_arch": "TestModel",
        "model_name_or_path": "tiny",
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_heads": num_kv_heads * 2,
        "num_kv_heads": num_kv_heads,
        "heavy_channels": heavy,
        "channel_type": "k",
        "indexing": "global_kv_head_id",
        "channels": {},
    }
    # Encode (layer, head) into the channel values for easy verification.
    for layer in range(num_layers):
        rows = []
        for h in range(num_kv_heads):
            row = [(layer * 100 + h * 10 + c) % head_dim for c in range(heavy)]
            # Ensure uniqueness within the row
            seen = []
            for v in row:
                while v in seen:
                    v = (v + 1) % head_dim
                seen.append(v)
            rows.append(seen)
        raw["channels"][str(layer)] = rows
    return parse_calibration_dict(raw)


class TestSliceForTp(CustomTestCase):
    def test_tp1_returns_full(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        sliced = slice_for_tp(calib, tp_size=1, tp_rank=0)
        self.assertEqual(set(sliced.keys()), set(calib.channels.keys()))
        for k, v in sliced.items():
            self.assertTrue(torch.equal(v, calib.channels[k]))

    def test_tp2_partition_is_contiguous(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        rank0 = slice_for_tp(calib, tp_size=2, tp_rank=0)
        rank1 = slice_for_tp(calib, tp_size=2, tp_rank=1)
        # 4 KV heads / 2 ranks = 2 heads per rank
        for layer in calib.channels:
            self.assertEqual(rank0[layer].shape, (2, 8))
            self.assertEqual(rank1[layer].shape, (2, 8))
            full = calib.channels[layer]
            self.assertTrue(torch.equal(rank0[layer], full[:2]))
            self.assertTrue(torch.equal(rank1[layer], full[2:4]))

    def test_tp4_one_head_per_rank(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        for rank in range(4):
            sl = slice_for_tp(calib, tp_size=4, tp_rank=rank)
            for layer in calib.channels:
                self.assertEqual(sl[layer].shape, (1, 8))
                self.assertTrue(
                    torch.equal(sl[layer], calib.channels[layer][rank : rank + 1])
                )

    def test_tp8_with_8_kv_heads(self):
        calib = _make_calib_with_n_kv_heads(8)
        for rank in range(8):
            sl = slice_for_tp(calib, tp_size=8, tp_rank=rank)
            for layer in calib.channels:
                self.assertEqual(sl[layer].shape, (1, 8))
                self.assertTrue(
                    torch.equal(sl[layer], calib.channels[layer][rank : rank + 1])
                )

    def test_indivisible_tp_raises(self):
        # 4 KV heads with TP=3 is not partitionable along the head axis.
        calib = parse_calibration_file(FIXTURE_PATH)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            slice_for_tp(calib, tp_size=3, tp_rank=0)

    def test_rank_out_of_range(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        with self.assertRaisesRegex(ValueError, "tp_rank"):
            slice_for_tp(calib, tp_size=2, tp_rank=2)

    def test_tp_size_zero(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        with self.assertRaisesRegex(ValueError, "tp_size"):
            slice_for_tp(calib, tp_size=0, tp_rank=0)

    def test_slicing_does_not_alias_source(self):
        # Mutating the slice should not affect the original calibration tensor.
        calib = parse_calibration_file(FIXTURE_PATH)
        sliced = slice_for_tp(calib, tp_size=2, tp_rank=0)
        before = calib.channels[0].clone()
        sliced[0][0, 0] = -1
        self.assertTrue(torch.equal(calib.channels[0], before))


class TestChannelIndicesForRuntime(CustomTestCase):
    def test_dtype_and_device(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        out = channel_indices_for_runtime(
            calib, tp_size=2, tp_rank=1, device=torch.device("cpu")
        )
        for layer, t in out.items():
            self.assertEqual(t.dtype, torch.int32)
            self.assertEqual(t.device.type, "cpu")
            self.assertEqual(t.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
