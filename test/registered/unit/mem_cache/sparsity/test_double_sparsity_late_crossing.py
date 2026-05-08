"""Late-crossing seq_len gating (v1.1).

A DS request admitted below `min_seq_len` (so prompt_lens-based gating
says "dense") that crosses `min_seq_len` during decode must flip to
sparse. v1's per-prompt-len gate kept it dense forever; v1.1 introduces
the `effective_sparse_mask` framework hook that DS overrides to gate
on current `forward_batch.seq_lens`.

This test pins both layers of the fix:
  1. `DoubleSparsityAlgorithm.effective_sparse_mask` returns True for a
     row whose `seq_lens >= min_seq_len`, regardless of `prompt_lens`.
  2. `BaseSparseAlgorithm.effective_sparse_mask` default returns the
     coordinator's mask unchanged — Quest-style algorithms keep their
     prompt-lens-based gating.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity import (
    DoubleSparsityAlgorithm,
)
from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
    DoubleSparsityRuntimeConfig,
    parse_calibration_file,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


FIXTURE_PATH = Path(__file__).parent / "_fixtures" / "tiny_ds_calibration.json"


def _build_ds_algo(min_seq_len: int = 4096):
    calib = parse_calibration_file(FIXTURE_PATH)
    rt = DoubleSparsityRuntimeConfig(
        heavy_channels=8,
        token_budget=16,
        recent_tokens=2,
        sink_tokens=1,
        min_seq_len=min_seq_len,
        max_selected_per_request=8192,
        gqa_reduction="max_abs",
        klabel_dtype="bf16",
    )
    sc = SparseConfig(algorithm="double_sparsity", backend="fa3", page_size=1)
    return DoubleSparsityAlgorithm(
        sc,
        torch.device("cpu"),
        runtime_config=rt,
        calibration=calib,
        tp_size=1,
        tp_rank=0,
        num_kv_heads_local=4,
        num_q_heads_local=8,
        head_dim=16,
    )


class TestDsLateCrossing(CustomTestCase):
    def test_late_crossing_seq_len_flips_to_sparse(self):
        algo = _build_ds_algo(min_seq_len=4096)
        fb = MagicMock()
        fb.seq_lens = torch.tensor([5000], dtype=torch.int64)
        # Default mask comes from the coordinator's prompt_lens path, which
        # would have said False here (admission-time prompt_len=100 < 4096).
        default_mask = torch.tensor([False])
        req_pool_indices = torch.tensor([0])
        eff = algo.effective_sparse_mask(fb, req_pool_indices, default_mask)
        self.assertTrue(eff[0].item())

    def test_short_seq_stays_dense(self):
        algo = _build_ds_algo(min_seq_len=4096)
        fb = MagicMock()
        fb.seq_lens = torch.tensor([200], dtype=torch.int64)
        default_mask = torch.tensor([True])  # would have been True; DS overrides
        eff = algo.effective_sparse_mask(fb, torch.tensor([0]), default_mask)
        self.assertFalse(eff[0].item())

    def test_mixed_batch(self):
        algo = _build_ds_algo(min_seq_len=4096)
        fb = MagicMock()
        fb.seq_lens = torch.tensor([100, 5000, 4095, 4096], dtype=torch.int64)
        default = torch.tensor([True, False, True, False])
        eff = algo.effective_sparse_mask(fb, torch.arange(4), default)
        # Only seq_lens >= 4096 should be sparse.
        self.assertEqual(eff.tolist(), [False, True, False, True])


class TestBaseDefaultPassthrough(CustomTestCase):
    """Quest/etc. inherit the default impl, which must return default_mask
    unchanged so we don't change their behavior."""

    def test_default_returns_default_mask(self):
        # BaseSparseAlgorithm is abstract because of retrieve_topk; build a
        # tiny concrete subclass that inherits the default effective_sparse_mask.
        class _Concrete(BaseSparseAlgorithm):
            def retrieve_topk(self, *a, **k):
                raise NotImplementedError

        algo = _Concrete(SparseConfig(), torch.device("cpu"))
        fb = MagicMock()
        default = torch.tensor([True, False, True])
        out = algo.effective_sparse_mask(fb, torch.arange(3), default)
        self.assertTrue(torch.equal(out, default))


if __name__ == "__main__":
    unittest.main()
