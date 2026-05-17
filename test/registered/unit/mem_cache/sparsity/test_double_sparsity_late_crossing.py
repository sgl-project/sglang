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


class TestCoordinatorThreadsEffectiveMask(CustomTestCase):
    """Pin the framework integration: the SAME effective mask reaches
    BOTH retrieve_topk and adapt_for_attn_metadata. A regression where
    the coordinator passes default_mask to the adaptor (the v1.1-2 bug
    being fixed) must fail this test.
    """

    def test_late_crossing_mask_reaches_adaptor(self):
        from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import (
            BaseSparseAlgorithm,
        )
        from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor
        from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
            SparseCoordinator,
        )

        # Fake algorithm: returns a mask flipped from default_mask, captures the
        # sparse_mask it receives in retrieve_topk.
        captured = {"retrieve_topk_mask": None, "adaptor_mask": None}

        class _FakeAlgo(BaseSparseAlgorithm):
            def __init__(self):
                self.config = SparseConfig()
                self.device = torch.device("cpu")
                self.req_to_token_pool = None
                self.states = None

            def initialize_representation_pool(self, *a, **k):
                pass

            def effective_sparse_mask(self, fb, rpi, default_mask):
                # The override the coordinator must respect.
                return ~default_mask

            def retrieve_topk(self, *, sparse_mask, **kwargs):
                captured["retrieve_topk_mask"] = sparse_mask.clone()
                bs = sparse_mask.shape[0]
                return (
                    torch.full((bs, 4), -1, dtype=torch.int32),
                    torch.zeros(bs, dtype=torch.int32),
                )

        class _FakeAdaptor(BackendAdaptor):
            def adapt_for_attn_metadata(self, *, sparse_mask, **kwargs):
                captured["adaptor_mask"] = sparse_mask.clone()
                return None

        # Minimal mocks for SparseCoordinator's __init__.
        bs = 3
        max_pool_size = 8
        req_to_token_pool = MagicMock()
        req_to_token_pool.req_to_token = torch.zeros(
            max_pool_size, 16, dtype=torch.int32
        )
        req_to_token_pool.max_context_len = 16
        token_to_kv_pool = MagicMock()
        token_to_kv_pool.get_key_buffer.return_value = torch.zeros(64, 1, 16)

        sc_config = SparseConfig(min_sparse_prompt_len=4096)
        coord = SparseCoordinator(
            config=sc_config,
            algorithm=_FakeAlgo(),
            backend_adaptor=_FakeAdaptor(torch.device("cpu")),
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            start_layer=0,
            end_layer=1,
            device=torch.device("cpu"),
        )
        # Default mask all-False (admission-time prompt_lens=0 < 4096).
        coord.states.prompt_lens[:bs] = 0

        layer = MagicMock()
        layer.layer_id = 0
        fb = MagicMock()
        fb.req_pool_indices = torch.arange(bs)
        fb.seq_lens = torch.tensor([100, 5000, 6000])

        coord._handle_sparse_retrieve(
            query=torch.zeros(bs, 4, 16),
            layer=layer,
            forward_batch=fb,
            attn_metadata=None,
        )

        # Default mask was [F, F, F]; algorithm flipped to [T, T, T].
        # BOTH retrieve_topk and the adaptor must have seen [T, T, T].
        self.assertIsNotNone(captured["retrieve_topk_mask"])
        self.assertIsNotNone(captured["adaptor_mask"])
        self.assertTrue(
            torch.equal(
                captured["retrieve_topk_mask"], torch.tensor([True, True, True])
            )
        )
        self.assertTrue(
            torch.equal(captured["adaptor_mask"], torch.tensor([True, True, True]))
        )
        # Same tensor object — proves the coordinator threads ONE mask, not two.
        self.assertTrue(
            torch.equal(captured["retrieve_topk_mask"], captured["adaptor_mask"])
        )


if __name__ == "__main__":
    unittest.main()
