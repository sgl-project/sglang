"""DS selection scratch preallocation regression test (Commit 0).

Pins the contract that `DoubleSparsityAlgorithm` preallocates the
six selection-pipeline scratch tensors at `initialize_representation_pool`
time and that `retrieve_topk` reuses those buffers on every decode step
(no fresh output-tensor allocation on the production CUDA path).

CPU portion (allocation correctness):
  - Before init, all six private scratch attrs are None.
  - After init, all six are allocated with the expected shapes/dtypes
    derived from `runtime_config` and `req_to_token_pool.req_to_token`.

CUDA portion (buffer reuse correctness):
  - `retrieve_topk` returns tensors whose `data_ptr()` matches the
    algorithm's preallocated `_selected_logical` / `_valid_lengths`.
  - A second `retrieve_topk` call returns the same tensors (no
    reallocation between steps).
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity import (
    DoubleSparsityAlgorithm,
)
from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
    DoubleSparsityRuntimeConfig,
    parse_calibration_file,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import SparseConfig
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Mixed CPU+CUDA file; the CUDA tests skip cleanly when CUDA is missing.
register_cuda_ci(est_time=4, suite="stage-b-test-1-gpu-small")


FIXTURE_PATH = Path(__file__).parent / "_fixtures" / "tiny_ds_calibration.json"


def _build_algorithm(device, *, scratch_max_bs=2, max_ctx=64, num_reqs=None):
    """Build a DoubleSparsityAlgorithm wired to mock pools."""
    calib = parse_calibration_file(FIXTURE_PATH)
    rt = DoubleSparsityRuntimeConfig(
        heavy_channels=calib.heavy_channels,  # 8 in the fixture
        token_budget=32,
        recent_tokens=4,
        sink_tokens=2,
        min_seq_len=16,
        max_selected_per_request=48,
        gqa_reduction="max_abs",
        klabel_dtype="bf16",
        block_t=256,
        k_block=16,
        scratch_max_bs=scratch_max_bs,
    )
    sc = SparseConfig(algorithm="double_sparsity", backend="fa3", page_size=1)
    algo = DoubleSparsityAlgorithm(
        sc,
        device,
        runtime_config=rt,
        calibration=calib,
        tp_size=1,
        tp_rank=0,
        num_kv_heads_local=calib.num_kv_heads_global,
        num_q_heads_local=calib.num_heads,
        head_dim=calib.head_dim,
    )

    # Minimal mock pools — only the attributes the algorithm actually reads.
    num_tokens_in_pool = 4096
    fake_key_buffer = torch.zeros(
        num_tokens_in_pool,
        calib.num_kv_heads_global,
        calib.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: fake_key_buffer,
    )
    # req_to_token rows are independent of scratch_max_bs: the test may pass a
    # `bs > scratch_max_bs` batch on purpose (oversize-batch guard test).
    req_rows = num_reqs if num_reqs is not None else scratch_max_bs
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(
            req_rows, max_ctx, dtype=torch.int32, device=device
        ),
    )

    algo.initialize_representation_pool(
        start_layer=0,
        end_layer=calib.num_layers,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        states=None,
    )
    return algo, rt, calib, req_to_token_pool


class TestSelectionScratchAllocationCPU(CustomTestCase):
    """Allocation correctness — runs on CPU."""

    def test_scratch_is_none_before_init(self):
        calib = parse_calibration_file(FIXTURE_PATH)
        rt = DoubleSparsityRuntimeConfig(
            heavy_channels=calib.heavy_channels,
            token_budget=32,
            recent_tokens=4,
            sink_tokens=2,
            min_seq_len=16,
            max_selected_per_request=48,
            gqa_reduction="max_abs",
            klabel_dtype="bf16",
            block_t=256,
            k_block=16,
            scratch_max_bs=2,
        )
        sc = SparseConfig(algorithm="double_sparsity", backend="fa3", page_size=1)
        algo = DoubleSparsityAlgorithm(
            sc,
            torch.device("cpu"),
            runtime_config=rt,
            calibration=calib,
            tp_size=1,
            tp_rank=0,
            num_kv_heads_local=calib.num_kv_heads_global,
            num_q_heads_local=calib.num_heads,
            head_dim=calib.head_dim,
        )
        # Pre-init: all six private scratch attrs must be None so the
        # CUDA path knows to fall back to the unthreaded variant (which
        # keeps any direct-retrieve_topk tests on existing fixtures green).
        self.assertIsNone(algo._block_topk_logical)
        self.assertIsNone(algo._block_topk_scores)
        self.assertIsNone(algo._merged_logical)
        self.assertIsNone(algo._merged_scores)
        self.assertIsNone(algo._selected_logical)
        self.assertIsNone(algo._valid_lengths)

    def test_scratch_shapes_after_init(self):
        device = torch.device("cpu")
        scratch_max_bs = 3
        max_ctx = 128
        algo, rt, _, _ = _build_algorithm(
            device, scratch_max_bs=scratch_max_bs, max_ctx=max_ctx
        )
        h_kv = algo.num_kv_heads_local
        num_blocks = (max_ctx + rt.block_t - 1) // rt.block_t
        effective_budget = min(rt.token_budget, num_blocks * rt.k_block)

        self.assertEqual(
            tuple(algo._block_topk_logical.shape),
            (scratch_max_bs, h_kv, num_blocks, rt.k_block),
        )
        self.assertEqual(algo._block_topk_logical.dtype, torch.int32)
        self.assertEqual(
            tuple(algo._block_topk_scores.shape),
            (scratch_max_bs, h_kv, num_blocks, rt.k_block),
        )
        self.assertEqual(algo._block_topk_scores.dtype, torch.float32)
        self.assertEqual(
            tuple(algo._merged_logical.shape),
            (scratch_max_bs, h_kv, effective_budget),
        )
        self.assertEqual(algo._merged_logical.dtype, torch.int32)
        self.assertEqual(
            tuple(algo._merged_scores.shape),
            (scratch_max_bs, h_kv, effective_budget),
        )
        self.assertEqual(algo._merged_scores.dtype, torch.float32)
        self.assertEqual(
            tuple(algo._selected_logical.shape),
            (scratch_max_bs, rt.max_selected_per_request),
        )
        self.assertEqual(algo._selected_logical.dtype, torch.int32)
        self.assertEqual(tuple(algo._valid_lengths.shape), (scratch_max_bs,))
        self.assertEqual(algo._valid_lengths.dtype, torch.int32)


class TestSelectionScratchReuseCUDA(CustomTestCase):
    """Buffer-reuse correctness — runs only with CUDA available."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for selection-scratch reuse test")
        self.device = torch.device("cuda")

    def _make_inputs(self, algo, bs, max_ctx, seq_len):
        """Build queries / req_pool_indices / seq_lens / K_label state."""
        torch.manual_seed(0)
        h_q = algo.num_q_heads_local
        d = algo.head_dim
        queries = torch.randn(bs, h_q, d, dtype=torch.bfloat16, device=self.device)
        req_pool_indices = torch.arange(bs, dtype=torch.int64, device=self.device)
        # All requests have the same seq_len for simplicity; >= min_seq_len so
        # we exercise the sparse branch (not the dense fallback).
        seq_lens = torch.full(
            (bs,), seq_len, dtype=torch.int64, device=self.device
        )

        # Populate req_to_token rows so logical positions map to valid pool ids.
        # Pool capacity in the fixture: 4096 (see _build_algorithm).
        for b in range(bs):
            algo.req_to_token_pool.req_to_token[b, :seq_len] = torch.arange(
                seq_len, dtype=torch.int32, device=self.device
            )

        # K_label for layer 0 needs *something* finite at the positions the
        # selection will score (avoid NaNs). Zero is fine.
        # Already zero-initialized in initialize_representation_pool.
        forward_batch = SimpleNamespace(seq_lens=seq_lens)
        return queries, req_pool_indices, forward_batch

    def test_retrieve_topk_returns_preallocated_buffers(self):
        scratch_max_bs = 2
        max_ctx = 256
        algo, rt, _, _ = _build_algorithm(
            self.device, scratch_max_bs=scratch_max_bs, max_ctx=max_ctx
        )
        bs = 2  # == scratch_max_bs → narrow returns the full preallocated row.
        seq_len = 64  # >= min_seq_len (16); exercise sparse branch.

        queries, req_pool_indices, forward_batch = self._make_inputs(
            algo, bs, max_ctx, seq_len
        )

        sparse_mask = torch.ones(bs, dtype=torch.bool, device=self.device)
        selected, valid = algo.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
        )

        # The returned tensors must share storage with the algorithm's
        # preallocated scratch — this is the contract that proves the
        # production CUDA path performs zero output-tensor allocation.
        self.assertEqual(
            selected.data_ptr(),
            algo._selected_logical.data_ptr(),
            "retrieve_topk must return a view of the preallocated _selected_logical",
        )
        self.assertEqual(
            valid.data_ptr(),
            algo._valid_lengths.data_ptr(),
            "retrieve_topk must return a view of the preallocated _valid_lengths",
        )
        # Shape sanity: first axis matches the narrowed batch size.
        self.assertEqual(selected.shape[0], bs)
        self.assertEqual(valid.shape[0], bs)
        self.assertEqual(selected.shape[1], rt.max_selected_per_request)

    def test_retrieve_topk_reuses_buffers_across_calls(self):
        """Two back-to-back calls must reuse the same storage."""
        scratch_max_bs = 2
        max_ctx = 256
        algo, _, _, _ = _build_algorithm(
            self.device, scratch_max_bs=scratch_max_bs, max_ctx=max_ctx
        )
        bs = 1  # sub-batch — exercises the .narrow(0, 0, bs) path.
        seq_len = 64

        queries, req_pool_indices, forward_batch = self._make_inputs(
            algo, bs, max_ctx, seq_len
        )
        sparse_mask = torch.ones(bs, dtype=torch.bool, device=self.device)

        sel1, val1 = algo.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
        )
        sel2, val2 = algo.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool_indices,
            sparse_mask=sparse_mask,
            forward_batch=forward_batch,
        )

        self.assertEqual(sel1.data_ptr(), sel2.data_ptr())
        self.assertEqual(val1.data_ptr(), val2.data_ptr())
        self.assertEqual(sel1.data_ptr(), algo._selected_logical.data_ptr())
        self.assertEqual(val1.data_ptr(), algo._valid_lengths.data_ptr())

    def test_oversize_batch_raises(self):
        """bs > scratch_max_bs must error cleanly (admission-cap mismatch)."""
        scratch_max_bs = 1
        max_ctx = 256
        bs = 2
        # Mock req_to_token must have >= bs rows; scratch is sized to 1 row.
        algo, _, _, _ = _build_algorithm(
            self.device,
            scratch_max_bs=scratch_max_bs,
            max_ctx=max_ctx,
            num_reqs=bs,
        )
        seq_len = 64
        queries, req_pool_indices, forward_batch = self._make_inputs(
            algo, bs, max_ctx, seq_len
        )
        sparse_mask = torch.ones(bs, dtype=torch.bool, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "scratch_max_bs"):
            algo.retrieve_topk(
                queries=queries,
                layer_id=0,
                req_pool_indices=req_pool_indices,
                sparse_mask=sparse_mask,
                forward_batch=forward_batch,
            )


if __name__ == "__main__":
    unittest.main()
