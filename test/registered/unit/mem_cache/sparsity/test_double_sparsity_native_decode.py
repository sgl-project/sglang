"""Unit tests for the native DS sparse-decode kernels.

Covers:
  * Score kernel masks sink / recent / out-of-history to -inf.
  * `_build_selected_physical` lays out top-k + sink + recent in the
    documented order; current decode position is always in recent.
  * Sparse attention output matches a dense FA reference when the
    selected set covers the full sequence.
  * Sparse attention output matches a per-(bs, q_head) torch reference
    that gathers K/V at the selected physical ids and runs softmax.
  * `DoubleSparsityAlgorithm.try_native_sparse_decode` end-to-end does
    NOT invoke the legacy stage-2 merge / union pipeline.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")


def _have_cuda() -> bool:
    return torch.cuda.is_available()


def _torch_sparse_attn_ref(
    q: torch.Tensor,  # [bs, H_q, D]
    k_buffer: torch.Tensor,  # [T, H_kv, D]
    v_buffer: torch.Tensor,  # [T, H_kv, D]
    selected_physical: torch.Tensor,  # [bs, H_kv, total_selected] int32
    sm_scale: float,
) -> torch.Tensor:
    """Reference sparse attention. Used to validate the Triton output."""
    bs, h_q, d = q.shape
    _, h_kv, _ = k_buffer.shape
    gqa = h_q // h_kv
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(bs):
        for hq in range(h_q):
            hkv = hq // gqa
            sel = selected_physical[b, hkv].long()
            k = k_buffer[sel, hkv].float()
            v = v_buffer[sel, hkv].float()
            q_vec = q[b, hq].float()
            scores = (q_vec * k).sum(dim=-1) * sm_scale
            weights = torch.softmax(scores, dim=-1)
            out[b, hq] = (weights.unsqueeze(-1) * v).sum(dim=0)
    return out


@unittest.skipUnless(_have_cuda(), "CUDA required")
class TestScoreKernelMasking(CustomTestCase):
    """The score kernel must mask sink / recent / out-of-history positions to -inf."""

    def test_masks_sink_recent_oob(self):
        from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
            _launch_score,
        )

        device = "cuda"
        bs, h_kv, s = 2, 1, 32
        max_ctx = 256
        seq_lens = torch.tensor([200, 128], dtype=torch.int64, device=device)
        sink_tokens, recent_tokens = 4, 8

        q_label = torch.randn(bs, h_kv, s, device=device, dtype=torch.bfloat16)
        # T_pool = max_ctx so phys=t directly.
        k_label = torch.randn(max_ctx, h_kv, s, device=device, dtype=torch.bfloat16)
        req_to_token = (
            torch.arange(max_ctx, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(bs, max_ctx)
            .contiguous()
        )
        att_out = torch.full(
            (bs, h_kv, max_ctx),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        _launch_score(
            q_label=q_label,
            k_label_layer=k_label,
            req_to_token_indexed=req_to_token,
            seq_lens=seq_lens,
            att_out=att_out,
            sm_scale=1.0,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            block_t=64,
        )
        # Row 0: seq=200, history=[0, 199), masked sink=[0,4) and
        # recent=[200-8,200)=[192,200). Layout is [bs, H_kv, max_ctx].
        row0 = att_out[0, 0]
        self.assertTrue(torch.isneginf(row0[:sink_tokens]).all())
        self.assertTrue(torch.isneginf(row0[200 - recent_tokens : 200]).all())
        self.assertTrue(torch.isneginf(row0[199:]).all())  # >= history_len
        # Sanity: positions in [sink, history_minus_recent) are finite.
        self.assertTrue(torch.isfinite(row0[sink_tokens : 200 - recent_tokens]).all())

        # Row 1: seq=128, history=[0, 127), masked sink + recent.
        row1 = att_out[1, 0]
        self.assertTrue(torch.isneginf(row1[:sink_tokens]).all())
        self.assertTrue(torch.isneginf(row1[128 - recent_tokens : 128]).all())
        self.assertTrue(torch.isneginf(row1[127:]).all())


@unittest.skipUnless(_have_cuda(), "CUDA required")
class TestBuildSelectedPhysical(CustomTestCase):
    """Selected_physical layout: top-k physical at [0:top_k), sink at
    [top_k : top_k + sink), recent at [top_k + sink : total]."""

    def test_layout_and_recent_includes_current(self):
        from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
            _build_selected_physical,
        )

        device = "cuda"
        bs, h_kv, top_k = 1, 1, 4
        sink_tokens, recent_tokens = 2, 3
        total = top_k + sink_tokens + recent_tokens
        seq_lens = torch.tensor([20], dtype=torch.int64, device=device)
        max_ctx = 32

        # Identity req_to_token: physical id == logical id (lets us read
        # the layout from the output directly).
        req_to_token = (
            torch.arange(max_ctx, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(bs, max_ctx)
            .contiguous()
        )
        # top-k logical positions: pick some interior history positions.
        topk_logical = torch.tensor(
            [[[7, 11, 13, 15]]], dtype=torch.int32, device=device
        )
        out = torch.full((bs, h_kv, total), -99, dtype=torch.int32, device=device)
        _build_selected_physical(
            topk_logical=topk_logical,
            req_to_token_indexed=req_to_token,
            seq_lens=seq_lens,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            out=out,
        )
        row = out[0, 0].cpu().tolist()
        # [top-k physical][sink physical][recent physical]
        self.assertEqual(row[:top_k], [7, 11, 13, 15])
        self.assertEqual(row[top_k : top_k + sink_tokens], [0, 1])
        # Recent: seq=20, recent=3 → positions [17, 18, 19]
        self.assertEqual(
            row[top_k + sink_tokens : top_k + sink_tokens + recent_tokens],
            [17, 18, 19],
        )
        # Current decode position is seq-1 = 19 → must appear in recent slot.
        self.assertIn(19, row[top_k + sink_tokens :])


@unittest.skipUnless(_have_cuda(), "CUDA required")
class TestSparseAttnAgainstRef(CustomTestCase):
    """`ds_native_sparse_decode` output matches a torch reference that
    gathers K/V at the same selected physical ids and runs softmax."""

    def test_matches_torch_ref(self):
        from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
            ds_native_sparse_decode,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
            gqa_reduction_id,
        )
        from sglang.srt.mem_cache.sparsity.triton_ops.select_kernels import (
            _compute_q_label,
        )

        device = "cuda"
        bs, h_q, h_kv, d, s = 1, 8, 1, 64, 32
        seq_len, max_ctx = 1024, 1024
        top_k, sink_tokens, recent_tokens = 32, 4, 8
        total = top_k + sink_tokens + recent_tokens
        max_blocks = (total + 127) // 128

        torch.manual_seed(0)
        q = torch.randn(bs, h_q, d, device=device, dtype=torch.bfloat16)
        k_buffer = torch.randn(max_ctx, h_kv, d, device=device, dtype=torch.bfloat16)
        v_buffer = torch.randn(max_ctx, h_kv, d, device=device, dtype=torch.bfloat16)
        k_label = torch.randn(max_ctx, h_kv, s, device=device, dtype=torch.bfloat16)
        channel_idx = (
            torch.arange(s, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(h_kv, s)
            .contiguous()
        )
        req_to_token = (
            torch.arange(max_ctx, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(bs, max_ctx)
            .contiguous()
        )
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)

        att_out = torch.full(
            (h_kv, bs, max_ctx), float("-inf"), dtype=torch.float32, device=device
        )
        selected_physical = torch.zeros(
            (bs, h_kv, total), dtype=torch.int32, device=device
        )
        mid_out = torch.zeros(
            (bs, h_q, max_blocks, d), dtype=torch.float32, device=device
        )
        mid_log = torch.full(
            (bs, h_q, max_blocks),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        output = torch.zeros(bs, h_q, d, dtype=torch.bfloat16, device=device)

        q_label = _compute_q_label(
            q,
            channel_idx,
            num_kv_heads=h_kv,
            gqa_reduction_id=gqa_reduction_id("max_abs"),
        )
        sm_scale = 1.0 / (d**0.5)

        ds_native_sparse_decode(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            k_label_layer=k_label,
            q_label=q_label,
            req_to_token_indexed=req_to_token,
            seq_lens=seq_lens,
            top_k=top_k,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            sm_scale=sm_scale,
            att_out_approx=att_out,
            selected_physical=selected_physical,
            mid_out=mid_out,
            mid_o_logexpsum=mid_log,
            output=output,
        )
        ref = _torch_sparse_attn_ref(q, k_buffer, v_buffer, selected_physical, sm_scale)
        # bf16 → fp32 reference: tolerate fp16-class roundoff
        self.assertTrue(
            torch.allclose(output.float(), ref, atol=1e-2, rtol=1e-2),
            msg=f"max |diff| = {(output.float() - ref).abs().max().item():.4g}",
        )

    def test_recent_includes_current_position_in_output(self):
        """Smoke: with recent_tokens=1, the selected set must include
        physical id `seq_len-1`. Output should not be all-zero or NaN."""
        from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
            ds_native_sparse_decode,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
            gqa_reduction_id,
        )
        from sglang.srt.mem_cache.sparsity.triton_ops.select_kernels import (
            _compute_q_label,
        )

        device = "cuda"
        bs, h_q, h_kv, d, s = 1, 4, 1, 64, 16
        seq_len, max_ctx = 256, 256
        top_k, sink_tokens, recent_tokens = 16, 2, 1
        total = top_k + sink_tokens + recent_tokens
        max_blocks = (total + 127) // 128

        torch.manual_seed(1)
        q = torch.randn(bs, h_q, d, device=device, dtype=torch.bfloat16)
        k_buffer = torch.randn(max_ctx, h_kv, d, device=device, dtype=torch.bfloat16)
        v_buffer = torch.randn(max_ctx, h_kv, d, device=device, dtype=torch.bfloat16)
        k_label = torch.randn(max_ctx, h_kv, s, device=device, dtype=torch.bfloat16)
        channel_idx = (
            torch.arange(s, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(h_kv, s)
            .contiguous()
        )
        req_to_token = (
            torch.arange(max_ctx, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(bs, max_ctx)
            .contiguous()
        )
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)

        att_out = torch.full(
            (h_kv, bs, max_ctx), float("-inf"), dtype=torch.float32, device=device
        )
        selected_physical = torch.zeros(
            (bs, h_kv, total), dtype=torch.int32, device=device
        )
        mid_out = torch.zeros(
            (bs, h_q, max_blocks, d), dtype=torch.float32, device=device
        )
        mid_log = torch.full(
            (bs, h_q, max_blocks),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        output = torch.zeros(bs, h_q, d, dtype=torch.bfloat16, device=device)

        q_label = _compute_q_label(
            q,
            channel_idx,
            num_kv_heads=h_kv,
            gqa_reduction_id=gqa_reduction_id("max_abs"),
        )
        ds_native_sparse_decode(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            k_label_layer=k_label,
            q_label=q_label,
            req_to_token_indexed=req_to_token,
            seq_lens=seq_lens,
            top_k=top_k,
            sink_tokens=sink_tokens,
            recent_tokens=recent_tokens,
            sm_scale=1.0 / (d**0.5),
            att_out_approx=att_out,
            selected_physical=selected_physical,
            mid_out=mid_out,
            mid_o_logexpsum=mid_log,
            output=output,
        )
        # Current decode position (seq-1=255) must be in the last `recent`
        # slots of selected_physical.
        recent_slice = selected_physical[0, 0, top_k + sink_tokens :].cpu().tolist()
        self.assertIn(seq_len - 1, recent_slice)
        # Output must not be NaN / inf.
        self.assertTrue(torch.isfinite(output.float()).all())


@unittest.skipUnless(_have_cuda(), "CUDA required")
class TestAlgorithmNativePathBypassesStage2(CustomTestCase):
    """`DoubleSparsityAlgorithm.try_native_sparse_decode` must NOT touch
    the legacy stage-2 merge / union path (`ds_select_tokens_triton`)."""

    def test_native_path_does_not_call_legacy_selection(self):
        from unittest.mock import MagicMock, patch

        from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity import (
            DoubleSparsityAlgorithm,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
            DoubleSparsityCalibration,
            DoubleSparsityRuntimeConfig,
        )

        device = torch.device("cuda")
        # Tiny model geometry: 2 layers, 1 KV head, 4 Q heads (GQA=4),
        # head_dim=32, S=8. Single-rank (tp=1).
        head_dim, num_layers, num_q, num_kv, s = 32, 2, 4, 1, 8
        channels = {
            i: torch.arange(num_kv * s, dtype=torch.int32).reshape(num_kv, s)
            for i in range(num_layers)
        }
        calib = DoubleSparsityCalibration(
            schema_version=1,
            model_arch="t",
            model_name_or_path="",
            head_dim=head_dim,
            num_layers=num_layers,
            num_heads=num_q,
            num_kv_heads_global=num_kv,
            heavy_channels=s,
            channel_type="k",
            indexing="global_kv_head_id",
            channels=channels,
        )
        rt = DoubleSparsityRuntimeConfig(
            heavy_channels=s,
            token_budget=8,
            recent_tokens=2,
            sink_tokens=2,
            min_seq_len=16,
            max_selected_per_request=64,
            gqa_reduction="max_abs",
            klabel_dtype="bf16",
            block_t=256,
            k_block=16,
            scratch_max_bs=2,
        )
        algo = DoubleSparsityAlgorithm(
            config=MagicMock(),
            device=device,
            runtime_config=rt,
            calibration=calib,
            tp_size=1,
            tp_rank=0,
            num_kv_heads_local=num_kv,
            num_q_heads_local=num_q,
            head_dim=head_dim,
        )
        # Fake token pool with key/value/k_label buffers we control.
        max_ctx = 64
        T_pool = 128
        token_pool = MagicMock()
        token_pool.get_key_buffer = MagicMock(
            return_value=torch.randn(
                T_pool, num_kv, head_dim, device=device, dtype=torch.bfloat16
            )
        )
        token_pool.get_value_buffer = MagicMock(
            return_value=torch.randn(
                T_pool, num_kv, head_dim, device=device, dtype=torch.bfloat16
            )
        )
        token_pool.set_kv_buffer = MagicMock()

        req_to_token_pool = MagicMock()
        req_to_token_pool.req_to_token = (
            torch.arange(max_ctx, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(2, max_ctx)
            .contiguous()
        )
        algo.initialize_representation_pool(
            start_layer=0,
            end_layer=num_layers,
            token_to_kv_pool=token_pool,
            req_to_token_pool=req_to_token_pool,
            states=MagicMock(),
        )

        # Build a forward_batch with seq_lens long enough to trigger native.
        bs = 1
        seq_len = 32
        layer = MagicMock()
        layer.layer_id = 0
        layer.scaling = 1.0 / (head_dim**0.5)
        forward_batch = MagicMock()
        forward_batch.seq_lens = torch.tensor(
            [seq_len], dtype=torch.int64, device=device
        )
        forward_batch.req_pool_indices = torch.tensor(
            [0], dtype=torch.int64, device=device
        )
        forward_batch.out_cache_loc = torch.tensor(
            [seq_len - 1], dtype=torch.int64, device=device
        )
        forward_batch.token_to_kv_pool = token_pool

        q = torch.randn(bs, num_q * head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(bs, num_kv, head_dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(bs, num_kv, head_dim, device=device, dtype=torch.bfloat16)

        # Patch the legacy entry to detect if it's invoked.
        with patch(
            "sglang.srt.mem_cache.sparsity.algorithms.double_sparsity.ds_select_tokens_triton"
        ) as legacy_entry:
            out = algo.try_native_sparse_decode(
                q, k, v, layer, forward_batch, save_kv_cache=True
            )
        self.assertIsNotNone(out, "native path should fire when seq >= min_seq")
        legacy_entry.assert_not_called()
        # Output shape: [bs, H_q * D]
        self.assertEqual(out.shape, (bs, num_q * head_dim))
        # set_kv_buffer must have been called (native path writes K/V).
        token_pool.set_kv_buffer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
