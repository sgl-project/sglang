"""Parity tests for the DS native top-k selector backends.

Each backend (torch / flashinfer_topk_page_table / sgl_fast_topk_transform)
must:

  * Produce the same SET of selected physical ids per ``(bs, h_kv)`` as
    the torch baseline (ordering inside the top-k slot is allowed to
    differ).
  * Lay out sink + recent in the documented slot order (top-k | sink |
    recent).

FlashInfer / sgl_kernel are skipped when the optional dependency is not
available; if the user explicitly requests them via the runtime config
flag, ``make_selector`` raises with a clear message (covered by a
separate construction test).
"""

from __future__ import annotations

import importlib.util
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")


def _have_cuda() -> bool:
    return torch.cuda.is_available()


def _have_flashinfer() -> bool:
    return importlib.util.find_spec("flashinfer") is not None


def _have_sgl_kernel_fast_topk() -> bool:
    spec = importlib.util.find_spec("sgl_kernel")
    if spec is None:
        return False
    import sgl_kernel

    return hasattr(sgl_kernel, "fast_topk_transform_fused")


def _make_inputs(
    bs: int,
    h_kv: int,
    max_ctx: int,
    seq_len: int,
    sink_tokens: int,
    recent_tokens: int,
    device: torch.device,
    seed: int = 0,
):
    """Identity req_to_token (phys == logical) so we can compare sets
    in logical-space directly even though backends emit physical."""
    torch.manual_seed(seed)
    att_out = torch.randn(bs, h_kv, max_ctx, dtype=torch.float32, device=device)
    # Mask sink / recent / oob to -inf (matches what the score kernel
    # does in production). The selector backends rely on this so they
    # can't pick masked positions.
    att_out[..., :sink_tokens] = float("-inf")
    history_len = seq_len - 1
    att_out[..., seq_len - recent_tokens : seq_len] = float("-inf")
    att_out[..., history_len:] = float("-inf")
    req_to_token_indexed = (
        torch.arange(max_ctx, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(bs, max_ctx)
        .contiguous()
    )
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int64, device=device)
    return att_out, req_to_token_indexed, seq_lens


def _run_select(backend_name: str, att_out, r2t, seq_lens, top_k, sink, recent):
    from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
        make_selector,
    )

    bs, h_kv, _ = att_out.shape
    total = top_k + sink + recent
    out = torch.zeros((bs, h_kv, total), dtype=torch.int32, device=att_out.device)
    selector = make_selector(backend_name, max_bs=bs, h_kv=h_kv, device=att_out.device)
    selector.select(
        att_out_approx=att_out,
        req_to_token_indexed=r2t,
        seq_lens=seq_lens,
        top_k=top_k,
        sink_tokens=sink,
        recent_tokens=recent,
        out=out,
    )
    return out


@unittest.skipUnless(_have_cuda(), "CUDA required")
class TestSelectorParity(CustomTestCase):
    """torch vs FlashInfer / SGL backends must agree on the SET of
    selected top-k physical ids and on the sink/recent layout."""

    def _assert_layouts_match(self, out_a, out_b, top_k, sink, recent):
        bs, h_kv, _ = out_a.shape
        for b in range(bs):
            for h in range(h_kv):
                # Compare top-k as sets (order may differ across backends).
                set_a = set(out_a[b, h, :top_k].cpu().tolist())
                set_b = set(out_b[b, h, :top_k].cpu().tolist())
                self.assertEqual(
                    set_a,
                    set_b,
                    f"top-k set mismatch at (b={b}, h={h})",
                )
                # Sink + recent are deterministic across backends.
                self.assertTrue(
                    torch.equal(
                        out_a[b, h, top_k : top_k + sink],
                        out_b[b, h, top_k : top_k + sink],
                    ),
                    f"sink mismatch at (b={b}, h={h})",
                )
                self.assertTrue(
                    torch.equal(
                        out_a[b, h, top_k + sink : top_k + sink + recent],
                        out_b[b, h, top_k + sink : top_k + sink + recent],
                    ),
                    f"recent mismatch at (b={b}, h={h})",
                )

    @unittest.skipUnless(_have_flashinfer(), "FlashInfer required")
    def test_flashinfer_matches_torch(self):
        device = torch.device("cuda")
        bs, h_kv, max_ctx, seq_len = 2, 1, 256, 128
        sink, recent, top_k = 4, 8, 16
        att_out, r2t, seq_lens = _make_inputs(
            bs, h_kv, max_ctx, seq_len, sink, recent, device, seed=0
        )
        out_torch = _run_select("torch", att_out, r2t, seq_lens, top_k, sink, recent)
        out_fi = _run_select(
            "flashinfer_topk_page_table",
            att_out,
            r2t,
            seq_lens,
            top_k,
            sink,
            recent,
        )
        self._assert_layouts_match(out_torch, out_fi, top_k, sink, recent)

    @unittest.skipUnless(_have_flashinfer(), "FlashInfer required")
    def test_flashinfer_multi_head_kv(self):
        """h_kv > 1 exercises the row_to_batch mapping path."""
        device = torch.device("cuda")
        bs, h_kv, max_ctx, seq_len = 2, 4, 256, 128
        sink, recent, top_k = 2, 4, 16
        att_out, r2t, seq_lens = _make_inputs(
            bs, h_kv, max_ctx, seq_len, sink, recent, device, seed=1
        )
        out_torch = _run_select("torch", att_out, r2t, seq_lens, top_k, sink, recent)
        out_fi = _run_select(
            "flashinfer_topk_page_table",
            att_out,
            r2t,
            seq_lens,
            top_k,
            sink,
            recent,
        )
        self._assert_layouts_match(out_torch, out_fi, top_k, sink, recent)

    def test_sgl_backend_currently_not_implemented(self):
        """`sgl_fast_topk_transform` is registered but not yet wired; the
        installed sgl_kernel.fast_topk_transform_fused signature lacks the
        row_to_batch parameter we need for the per-h_kv broadcast.
        ``make_selector`` must fail loud rather than constructing a
        crippled selector. Once the kernel exposes a compatible API
        (or we land the score-row-duplication adaptor), flip this to a
        parity test."""
        from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
            make_selector,
        )

        with self.assertRaises(NotImplementedError):
            make_selector(
                "sgl_fast_topk_transform",
                max_bs=1,
                h_kv=1,
                device=torch.device("cpu"),
            )


@unittest.skipUnless(_have_cuda(), "CUDA required")
class TestSelectorConstruction(CustomTestCase):
    """``make_selector`` resolves backend names and fails loud on unknown
    names / unavailable deps."""

    def test_torch_always_available(self):
        from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
            make_selector,
        )

        sel = make_selector("torch")
        self.assertEqual(sel.name, "torch")

    def test_unknown_backend_raises(self):
        from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
            make_selector,
        )

        with self.assertRaises(ValueError):
            make_selector("not_a_real_backend")

    def test_jit_fused_not_implemented(self):
        from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
            make_selector,
        )

        with self.assertRaises(NotImplementedError):
            make_selector(
                "jit_fused_selector",
                max_bs=1,
                h_kv=1,
                device=torch.device("cpu"),
            )

    def test_flashinfer_topk_ceiling_validated_in_runtime_config(self):
        """``DoubleSparsityRuntimeConfig.validate()`` must reject the
        combination of ``selector_backend='flashinfer_topk_page_table'``
        with ``token_budget`` above the FlashInfer top-k ceiling."""
        from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
            DoubleSparsityRuntimeConfig,
        )
        from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
            FLASHINFER_TOPK_MAX,
        )

        bad = DoubleSparsityRuntimeConfig(
            heavy_channels=32,
            token_budget=FLASHINFER_TOPK_MAX + 1024,
            recent_tokens=64,
            sink_tokens=4,
            min_seq_len=4096,
            max_selected_per_request=16384,
            gqa_reduction="max_abs",
            klabel_dtype="bf16",
            block_t=1024,
            k_block=64,
            scratch_max_bs=16,
            selector_backend="flashinfer_topk_page_table",
        )
        with self.assertRaisesRegex(ValueError, "FlashInfer"):
            bad.validate()

        # Same budget with torch backend is fine.
        ok = DoubleSparsityRuntimeConfig(
            heavy_channels=32,
            token_budget=FLASHINFER_TOPK_MAX + 1024,
            recent_tokens=64,
            sink_tokens=4,
            min_seq_len=4096,
            max_selected_per_request=16384,
            gqa_reduction="max_abs",
            klabel_dtype="bf16",
            block_t=1024,
            k_block=64,
            scratch_max_bs=16,
            selector_backend="torch",
        )
        ok.validate()  # must not raise


if __name__ == "__main__":
    unittest.main()
