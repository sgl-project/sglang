"""Gemma-3 bidirectional image attention on XPU (sgl-kernel-xpu bidirectional_block_ids).

Guards the cases the kernel mask cannot express directly: a sliding-window layer
whose prompt exceeds the window must keep local attention, and image tokens in an
extend after a cached prefix must still attend bidirectionally.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

PAGE, H, HKV, D = 128, 8, 4, 128


def _paged_inputs(seq_len, dev):
    q = torch.randn(seq_len, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(seq_len, HKV, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(seq_len, HKV, D, device=dev, dtype=torch.bfloat16)
    npages = (seq_len + PAGE - 1) // PAGE
    k_cache = torch.zeros(npages * PAGE, HKV, D, device=dev, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    k_cache[:seq_len], v_cache[:seq_len] = k, v
    return (
        q,
        k,
        v,
        k_cache.view(npages, PAGE, HKV, D),
        v_cache.view(npages, PAGE, HKV, D),
    )


def _reference(q, k, v, allow, scale):
    k = k.float().repeat_interleave(H // HKV, dim=1)
    v = v.float().repeat_interleave(H // HKV, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    scores = scores.masked_fill(~allow.unsqueeze(0), float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)


def _image_batch(prefix_lens, extend_lens, image_spans):
    image_items = [
        SimpleNamespace(mm_items=[SimpleNamespace(is_image=lambda: True, offsets=s)])
        for s in image_spans
    ]
    return SimpleNamespace(
        batch_size=len(prefix_lens),
        extend_prefix_lens_cpu=prefix_lens,
        extend_seq_lens_cpu=extend_lens,
        mm_inputs=image_items,
    )


def _build_block_ids(forward_batch, device):
    from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration

    return Gemma3ForConditionalGeneration._build_xpu_image_block_ids(
        forward_batch=forward_batch, device=device
    )


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an Intel XPU"
)
class TestXPUGemma3SlidingWindowBound(CustomTestCase):
    WINDOW_LEFT = 63
    SPAN = (8, 40)

    def _run(self, seq_len):
        from sgl_kernel.flash_attn import (
            build_bidirectional_block_ids,
            flash_attn_with_kvcache,
        )

        from sglang.srt.layers.attention.xpu_backend import (
            _bidirectional_image_attn_args,
        )

        torch.manual_seed(0)
        dev = "xpu"
        scale = D**-0.5
        q, k, v, k_cache, v_cache = _paged_inputs(seq_len, dev)
        block_ids, window_size = _bidirectional_image_attn_args(
            block_ids=build_bidirectional_block_ids([seq_len], [[self.SPAN]], dev),
            window_size=(self.WINDOW_LEFT, 0),
            max_seqlen_k=seq_len,
        )
        out = flash_attn_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=torch.arange(
                k_cache.shape[0], device=dev, dtype=torch.int32
            ).view(1, -1),
            cache_seqlens=torch.tensor([seq_len], device=dev, dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, seq_len], device=dev, dtype=torch.int32),
            max_seqlen_q=seq_len,
            softmax_scale=scale,
            causal=block_ids is None,
            window_size=window_size,
            bidirectional_block_ids=block_ids,
        )
        torch.xpu.synchronize()
        rows = torch.arange(seq_len, device=dev).view(-1, 1)
        cols = torch.arange(seq_len, device=dev).view(1, -1)
        return out.float(), q, k, v, rows, cols, scale

    def test_prompt_longer_than_window_keeps_local_attention(self):
        """Keys beyond the sliding window must stay masked once the prompt outgrows
        the window, even though the batch carries image block ids."""
        out, q, k, v, rows, cols, scale = self._run(seq_len=200)
        local = (cols <= rows) & (rows - cols <= self.WINDOW_LEFT)
        want = _reference(q, k, v, local, scale)
        torch.testing.assert_close(out, want, rtol=2e-2, atol=2e-2)

    def test_prompt_within_window_keeps_bidirectional_image(self):
        """At kv length == left window + 1 the window clips nothing, so the image
        span must still be attended bidirectionally."""
        out, q, k, v, rows, cols, scale = self._run(seq_len=self.WINDOW_LEFT + 1)
        allow = cols <= rows
        begin, last = self.SPAN
        allow[begin : last + 1, begin : last + 1] = True
        want = _reference(q, k, v, allow, scale)
        torch.testing.assert_close(out, want, rtol=2e-2, atol=2e-2)


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an Intel XPU"
)
class TestXPUGemma3ImageAfterCachedPrefix(CustomTestCase):
    def test_matches_full_bidirectional_attention(self):
        """With a cached prefix (radix hit or an earlier chunk), image tokens in the
        extend must attend bidirectionally and see the whole prefix. The batch mixes
        a page-aligned prefix, an unaligned prefix with an image straddling it, and
        no prefix with an image cut by the chunk end."""
        from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend

        torch.manual_seed(0)
        dev = "xpu"
        scale = D**-0.5
        prefix_lens, extend_lens = [128, 50, 0], [100, 90, 70]
        image_spans = [[(140, 190)], [(30, 100)], [(40, 90)]]
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        pages_per_seq = [(s + PAGE - 1) // PAGE for s in seq_lens]
        num_pages = sum(pages_per_seq)

        # Hand pages out in reverse so each sequence's pages are non-contiguous.
        free_pages = list(range(num_pages))[::-1]
        req_to_token = torch.zeros(len(seq_lens), 256, dtype=torch.int32, device=dev)
        for i, n in enumerate(pages_per_seq):
            pages = torch.tensor([free_pages.pop(0) for _ in range(n)], device=dev)
            slots = (pages[:, None] * PAGE + torch.arange(PAGE, device=dev)).flatten()
            req_to_token[i, : n * PAGE] = slots.to(torch.int32)
        k_flat = torch.randn(num_pages * PAGE, HKV, D, device=dev, dtype=torch.bfloat16)
        v_flat = torch.randn_like(k_flat)
        q = torch.randn(sum(extend_lens), H, D, device=dev, dtype=torch.bfloat16)

        backend = XPUAttentionBackend.__new__(XPUAttentionBackend)
        backend.num_splits = 0
        to_dev = lambda x: torch.tensor(x, device=dev, dtype=torch.int32)
        block_ids = _build_block_ids(
            _image_batch(prefix_lens, extend_lens, image_spans), dev
        )
        out = backend._forward_image_blocks_after_prefix(
            q=q,
            key_cache=k_flat.view(num_pages, PAGE, HKV, D),
            value_cache=v_flat.view(num_pages, PAGE, HKV, D),
            page_table=req_to_token[:, ::PAGE] // PAGE,
            forward_batch=SimpleNamespace(
                extend_prefix_lens_cpu=prefix_lens,
                extend_prefix_lens=to_dev(prefix_lens),
                extend_seq_lens=to_dev(extend_lens),
            ),
            metadata=SimpleNamespace(
                cache_seqlens_int32=to_dev(seq_lens),
                cu_seqlens_k=torch.nn.functional.pad(
                    torch.cumsum(to_dev(seq_lens), dim=0, dtype=torch.int32), (1, 0)
                ),
                max_seq_len_k=max(seq_lens),
            ),
            block_ids=block_ids,
            scale=scale,
            softcap=0.0,
        )
        torch.xpu.synchronize()

        q_start = 0
        for i, (prefix_len, seq_len) in enumerate(zip(prefix_lens, seq_lens)):
            slots = req_to_token[i, :seq_len].long()
            rows = torch.arange(prefix_len, seq_len, device=dev).view(-1, 1)
            cols = torch.arange(seq_len, device=dev).view(1, -1)
            allow = cols <= rows
            for begin, last in image_spans[i]:
                allow |= (
                    (rows >= begin) & (rows <= last) & (cols >= begin) & (cols <= last)
                )
            q_end = q_start + seq_len - prefix_len
            want = _reference(
                q[q_start:q_end], k_flat[slots], v_flat[slots], allow, scale
            )
            torch.testing.assert_close(
                out[q_start:q_end].float(), want, rtol=2e-2, atol=2e-2
            )
            q_start = q_end


class TestXPUGemma3ImageBlockIds(CustomTestCase):
    def test_image_only_in_cached_prefix_needs_no_mask(self):
        """A text-only extend after a cached image is exact under causal attention
        and must take the plain causal path."""
        self.assertIsNone(
            _build_block_ids(_image_batch([256], [50], [[(10, 200)]]), "cpu")
        )


if __name__ == "__main__":
    unittest.main()
