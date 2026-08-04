# SPDX-License-Identifier: Apache-2.0
"""SM120 sparse-MLA topk bucket alignment (coverage for #33407).

Test authored by @efschu (github.com/efschu/htsglang), contributed here with
their attribution; import paths adapted from that fork's layout to this tree
(sglang.kernels.ops.attention) and _next_topk_bucket exposed as a named helper.


The CUTLASS SM120 sparse-MLA decode kernels are instantiated for a fixed set
of ``(num_heads, topk)`` pairs -- ``topk in {128, 512, 1024}`` -- and DSpark's
draft indexer emits ``topk=192``, which is in no bucket. Before the port,
``_flash_mla_flashinfer`` handed that width straight to
``sparse_mla_sm120_decode_dsv4`` and the server died at boot on the first
DSpark call.

Everything here is hermetic (``CUDA_VISIBLE_DEVICES=99``, CPU tensors, no
kernel launch): the fix is index arithmetic plus a dispatch decision, and both
are observable without a card. The kernel and the Triton fallback are replaced
by recorders, so "which path was taken, with which index width" is an
observation rather than an inference.

The KV cache in these tests already has ``page_size == 64``, which is the
production short-circuit that skips the page-split (``_PBS_DST``); that keeps
the test free of any Triton launch.
"""

from __future__ import annotations

import unittest

import flashinfer.mla._sparse_mla_sm120 as fi
import torch

from sglang.kernels.ops.attention import flash_mla_sm120 as fmod
from sglang.kernels.ops.attention import flash_mla_sm120_triton as tmod
from sglang.kernels.ops.attention.flash_mla_sm120 import (
    _NOPE_ROPE_STRIDE,
    _PBS_DST,
    _SCALE_STRIDE,
    _SUPPORTED_TOPK_WIDTHS,
    _next_topk_bucket,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_BYTES_PER_TOKEN = _NOPE_ROPE_STRIDE + _SCALE_STRIDE
_HEAD_DIM_QK = 512
_HEAD_DIM_V = 448


class TestBucketArithmetic(unittest.TestCase):
    """The whole correctness argument of the pad, without a kernel."""

    def test_dspark_192_pads_to_the_next_instantiated_width(self):
        self.assertEqual(_next_topk_bucket(192), 512)

    def test_an_instantiated_width_is_its_own_bucket(self):
        for width in _SUPPORTED_TOPK_WIDTHS:
            self.assertEqual(_next_topk_bucket(width), width)

    def test_the_bucket_is_never_narrower_than_the_request(self):
        for topk in range(1, 2049):
            bucket = _next_topk_bucket(topk)
            self.assertIsNotNone(bucket, topk)
            self.assertGreaterEqual(bucket, topk, topk)

    def test_the_bucket_is_the_smallest_that_fits(self):
        for topk in (1, 127, 128, 129, 192, 511, 513, 1025, 2048):
            bucket = _next_topk_bucket(topk)
            narrower = [w for w in _SUPPORTED_TOPK_WIDTHS if w >= topk and w < bucket]
            self.assertEqual(narrower, [], topk)

    def test_wider_than_every_kernel_has_no_bucket(self):
        """Padding cannot rescue this; the dispatch check must catch it."""
        self.assertIsNone(_next_topk_bucket(2049))

    def test_the_widths_match_the_installed_flashinfer_decode_table(self):
        """Spread precondition: the table this port targets is really this one.

        A constant copied out of an upstream comment is worth nothing if the
        installed kernels were built for other widths.
        """
        table_widths = {topk for _, topk in fi._DECODE_DSV4_DISPATCH}
        self.assertTrue(table_widths)
        self.assertNotIn(192, table_widths, "the defect this port fixes is gone")
        self.assertTrue(
            table_widths.issubset(set(_SUPPORTED_TOPK_WIDTHS)),
            f"instantiated decode widths {sorted(table_widths)} are not covered "
            f"by _SUPPORTED_TOPK_WIDTHS {_SUPPORTED_TOPK_WIDTHS}",
        )


class _Recorders:
    """Replace the CUTLASS kernel and the Triton fallback with recorders."""

    def __enter__(self):
        self.decode_calls = []
        self.triton_calls = []
        self._fi = fi.sparse_mla_sm120_decode_dsv4
        self._tr = tmod.flash_mla_sparse_decode_triton

        def decode(**kwargs):
            self.decode_calls.append(kwargs)
            return None

        def triton(q, k_cache, indices, topk_length, *args, **kwargs):
            self.triton_calls.append((indices.shape, topk_length))
            b = q.shape[0]
            h = q.shape[2] if q.ndim == 4 else q.shape[1]
            return (
                torch.zeros(b, 1, h, _HEAD_DIM_V, dtype=torch.bfloat16),
                torch.zeros(b, h, dtype=torch.float32),
            )

        fi.sparse_mla_sm120_decode_dsv4 = decode
        tmod.flash_mla_sparse_decode_triton = triton
        return self

    def __exit__(self, *exc):
        fi.sparse_mla_sm120_decode_dsv4 = self._fi
        tmod.flash_mla_sparse_decode_triton = self._tr
        return False


def _call(topk: int, heads: int = 128, batch: int = 1, d_qk: int = _HEAD_DIM_QK):
    """Drive ``_flash_mla_flashinfer`` on CPU and report what it reached."""
    q = torch.zeros(batch, 1, heads, d_qk, dtype=torch.bfloat16)
    # page_size == _PBS_DST short-circuits the page-split, so no Triton kernel
    # is launched anywhere on this path.
    k_cache = torch.zeros(4, _PBS_DST, 1, _BYTES_PER_TOKEN, dtype=torch.uint8)
    indices = torch.zeros(batch, topk, dtype=torch.int32)
    with _Recorders() as rec:
        out = fmod._flash_mla_flashinfer(
            q,
            k_cache,
            indices,
            None,  # topk_length
            None,  # attn_sink
            _HEAD_DIM_V,
            d_qk ** (-0.5),
            None,  # extra_k_cache
            None,  # extra_indices
            None,  # extra_topk_length
        )
    return rec, out


class TestDispatchOnCall(unittest.TestCase):
    def setUp(self):
        # The two "log once" latches are module state; reset them so a test
        # order cannot decide whether a branch logs.
        fmod._noted_bucket_pad = False
        fmod._warned_triton_fb = False

    def test_topk_192_reaches_the_kernel_padded_to_512(self):
        rec, _ = _call(192)
        self.assertEqual(len(rec.decode_calls), 1)
        self.assertEqual(rec.triton_calls, [])
        kwargs = rec.decode_calls[0]
        self.assertEqual(kwargs["indices"].shape[-1], 512)
        self.assertTrue(
            fi._decode_dsv4_dispatchable(
                1, 128, kwargs["indices"].shape[-1], _HEAD_DIM_QK, _PBS_DST, 0
            ),
            "the padded width must be dispatchable -- that is the whole fix",
        )

    def test_the_padding_is_the_minus_one_skip_sentinel(self):
        rec, _ = _call(192)
        idx = rec.decode_calls[0]["indices"]
        self.assertTrue(bool((idx[:, :192] == 0).all()), "real indices were altered")
        self.assertTrue(bool((idx[:, 192:] == -1).all()), "pad is not the sentinel")

    def test_the_scan_is_capped_at_the_true_width(self):
        """Without topk_length the kernel would read the -1 padding."""
        rec, _ = _call(192)
        capped = rec.decode_calls[0]["topk_length"]
        self.assertIsNotNone(capped, "topk_length must be synthesised by the pad")
        self.assertEqual(capped.dtype, torch.int32)
        self.assertEqual(capped.tolist(), [192])

    def test_an_instantiated_width_is_passed_through_untouched(self):
        """Neutrality: the pre-port behaviour for every width that worked."""
        rec, _ = _call(512)
        self.assertEqual(len(rec.decode_calls), 1)
        self.assertEqual(rec.decode_calls[0]["indices"].shape[-1], 512)
        self.assertIsNone(
            rec.decode_calls[0]["topk_length"],
            "an already-instantiated width must not grow a synthetic cap",
        )

    def test_the_split_k_scratch_covers_the_padded_width(self):
        """mid_out/mid_lse are sized from the width the kernel actually scans."""
        rec, _ = _call(192)
        kwargs = rec.decode_calls[0]
        self.assertEqual(kwargs["mid_out"].shape[2], 512 // 64)
        self.assertEqual(kwargs["mid_lse"].shape[2], 512 // 64)

    def test_an_undispatchable_geometry_falls_back_to_triton(self):
        """d_qk != 512 is out of the CUTLASS table at every topk width."""
        rec, out = _call(128, d_qk=256)
        self.assertEqual(rec.decode_calls, [])
        self.assertEqual(len(rec.triton_calls), 1)
        self.assertIsNotNone(out[0])

    def test_a_batch_above_the_decode_maximum_falls_back_to_triton(self):
        rec, _ = _call(128, batch=fi._DECODE_MAX_TOKENS + 1)
        self.assertEqual(rec.decode_calls, [])
        self.assertEqual(len(rec.triton_calls), 1)

    def test_an_uninstantiated_head_count_falls_back_to_triton(self):
        heads = next(
            h
            for h in range(1, 4096)
            if (h, 512) not in fi._DECODE_DSV4_DISPATCH
            and (h, 128) not in fi._DECODE_DSV4_DISPATCH
        )
        rec, _ = _call(128, heads=heads)
        self.assertEqual(rec.decode_calls, [])
        self.assertEqual(len(rec.triton_calls), 1)

    def test_the_fallback_gets_the_unpadded_indices(self):
        """Triton reads the real width; handing it the -1 pad would be wrong."""
        rec, _ = _call(192, d_qk=256)
        self.assertEqual(len(rec.triton_calls), 1)
        shape, _ = rec.triton_calls[0]
        self.assertEqual(shape[-1], 192)


if __name__ == "__main__":
    unittest.main()
