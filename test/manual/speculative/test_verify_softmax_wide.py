# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""verify_softmax_wide must return flashinfer.sampling.softmax's bits.

The reference is the served call it replaces (dspark_accept.softmax_temp_flashinfer
with the knob off: torch.repeat_interleave of the per-request temperatures, then
flashinfer 0.7.2's vocabulary-split OnlineSoftmax). Equality is exact (torch.equal
on the fp32 bit patterns, NaN included), no tolerance. A float64 softmax checks
the inputs are sane (|p - p64| <= 1e-5 absolute), independently of flashinfer.

What a failure means:
  test_bitwise_direct     the wide kernel's arithmetic, merge order, temperature
                          index, slice / tile split or flags differ from flashinfer's
  test_bitwise_dispatch   the knob does not reach the wide kernel, or reaches it with
                          different arguments than the served call
Run: python3 test_verify_softmax_wide.py -v  (one CUDA GPU, flashinfer importable)
"""

import os
import unittest

import torch

os.environ.setdefault("SGLANG_OPT_DFLASH_VERIFY_SOFTMAX_WIDE", "0")

import flashinfer.sampling  # noqa: E402

from sglang.kernels.ops.speculative.dspark import dspark_accept  # noqa: E402
from sglang.kernels.ops.speculative.verify_softmax_wide import (  # noqa: E402
    verify_softmax_wide,
    verify_softmax_wide_supported,
)

VOCAB = 163840


def _reference(logits, temperatures, rows_per_request):
    bs = logits.shape[0] // rows_per_request
    temp_per_row = torch.repeat_interleave(
        temperatures.reshape(bs).to(torch.float32), rows_per_request, dim=0
    ).contiguous()
    return flashinfer.sampling.softmax(
        logits=logits.to(torch.float32).contiguous(), temperature=temp_per_row
    )


def _same_bits(a, b):
    return a.dtype == b.dtype == torch.float32 and a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


def _poison(*shapes):
    # Freed NaN blocks of the output / partials sizes: a cell the kernel fails to
    # write then reads NaN instead of a stale correct value from the allocator.
    for shape in shapes:
        x = torch.full(shape, float("nan"), dtype=torch.float32, device="cuda")
        del x


def _cases():
    g = torch.Generator(device="cuda").manual_seed(20261031)

    def randn(rows, d, scale):
        return torch.randn(rows, d, generator=g, device="cuda", dtype=torch.float32) * scale

    out = []
    # (bs, rows_per_request, vocab): served bs 1 x 8, bs 2, bs 16 (128 rows, the split
    # path's limit), the smallest split vocabulary, an odd vocabulary (slice stride
    # 7784, partial last tile).
    shapes = [(1, 8, VOCAB), (2, 8, VOCAB), (16, 8, VOCAB), (1, 8, 24576), (3, 7, 155652)]
    temp_sets = {
        1: [[1.0], [0.7], [1e-5]],
        2: [[1.0, 1.0], [1.0, 0.7]],
        3: [[1.0, 2.0, 0.5]],
        16: [[1.0] * 16, [1.0, 0.7, 2.0, 0.0, 1e-5, 1.3, 0.9, 1.0] * 2],
    }
    for bs, r, d in shapes:
        rows = bs * r
        dists = {
            "randn1": randn(rows, d, 1.0),
            "randn8": randn(rows, d, 8.0),
            "softcap30": 30.0 * torch.tanh(randn(rows, d, 6.0) / 30.0),
        }
        peaked = randn(rows, d, 1.0)
        peaked[torch.arange(rows, device="cuda"), (torch.arange(rows, device="cuda") * 7919) % d] += 40.0
        dists["peaked"] = peaked
        neg = randn(rows, d, 3.0)
        neg[torch.rand(rows, d, generator=g, device="cuda") < 0.01] = float("-inf")
        neg[0, : min(8192, d // 2)] = float("-inf")  # a whole slice of -inf in row 0
        dists["neginf"] = neg
        for name, logits in dists.items():
            for temps in temp_sets[bs]:
                t = torch.tensor(temps, dtype=torch.float32, device="cuda").view(bs, 1)
                out.append((f"bs{bs}_r{r}_d{d}_{name}_T{temps[:3]}", logits, t, r))
    return out


class TestVerifySoftmaxWide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _cases()

    def test_sane_inputs(self):
        # The reference is a softmax at all (independent float64 check).
        for name, logits, t, r in self.cases[:12]:
            ref = _reference(logits, t, r)
            tr = torch.repeat_interleave(t.view(-1).double(), r)[:, None]
            scaled = torch.where(tr > 0, logits.double() / tr.clamp_min(1e-30), torch.zeros_like(logits.double()))
            p64 = torch.softmax(scaled, dim=-1)
            err = (ref.double() - p64).abs().max().item()
            self.assertLessEqual(err, 1e-5, f"{name}: flashinfer vs float64 {err}")

    def test_bitwise_direct(self):
        bad = []
        for name, logits, t, r in self.cases:
            self.assertTrue(verify_softmax_wide_supported(logits, r), name)
            ref = _reference(logits, t, r)
            _poison(tuple(logits.shape), (logits.shape[0], (logits.shape[1] + 8191) // 8192, 2))
            got = verify_softmax_wide(logits=logits, temperatures=t, rows_per_request=r)
            if not _same_bits(got, ref):
                diff = got.view(torch.int32) != ref.view(torch.int32)
                bad.append(f"{name}: {int(diff.sum())} cells differ")
        self.assertEqual(bad, [], "\n".join(bad))

    def test_bitwise_dispatch(self):
        # The served entry with the knob on returns the same bits as with it off.
        os.environ["SGLANG_OPT_DFLASH_VERIFY_SOFTMAX_WIDE"] = "1"
        dspark_accept._verify_softmax_wide_enabled.cache_clear()
        try:
            self.assertTrue(dspark_accept._verify_softmax_wide_enabled())
            bad = []
            for name, logits, t, r in self.cases[::3]:
                ref = _reference(logits, t, r)
                _poison(tuple(logits.shape), (logits.shape[0], (logits.shape[1] + 8191) // 8192, 2))
                got = dspark_accept.softmax_temp_flashinfer(
                    logits=logits, temperatures=t, rows_per_request=r
                )
                if not _same_bits(got, ref):
                    bad.append(name)
            self.assertEqual(bad, [], f"dispatch differs: {bad}")
        finally:
            os.environ["SGLANG_OPT_DFLASH_VERIFY_SOFTMAX_WIDE"] = "0"
            dspark_accept._verify_softmax_wide_enabled.cache_clear()

    def test_unsupported_shapes_fall_back(self):
        # Shapes flashinfer runs on its single-CTA path are refused (the caller keeps flashinfer).
        x = torch.zeros(136, VOCAB, device="cuda")
        self.assertFalse(verify_softmax_wide_supported(x, 8))
        self.assertFalse(verify_softmax_wide_supported(torch.zeros(8, 16384, device="cuda"), 8))
        self.assertFalse(verify_softmax_wide_supported(torch.zeros(8, VOCAB + 2, device="cuda"), 8))


if __name__ == "__main__":
    unittest.main()
