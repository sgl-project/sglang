"""Vocab-shard argmax partials + sum all-reduce + finalize equal the gathered
logits path bit for bit (TP emulated on one GPU)."""

import unittest

import torch

from sglang.kernels.ops.speculative.topk1 import (
    draft_topk1_argmax_from_partials,
    draft_topk1_postprocess,
    draft_topk1_postprocess_from_partials,
    draft_topk1_shard_partials,
    draft_topk1_shard_partials_width,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BF16 = torch.bfloat16
DEV = "cuda"

# (name, tp, vocab, logits dtype)
LAYOUTS = [
    ("tp8", 8, 154880, BF16),
    ("tp4", 4, 154880, BF16),
    ("tp2", 2, 154880, BF16),
    ("tp8-padded", 8, 154870, BF16),  # the last rank holds LM-head padding columns
    ("tp4-odd", 4, 100003, BF16),
    ("tp8-fp32", 8, 154880, torch.float32),
    ("tp4-one-split", 4, 20000, BF16),
]
ROWS = [1, 3, 16, 96]
PATTERNS = [
    "randn",
    "coarse",
    "tie_ranks",
    "tie_splits",
    "max_in_padding",
    "nan_one",
    "nan_many",
    "all_neg_inf",
    "pos_inf",
    "signed_zero",
    "denormal_max",
    "nan_only_padding",
]
ALL_REDUCES = ["sum", "reverse", "stacked", "negzero"]


def shard_width(vocab, tp, padding=64):
    """VocabParallelEmbedding's per-rank width without added vocab."""
    if -(-vocab // padding) * padding % tp != 0:
        padding *= tp
    return -(-vocab // padding) * padding // tp


def make_logits(pattern, rows, tp, width, vocab, dtype, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    total = tp * width
    x = torch.randn(rows, total, generator=g, device=DEV) * 2.0
    if pattern == "coarse":
        x = torch.randint(-3, 4, (rows, total), generator=g, device=DEV).float()
    x = x.to(dtype)
    rr = torch.arange(rows, device=DEV)

    def cols(n, hi=vocab):
        return torch.randint(0, hi, (rows, n), generator=g, device=DEV)

    if pattern in ("tie_ranks", "tie_splits"):
        owner = torch.randint(0, tp, (rows,), generator=g, device=DEV)
        offsets = (
            [
                torch.randint(0, width, (rows,), generator=g, device=DEV)
                for _ in range(3)
            ]
            if pattern == "tie_ranks"
            else [torch.full_like(owner, min(c, width - 1)) for c in (8191, 8192, 0)]
        )
        for k, off in enumerate(offsets):
            rank = owner if pattern == "tie_splits" else (owner + k) % tp
            x[rr, (rank * width + off).clamp(max=vocab - 1)] = 50.0
    elif pattern == "max_in_padding":
        x[:, vocab:] = 1000.0
        x[rr, cols(1)[:, 0]] = 60.0
    elif pattern == "nan_one":
        x[rr, cols(1)[:, 0]] = float("nan")
    elif pattern == "nan_many":
        x[rr.unsqueeze(1), cols(3)] = float("nan")
        x[rr, cols(1)[:, 0]] = float("inf")
    elif pattern == "all_neg_inf":
        x.fill_(float("-inf"))
        x[1::2, cols(1)[1::2, 0]] = float("nan")
    elif pattern == "pos_inf":
        x[rr.unsqueeze(1), cols(2)] = float("inf")
    elif pattern in ("signed_zero", "denormal_max"):
        x = -(x.float().abs() + 1.0).to(dtype)
        x[rr, cols(1)[:, 0]] = 0.0 if pattern == "denormal_max" else -0.0
        tiny = 1e-40 if pattern == "denormal_max" else 0.0
        x[rr, cols(1)[:, 0]] = tiny
    elif pattern == "nan_only_padding":
        x[:, vocab:] = float("nan")
    return x.contiguous()


def all_reduce_sum(contribs, mode):
    """Any fp32 sum all-reduce, every slot having one nonzero owner: plain and
    reversed rank order, a stacked sum, and a backend that sends +0 as -0."""
    if mode == "negzero":
        flipped = []
        for t in contribs:
            bits = t.view(torch.int32)
            neg_zero = torch.full_like(bits, -(2**31))
            flipped.append(torch.where(bits == 0, neg_zero, bits).view(torch.float32))
        acc = torch.full_like(contribs[0], -0.0)
        for t in flipped:
            acc = acc + t
        return acc
    if mode == "stacked":
        return torch.stack(contribs).sum(0)
    acc = torch.zeros_like(contribs[0])
    for t in contribs[::-1] if mode == "reverse" else contribs:
        acc = acc + t
    return acc


class TestDraftTopk1Partials(CustomTestCase):
    def assert_equal(self, a, b, tag):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
            where = (a != b).nonzero()[:4].tolist() if a.shape == b.shape else "shape"
            self.fail(f"{tag}: differs at {where}")

    def check_case(self, tp, vocab, dtype, rows, pattern, mode, seed, tag):
        width = shard_width(vocab, tp)
        logits = make_logits(pattern, rows, tp, width, vocab, dtype, seed)
        contribs = [
            draft_topk1_shard_partials(
                logits[:, r * width : (r + 1) * width],
                tp_rank=r,
                tp_size=tp,
                vocab_size=vocab,
            )
            for r in range(tp)
        ]
        self.assertEqual(
            contribs[0].shape,
            (rows, draft_topk1_shard_partials_width(local_width=width, tp_size=tp)),
        )
        partials = all_reduce_sum(contribs, mode)
        # The gathered path: logits[:, :vocab] in the fp32 buffer.
        gathered = logits[:, :vocab].float()

        # Draft decode step: tokens, topk_p, positions and the draft-token column.
        pos0 = torch.randint(0, 1 << 20, (rows,), device=DEV)
        tok0 = torch.randint(0, vocab, (rows, 4), device=DEV)
        pos_ref, tok_ref, pos_new, tok_new = (
            pos0.clone(),
            tok0.clone(),
            pos0.clone(),
            tok0.clone(),
        )
        p_ref, i_ref = draft_topk1_postprocess(gathered, pos_ref, tok_ref, 2)
        p_new, i_new = draft_topk1_postprocess_from_partials(
            partials, pos_new, tok_new, 2
        )
        for what, a, b in (
            ("topk_index", i_ref, i_new),
            ("topk_p", p_ref, p_new),
            ("positions", pos_ref, pos_new),
            ("draft_tokens", tok_ref, tok_new),
        ):
            self.assert_equal(a, b, f"decode {tag} {what}")
        # Draft extend: torch.argmax order (NaN first, lowest index on ties).
        p_new, i_new = draft_topk1_argmax_from_partials(partials)
        self.assert_equal(
            torch.argmax(gathered, dim=-1, keepdim=True), i_new, f"extend {tag}"
        )
        self.assert_equal(torch.ones_like(p_new), p_new, f"extend {tag} topk_p")

    def test_matches_gathered_logits(self):
        seed = 0
        for name, tp, vocab, dtype in LAYOUTS:
            for rows in ROWS:
                for pattern in PATTERNS:
                    modes = ALL_REDUCES if rows in (1, 16) else ["sum"]
                    for mode in modes:
                        seed += 1
                        tag = f"{name} rows={rows} {pattern} ar={mode}"
                        with self.subTest(tag=tag):
                            self.check_case(
                                tp, vocab, dtype, rows, pattern, mode, seed, tag
                            )

    def test_empty_batch(self):
        partials = draft_topk1_shard_partials(
            torch.empty(0, 19360, dtype=BF16, device=DEV),
            tp_rank=0,
            tp_size=8,
            vocab_size=154880,
        )
        topk_p, topk_index = draft_topk1_argmax_from_partials(partials)
        self.assertEqual(partials.shape[0], 0)
        self.assertEqual(tuple(topk_index.shape), (0, 1))
        self.assertEqual(tuple(topk_p.shape), (0, 1))


if __name__ == "__main__":
    unittest.main()
