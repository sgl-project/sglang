"""Kimi-K3 row-sharded latent up_proj tail against the replicated tail."""

import unittest

import torch

from sglang.srt.models.kimi_k3 import (
    _sharded_up_proj_eligible,
    _sharded_up_proj_tail,
    _tp_row_slice,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOKENS = 5
LATENT = 16
# Divisible by every TP size from 1 to 8.
HIDDEN = 840
SHARDED_TP_SIZES = range(2, 9)
# Kimi-K3's up_proj output rows: 7168 = 2**10 * 7.
K3_UP_PROJ_ROWS = 7168
K3_SHARDED_TP_SIZES = (2, 4, 7, 8)


def _rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)


def _partial_sums(dtype, tp_size):
    generator = torch.Generator().manual_seed(tp_size)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64).to(dtype)

    latents = [randn(TOKENS, LATENT) for _ in range(tp_size)]
    shared_partials = [randn(TOKENS, HIDDEN) for _ in range(tp_size)]
    up_weight = randn(HIDDEN, LATENT)
    return latents, shared_partials, up_weight


def _replicated_tail(latents, shared_partials, up_weight):
    latent = _rms_norm(torch.stack(latents).sum(dim=0))
    shared = torch.stack(shared_partials).sum(dim=0)
    return torch.nn.functional.linear(latent, up_weight) + shared


def _sharded_tail_sum(latents, shared_partials, up_weight):
    """Each rank's tail with the shared all-reduce replaced by identity, summed
    over ranks: the value the real all-reduce would return on every rank."""
    tp_size = len(latents)
    reduced_latent = _rms_norm(torch.stack(latents).sum(dim=0))
    contributions = [
        _sharded_up_proj_tail(
            latent=latents[rank].clone(),
            shared_output=shared_partials[rank].clone(),
            up_weight=up_weight,
            rows=_tp_row_slice(HIDDEN, tp_size=tp_size, tp_rank=rank),
            reduce_latent=lambda _: reduced_latent,
            reduce_shared=lambda shared: shared,
            prefix_sum=None,
        )
        for rank in range(tp_size)
    ]
    return torch.stack(contributions).sum(dim=0)


def _dense_up_proj(out_features: int, dtype=torch.bfloat16) -> torch.nn.Module:
    return torch.nn.Linear(4, out_features, bias=False, dtype=dtype, device="meta")


class TestKimiK3ShardedUpProjTail(CustomTestCase):
    def test_matches_replicated_tail_in_float64(self):
        for tp_size in SHARDED_TP_SIZES:
            with self.subTest(tp=tp_size):
                inputs = _partial_sums(torch.float64, tp_size)
                torch.testing.assert_close(
                    _sharded_tail_sum(*inputs),
                    _replicated_tail(*inputs),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_matches_replicated_tail_within_bf16_rounding(self):
        for tp_size in SHARDED_TP_SIZES:
            with self.subTest(tp=tp_size):
                latents, shared_partials, up_weight = _partial_sums(
                    torch.bfloat16, tp_size
                )
                expected = _replicated_tail(
                    [x.double() for x in latents],
                    [x.double() for x in shared_partials],
                    up_weight.double(),
                )
                actual = _sharded_tail_sum(latents, shared_partials, up_weight)
                torch.testing.assert_close(
                    actual.double(), expected, rtol=0.05, atol=0.1
                )

    def test_rank_row_slices_partition_the_output_rows(self):
        cases = [(HIDDEN, tp) for tp in SHARDED_TP_SIZES] + [
            (K3_UP_PROJ_ROWS, tp) for tp in K3_SHARDED_TP_SIZES
        ]
        for num_rows, tp_size in cases:
            with self.subTest(num_rows=num_rows, tp=tp_size):
                covered = []
                for rank in range(tp_size):
                    rows = _tp_row_slice(num_rows, tp_size=tp_size, tp_rank=rank)
                    covered.extend(range(rows.start, rows.stop))
                self.assertEqual(covered, list(range(num_rows)))

    def test_adds_prefix_sum_after_the_shared_reduction(self):
        tp_size = 8
        reduced = torch.full((TOKENS, HIDDEN), 2.0, dtype=torch.float64)
        prefix_sum = torch.full((TOKENS, HIDDEN), 3.0, dtype=torch.float64)
        latents, shared_partials, up_weight = _partial_sums(torch.float64, tp_size)
        out = _sharded_up_proj_tail(
            latent=latents[0],
            shared_output=shared_partials[0],
            up_weight=up_weight,
            rows=_tp_row_slice(HIDDEN, tp_size=tp_size, tp_rank=0),
            reduce_latent=lambda latent: latent,
            reduce_shared=lambda _: reduced,
            prefix_sum=prefix_sum,
        )
        torch.testing.assert_close(out, reduced + prefix_sum, rtol=0, atol=0)

    def test_k3_eligibility_for_every_tp_size(self):
        """TP sizes that do not split 7168 rows evenly keep the stock tail."""
        layer = _dense_up_proj(K3_UP_PROJ_ROWS)
        for tp_size in range(1, 9):
            with self.subTest(tp=tp_size):
                self.assertIs(
                    _sharded_up_proj_eligible(layer, tp_size, on_hip=True),
                    tp_size in K3_SHARDED_TP_SIZES,
                )

    def test_eligibility(self):
        cases = [
            (torch.bfloat16, 7168, 8, True),
            (torch.float16, 7168, 8, True),
            (torch.bfloat16, 7168, 1, False),
            (torch.bfloat16, 7170, 8, False),
            (torch.float32, 7168, 8, False),
        ]
        for dtype, out_features, tp_size, eligible in cases:
            with self.subTest(dtype=dtype, out_features=out_features, tp=tp_size):
                self.assertIs(
                    _sharded_up_proj_eligible(
                        _dense_up_proj(out_features, dtype), tp_size, on_hip=True
                    ),
                    eligible,
                )
        self.assertFalse(_sharded_up_proj_eligible(None, 8, on_hip=True))
        packed = torch.nn.Module()
        packed.weight = torch.nn.Parameter(
            torch.empty(7168, 2, dtype=torch.uint8, device="meta"),
            requires_grad=False,
        )
        self.assertFalse(_sharded_up_proj_eligible(packed, 8, on_hip=True))

    def test_not_eligible_with_bias(self):
        """The sharded tail calls the GEMM directly and would drop a bias."""
        layer = torch.nn.Linear(
            4, K3_UP_PROJ_ROWS, bias=True, dtype=torch.bfloat16, device="meta"
        )
        self.assertFalse(_sharded_up_proj_eligible(layer, 8, on_hip=True))

    def test_not_eligible_off_rocm(self):
        for tp_size in range(1, 9):
            with self.subTest(tp=tp_size):
                self.assertFalse(
                    _sharded_up_proj_eligible(
                        _dense_up_proj(K3_UP_PROJ_ROWS), tp_size, on_hip=False
                    )
                )


if __name__ == "__main__":
    unittest.main()
