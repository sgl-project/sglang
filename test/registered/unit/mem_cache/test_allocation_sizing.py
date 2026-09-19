"""Unit tests for allocation_sizing.py -- CPU only, no GPU required.

Covers the shared max_running_requests estimation and the req_to_token pool
byte deduction used by KVCacheConfigurator._resolve_memory_pool_config,
including the one-iteration fixed-point invariant (the deduction computed from
a provisional token capacity must always cover the req pool sized from the
final, smaller token capacity).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.kv_cache_configurator import (
    estimate_max_running_requests,
    estimate_req_to_token_pool_bytes,
    get_req_to_token_pool_num_slots,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_server_args(
    *,
    max_running_requests=None,
    disaggregation_mode="null",
    disaggregation_decode_extra_slots=None,
    speculative_algorithm=None,
    max_speculative_num_draft_tokens=None,
    page_size=1,
):
    return SimpleNamespace(
        max_running_requests=max_running_requests,
        disaggregation_mode=disaggregation_mode,
        disaggregation_decode_extra_slots=disaggregation_decode_extra_slots,
        speculative_algorithm=speculative_algorithm,
        max_speculative_num_draft_tokens=max_speculative_num_draft_tokens,
        page_size=page_size,
    )


class TestEstimateMaxRunningRequests(unittest.TestCase):
    def test_long_context_hits_2048_floor(self):
        # MiniMax-M3 shape: ~901k tokens over a 1M context -> estimate ~440,
        # clamped up to the 2048 floor.
        args = _make_server_args()
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=901_120,
                context_len=1_048_576,
                server_args=args,
                attn_dp_size=1,
            ),
            2048,
        )

    def test_short_context_hits_4096_ceiling(self):
        args = _make_server_args()
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=10_000_000,
                context_len=8192,
                server_args=args,
                attn_dp_size=1,
            ),
            4096,
        )

    def test_half_token_capacity_clamp(self):
        args = _make_server_args()
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=1000,
                context_len=8192,
                server_args=args,
                attn_dp_size=1,
            ),
            500,
        )

    def test_user_requested_and_dp_split(self):
        args = _make_server_args(max_running_requests=512)
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=901_120,
                context_len=1_048_576,
                server_args=args,
                attn_dp_size=1,
            ),
            512,
        )
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=901_120,
                context_len=1_048_576,
                server_args=args,
                attn_dp_size=2,
            ),
            256,
        )

    def test_user_requested_capped_by_half_token_capacity(self):
        args = _make_server_args(max_running_requests=4096)
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=1000,
                context_len=8192,
                server_args=args,
                attn_dp_size=1,
            ),
            500,
        )

    def test_mamba_req_cap(self):
        args = _make_server_args()
        self.assertEqual(
            estimate_max_running_requests(
                token_capacity=901_120,
                context_len=1_048_576,
                server_args=args,
                attn_dp_size=1,
                mamba_req_cap=300,
            ),
            300,
        )


class TestGetReqToTokenPoolNumSlots(unittest.TestCase):
    def test_default_padding_row(self):
        args = _make_server_args()
        self.assertEqual(get_req_to_token_pool_num_slots(2048, args), 2049)

    def test_decode_mode_extra_slots(self):
        args = _make_server_args(
            disaggregation_mode="decode", disaggregation_decode_extra_slots=10
        )
        self.assertEqual(get_req_to_token_pool_num_slots(2048, args), 2059)

    def test_decode_mode_unset_extra_slots(self):
        args = _make_server_args(
            disaggregation_mode="decode", disaggregation_decode_extra_slots=None
        )
        self.assertEqual(get_req_to_token_pool_num_slots(2048, args), 2049)


class TestEstimateReqToTokenPoolBytes(unittest.TestCase):
    def test_minimax_m3_shape(self):
        # MiniMax-M3: 1M context, 2048-request floor -> ~8 GiB req_to_token map.
        args = _make_server_args()
        nbytes = estimate_req_to_token_pool_bytes(
            token_capacity=901_120,
            context_len=1_048_576,
            server_args=args,
            attn_dp_size=1,
        )
        # extra_context_len = 4 (no spec decoding) -> 2049 x (1048576 + 4) x 4B
        self.assertEqual(nbytes, 2049 * (1_048_576 + 4) * 4)
        self.assertGreater(nbytes, 8 * (1 << 30))

    def test_one_iteration_fixed_point_never_under_deducts(self):
        # Emulate KVCacheConfigurator._resolve_memory_pool_config:
        #   T0 = budget // cell (upper bound), R = deduction(T0),
        #   T1 = (budget - R) // cell (final).
        # The req pool actually allocated for T1 must fit within R for every
        # budget/cell/context combination, i.e. deduction(T1) <= R.
        args = _make_server_args()
        cell_size = 45_312  # MiniMax-M3 per-token KV bytes (tp16, bf16)
        context_len = 1_048_576
        for budget_gb in (5, 10, 20, 30, 38, 50, 100):
            budget = budget_gb * (1 << 30)
            t0 = budget // cell_size
            deduction = estimate_req_to_token_pool_bytes(
                token_capacity=t0,
                context_len=context_len,
                server_args=args,
                attn_dp_size=1,
            )
            t1 = max(budget - deduction, 0) // cell_size
            actual = estimate_req_to_token_pool_bytes(
                token_capacity=t1,
                context_len=context_len,
                server_args=args,
                attn_dp_size=1,
            )
            self.assertLessEqual(actual, deduction, f"budget={budget_gb} GB")


if __name__ == "__main__":
    unittest.main()
