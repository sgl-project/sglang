"""Per-rank-local non-padded token count across attn x MoE parallelism layouts.

``compute_local_num_token_non_padded`` (GPU tensor) and
``compute_local_num_token_non_padded_cpu`` (host int) convert a dp-group-global
real-token count into this attention-TP rank's local count. Each rank owns a
contiguous ``padded_bucket // attn_tp_size`` slice of the padded sequence, so the
localizer clamps ``real - chunk * attn_tp_rank`` into ``[0, chunk]``: a replicated
(non-sharded) rank keeps the full count and SP ranks split it. The value is
identical whether the MoE runs TP or EP -- it is an attention-side quantity both
backends consume. This table locks the exact per-rank counts and that the GPU
tensor and host-int twin agree, so a change to the sharding math fails loudly.

``ForwardBatch.moe_num_token_non_padded()`` decides whether that count may bound
a sparse MoE's input at all. Masking a gathered buffer with it truncates every
peer rank's rows, which collapsed gsm8k accuracy under DP attention with
``--moe-a2a-backend none``, where each rank sums its own partial output. The
second table locks that scatter mode x layout x CP decision.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_local_num_token_non_padded,
    compute_local_num_token_non_padded_cpu,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase


class TestNumTokenNonPaddedLayoutTable(CustomTestCase):
    # (label, attn_tp_size, sharded, padded_bucket, real-per-dp-group,
    #  expected [per attn-tp rank] per dp group)
    _LAYOUTS = [
        # Each dp rank gets a 10-token request, cuda graph pads it to 16.
        ("TP4", 4, False, 16, [10], [[10, 10, 10, 10]]),
        ("TP4.SP4", 4, True, 16, [10], [[4, 4, 2, 0]]),
        ("DP4", 1, False, 16, [10, 10, 10, 10], [[10], [10], [10], [10]]),
        ("TP2.DP2.SP2", 2, True, 16, [10, 10], [[8, 2], [8, 2]]),
        # dp0/dp2 get 10 tokens, dp1/dp3 get 20; all padded to 32.
        ("DP4.EP4", 1, False, 32, [10, 20, 10, 20], [[10], [20], [10], [20]]),
        ("TP2.DP2.SP2.EP2", 2, True, 32, [10, 20], [[10, 0], [16, 4]]),
    ]

    def test_layouts_match_expected_per_rank(self):
        for label, attn_tp, sharded, bucket, dp_reals, expected in self._LAYOUTS:
            for dp_idx, real in enumerate(dp_reals):
                for rank in range(attn_tp):
                    want = expected[dp_idx][rank]
                    with (
                        self.subTest(layout=label, dp=dp_idx, rank=rank),
                        get_parallel().override(
                            attn_tp_size=attn_tp, attn_tp_rank=rank
                        ),
                    ):
                        got_cpu = compute_local_num_token_non_padded_cpu(
                            global_num_token_non_padded=real,
                            num_tokens_per_dp=bucket,
                            sharded=sharded,
                        )
                        got_gpu = compute_local_num_token_non_padded(
                            global_num_token_non_padded=torch.tensor(real),
                            num_tokens_per_dp=bucket,
                            sharded=sharded,
                        )
                        self.assertEqual(got_cpu, want)
                        self.assertEqual(int(got_gpu), want)


LOCAL, GLOBAL = 7, 30


def _forward_batch(
    *,
    sharded: bool,
    local=LOCAL,
    glob=GLOBAL,
    forward_mode=ForwardMode.DECODE,
    attn_cp_metadata=None,
) -> ForwardBatch:
    empty = torch.empty(0, dtype=torch.int32)
    batch = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=1,
        input_ids=empty,
        req_pool_indices=empty,
        seq_lens=empty,
        out_cache_loc=empty,
        seq_lens_sum=0,
    )
    batch.num_token_non_padded = (
        None if local is None else torch.tensor(local, dtype=torch.int32)
    )
    batch.global_num_token_non_padded = (
        None if glob is None else torch.tensor(glob, dtype=torch.int32)
    )
    batch.attn_tp_sequence_sharded = sharded
    batch.attn_cp_metadata = attn_cp_metadata
    return batch


def _value(batch: ForwardBatch):
    got = batch.moe_num_token_non_padded()
    return None if got is None else int(got)


class TestMoeNumTokenNonPaddedTable(CustomTestCase):
    # (label, mode, sharded, attn_dp_size, expected). Decode forwards, so the
    # MOE_FULL rows follow the FULL layout: that config all-gathers over the
    # MoE-CP group on a context-parallel extend only, which is the case below.
    _TABLE = [
        ("scattered.replicated", ScatterMode.SCATTERED, False, 1, LOCAL),
        ("scattered.sharded", ScatterMode.SCATTERED, True, 1, LOCAL),
        ("scattered.dp", ScatterMode.SCATTERED, True, 4, LOCAL),
        ("moe_full.decode.replicated", ScatterMode.MOE_FULL, False, 1, LOCAL),
        ("moe_full.decode.sharded", ScatterMode.MOE_FULL, True, 1, GLOBAL),
        ("moe_full.decode.dp", ScatterMode.MOE_FULL, True, 4, None),
        ("full.dp_gathered", ScatterMode.FULL, True, 4, None),
        ("full.dp_gathered.replicated", ScatterMode.FULL, False, 4, None),
        ("full.tp_sharded", ScatterMode.FULL, True, 1, GLOBAL),
        ("full.replicated", ScatterMode.FULL, False, 1, LOCAL),
    ]

    def test_table(self):
        for label, mode, sharded, attn_dp_size, expected in self._TABLE:
            with (
                self.subTest(case=label),
                get_parallel().override(attn_dp_size=attn_dp_size, attn_cp_size=1),
                patch.object(comm, "sparse_mlp_scatter_mode", return_value=mode),
            ):
                self.assertEqual(_value(_forward_batch(sharded=sharded)), expected)

    def test_cp_gathered_full_is_unmasked(self):
        """A CP prefill must route every row: DSA / MLA CP take the FULL mode
        but all-gather across CP, which zigzag-permutes the real rows."""
        for sharded in (False, True):
            for dsa_cp, mla_cp in ((True, False), (False, True)):
                with (
                    self.subTest(sharded=sharded, dsa_cp=dsa_cp),
                    get_parallel().override(attn_dp_size=1, attn_cp_size=2),
                    patch.object(
                        comm, "sparse_mlp_scatter_mode", return_value=ScatterMode.FULL
                    ),
                    patch(
                        "sglang.srt.layers.attention.dsa.utils.dsa_use_prefill_cp",
                        return_value=dsa_cp,
                    ),
                    patch(
                        "sglang.srt.layers.cp.utils.is_mla_cp_active",
                        return_value=mla_cp,
                    ),
                ):
                    self.assertIsNone(_value(_forward_batch(sharded=sharded)))

    def test_moe_full_gathers_only_on_a_context_parallel_extend(self):
        """A moe-cp config keeps its bound on every forward but the CP extend,
        which is the only one that all-gathers over the MoE-CP group."""
        # get_moe_cp_size reads a live process group, so it is stubbed rather
        # than reached through a topology published without distributed init.
        for label, forward_mode, metadata, expected in [
            ("cp_extend", ForwardMode.EXTEND, object(), None),
            ("extend_without_cp_metadata", ForwardMode.EXTEND, None, LOCAL),
            ("decode", ForwardMode.DECODE, object(), LOCAL),
        ]:
            with (
                self.subTest(case=label),
                get_parallel().override(attn_dp_size=1, attn_cp_size=2),
                patch.object(
                    comm, "sparse_mlp_scatter_mode", return_value=ScatterMode.MOE_FULL
                ),
                patch("sglang.srt.layers.dp_attention.get_moe_cp_size", return_value=2),
                patch(
                    "sglang.srt.layers.attention.dsa.utils.dsa_use_prefill_cp",
                    return_value=False,
                ),
                patch(
                    "sglang.srt.layers.cp.utils.is_mla_cp_active", return_value=False
                ),
            ):
                got = _value(
                    _forward_batch(
                        sharded=False,
                        forward_mode=forward_mode,
                        attn_cp_metadata=metadata,
                    )
                )
                self.assertEqual(got, expected)

    def test_graph_replay_without_global_count_skips_masking(self):
        """A captured batch carries the LOCAL buffer alone, so the attn-TP
        sharded FULL case has no bound to mask with and must not use it."""
        with (
            get_parallel().override(attn_dp_size=1, attn_cp_size=1),
            patch.object(
                comm, "sparse_mlp_scatter_mode", return_value=ScatterMode.FULL
            ),
        ):
            self.assertIsNone(_value(_forward_batch(sharded=True, glob=None)))

    def test_absent_local_count_stays_absent(self):
        """Without expert parallelism the count is never filled, and routing
        must not mask every row against the unfilled buffer."""
        for mode in (ScatterMode.SCATTERED, ScatterMode.FULL, ScatterMode.MOE_FULL):
            with (
                self.subTest(mode=mode),
                get_parallel().override(attn_dp_size=1, attn_cp_size=1),
                patch.object(comm, "sparse_mlp_scatter_mode", return_value=mode),
            ):
                self.assertIsNone(_value(_forward_batch(sharded=False, local=None)))


if __name__ == "__main__":
    unittest.main()
