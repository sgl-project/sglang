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
second table locks that MoE input x layout x CP decision.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    compute_local_num_token_non_padded,
    compute_local_num_token_non_padded_cpu,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.communicator_patch import patch_communicator
from sglang.test.test_utils import CustomTestCase


@contextmanager
def sparse_moe_input(rows):
    """Where a sparse MoE's input is: on each rank's local rows ("local": a2a
    dispatch, FP4 all-gather or dwdp), on rows a GQA prefill CP gathers over the
    MoE-CP group ("moe_cp", on a CP extend only), or on the FFN's rows ("ffn")."""
    local = rows == "local"
    with (
        patch.object(
            moe_utils, "is_moe_input_scattered_across_dp_ranks", return_value=local
        ),
        patch_communicator(
            "is_moe_input_scattered_across_dp_ranks", return_value=local
        ),
        patch_communicator("is_enable_moe_cp_allgather", return_value=rows == "moe_cp"),
        patch_communicator("_cp_gathers_over_attn_cp", return_value=False),
    ):
        yield


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
    def test_interleave_cp8_ep8_masks_only_physical_padding(self):
        from unittest.mock import PropertyMock

        from sglang.srt.layers.cp.base import ContextParallelStrategy
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy

        strategy = InterleaveCPStrategy(cp_size=8)
        for total in (8, 11, 64, 65):
            for rank in range(8):
                with (
                    self.subTest(total=total, rank=rank),
                    patch.object(
                        ContextParallelStrategy,
                        "cp_rank",
                        new_callable=PropertyMock,
                        return_value=rank,
                    ),
                    patch(
                        "sglang.srt.layers.cp.base.get_cp_strategy",
                        return_value=strategy,
                    ),
                    sparse_moe_input("local"),
                ):
                    metadata = strategy.build_metadata(total, [total])
                    logical = list(metadata.per_rank_actual_token)
                    metadata.per_rank_logical_token = logical
                    metadata.per_rank_actual_token = [16] * 8
                    batch = _forward_batch(
                        sharded=False,
                        local=total,
                        forward_mode=ForwardMode.EXTEND,
                        attn_cp_metadata=metadata,
                    )
                    self.assertEqual(_value(batch), len(range(rank, total, 8)))
                    cached = batch.moe_num_token_non_padded()
                    self.assertIs(batch.moe_num_token_non_padded(), cached)
                    batch.forward_mode = ForwardMode.DECODE
                    self.assertEqual(_value(batch), total)

    # (label, MoE input, sharded, attn_dp_size, expected). Decode forwards, so
    # the MoE-CP rows follow the FFN rows' layout: that config all-gathers over
    # the MoE-CP group on a context-parallel extend only, which is the case below.
    _TABLE = [
        ("scattered.replicated", "local", False, 1, LOCAL),
        ("scattered.sharded", "local", True, 1, LOCAL),
        ("scattered.dp", "local", True, 4, LOCAL),
        ("moe_full.decode.replicated", "moe_cp", False, 1, LOCAL),
        ("moe_full.decode.sharded", "moe_cp", True, 1, GLOBAL),
        ("moe_full.decode.dp", "moe_cp", True, 4, None),
        ("full.dp_gathered", "ffn", True, 4, None),
        ("full.dp_gathered.replicated", "ffn", False, 4, None),
        ("full.tp_sharded", "ffn", True, 1, GLOBAL),
        ("full.replicated", "ffn", False, 1, LOCAL),
    ]

    def test_table(self):
        for label, mode, sharded, attn_dp_size, expected in self._TABLE:
            with (
                self.subTest(case=label),
                get_parallel().override(attn_dp_size=attn_dp_size, attn_cp_size=1),
                sparse_moe_input(mode),
            ):
                self.assertEqual(_value(_forward_batch(sharded=sharded)), expected)

    def test_cp_gathered_full_is_unmasked(self):
        """A CP prefill must route every row: DSA / MLA CP put a MoE on the
        FFN rows but all-gather across CP, which zigzag-permutes the real rows."""
        for sharded in (False, True):
            for dsa_cp, mla_cp in ((True, False), (False, True)):
                with (
                    self.subTest(sharded=sharded, dsa_cp=dsa_cp),
                    get_parallel().override(attn_dp_size=1, attn_cp_size=2),
                    sparse_moe_input("ffn"),
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
        cp_metadata = SimpleNamespace(per_rank_actual_token=[2, 2])
        """A moe-cp config keeps its bound on every forward but the CP extend,
        which is the only one that all-gathers over the MoE-CP group."""
        # get_moe_cp_size reads a live process group, so it is stubbed rather
        # than reached through a topology published without distributed init.
        for label, forward_mode, metadata, expected in [
            ("cp_extend", ForwardMode.EXTEND, cp_metadata, None),
            ("extend_without_cp_metadata", ForwardMode.EXTEND, None, LOCAL),
            ("decode", ForwardMode.DECODE, cp_metadata, LOCAL),
        ]:
            with (
                self.subTest(case=label),
                get_parallel().override(attn_dp_size=1, attn_cp_size=2),
                sparse_moe_input("moe_cp"),
                patch_communicator("get_moe_cp_size", return_value=2),
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
        sharded case on the FFN rows has no bound to mask with and must not use it."""
        with (
            get_parallel().override(attn_dp_size=1, attn_cp_size=1),
            sparse_moe_input("ffn"),
        ):
            self.assertIsNone(_value(_forward_batch(sharded=True, glob=None)))

    def test_absent_local_count_stays_absent(self):
        """Without expert parallelism the count is never filled, and routing
        must not mask every row against the unfilled buffer."""
        for mode in ("local", "ffn", "moe_cp"):
            with (
                self.subTest(mode=mode),
                get_parallel().override(attn_dp_size=1, attn_cp_size=1),
                sparse_moe_input(mode),
            ):
                self.assertIsNone(_value(_forward_batch(sharded=False, local=None)))


if __name__ == "__main__":
    unittest.main()
