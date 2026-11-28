"""Regression: the TBO split must not resurrect masked idle-rank dummy tokens.

Under DP attention an idle rank's batch is rewritten into a fabricated EXTEND
batch and then padded, and ``prepare_mlp_sync_batch`` masks the fabricated rows
by setting ``num_token_non_padded(_cpu)`` to 0 so MoE top-k skips them and
attention returns early. ``TboForwardBatchPreparer`` runs right after that, and
it used to (a) recompute the children counts from ``len(batch.input_ids)`` --
the *padded* token count, which restores the dummy rows as real tokens for MoE
-- and (b) hardcode each child's ``global_num_token_non_padded_cpu`` to ``None``, so
the ``real_num_tokens == 0`` attention skip compared ``None == 0`` and never
fired. Net effect: DeepEP/pplx MAX_LEN + ``--enable-two-batch-overlap``
silently lost the whole idle-rank optimization.

CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap import two_batch_overlap
from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    compute_split_seq_index,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import (
    get_context,
    get_device,
    get_flags,
    get_parallel,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_extend_batch(*, padded_num_tokens: int, global_num_token_non_padded_cpu: int):
    # Only the fields compute_tbo_children_num_token_non_padded reads.
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        spec_info=None,
        tbo_split_seq_index=1,
        extend_seq_lens_cpu=[4, 4],
        input_ids=torch.zeros(padded_num_tokens, dtype=torch.long),
        global_num_token_non_padded_cpu=global_num_token_non_padded_cpu,
    )


def _make_decode_capture_batch(*, num_tokens: int):
    # Decode CUDA-graph capture batches do not populate the CPU mirror.
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        spec_info=None,
        tbo_split_seq_index=2,
        extend_seq_lens_cpu=None,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        global_num_token_non_padded_cpu=None,
    )


class TestTboChildrenDummyTokenMask(CustomTestCase):
    def _children_counts(self, batch):
        with get_context().override_server_args():
            with get_device().override(device="cpu"):
                return (
                    TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(
                        batch
                    ).tolist()
                )

    def test_masked_idle_parent_yields_zero_token_children(self):
        # An idle rank masked to 0 real tokens must split into two 0-token
        # children even though input_ids still holds the padded dummy rows.
        batch = _make_extend_batch(
            padded_num_tokens=16, global_num_token_non_padded_cpu=0
        )
        self.assertEqual(self._children_counts(batch), [0, 0])

    def test_padding_is_not_counted_as_real_tokens(self):
        # A busy rank padded up to a larger bucket must split on its real token
        # count, not on the padded input_ids length.
        batch = _make_extend_batch(
            padded_num_tokens=16, global_num_token_non_padded_cpu=8
        )
        self.assertEqual(self._children_counts(batch), [4, 4])

    def test_cpu_pair_matches_device_pair(self):
        # prepare() derives the children's CPU counts separately from the device
        # tensor; the two must not drift apart.
        batch = _make_extend_batch(
            padded_num_tokens=16, global_num_token_non_padded_cpu=5
        )
        cpu_pair = TboForwardBatchPreparer._split_num_token_non_padded(
            tbo_split_token_index=TboForwardBatchPreparer._compute_split_token_index(
                batch
            ),
            num_token_non_padded=batch.global_num_token_non_padded_cpu,
        )
        self.assertEqual(list(cpu_pair), self._children_counts(batch))

    def test_filter_batch_propagates_cpu_count_to_child(self):
        # Without this the attention 0-token skip (which reads
        # global_num_token_non_padded_cpu) compares None == 0 and never fires.
        bs = 8
        parent = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=torch.zeros(bs, dtype=torch.long),
            positions=torch.zeros(bs, dtype=torch.long),
            out_cache_loc=torch.zeros(bs, dtype=torch.long),
            req_pool_indices=torch.zeros(bs, dtype=torch.long),
            seq_lens=torch.ones(bs, dtype=torch.int32),
            seq_lens_cpu=torch.ones(bs, dtype=torch.int32),
            seq_lens_sum=bs,
            spec_info=None,
        )
        with (
            get_context().override_server_args(
                attention_backend="fa3", moe_dense_tp_size=None
            ),
            get_parallel().override(attn_tp_size=1),
        ):
            child = TboForwardBatchPreparer.filter_batch(
                parent,
                start_token_index=0,
                end_token_index=4,
                start_seq_index=0,
                end_seq_index=4,
                out_num_token_non_padded=torch.tensor(0),
                out_num_token_non_padded_cpu=0,
            )
        self.assertEqual(child.global_num_token_non_padded_cpu, 0)

    def test_capture_count_falls_back_to_physical_rows(self):
        batch = _make_decode_capture_batch(num_tokens=8)
        self.assertEqual(self._children_counts(batch), [2, 6])

    def test_prepare_falls_back_to_physical_rows_for_missing_cpu_count(self):
        batch = _make_decode_capture_batch(num_tokens=8)

        with (
            patch.object(
                TboForwardBatchPreparer,
                "compute_tbo_children_num_token_non_padded",
                return_value=torch.tensor([2, 6], dtype=torch.int32),
            ),
            patch.object(TboForwardBatchPreparer, "prepare_raw") as prepare_raw,
        ):
            TboForwardBatchPreparer.prepare(batch)

        prepare_raw.assert_called_once()
        self.assertEqual(
            prepare_raw.call_args.kwargs["tbo_children_num_token_non_padded_cpu"],
            (2, 6),
        )


class TestTboTwoChunkSplit(CustomTestCase):
    """The children's seq range must follow the token split prepare_raw is given."""

    def _child_a_end(self, *, forward_mode, extend_seq_lens):
        num_tokens = sum(extend_seq_lens)
        batch = SimpleNamespace(
            forward_mode=forward_mode,
            spec_info=None,
            extend_seq_lens_cpu=extend_seq_lens,
            input_ids=torch.zeros(num_tokens, dtype=torch.long),
            batch_size=len(extend_seq_lens),
            tbo_children=None,
        )
        with (
            get_context().override_server_args(),
            patch.object(
                two_batch_overlap,
                "get_attn_backend",
                return_value=Mock(spec=TboAttnBackend),
            ),
            patch.object(TboForwardBatchPreparer, "filter_batch") as filter_batch,
            patch.object(
                TboForwardBatchPreparer,
                "derive_fields_related_to_seq_len_for_two_chunk",
            ),
        ):
            batch.tbo_split_seq_index = compute_split_seq_index(
                forward_mode=forward_mode,
                num_tokens=num_tokens,
                extend_lens=extend_seq_lens,
                token_num_per_seq=None,
            )
            TboForwardBatchPreparer.prepare_raw(
                batch, tbo_children_num_token_non_padded=(None, None)
            )
        child_a = filter_batch.call_args_list[0].kwargs
        return child_a["end_token_index"], child_a["end_seq_index"]

    def test_mixed_cut_inside_a_seq_keeps_that_seq_in_child_a(self):
        # A prefill chunk plus two decode rows splits 11 / 11 tokens inside seq 0,
        # so child_a ends with seq 0.
        self.assertEqual(
            self._child_a_end(
                forward_mode=ForwardMode.MIXED, extend_seq_lens=[20, 1, 1]
            ),
            (11, 1),
        )

    def test_cut_on_a_seq_boundary_stays_seq_aligned(self):
        # 11 // 2 lands exactly after seq 0; child_a must not end on a seq with
        # zero tokens, so the split moves to the seq boundary after seq 1.
        self.assertEqual(
            self._child_a_end(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=[5, 1, 1, 1, 1, 1, 1],
            ),
            (6, 2),
        )


class TestTboPlainTpSplit(CustomTestCase):
    """On plain TP only an eager extend or mixed batch of at least
    --tbo-min-extend-tokens tokens splits, and only into two non-empty halves;
    speculative batches stay whole."""

    def _split(self, extend_seq_lens, *, spec=SpeculativeAlgorithm.NONE):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            spec_algorithm=spec,
            extend_num_tokens=sum(extend_seq_lens),
            extend_seq_lens_cpu=extend_seq_lens,
            tbo_split_seq_index=None,
        )
        with (
            get_context().override_server_args(tbo_min_extend_tokens=8),
            get_flags().moe.override(
                tbo_enabled=True, tbo_token_distribution_threshold=0
            ),
            patch.object(TboForwardBatchPreparer, "prepare") as prepare,
        ):
            TboForwardBatchPreparer.prepare_plain_tp(batch)
        self.assertEqual(prepare.called, batch.tbo_split_seq_index is not None)
        return batch.tbo_split_seq_index

    def test_only_a_large_target_batch_splits(self):
        self.assertEqual(self._split([4, 4]), 1)
        self.assertIsNone(self._split([4, 3]))
        self.assertIsNone(self._split([4, 4], spec=SpeculativeAlgorithm.EAGLE))
        # Without a two-chunk split one request would leave child_a empty.
        self.assertIsNone(self._split([8]))


if __name__ == "__main__":
    unittest.main()
