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
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboDPAttentionPreparer,
    TboForwardBatchPreparer,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context, get_device, get_parallel
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
        parent.tbo_disable_reason = "batch_too_small"
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
        self.assertEqual(child.tbo_disable_reason, "batch_too_small")

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


class TestTboAdmissionPolicy(CustomTestCase):
    def _admit(
        self,
        *,
        num_tokens,
        threshold,
        min_child_tokens,
        forward_mode=ForwardMode.DECODE,
        extend_seq_lens=None,
    ):
        preparer = TboDPAttentionPreparer()
        preparer.local_tbo_split_seq_index = num_tokens // 2
        batch = SimpleNamespace(
            forward_mode=forward_mode,
            extend_seq_lens=extend_seq_lens,
        )
        with get_context().override_server_args(
            enable_two_batch_overlap=True,
            tbo_admission_policy="threshold",
            tbo_prefill_token_threshold=threshold,
            tbo_decode_token_threshold=threshold,
            tbo_min_child_tokens=min_child_tokens,
        ):
            token_num_per_seq = 1 if forward_mode.is_decode() else None
            return preparer._admit_by_threshold(batch, num_tokens, token_num_per_seq)

    def test_small_batch_is_rejected(self):
        self.assertEqual(
            self._admit(num_tokens=8, threshold=16, min_child_tokens=1),
            (False, "batch_too_small"),
        )

    def test_child_minimum_is_enforced(self):
        self.assertEqual(
            self._admit(num_tokens=8, threshold=0, min_child_tokens=8),
            (False, "child_too_small"),
        )

    def test_empty_child_is_rejected(self):
        self.assertEqual(
            self._admit(num_tokens=1, threshold=0, min_child_tokens=0),
            (False, "empty_child"),
        )

    def test_prefill_uses_prefill_threshold(self):
        self.assertEqual(
            self._admit(
                num_tokens=8,
                threshold=16,
                min_child_tokens=1,
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=[4, 4],
            ),
            (False, "batch_too_small"),
        )

    def test_always_policy_preserves_current_admission(self):
        preparer = TboDPAttentionPreparer()
        preparer.local_tbo_split_seq_index = 4
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            extend_seq_lens=None,
        )
        with get_context().override_server_args(
            enable_two_batch_overlap=True,
            tbo_admission_policy="always",
            tbo_decode_token_threshold=1024,
            tbo_min_child_tokens=1024,
        ):
            self.assertEqual(
                preparer._admit_by_threshold(batch, 8, 1),
                (True, "none"),
            )

    def test_balanced_batch_is_admitted(self):
        self.assertEqual(
            self._admit(num_tokens=32, threshold=16, min_child_tokens=8),
            (True, "none"),
        )

    def test_global_threshold_uses_token_column(self):
        preparer = TboDPAttentionPreparer()
        preparer.enable_two_batch_overlap = True
        preparer.local_tbo_split_seq_index = 4
        info = torch.tensor(
            [
                [8, 0, 0, 0, 1, ForwardMode.DECODE.value],
                [0, 0, 0, 0, 1, ForwardMode.IDLE.value],
            ],
            dtype=torch.int64,
        )
        with get_context().override_server_args(
            enable_two_batch_overlap=True,
            tbo_admission_policy="threshold",
            tbo_decode_token_threshold=16,
        ):
            split_seq_index, global_forward_mode = preparer.compute_output(info)
        self.assertIsNone(split_seq_index)
        self.assertIsNone(global_forward_mode)
        self.assertEqual(preparer.last_disable_reason, "batch_too_small")


if __name__ == "__main__":
    unittest.main()
