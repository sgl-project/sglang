"""Unit tests for DSpark speculative-decoding registration and arg handling."""

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import _handle_dspark
from sglang.srt.speculative.spec_info import SpecInputType, SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSparkPredicates(CustomTestCase):
    """Lock the DSpark predicate truth table that the scheduler / model_runner /
    attention backends dispatch on."""

    def setUp(self):
        self.algo = SpeculativeAlgorithm.DSPARK

    def test_identity_predicates(self):
        self.assertTrue(self.algo.is_dspark())
        self.assertTrue(self.algo.is_some())
        self.assertTrue(self.algo.is_speculative())
        for other in (
            self.algo.is_none,
            self.algo.is_eagle,
            self.algo.is_eagle3,
            self.algo.is_frozen_kv_mtp,
            self.algo.is_dflash,
            self.algo.is_standalone,
            self.algo.is_ngram,
        ):
            self.assertFalse(other(), other.__name__)

    def test_capability_predicates(self):
        # Block drafter verified by the target (like DFLASH), with its own draft
        # KV chains and target hidden states carried in, and no topk tree.
        self.assertTrue(self.algo.supports_target_verify_for_draft())
        self.assertTrue(self.algo.has_draft_kv())
        self.assertTrue(self.algo.carries_draft_hidden_states())
        self.assertFalse(self.algo.need_topk())

    def test_from_string_resolves(self):
        self.assertIs(
            SpeculativeAlgorithm.from_string("DSPARK"), SpeculativeAlgorithm.DSPARK
        )
        self.assertIs(
            SpeculativeAlgorithm.from_string("dspark"), SpeculativeAlgorithm.DSPARK
        )

    def test_spec_input_types_classified(self):
        from sglang.srt.speculative.spec_info import SpecInput

        draft = SimpleNamespace(spec_input_type=SpecInputType.DSPARK_DRAFT)
        verify = SimpleNamespace(spec_input_type=SpecInputType.DSPARK_VERIFY)
        self.assertTrue(SpecInput.is_draft_input(draft))
        self.assertTrue(SpecInput.is_verify_input(verify))
        self.assertFalse(SpecInput.is_verify_input(draft))
        self.assertFalse(SpecInput.is_draft_input(verify))


def _make_server_args(**overrides):
    base = dict(
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        pp_size=1,
        speculative_draft_model_path="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        speculative_draft_model_revision="main",
        model_path="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        revision="main",
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_dspark_block_size=5,
        speculative_num_draft_tokens=None,
        speculative_dspark_confidence_threshold=0.0,
        max_running_requests=None,
        enable_mixed_chunk=False,
        trust_remote_code=False,
        json_model_override_args="{}",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestHandleDspark(CustomTestCase):
    def test_pins_steps_and_topk_and_block_size(self):
        args = _make_server_args()
        _handle_dspark(args)
        self.assertEqual(args.speculative_num_steps, 1)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_draft_tokens, 5)
        self.assertEqual(args.max_running_requests, 48)

    def test_explicit_num_draft_tokens_used_when_block_size_alias_unset(self):
        args = _make_server_args(
            speculative_dspark_block_size=None,
            speculative_num_draft_tokens=6,
        )
        _handle_dspark(args)
        self.assertEqual(args.speculative_num_draft_tokens, 6)

    def test_req_to_token_headroom_covers_dspark_decode_reserve_topk_one(self):
        from sglang.srt.mem_cache.common import get_req_to_token_extra_context_len

        args = SimpleNamespace(
            speculative_algorithm="DSPARK",
            speculative_num_steps=1,
            speculative_eagle_topk=1,
            max_speculative_num_draft_tokens=6,
            page_size=1,
        )

        self.assertEqual(get_req_to_token_extra_context_len(args), 12)

    def test_spec_prepare_routes_dspark_to_draft_input_prepare(self):
        from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

        calls = []
        spec_info = SimpleNamespace(prepare_for_decode=lambda batch: calls.append(batch))
        spec_algorithm = SimpleNamespace(
            is_dflash=lambda: False,
            is_dspark=lambda: True,
        )
        batch = SimpleNamespace(spec_algorithm=spec_algorithm, spec_info=spec_info)

        with patch(
            "sglang.srt.speculative.eagle_utils.eagle_prepare_for_decode",
            side_effect=AssertionError("DSpark must not use EAGLE prepare"),
        ):
            spec_prepare_for_decode(batch)

        self.assertEqual(calls, [batch])

    def test_overrides_steps_and_topk_but_preserves_explicit_max_requests(self):
        args = _make_server_args(
            speculative_num_steps=3,
            speculative_eagle_topk=4,
            max_running_requests=96,
        )
        _handle_dspark(args)
        self.assertEqual(args.speculative_num_steps, 1)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.max_running_requests, 96)

    def test_block_size_alias_conflict_raises(self):
        args = _make_server_args(
            speculative_dspark_block_size=5, speculative_num_draft_tokens=8
        )
        with self.assertRaisesRegex(ValueError, "must match"):
            _handle_dspark(args)

    def test_nonpositive_block_size_raises(self):
        args = _make_server_args(speculative_dspark_block_size=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            _handle_dspark(args)

    def test_confidence_threshold_bounds(self):
        args = _make_server_args(speculative_dspark_confidence_threshold=1.5)
        with self.assertRaisesRegex(ValueError, "confidence-threshold"):
            _handle_dspark(args)

    def test_confidence_threshold_accepts_endpoints_and_rejects_negative(self):
        for threshold in (0.0, 1.0):
            args = _make_server_args(speculative_dspark_confidence_threshold=threshold)
            _handle_dspark(args)
            self.assertEqual(args.speculative_dspark_confidence_threshold, threshold)

        args = _make_server_args(speculative_dspark_confidence_threshold=-0.1)
        with self.assertRaisesRegex(ValueError, "confidence-threshold"):
            _handle_dspark(args)

    def test_draft_path_defaults_to_target(self):
        args = _make_server_args(
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
            model_path="my/target",
            revision="abc",
        )
        _handle_dspark(args)
        self.assertEqual(args.speculative_draft_model_path, "my/target")
        self.assertEqual(args.speculative_draft_model_revision, "abc")

    def test_dp_attention_enables_dp_lm_head(self):
        args = _make_server_args(enable_dp_attention=True)
        _handle_dspark(args)
        self.assertTrue(args.enable_dp_lm_head)

    def test_pipeline_parallel_rejected(self):
        args = _make_server_args(pp_size=2)
        with self.assertRaisesRegex(ValueError, "pp_size"):
            _handle_dspark(args)

    def test_mixed_chunk_disabled(self):
        args = _make_server_args(enable_mixed_chunk=True)
        _handle_dspark(args)
        self.assertFalse(args.enable_mixed_chunk)


class TestDSparkDraftInputBatch(CustomTestCase):
    def setUp(self):
        try:
            import torch

            from sglang.srt.speculative.dspark_info import DSparkDraftInputV2
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark_info unavailable on this runner: {e}")
        self.torch = torch
        self.cls = DSparkDraftInputV2

    def _make(self, bs, with_cur=True):
        t = self.torch
        return self.cls(
            bonus_tokens=t.arange(bs, dtype=t.int64),
            new_seq_lens=t.full((bs,), 10, dtype=t.int64),
            cur_allocated_seq_lens_cpu=(
                t.full((bs,), 12, dtype=t.int32) if with_cur else None
            ),
        )

    def test_merge_then_filter_asymmetric_cur_allocated(self):
        # Regression: merging an input without cur_allocated_seq_lens_cpu into one
        # that has it must drop it to None rather than leave a stale-sized array that
        # filter_batch then indexes out of bounds (the serving crash in merge_batch).
        a = self._make(2, with_cur=True)
        b = self._make(1, with_cur=False)
        a.merge_batch(b)
        self.assertEqual(len(a.bonus_tokens), 3)
        self.assertIsNone(a.cur_allocated_seq_lens_cpu)
        a.filter_batch(self.torch.tensor([0, 2], dtype=self.torch.int64))
        self.assertEqual(len(a.bonus_tokens), 2)

    def test_filter_batch_indexes_cur_allocated(self):
        a = self._make(4, with_cur=True)
        a.filter_batch(self.torch.tensor([1, 3], dtype=self.torch.int64))
        self.assertEqual(a.bonus_tokens.tolist(), [1, 3])
        self.assertEqual(len(a.cur_allocated_seq_lens_cpu), 2)

    def test_filter_batch_drops_subset_prefill_tail_payload(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10, 11, 12], dtype=t.int64),
            new_seq_lens=t.tensor([100, 101, 102], dtype=t.int64),
            prefill_tail_hidden_states=t.tensor([[[1.0, 2.0]]]),
            prefill_tail_valid_mask=t.tensor([[True]]),
            transfer_warmup_rounds=t.tensor([2, 1, 0], dtype=t.int32),
        )

        a.filter_batch(t.tensor([2, 0], dtype=t.int64))

        self.assertEqual(a.bonus_tokens.tolist(), [12, 10])
        self.assertEqual(tuple(a.prefill_tail_hidden_states.shape), (0, 1, 2))
        self.assertEqual(tuple(a.prefill_tail_valid_mask.shape), (0, 1))
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [0, 2])

    def test_filter_batch_drops_subset_prefill_tail_payload_with_future_indices(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10, 11, 12], dtype=t.int64),
            new_seq_lens=t.tensor([100, 101, 102], dtype=t.int64),
            prefill_tail_hidden_states=t.tensor([[[1.0, 2.0]]]),
            prefill_tail_valid_mask=t.tensor([[True]]),
            transfer_warmup_rounds=t.tensor([2, 1, 0], dtype=t.int32),
            future_indices=t.tensor([5, 6, 7], dtype=t.int64),
        )

        a.filter_batch(t.tensor([2, 0], dtype=t.int64))

        self.assertEqual(a.future_indices.tolist(), [7, 5])
        self.assertEqual(tuple(a.prefill_tail_hidden_states.shape), (0, 1, 2))
        self.assertEqual(tuple(a.prefill_tail_valid_mask.shape), (0, 1))
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [0, 2])

    def test_prepare_for_decode_uses_req_allocated_len_over_stale_carried_len(self):
        t = self.torch
        spec = self.cls(
            bonus_tokens=t.tensor([1], dtype=t.int64),
            new_seq_lens=t.tensor([10], dtype=t.int64),
            cur_allocated_seq_lens_cpu=t.tensor([999], dtype=t.int32),
        )
        req = SimpleNamespace(rid="r0", kv_committed_len=10, kv_allocated_len=12)
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            device=t.device("cpu"),
            reqs=[req],
            token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
            tree_cache=SimpleNamespace(),
            req_pool_indices=t.tensor([1], dtype=t.int64),
            req_to_token_pool=SimpleNamespace(
                req_to_token=t.zeros((2, 64), dtype=t.int32)
            ),
            seq_lens=t.tensor([10], dtype=t.int64),
            orig_seq_lens=t.tensor([10], dtype=t.int32),
            seq_lens_cpu=t.tensor([10], dtype=t.int32),
            seq_lens_sum=10,
        )
        calls = []

        def _record_assign_call(*args):
            calls.append(
                {
                    "start_offset": args[2].clone(),
                    "end_offset": args[3].clone(),
                }
            )

        with (
            patch(
                "sglang.srt.speculative.dspark_info.get_global_server_args",
                return_value=SimpleNamespace(speculative_num_draft_tokens=6),
            ),
            patch(
                "sglang.srt.speculative.dspark_info.alloc_token_slots",
                return_value=t.arange(10, dtype=t.int64),
            ),
            patch(
                "sglang.srt.speculative.dspark_info.assign_req_to_token_pool_func",
                side_effect=_record_assign_call,
            ),
        ):
            spec.prepare_for_decode(batch)

        self.assertEqual(calls[0]["start_offset"].tolist(), [12])
        self.assertEqual(calls[0]["end_offset"].tolist(), [22])
        self.assertEqual(req.kv_allocated_len, 22)
        self.assertEqual(batch.seq_lens.tolist(), [10])

    def test_prepare_for_decode_paged_last_loc_uses_allocated_len(self):
        t = self.torch
        spec = self.cls(
            bonus_tokens=t.tensor([1], dtype=t.int64),
            new_seq_lens=t.tensor([10], dtype=t.int64),
        )
        req_to_token = t.zeros((2, 64), dtype=t.int32)
        req_to_token[1, 9] = 109
        req_to_token[1, 11] = 111
        req = SimpleNamespace(rid="r0", kv_committed_len=10, kv_allocated_len=12)
        batch = SimpleNamespace(
            batch_size=lambda: 1,
            device=t.device("cpu"),
            reqs=[req],
            token_to_kv_pool_allocator=SimpleNamespace(page_size=16),
            tree_cache=SimpleNamespace(),
            req_pool_indices=t.tensor([1], dtype=t.int64),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            seq_lens=t.tensor([10], dtype=t.int64),
            orig_seq_lens=t.tensor([10], dtype=t.int32),
            seq_lens_cpu=t.tensor([10], dtype=t.int32),
            seq_lens_sum=10,
        )
        paged_calls = []

        def _record_paged_alloc(*args, **_kwargs):
            paged_calls.append({"last_loc": args[5].clone()})
            return t.arange(10, dtype=t.int64)

        with (
            patch(
                "sglang.srt.speculative.dspark_info.get_global_server_args",
                return_value=SimpleNamespace(speculative_num_draft_tokens=6),
            ),
            patch(
                "sglang.srt.speculative.dspark_info.alloc_paged_token_slots_extend",
                side_effect=_record_paged_alloc,
            ),
            patch(
                "sglang.srt.speculative.dspark_info.assign_req_to_token_pool_func",
            ),
        ):
            spec.prepare_for_decode(batch)

        self.assertEqual(paged_calls[0]["last_loc"].tolist(), [111])
        self.assertEqual(batch.seq_lens.tolist(), [10])

    def test_overlap_placeholders_inert_without_future_indices(self):
        a = self._make(2)
        self.assertTrue(a.direct_carry_valid)
        self.assertIsNone(a.future_indices)
        self.assertEqual(a.topk_p.numel(), 0)
        a.filter_batch(self.torch.tensor([1], dtype=self.torch.int64))
        self.assertEqual(len(a.bonus_tokens), 1)

    def test_transfer_warmup_rounds_survive_batch_ops(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10, 11], dtype=t.int64),
            new_seq_lens=t.tensor([100, 101], dtype=t.int64),
            transfer_warmup_rounds=t.tensor([2, 1], dtype=t.int32),
        )
        b = self.cls(
            bonus_tokens=t.tensor([12], dtype=t.int64),
            new_seq_lens=t.tensor([102], dtype=t.int64),
            transfer_warmup_rounds=t.tensor([0], dtype=t.int32),
        )
        a.merge_batch(b)
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [2, 1, 0])

        a.filter_batch(t.tensor([0, 2], dtype=t.int64))
        self.assertEqual(a.bonus_tokens.tolist(), [10, 12])
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [2, 0])

    def test_future_indices_merge_and_filter_carry_payload_by_future_row(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10, 11], dtype=t.int64),
            new_seq_lens=t.tensor([100, 101], dtype=t.int64),
            hidden_states=t.arange(2 * 2, dtype=t.float32).view(2, 2),
            hidden_valid_mask=t.tensor([[True, False], [True, True]]),
            prefill_tail_hidden_states=t.arange(2 * 2, dtype=t.float32).view(2, 1, 2),
            prefill_tail_valid_mask=t.tensor([[True], [False]]),
            transfer_warmup_rounds=t.tensor([2, 1], dtype=t.int32),
            future_indices=t.tensor([5, 6], dtype=t.int64),
        )
        b = self.cls(
            bonus_tokens=t.tensor([12], dtype=t.int64),
            new_seq_lens=t.tensor([102], dtype=t.int64),
            hidden_states=t.tensor([[4.0, 5.0]]),
            hidden_valid_mask=t.tensor([[False, True]]),
            prefill_tail_hidden_states=t.tensor([[[4.0, 5.0]]]),
            prefill_tail_valid_mask=t.tensor([[True]]),
            transfer_warmup_rounds=t.tensor([0], dtype=t.int32),
            future_indices=t.tensor([7], dtype=t.int64),
        )

        a.merge_batch(b)
        self.assertFalse(a.direct_carry_valid)
        self.assertEqual(a.future_indices.tolist(), [5, 6, 7])
        self.assertEqual(
            a.hidden_states.tolist(), [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
        )
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [2, 1, 0])

        a.filter_batch(t.tensor([2, 0], dtype=t.int64))
        self.assertFalse(a.direct_carry_valid)
        self.assertEqual(a.future_indices.tolist(), [7, 5])
        self.assertEqual(a.hidden_valid_mask.tolist(), [[False, True], [True, False]])
        self.assertEqual(a.prefill_tail_valid_mask.tolist(), [[True], [True]])
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [0, 2])

    def test_future_indices_merge_pads_variable_prefill_tail_width(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10], dtype=t.int64),
            new_seq_lens=t.tensor([100], dtype=t.int64),
            prefill_tail_hidden_states=t.tensor([[[1.0], [2.0]]]),
            prefill_tail_valid_mask=t.tensor([[True, True]]),
            transfer_warmup_rounds=t.tensor([1], dtype=t.int32),
            future_indices=t.tensor([5], dtype=t.int64),
        )
        b = self.cls(
            bonus_tokens=t.tensor([11], dtype=t.int64),
            new_seq_lens=t.tensor([101], dtype=t.int64),
            prefill_tail_hidden_states=t.tensor([[[3.0], [4.0], [5.0]]]),
            prefill_tail_valid_mask=t.tensor([[True, True, True]]),
            transfer_warmup_rounds=t.tensor([0], dtype=t.int32),
            future_indices=t.tensor([6], dtype=t.int64),
        )

        a.merge_batch(b)

        self.assertEqual(tuple(a.prefill_tail_hidden_states.shape), (2, 3, 1))
        self.assertEqual(
            a.prefill_tail_hidden_states.tolist(),
            [[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]]],
        )
        self.assertEqual(
            a.prefill_tail_valid_mask.tolist(),
            [[False, True, True], [True, True, True]],
        )

    def test_merge_and_filter_rich_payloads_when_both_sides_have_fields(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10, 11], dtype=t.int64),
            new_seq_lens=t.tensor([100, 101], dtype=t.int64),
            main_hidden=t.arange(4, dtype=t.float32).view(2, 2),
            confidence=t.tensor([[1.0, 2.0], [3.0, 4.0]]),
            topk_p=t.tensor([[7.0], [8.0]]),
            topk_index=t.tensor([[7], [8]], dtype=t.int64),
            hidden_states=t.tensor([[1.0], [2.0]]),
            hidden_valid_mask=t.tensor([[True], [False]]),
            prefill_tail_hidden_states=t.tensor([[[1.0]], [[2.0]]]),
            prefill_tail_valid_mask=t.tensor([[True], [False]]),
            transfer_warmup_rounds=t.tensor([2, 1], dtype=t.int32),
            reserved_seq_lens_cpu=t.tensor([20, 21], dtype=t.int32),
            reserved_seq_lens_sum=41,
        )
        b = self.cls(
            bonus_tokens=t.tensor([12], dtype=t.int64),
            new_seq_lens=t.tensor([102], dtype=t.int64),
            main_hidden=t.tensor([[4.0, 5.0]]),
            confidence=t.tensor([[5.0, 6.0]]),
            topk_p=t.tensor([[9.0]]),
            topk_index=t.tensor([[9]], dtype=t.int64),
            hidden_states=t.tensor([[3.0]]),
            hidden_valid_mask=t.tensor([[True]]),
            prefill_tail_hidden_states=t.tensor([[[3.0]]]),
            prefill_tail_valid_mask=t.tensor([[True]]),
            transfer_warmup_rounds=t.tensor([0], dtype=t.int32),
            reserved_seq_lens_cpu=t.tensor([22], dtype=t.int32),
            reserved_seq_lens_sum=22,
        )

        a.merge_batch(b)
        self.assertEqual(a.reserved_seq_lens_cpu.tolist(), [20, 21, 22])
        self.assertEqual(a.reserved_seq_lens_sum, 63)
        self.assertEqual(
            a.main_hidden.tolist(), [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
        )
        self.assertEqual(a.confidence.tolist(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(a.topk_p.tolist(), [[7.0], [8.0], [9.0]])
        self.assertEqual(a.topk_index.tolist(), [[7], [8], [9]])

        a.filter_batch(t.tensor([2, 0], dtype=t.int64))
        self.assertEqual(a.bonus_tokens.tolist(), [12, 10])
        self.assertEqual(a.reserved_seq_lens_cpu.tolist(), [22, 20])
        self.assertEqual(a.reserved_seq_lens_sum, 42)
        self.assertEqual(a.main_hidden.tolist(), [[4.0, 5.0], [0.0, 1.0]])
        self.assertEqual(a.confidence.tolist(), [[5.0, 6.0], [1.0, 2.0]])
        self.assertEqual(a.topk_p.tolist(), [[9.0], [7.0]])
        self.assertEqual(a.hidden_valid_mask.tolist(), [[True], [True]])
        self.assertEqual(a.prefill_tail_hidden_states.tolist(), [[[3.0]], [[1.0]]])

    def test_merge_pads_variable_prefill_tail_width(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10, 11], dtype=t.int64),
            new_seq_lens=t.tensor([100, 101], dtype=t.int64),
            prefill_tail_hidden_states=t.tensor(
                [[[1.0], [2.0]], [[3.0], [4.0]]]
            ),
            prefill_tail_valid_mask=t.tensor([[True, True], [False, True]]),
            transfer_warmup_rounds=t.tensor([1, 1], dtype=t.int32),
        )
        b = self.cls(
            bonus_tokens=t.tensor([12], dtype=t.int64),
            new_seq_lens=t.tensor([102], dtype=t.int64),
            prefill_tail_hidden_states=t.tensor([[[5.0], [6.0], [7.0]]]),
            prefill_tail_valid_mask=t.tensor([[True, True, True]]),
            transfer_warmup_rounds=t.tensor([0], dtype=t.int32),
        )

        a.merge_batch(b)

        self.assertEqual(tuple(a.prefill_tail_hidden_states.shape), (3, 3, 1))
        self.assertEqual(
            a.prefill_tail_hidden_states.tolist(),
            [
                [[0.0], [1.0], [2.0]],
                [[0.0], [3.0], [4.0]],
                [[5.0], [6.0], [7.0]],
            ],
        )
        self.assertEqual(
            a.prefill_tail_valid_mask.tolist(),
            [[False, True, True], [False, False, True], [True, True, True]],
        )

    def test_merge_preserves_existing_hidden_payload_when_peer_payload_empty(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10], dtype=t.int64),
            new_seq_lens=t.tensor([100], dtype=t.int64),
            hidden_states=t.tensor([[1.0, 2.0]]),
            hidden_valid_mask=t.tensor([[True, False]]),
            prefill_tail_hidden_states=t.tensor([[[3.0, 4.0]]]),
            prefill_tail_valid_mask=t.tensor([[True]]),
            transfer_warmup_rounds=t.tensor([1], dtype=t.int32),
        )
        b = self.cls(
            bonus_tokens=t.tensor([11], dtype=t.int64),
            new_seq_lens=t.tensor([101], dtype=t.int64),
            transfer_warmup_rounds=t.tensor([0], dtype=t.int32),
        )

        a.merge_batch(b)

        self.assertEqual(a.bonus_tokens.tolist(), [10, 11])
        self.assertEqual(a.hidden_states.tolist(), [[1.0, 2.0]])
        self.assertEqual(a.hidden_valid_mask.tolist(), [[True, False]])
        self.assertEqual(a.prefill_tail_hidden_states.tolist(), [[[3.0, 4.0]]])
        self.assertEqual(a.prefill_tail_valid_mask.tolist(), [[True]])
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [1, 0])

    def test_merge_adopts_peer_hidden_payload_when_self_payload_empty(self):
        t = self.torch
        a = self.cls(
            bonus_tokens=t.tensor([10], dtype=t.int64),
            new_seq_lens=t.tensor([100], dtype=t.int64),
            transfer_warmup_rounds=t.tensor([1], dtype=t.int32),
        )
        b = self.cls(
            bonus_tokens=t.tensor([11], dtype=t.int64),
            new_seq_lens=t.tensor([101], dtype=t.int64),
            hidden_states=t.tensor([[5.0, 6.0]]),
            hidden_valid_mask=t.tensor([[False, True]]),
            prefill_tail_hidden_states=t.tensor([[[7.0, 8.0]]]),
            prefill_tail_valid_mask=t.tensor([[True]]),
            transfer_warmup_rounds=t.tensor([0], dtype=t.int32),
        )

        a.merge_batch(b)

        self.assertEqual(a.bonus_tokens.tolist(), [10, 11])
        self.assertEqual(a.hidden_states.tolist(), [[5.0, 6.0]])
        self.assertEqual(a.hidden_valid_mask.tolist(), [[False, True]])
        self.assertEqual(a.prefill_tail_hidden_states.tolist(), [[[7.0, 8.0]]])
        self.assertEqual(a.prefill_tail_valid_mask.tolist(), [[True]])
        self.assertEqual(a.transfer_warmup_rounds.tolist(), [1, 0])

    def test_merge_drops_reserved_lengths_when_peer_missing_reserved_lengths(self):
        a = self.cls(
            bonus_tokens=self.torch.tensor([10], dtype=self.torch.int64),
            new_seq_lens=self.torch.tensor([100], dtype=self.torch.int64),
            reserved_seq_lens_cpu=self.torch.tensor([120], dtype=self.torch.int32),
            reserved_seq_lens_sum=120,
        )
        b = self.cls(
            bonus_tokens=self.torch.tensor([11], dtype=self.torch.int64),
            new_seq_lens=self.torch.tensor([101], dtype=self.torch.int64),
        )

        a.merge_batch(b)

        self.assertIsNone(a.reserved_seq_lens_cpu)
        self.assertIsNone(a.reserved_seq_lens_sum)

    def test_future_indices_merge_requires_peer_future_indices(self):
        a = self.cls(
            bonus_tokens=self.torch.tensor([10], dtype=self.torch.int64),
            new_seq_lens=self.torch.tensor([100], dtype=self.torch.int64),
            future_indices=self.torch.tensor([5], dtype=self.torch.int64),
        )
        b = self._make(1)

        with self.assertRaises(AssertionError):
            a.merge_batch(b)

    def test_create_idle_input_and_carry_prepare_buffers(self):
        t = self.torch
        idle = self.cls.create_idle_input(t.device("cpu"))

        self.assertTrue(idle.is_draft_input())
        self.assertEqual(idle.get_spec_adjust_token_coefficient(), (1, 1))
        self.assertEqual(idle.bonus_tokens.dtype, t.int64)
        self.assertEqual(tuple(idle.prefill_tail_hidden_states.shape), (0, 0, 0))
        self.assertIsNone(idle.verify_done)

        source = self._make(1)
        source._prepare_batch_seq_lens_cpu_buf = t.tensor([1, 2], dtype=t.int64)
        source._prepare_cur_kv_lens_cpu_buf = t.tensor([3, 4], dtype=t.int32)
        idle.carry_prepare_buffers_from(source)
        self.assertIs(
            idle._prepare_batch_seq_lens_cpu_buf,
            source._prepare_batch_seq_lens_cpu_buf,
        )
        self.assertIs(
            idle._prepare_cur_kv_lens_cpu_buf,
            source._prepare_cur_kv_lens_cpu_buf,
        )


class TestDSparkSpecInputDataclasses(CustomTestCase):
    def setUp(self):
        try:
            import torch

            from sglang.srt.speculative.dspark_info import (
                DSparkDraftBlockInput,
                DSparkDraftExtendInput,
                DSparkVerifyInput,
            )
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark_info unavailable on this runner: {e}")
        self.torch = torch
        self.draft_block_cls = DSparkDraftBlockInput
        self.draft_extend_cls = DSparkDraftExtendInput
        self.verify_cls = DSparkVerifyInput

    def test_block_and_verify_inputs_expose_spec_adjust_coefficients(self):
        t = self.torch
        draft = self.draft_block_cls(
            draft_token=t.tensor([1, 2, 3], dtype=t.int64),
            positions=t.tensor([10, 11, 12], dtype=t.int64),
            draft_token_num=3,
        )
        verify = self.verify_cls(
            draft_token=t.tensor([1, 2, 3, 4], dtype=t.int64),
            positions=t.tensor([10, 11, 12, 13], dtype=t.int64),
            draft_token_num=4,
        )

        self.assertEqual(draft.get_spec_adjust_token_coefficient(), (3, 3))
        self.assertEqual(verify.get_spec_adjust_token_coefficient(), (4, 4))
        self.assertEqual(draft.spec_input_type, SpecInputType.DSPARK_DRAFT_BLOCK)
        self.assertFalse(draft.is_draft_input())
        self.assertTrue(verify.is_verify_input())
        self.assertEqual(draft.num_tokens_per_batch, 3)
        self.assertEqual(verify.num_tokens_per_req, 4)

    def test_draft_extend_defaults_and_explicit_logprob_counts(self):
        t = self.torch
        default_extend = self.draft_extend_cls(hidden_states=t.zeros((2, 3)))
        explicit_extend = self.draft_extend_cls(
            hidden_states=t.zeros((2, 3)),
            num_tokens_per_req=2,
            num_tokens_for_logprob_per_req=1,
        )

        self.assertTrue(default_extend.is_draft_input())
        self.assertEqual(default_extend.get_spec_adjust_token_coefficient(), (1, 1))
        self.assertEqual(explicit_extend.get_spec_adjust_token_coefficient(), (2, 1))


class TestDSparkPrefillHandoff(CustomTestCase):
    def setUp(self):
        try:
            import torch

            from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark worker unavailable on this runner: {e}")
        self.torch = torch
        self.worker_cls = DSparkWorkerV2

    def _worker(self, prefill_transfer_warmup_rounds):
        worker = self.worker_cls.__new__(self.worker_cls)
        worker._prefill_transfer_warmup_rounds = prefill_transfer_warmup_rounds
        return worker

    def _accept_worker(self, *, use_confidence_gate=False, threshold=0.5):
        worker = self._worker(0)
        worker._use_confidence_gate = use_confidence_gate
        worker.confidence_threshold = threshold
        return worker

    def _make_prefill_input(self, rounds):
        t = self.torch
        worker = self._worker(rounds)
        return self.worker_cls._make_next_draft_input_prefill(
            worker,
            bonus_tokens=t.tensor([1262, 2808], dtype=t.int32),
            seq_lens=t.tensor([3676, 42], dtype=t.int32),
        )

    def test_prefill_handoff_matches_deepspec_reference_no_warmup(self):
        # DeepSpec first proposal uses prompt context hidden plus the
        # prefill-sampled anchor immediately. In SGLang that contract is encoded
        # as transfer_warmup_rounds == 0 for the handoff draft input.
        draft_input = self._make_prefill_input(rounds=0)
        self.assertEqual(draft_input.bonus_tokens.tolist(), [1262, 2808])
        self.assertEqual(draft_input.new_seq_lens.tolist(), [3676, 42])
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [0, 0])
        self.assertEqual(
            tuple(draft_input.prefill_tail_hidden_states.shape), (0, 0, 0)
        )
        self.assertEqual(tuple(draft_input.prefill_tail_valid_mask.shape), (0, 0))

    def test_prefill_handoff_preserves_explicit_tail_payload_and_cur_alloc(self):
        t = self.torch
        worker = self._worker(1)
        tail_hidden = t.arange(2 * 3 * 4, dtype=t.float32).view(2, 3, 4)
        tail_mask = t.tensor(
            [[False, True, True], [True, True, True]],
            dtype=t.bool,
        )
        cur_alloc = t.tensor([4096, 128], dtype=t.int32)

        draft_input = self.worker_cls._make_next_draft_input_prefill(
            worker,
            bonus_tokens=t.tensor([1262, 2808], dtype=t.int32),
            seq_lens=t.tensor([3676, 42], dtype=t.int32),
            cur_allocated_seq_lens_cpu=cur_alloc,
            prefill_tail_hidden_states=tail_hidden,
            prefill_tail_valid_mask=tail_mask,
        )

        self.assertEqual(draft_input.bonus_tokens.dtype, t.int64)
        self.assertEqual(draft_input.new_seq_lens.dtype, t.int64)
        self.assertIs(draft_input.cur_allocated_seq_lens_cpu, cur_alloc)
        self.assertIs(draft_input.prefill_tail_hidden_states, tail_hidden)
        self.assertIs(draft_input.prefill_tail_valid_mask, tail_mask)
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [1, 1])

    def test_service_warmup_prefill_handoff_uses_configured_rounds(self):
        draft_input = self._make_prefill_input(rounds=2)
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [2, 2])

        recovered = self.worker_cls._get_transfer_warmup_rounds(
            self._worker(0),
            draft_input,
            bs=2,
            device=self.torch.device("cpu"),
        )
        self.assertEqual(recovered.tolist(), [2, 2])

    def test_mismatched_warmup_shape_falls_back_to_zero(self):
        draft_input = self._make_prefill_input(rounds=1)
        draft_input.transfer_warmup_rounds = self.torch.tensor(
            [1], dtype=self.torch.int32
        )
        recovered = self.worker_cls._get_transfer_warmup_rounds(
            self._worker(0),
            draft_input,
            bs=2,
            device=self.torch.device("cpu"),
        )
        self.assertEqual(recovered.tolist(), [0, 0])

    def test_transfer_warmup_rounds_casts_dtype_without_changing_values(self):
        t = self.torch
        draft_input = self._make_prefill_input(rounds=1)
        draft_input.transfer_warmup_rounds = t.tensor([3, 0], dtype=t.int64)

        recovered = self.worker_cls._get_transfer_warmup_rounds(
            self._worker(0),
            draft_input,
            bs=2,
            device=t.device("cpu"),
        )

        self.assertEqual(recovered.dtype, t.int32)
        self.assertEqual(recovered.tolist(), [3, 0])

    def test_prefill_tail_hidden_packs_variable_length_batch(self):
        t = self.torch
        hidden = t.arange(10 * 2, dtype=t.float32).view(10, 2)
        tail_hidden, tail_mask = self.worker_cls._pack_prefill_tail_hidden(
            self._worker(0),
            hidden=hidden,
            extend_lens=[3, 0, 7],
        )

        self.assertEqual(tuple(tail_hidden.shape), (3, 7, 2))
        self.assertEqual(tuple(tail_mask.shape), (3, 7))
        self.assertEqual(
            tail_mask[0].tolist(), [False, False, False, False, True, True, True]
        )
        self.assertEqual(tail_mask[1].tolist(), [False] * 7)
        self.assertEqual(tail_mask[2].tolist(), [True] * 7)
        self.assertTrue(t.equal(tail_hidden[0, 4:], hidden[:3]))
        self.assertTrue(t.equal(tail_hidden[1], t.zeros_like(tail_hidden[1])))
        self.assertTrue(t.equal(tail_hidden[2], hidden[3:10]))

    def test_prefill_tail_hidden_caps_to_last_128_tokens(self):
        t = self.torch
        hidden = t.arange(130, dtype=t.float32).view(130, 1)
        tail_hidden, tail_mask = self.worker_cls._pack_prefill_tail_hidden(
            self._worker(0),
            hidden=hidden,
            extend_lens=[130],
        )

        self.assertEqual(tuple(tail_hidden.shape), (1, 128, 1))
        self.assertTrue(tail_mask.all())
        self.assertEqual(tail_hidden[:, 0, 0].item(), 2.0)
        self.assertEqual(tail_hidden[:, -1, 0].item(), 129.0)

    def test_prefill_tail_hidden_empty_or_nonpositive_lens_returns_empty_payload(self):
        t = self.torch
        hidden = t.empty((0, 3), dtype=t.float32)
        tail_hidden, tail_mask = self.worker_cls._pack_prefill_tail_hidden(
            self._worker(0),
            hidden=hidden,
            extend_lens=[4],
        )

        self.assertEqual(tuple(tail_hidden.shape), (0, 0))
        self.assertEqual(tuple(tail_mask.shape), (0, 0))
        self.assertEqual(tail_hidden.dtype, hidden.dtype)
        self.assertEqual(tail_mask.dtype, t.bool)

        hidden = t.arange(6, dtype=t.float32).view(3, 2)
        tail_hidden, tail_mask = self.worker_cls._pack_prefill_tail_hidden(
            self._worker(0),
            hidden=hidden,
            extend_lens=[0, -2],
        )
        self.assertEqual(tuple(tail_hidden.shape), (0, 0))
        self.assertEqual(tuple(tail_mask.shape), (0, 0))

    def test_decode_anchor_tokens_mix_prompt_output_and_fallback(self):
        t = self.torch
        worker = self._worker(0)
        worker.noise_token_id = 999
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(origin_input_ids=[10, 11, 12], output_ids=[]),
                SimpleNamespace(origin_input_ids=[20, 21], output_ids=[200, 201]),
                SimpleNamespace(origin_input_ids=[30, 31], output_ids=[]),
                SimpleNamespace(origin_input_ids=[], output_ids=[]),
            ],
            seq_lens_cpu=None,
        )
        anchors = self.worker_cls._get_decode_anchor_tokens(
            worker,
            batch=batch,
            prefix_lens=t.tensor([3, 4, 5, 0], dtype=t.int64),
            fallback_tokens=t.tensor([1000, 1001, 1002, 1003], dtype=t.int64),
            prefer_fallback_tokens=t.tensor([False, False, False, True]),
            bs=4,
            device=t.device("cpu"),
        )

        self.assertEqual(anchors.tolist(), [12, 201, 1002, 1003])

    def test_decode_anchor_tokens_prefers_seq_lens_cpu_and_safe_fallbacks(self):
        t = self.torch
        worker = self._worker(0)
        worker.noise_token_id = 999
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(origin_input_ids=[10, 11], output_ids=[200]),
                SimpleNamespace(origin_input_ids=[20], output_ids=[]),
                SimpleNamespace(origin_input_ids=None, output_ids=[300]),
                SimpleNamespace(origin_input_ids=[], output_ids=[]),
            ],
            seq_lens_cpu=[2, 99, 1, 0],
        )

        anchors = self.worker_cls._get_decode_anchor_tokens(
            worker,
            batch=batch,
            prefix_lens=t.tensor([99, 1, 99, 99], dtype=t.int64),
            fallback_tokens=t.tensor([1000, 1001, 1002], dtype=t.int64),
            prefer_fallback_tokens=t.tensor([False, True, False, False]),
            bs=4,
            device=t.device("cpu"),
        )

        self.assertEqual(anchors.tolist(), [11, 20, 300, 999])

    def test_decode_anchor_tokens_handles_malformed_reqs_and_missing_fallbacks(self):
        t = self.torch
        worker = self._worker(0)
        worker.noise_token_id = 777

        class _BadReq:
            @property
            def origin_input_ids(self):
                raise RuntimeError("bad origin")

            @property
            def output_ids(self):
                raise RuntimeError("bad output")

        batch = SimpleNamespace(
            reqs=[
                _BadReq(),
                SimpleNamespace(origin_input_ids=[], output_ids=[]),
                SimpleNamespace(origin_input_ids=[5], output_ids=[6]),
            ],
            seq_lens_cpu=[1, 0, 99],
        )

        anchors = self.worker_cls._get_decode_anchor_tokens(
            worker,
            batch=batch,
            prefix_lens=t.tensor([1, 0, 99], dtype=t.int64),
            fallback_tokens=t.tensor([1000, 1001], dtype=t.int64),
            prefer_fallback_tokens=t.tensor([False, False], dtype=t.bool),
            bs=3,
            device=t.device("cpu"),
        )

        self.assertEqual(anchors.tolist(), [777, 777, 6])

    def _pd_materialize_worker(self):
        t = self.torch
        worker = self._worker(0)
        worker.device = t.device("cpu")
        worker._get_target_aux_hidden_size = lambda: 2
        req_to_token = t.full((64, 8), -1, dtype=t.int64)
        for req_pool_idx in range(1, 64):
            req_to_token[req_pool_idx, 4] = req_pool_idx * 100 + 4
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
        )
        return worker

    def test_disagg_hidden_bootstrap_maps_sparse_future_payload(self):
        t = self.torch
        from sglang.srt.speculative.dspark_info import DSparkDraftInputV2

        worker = self._pd_materialize_worker()
        captures = {}

        def make_forward_batch(**kwargs):
            captures["forward_kwargs"] = kwargs
            return SimpleNamespace()

        def materialize(**kwargs):
            captures["materialize_kwargs"] = kwargs

        worker._make_draft_decode_forward_batch_for_materialize = make_forward_batch
        worker._materialize_main_hidden_to_draft_state = materialize

        req_pool_indices = t.arange(1, 53, dtype=t.int64)
        draft_input = DSparkDraftInputV2(
            bonus_tokens=t.tensor([101], dtype=t.int64),
            new_seq_lens=t.tensor([5], dtype=t.int64),
            hidden_states=t.tensor([[1.0, 2.0]]),
            hidden_valid_mask=t.tensor([True]),
            future_indices=t.tensor([17], dtype=t.int64),
        )
        batch = SimpleNamespace(req_pool_indices=req_pool_indices)

        self.worker_cls._materialize_disagg_prefill_hidden_to_draft_state(
            worker,
            draft_input=draft_input,
            batch=batch,
            prefix_lens=t.full((52,), 5, dtype=t.int64),
        )

        self.assertEqual(
            captures["forward_kwargs"]["req_pool_indices"].tolist(), [17]
        )
        self.assertEqual(captures["forward_kwargs"]["cache_loc"].tolist(), [1704])
        self.assertEqual(captures["forward_kwargs"]["seq_lens_before"].tolist(), [4])
        self.assertEqual(
            captures["materialize_kwargs"]["main_hidden"].tolist(), [[1.0, 2.0]]
        )
        self.assertEqual(captures["materialize_kwargs"]["positions"].tolist(), [4])

    def test_disagg_hidden_bootstrap_maps_sparse_tail_payload_after_merge(self):
        t = self.torch
        from sglang.srt.speculative.dspark_info import DSparkDraftInputV2

        worker = self._pd_materialize_worker()
        captures = {}
        worker._make_draft_decode_forward_batch_for_materialize = (
            lambda **kwargs: captures.setdefault("forward_kwargs", kwargs)
            or SimpleNamespace()
        )
        worker._materialize_main_hidden_to_draft_state = (
            lambda **kwargs: captures.setdefault("materialize_kwargs", kwargs)
        )

        draft_input = DSparkDraftInputV2(
            bonus_tokens=t.tensor([101], dtype=t.int64),
            new_seq_lens=t.tensor([5], dtype=t.int64),
            hidden_states=t.tensor([[3.0, 4.0]]),
            hidden_valid_mask=t.tensor([True]),
        )
        batch = SimpleNamespace(req_pool_indices=t.arange(1, 53, dtype=t.int64))

        self.worker_cls._materialize_disagg_prefill_hidden_to_draft_state(
            worker,
            draft_input=draft_input,
            batch=batch,
            prefix_lens=t.full((52,), 5, dtype=t.int64),
        )

        self.assertEqual(
            captures["forward_kwargs"]["req_pool_indices"].tolist(), [52]
        )
        self.assertEqual(captures["forward_kwargs"]["cache_loc"].tolist(), [5204])
        self.assertEqual(
            captures["materialize_kwargs"]["main_hidden"].tolist(), [[3.0, 4.0]]
        )

    def test_prefill_next_input_defaults_empty_tail_payloads(self):
        t = self.torch
        worker = self._worker(2)

        draft_input = self.worker_cls._make_next_draft_input_prefill(
            worker,
            bonus_tokens=t.tensor([1, 2], dtype=t.int32),
            seq_lens=t.tensor([10, 11], dtype=t.int32),
        )

        self.assertEqual(draft_input.bonus_tokens.dtype, t.int64)
        self.assertEqual(draft_input.new_seq_lens.dtype, t.int64)
        self.assertEqual(tuple(draft_input.prefill_tail_hidden_states.shape), (0, 0, 0))
        self.assertEqual(tuple(draft_input.prefill_tail_valid_mask.shape), (0, 0))
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [2, 2])

    def test_decode_next_input_carries_warmup_rounds(self):
        t = self.torch
        draft_input = self.worker_cls._make_next_draft_input_decode(
            self._worker(0),
            bonus_tokens=t.tensor([301, 302], dtype=t.int32),
            new_seq_lens=t.tensor([3681, 9], dtype=t.int32),
            transfer_warmup_rounds=t.tensor([1, 0], dtype=t.int64),
        )

        self.assertEqual(draft_input.bonus_tokens.dtype, t.int64)
        self.assertEqual(draft_input.new_seq_lens.dtype, t.int64)
        self.assertEqual(draft_input.transfer_warmup_rounds.dtype, t.int32)
        self.assertEqual(draft_input.bonus_tokens.tolist(), [301, 302])
        self.assertEqual(draft_input.new_seq_lens.tolist(), [3681, 9])
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [1, 0])

    def test_decode_next_input_defaults_warmup_zero_and_preserves_cur_alloc(self):
        t = self.torch
        cur_alloc = t.tensor([4096, 128], dtype=t.int32)

        draft_input = self.worker_cls._make_next_draft_input_decode(
            self._worker(0),
            bonus_tokens=t.tensor([301, 302], dtype=t.int32),
            new_seq_lens=t.tensor([3681, 9], dtype=t.int32),
            cur_allocated_seq_lens_cpu=cur_alloc,
        )

        self.assertEqual(draft_input.bonus_tokens.dtype, t.int64)
        self.assertEqual(draft_input.new_seq_lens.dtype, t.int64)
        self.assertIs(draft_input.cur_allocated_seq_lens_cpu, cur_alloc)
        self.assertEqual(draft_input.transfer_warmup_rounds.dtype, t.int32)
        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [0, 0])

    def _make_disagg_batch(self):
        t = self.torch
        return SimpleNamespace(
            reqs=[
                SimpleNamespace(hidden_states_tensor=t.tensor([1.0, 2.0])),
                SimpleNamespace(hidden_states_tensor=t.tensor([3.0, 4.0])),
            ],
            batch_size=lambda: 2,
            device=t.device("cpu"),
            seq_lens=t.tensor([10, 11], dtype=t.int64),
            seq_lens_cpu=t.tensor([10, 11], dtype=t.int64),
            enable_overlap=False,
        )

    def test_dspark_disagg_warmup_matches_deepspec_handoff_default(self):
        t = self.torch
        from sglang.srt.speculative.dspark_disaggregation import (
            build_dspark_disagg_draft_input,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_DSPARK_DEEPSPEC_PREFILL_HANDOFF": "1",
                "SGLANG_DSPARK_PREFILL_TRANSFER_WARMUP_ROUNDS": "7",
            },
        ):
            draft_input = build_dspark_disagg_draft_input(
                self._make_disagg_batch(),
                SimpleNamespace(),
                t.tensor([101, 102], dtype=t.int64),
                SimpleNamespace(),
            )

        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [0, 0])

    def test_dspark_disagg_warmup_uses_prefill_env_when_handoff_disabled(self):
        t = self.torch
        from sglang.srt.speculative.dspark_disaggregation import (
            build_dspark_disagg_draft_input,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_DSPARK_DEEPSPEC_PREFILL_HANDOFF": "0",
                "SGLANG_DSPARK_PREFILL_TRANSFER_WARMUP_ROUNDS": "3",
                "SGLANG_DSPARK_TRANSFER_WARMUP_ROUNDS": "5",
            },
        ):
            draft_input = build_dspark_disagg_draft_input(
                self._make_disagg_batch(),
                SimpleNamespace(),
                t.tensor([101, 102], dtype=t.int64),
                SimpleNamespace(),
            )

        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [3, 3])

    def test_dspark_disagg_warmup_supports_legacy_env_alias(self):
        t = self.torch
        from sglang.srt.speculative.dspark_disaggregation import (
            build_dspark_disagg_draft_input,
        )

        with patch.dict(
            os.environ,
            {
                "SGLANG_DSPARK_DEEPSPEC_PREFILL_HANDOFF": "0",
                "SGLANG_DSPARK_TRANSFER_WARMUP_ROUNDS": "4",
            },
        ):
            with patch.dict(
                os.environ,
                {"SGLANG_DSPARK_PREFILL_TRANSFER_WARMUP_ROUNDS": ""},
            ):
                os.environ.pop("SGLANG_DSPARK_PREFILL_TRANSFER_WARMUP_ROUNDS", None)
                draft_input = build_dspark_disagg_draft_input(
                    self._make_disagg_batch(),
                    SimpleNamespace(),
                    t.tensor([101, 102], dtype=t.int64),
                    SimpleNamespace(),
                )

        self.assertEqual(draft_input.transfer_warmup_rounds.tolist(), [4, 4])

    def test_accept_bonus_uses_longest_contiguous_draft_match(self):
        t = self.torch
        candidates = t.tensor(
            [
                [10, 20, 30, 40],
                [11, 21, 31, 41],
            ],
            dtype=t.int64,
        )
        target_predict = t.tensor(
            [
                [20, 30, 99, 77],
                [21, 99, 31, 88],
            ],
            dtype=t.int64,
        )
        confidence = t.zeros((2, 3), dtype=t.float32)

        commit_lens, bonus_tokens, out_tokens = (
            self.worker_cls._compute_accept_bonus_eager(
                self._accept_worker(),
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )
        )

        self.assertEqual(commit_lens.tolist(), [3, 2])
        self.assertEqual(bonus_tokens.tolist(), [99, 99])
        self.assertEqual(out_tokens.tolist(), [[20, 30, 99, 0], [21, 99, 41, 0]])
        if os.getenv("SGLANG_DSPARK_TEST_VERBOSE", "0") == "1":
            print(
                "DSpark accept parity debug: "
                + json.dumps(
                    {
                        "case": "longest_contiguous_match",
                        "candidates": candidates.tolist(),
                        "target_predict": target_predict.tolist(),
                        "commit_lens": commit_lens.tolist(),
                        "bonus_tokens": bonus_tokens.tolist(),
                        "out_tokens": out_tokens.tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    def test_accept_bonus_zero_draft_match_commits_target_first_token(self):
        t = self.torch
        candidates = t.tensor([[10, 20, 30, 40]], dtype=t.int64)
        target_predict = t.tensor([[99, 30, 40, 77]], dtype=t.int64)
        confidence = t.zeros((1, 3), dtype=t.float32)

        commit_lens, bonus_tokens, out_tokens = (
            self.worker_cls._compute_accept_bonus_eager(
                self._accept_worker(),
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )
        )

        self.assertEqual(commit_lens.tolist(), [1])
        self.assertEqual(bonus_tokens.tolist(), [99])
        self.assertEqual(out_tokens.tolist(), [[99, 30, 40, 0]])

    def test_accept_bonus_all_drafts_match_uses_final_target_as_bonus(self):
        t = self.torch
        candidates = t.tensor([[10, 20, 30, 40]], dtype=t.int64)
        target_predict = t.tensor([[20, 30, 40, 99]], dtype=t.int64)
        confidence = t.zeros((1, 3), dtype=t.float32)

        commit_lens, bonus_tokens, out_tokens = (
            self.worker_cls._compute_accept_bonus_eager(
                self._accept_worker(),
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )
        )

        self.assertEqual(commit_lens.tolist(), [4])
        self.assertEqual(bonus_tokens.tolist(), [99])
        self.assertEqual(out_tokens.tolist(), [[20, 30, 40, 99]])

    def test_accept_bonus_confidence_gate_truncates_match_prefix(self):
        t = self.torch
        candidates = t.tensor([[10, 20, 30, 40]], dtype=t.int64)
        target_predict = t.tensor([[20, 30, 40, 99]], dtype=t.int64)
        confidence = t.tensor([[10.0, -10.0, 10.0]], dtype=t.float32)

        commit_lens, bonus_tokens, out_tokens = (
            self.worker_cls._compute_accept_bonus_eager(
                self._accept_worker(use_confidence_gate=True, threshold=0.5),
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )
        )

        self.assertEqual(commit_lens.tolist(), [2])
        self.assertEqual(bonus_tokens.tolist(), [30])
        self.assertEqual(out_tokens.tolist(), [[20, 30, 40, 0]])

    def test_confident_prefix_handles_disabled_gate_and_threshold_boundary(self):
        t = self.torch
        confidence = t.tensor(
            [
                [0.0, 10.0, -10.0],
                [10.0, -10.0, 10.0],
                [-10.0, 10.0, 10.0],
            ],
            dtype=t.float32,
        )

        worker = self._accept_worker()
        self.assertEqual(
            self.worker_cls._confident_prefix(worker, confidence).tolist(), [3, 3, 3]
        )

        worker = self._accept_worker(use_confidence_gate=True, threshold=0.5)
        self.assertEqual(
            self.worker_cls._confident_prefix(worker, confidence).tolist(), [2, 1, 0]
        )

    def test_accept_bonus_confidence_gate_is_applied_per_batch_row(self):
        t = self.torch
        candidates = t.tensor(
            [
                [10, 20, 30, 40],
                [11, 21, 31, 41],
            ],
            dtype=t.int64,
        )
        target_predict = t.tensor(
            [
                [20, 30, 40, 99],
                [21, 31, 41, 88],
            ],
            dtype=t.int64,
        )
        confidence = t.tensor(
            [
                [10.0, 10.0, -10.0],
                [-10.0, 10.0, 10.0],
            ],
            dtype=t.float32,
        )

        commit_lens, bonus_tokens, out_tokens = (
            self.worker_cls._compute_accept_bonus_eager(
                self._accept_worker(use_confidence_gate=True, threshold=0.5),
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )
        )

        self.assertEqual(commit_lens.tolist(), [3, 1])
        self.assertEqual(bonus_tokens.tolist(), [40, 21])
        self.assertEqual(
            out_tokens.tolist(), [[20, 30, 40, 0], [21, 31, 41, 0]]
        )

    def test_accept_bonus_verify_stride_one_commits_only_target_bonus(self):
        t = self.torch
        candidates = t.tensor([[10], [11]], dtype=t.int64)
        target_predict = t.tensor([[20], [21]], dtype=t.int64)
        confidence = t.empty((2, 0), dtype=t.float32)

        commit_lens, bonus_tokens, out_tokens = (
            self.worker_cls._compute_accept_bonus_eager(
                self._accept_worker(),
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )
        )

        self.assertEqual(commit_lens.tolist(), [1, 1])
        self.assertEqual(bonus_tokens.tolist(), [20, 21])
        self.assertEqual(out_tokens.tolist(), [[20], [21]])

    def test_accept_bonus_rejects_mismatched_shapes(self):
        t = self.torch
        worker = self._accept_worker()
        candidates = t.tensor([[10, 20, 30]], dtype=t.int64)
        target_predict = t.tensor([[20, 30]], dtype=t.int64)
        confidence = t.zeros((1, 2), dtype=t.float32)

        with self.assertRaisesRegex(ValueError, "target_predict must match"):
            self.worker_cls._compute_accept_bonus_eager(
                worker,
                candidates=candidates,
                target_predict=target_predict,
                confidence=confidence,
            )

        with self.assertRaisesRegex(ValueError, "confidence must have"):
            self.worker_cls._compute_accept_bonus_eager(
                worker,
                candidates=candidates,
                target_predict=candidates,
                confidence=t.zeros((1, 1), dtype=t.float32),
            )


class TestDSparkRuntimeDebugEvidence(CustomTestCase):
    def setUp(self):
        try:
            import torch

            from sglang.srt.speculative.dspark_info import DSparkDraftInputV2
            from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark worker unavailable on this runner: {e}")
        self.torch = torch
        self.worker_cls = DSparkWorkerV2
        self.draft_input_cls = DSparkDraftInputV2

    def _worker(self):
        worker = self.worker_cls.__new__(self.worker_cls)
        worker.device = self.torch.device("cpu")
        worker._accept_anomaly_enabled = True
        worker._swa_write_source_by_req_pool = {}
        worker._swa_write_source_snapshot_by_req_pool = None
        worker._prefill_tail_replay_debug_by_req_pool = {}
        worker._target_aux_debug_by_req_pool = {}
        worker._boundary_debug_by_req_pool = {}
        worker._is_tp0 = lambda: True
        return worker

    def test_target_aux_payload_splits_raw_target_layers_and_norm_summaries(self):
        t = self.torch
        worker = self._worker()
        worker._draft_inner = SimpleNamespace(
            target_layer_ids=[2, 5],
            hidden_size=3,
            main_norm=SimpleNamespace(weight=t.tensor([1.0, 2.0, 3.0])),
            shared_head=SimpleNamespace(
                norm=SimpleNamespace(weight=t.tensor([4.0, 5.0, 6.0]))
            ),
        )
        raw = t.arange(12, dtype=t.float32).view(2, 6)
        projected = raw[:, :3] + 1.0

        payload = self.worker_cls._target_aux_payload(
            worker,
            raw_hidden_rows=raw,
            projected_rows=projected,
        )

        self.assertEqual(payload["target_layer_ids"], [2, 5])
        self.assertEqual(payload["decoder_layer_ids"], [2, 5])
        self.assertEqual(payload["raw"]["shape"], [2, 6])
        self.assertEqual(payload["projected"]["shape"], [2, 3])
        self.assertEqual([x["target_layer_id"] for x in payload["raw_layers"]], [2, 5])
        self.assertEqual(payload["raw_layers"][0]["hidden"]["shape"], [2, 3])
        self.assertEqual(payload["main_norm_weight"]["shape"], [3])
        self.assertEqual(payload["shared_norm_weight"]["shape"], [3])
        self.assertGreater(payload["projected_to_raw_norm_ratio_first_last"][0], 0.0)

    def test_target_aux_payload_reports_expected_width_on_shape_mismatch(self):
        t = self.torch
        worker = self._worker()
        worker._draft_inner = SimpleNamespace(
            target_layer_ids=[1, 3],
            hidden_size=4,
            main_norm=None,
            shared_head=None,
        )

        payload = self.worker_cls._target_aux_payload(
            worker,
            raw_hidden_rows=t.ones((2, 7)),
            projected_rows=t.ones((2, 4)),
        )

        self.assertEqual(payload["expected_raw_width"], 8)
        self.assertNotIn("raw_layers", payload)
        self.assertIsNone(payload["main_norm_weight"])
        self.assertIsNone(payload["shared_norm_weight"])

    def test_boundary_loc_payload_translates_full_cache_locs_to_swa_locs(self):
        t = self.torch
        worker = self._worker()

        class _TokenToKVPool:
            def translate_loc_from_full_to_swa(self, locs):
                return locs + 100

        worker.draft_model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(token_to_kv_pool=_TokenToKVPool())
        )

        payload = self.worker_cls._boundary_loc_payload(
            worker,
            positions=t.tensor([10, 11], dtype=t.int32),
            cache_locs=t.tensor([3, 4], dtype=t.int32),
            hidden_rows=t.tensor([[1.0, -2.0], [3.0, 4.0]]),
        )

        self.assertEqual(payload["positions"], [10, 11])
        self.assertEqual(payload["full_locs"], [3, 4])
        self.assertEqual(payload["swa_locs"], [103, 104])
        self.assertEqual(payload["hidden"]["shape"], [2, 2])
        self.assertEqual(payload["hidden"]["max_abs"], 4.0)

    def test_swa_write_source_tracks_tail_window_and_visible_counts(self):
        worker = self._worker()

        self.worker_cls._record_swa_locs_source(
            worker,
            req_pool_idx=7,
            swa_locs=list(range(600)),
            source="prefill_source",
        )
        self.assertLessEqual(len(worker._swa_write_source_by_req_pool[7]), 256)

        self.worker_cls._record_swa_locs_source(
            worker,
            req_pool_idx=7,
            swa_locs=[590, 591, 700],
            source="decode_verify_write",
        )
        source_debug = self.worker_cls._visible_window_source_debug(
            worker,
            7,
            [588, 589, 590, 591, 700, 701],
        )

        self.assertEqual(source_debug["visible_len"], 6)
        self.assertEqual(source_debug["counts"]["prefill_source"], 2)
        self.assertEqual(source_debug["counts"]["decode_verify_write"], 3)
        self.assertEqual(source_debug["counts"]["unknown"], 1)

    def test_visible_window_source_debug_prefers_snapshot_and_handles_empty(self):
        worker = self._worker()
        worker._swa_write_source_by_req_pool = {
            1: {10: "live_source", 11: "live_source"}
        }
        worker._swa_write_source_snapshot_by_req_pool = {
            1: {10: "snapshot_source", 12: "snapshot_source"}
        }

        self.assertIsNone(self.worker_cls._visible_window_source_debug(worker, 1, []))
        source_debug = self.worker_cls._visible_window_source_debug(
            worker,
            1,
            [10, 11, 12],
            max_sources=2,
        )

        self.assertEqual(source_debug["visible_len"], 3)
        self.assertEqual(source_debug["sources_first"], ["snapshot_source", "unknown"])
        self.assertEqual(source_debug["sources_last"], ["unknown", "snapshot_source"])
        self.assertEqual(source_debug["counts"]["snapshot_source"], 2)
        self.assertEqual(source_debug["counts"]["unknown"], 1)

    def test_record_full_cache_locs_source_uses_swa_translation(self):
        t = self.torch
        worker = self._worker()

        class _TokenToKVPool:
            def translate_loc_from_full_to_swa(self, locs):
                return locs + 10

        worker.draft_model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(token_to_kv_pool=_TokenToKVPool())
        )

        self.worker_cls._record_full_cache_locs_source(
            worker,
            req_pool_idx=3,
            cache_locs=t.tensor([1, 2, 3], dtype=t.int32),
            source="decode_verify_write",
        )

        self.assertEqual(
            worker._swa_write_source_by_req_pool[3],
            {
                11: "decode_verify_write",
                12: "decode_verify_write",
                13: "decode_verify_write",
            },
        )

    def test_boundary_debug_records_history_and_write_sources(self):
        t = self.torch
        worker = self._worker()

        class _TokenToKVPool:
            def translate_loc_from_full_to_swa(self, locs):
                return locs + 100

        worker.draft_model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(token_to_kv_pool=_TokenToKVPool())
        )
        for i in range(18):
            self.worker_cls._record_boundary_debug(
                worker,
                req_pool_idx=4,
                stage="decode_verify_write",
                positions=t.tensor([i], dtype=t.int64),
                cache_locs=t.tensor([i + 10], dtype=t.int64),
                hidden_rows=t.tensor([[float(i), 1.0]]),
                extra={"round": i},
            )

        debug = worker._boundary_debug_by_req_pool[4]
        history = debug["decode_verify_write_history"]
        self.assertEqual(len(history), 16)
        self.assertEqual(history[0]["round"], 2)
        self.assertEqual(history[-1]["round"], 17)
        self.assertEqual(
            worker._swa_write_source_by_req_pool[4][127],
            "decode_verify_write",
        )

        self.worker_cls._record_boundary_debug(
            worker,
            req_pool_idx=5,
            stage="decode_tail_replay",
            positions=t.tensor([1], dtype=t.int64),
            cache_locs=t.tensor([2], dtype=t.int64),
            hidden_rows=t.tensor([[1.0, 2.0]]),
        )
        self.assertEqual(
            worker._swa_write_source_by_req_pool[5][102],
            "prefill_tail_replay",
        )

    def test_record_target_aux_debug_respects_gate_and_records_errors(self):
        t = self.torch
        worker = self._worker()
        worker._accept_anomaly_enabled = False
        worker._draft_inner = SimpleNamespace(
            target_layer_ids=[0],
            hidden_size=2,
            main_norm=None,
            shared_head=None,
        )

        self.worker_cls._record_target_aux_debug(
            worker,
            req_pool_idx=1,
            stage="gated",
            raw_hidden_rows=t.ones((1, 2)),
            projected_rows=t.ones((1, 2)),
        )
        self.assertEqual(worker._target_aux_debug_by_req_pool, {})

        worker._accept_anomaly_enabled = True
        worker.device = "not-a-device"
        self.worker_cls._record_target_aux_debug(
            worker,
            req_pool_idx=1,
            stage="error_stage",
            raw_hidden_rows=t.ones((1, 2)),
            projected_rows=t.ones((1, 2)),
        )

        self.assertIn("error", worker._target_aux_debug_by_req_pool[1]["error_stage"])

    def test_prefill_tail_replay_debug_records_payload_shape_and_mask_counts(self):
        t = self.torch
        worker = self._worker()
        draft_input = self.draft_input_cls(
            bonus_tokens=t.tensor([10, 11], dtype=t.int64),
            new_seq_lens=t.tensor([100, 101], dtype=t.int64),
            prefill_tail_hidden_states=t.zeros((2, 3, 4)),
            prefill_tail_valid_mask=t.tensor(
                [[True, False, True], [False, False, False]]
            ),
        )

        self.worker_cls._init_prefill_tail_replay_round_debug(
            worker,
            req_pool_indices=t.tensor([5, 6], dtype=t.int64),
            draft_input=draft_input,
        )

        first = worker._prefill_tail_replay_debug_by_req_pool[5]
        second = worker._prefill_tail_replay_debug_by_req_pool[6]
        self.assertTrue(first["round_has_payload"])
        self.assertEqual(first["mask_valid_count"], 2)
        self.assertEqual(first["hidden_shape"], [2, 3, 4])
        self.assertEqual(second["mask_valid_count"], 0)
        self.assertEqual(second["skip_reason"], "empty_mask_row")

    def test_compressor_materialization_rebuilds_and_restores_backend_metadata(self):
        import contextlib

        t = self.torch
        worker = self._worker()
        calls = []
        old_metadata = {"old": True}

        class _DraftModel:
            def project_main_hidden(self, hidden):
                calls.append(("project", hidden.clone()))
                return hidden + 10.0

        class _AttnBackend:
            def __init__(self):
                self.forward_metadata = old_metadata

            def init_forward_metadata(self, forward_batch):
                calls.append(("init_metadata", forward_batch.name))
                self.forward_metadata = {"batch": forward_batch.name}

            def forward_core_compressor(
                self,
                main_x,
                forward_batch,
                layer_id,
                compressor,
            ):
                calls.append(
                    (
                        "compressor",
                        layer_id,
                        compressor,
                        main_x.clone(),
                        forward_batch.name,
                        self.forward_metadata,
                    )
                )

        attn_backend = _AttnBackend()
        worker.draft_model = _DraftModel()
        worker.draft_model_runner = SimpleNamespace(
            tp_group="tp0",
            attn_backend=attn_backend,
        )
        worker.draft_tp_context = lambda tp_group: contextlib.nullcontext(tp_group)
        worker._draft_inner = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(layer_id=0, compressor="c0")
                ),
                SimpleNamespace(
                    self_attn=SimpleNamespace(layer_id=1, compressor=None)
                ),
                SimpleNamespace(
                    self_attn=SimpleNamespace(layer_id=2, compressor="c2")
                ),
            ]
        )
        hidden = t.arange(6, dtype=t.float32).view(2, 3)
        forward_batch = SimpleNamespace(name="materialize_batch")

        self.worker_cls._materialize_main_hidden_to_draft_compressors(
            worker,
            main_hidden=hidden,
            draft_forward_batch=forward_batch,
            projected=False,
        )

        self.assertIs(attn_backend.forward_metadata, old_metadata)
        self.assertEqual(calls[0][0], "init_metadata")
        self.assertEqual(calls[1][0], "project")
        compressor_calls = [call for call in calls if call[0] == "compressor"]
        self.assertEqual([call[1] for call in compressor_calls], [0, 2])
        self.assertTrue(t.equal(compressor_calls[0][3], hidden + 10.0))
        self.assertEqual(compressor_calls[0][5], {"batch": "materialize_batch"})

    def test_compressor_materialization_projected_hidden_skips_projection(self):
        import contextlib

        t = self.torch
        worker = self._worker()
        calls = []

        class _DraftModel:
            def project_main_hidden(self, _hidden):
                raise AssertionError("project_main_hidden should not be called")

        class _AttnBackend:
            forward_metadata = None

            def init_forward_metadata(self, forward_batch):
                calls.append(("init_metadata", forward_batch.name))

            def forward_core_compressor(
                self,
                main_x,
                _forward_batch,
                layer_id,
                _compressor,
            ):
                calls.append(("compressor", layer_id, main_x.clone()))

        worker.draft_model = _DraftModel()
        worker.draft_model_runner = SimpleNamespace(
            tp_group=None,
            attn_backend=_AttnBackend(),
        )
        worker.draft_tp_context = lambda tp_group: contextlib.nullcontext()
        worker._draft_inner = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(layer_id=7, compressor="compressor")
                )
            ]
        )
        projected_hidden = t.ones((2, 3), dtype=t.float32)

        self.worker_cls._materialize_main_hidden_to_draft_compressors(
            worker,
            main_hidden=projected_hidden,
            draft_forward_batch=SimpleNamespace(name="projected_batch"),
            projected=True,
        )

        self.assertEqual(calls[0], ("init_metadata", "projected_batch"))
        self.assertEqual(calls[1][0], "compressor")
        self.assertEqual(calls[1][1], 7)
        self.assertTrue(t.equal(calls[1][2], projected_hidden))

    def test_materialize_state_continues_kv_when_compressor_replay_fails(self):
        t = self.torch
        worker = self._worker()
        calls = []

        def fail_compressors(**kwargs):
            calls.append(("compressor_failed", kwargs["main_hidden"].clone()))
            raise RuntimeError("compressor boom")

        def record_kv(**kwargs):
            calls.append(
                (
                    "kv",
                    kwargs["main_hidden"].clone(),
                    kwargs["cache_loc"].clone(),
                    kwargs["positions"].clone(),
                    kwargs["projected"],
                )
            )

        worker._materialize_main_hidden_to_draft_compressors = fail_compressors
        worker._materialize_main_hidden_to_draft_kv = record_kv
        main_hidden = t.ones((2, 3))
        kv_hidden = t.full((2, 3), 5.0)
        cache_loc = t.tensor([11, 12], dtype=t.int32)
        positions = t.tensor([101, 102], dtype=t.int32)

        self.worker_cls._materialize_main_hidden_to_draft_state(
            worker,
            main_hidden=main_hidden,
            cache_loc=cache_loc,
            positions=positions,
            draft_forward_batch=SimpleNamespace(name="batch"),
            kv_main_hidden=kv_hidden,
            projected=True,
        )

        self.assertEqual(calls[0][0], "compressor_failed")
        self.assertEqual(calls[1][0], "kv")
        self.assertTrue(t.equal(calls[1][1], kv_hidden))
        self.assertTrue(t.equal(calls[1][2], cache_loc))
        self.assertTrue(t.equal(calls[1][3], positions))
        self.assertTrue(calls[1][4])

    def test_prefill_tail_replay_flattens_valid_rows_for_projected_materialization(self):
        t = self.torch
        worker = self._worker()
        worker._accept_anomaly_enabled = False
        materialize_calls = []

        def record_materialize(**kwargs):
            materialize_calls.append(
                {
                    "main_hidden": kwargs["main_hidden"].clone(),
                    "cache_loc": kwargs["cache_loc"].clone(),
                    "positions": kwargs["positions"].clone(),
                    "draft_forward_batch": kwargs["draft_forward_batch"],
                    "projected": kwargs["projected"],
                }
            )

        worker._materialize_main_hidden_to_draft_state = record_materialize
        req_to_token = t.full((4, 16), -1, dtype=t.int64)
        req_to_token[1, 2] = 102
        req_to_token[1, 4] = 104
        req_to_token[1, 5] = 105
        req_to_token[2, 1] = 201
        req_to_token[2, 2] = -1
        req_to_token[2, 4] = 204
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token)
        )
        hidden = t.arange(2 * 4 * 3, dtype=t.float32).view(2, 4, 3)
        mask = t.tensor(
            [
                [True, False, True, True],
                [True, True, False, True],
            ],
            dtype=t.bool,
        )
        draft_input = self.draft_input_cls(
            bonus_tokens=t.tensor([0, 0], dtype=t.int64),
            new_seq_lens=t.tensor([6, 5], dtype=t.int64),
            prefill_tail_hidden_states=hidden,
            prefill_tail_valid_mask=mask,
        )
        batch = SimpleNamespace(req_pool_indices=t.tensor([1, 2], dtype=t.int64))

        self.worker_cls._materialize_prefill_tail_hidden_to_draft_state(
            worker,
            draft_input=draft_input,
            batch=batch,
            prefix_lens=t.tensor([6, 5], dtype=t.int64),
        )

        self.assertEqual(len(materialize_calls), 1)
        call = materialize_calls[0]
        self.assertIsNone(call["draft_forward_batch"])
        self.assertTrue(call["projected"])
        self.assertEqual(call["positions"].tolist(), [2, 4, 5, 1, 4])
        self.assertEqual(call["cache_loc"].tolist(), [102, 104, 105, 201, 204])
        expected_hidden = t.stack(
            [
                hidden[0, 0],
                hidden[0, 2],
                hidden[0, 3],
                hidden[1, 0],
                hidden[1, 3],
            ],
            dim=0,
        )
        self.assertTrue(t.equal(call["main_hidden"], expected_hidden))

    def test_prefill_tail_replay_skips_when_positions_are_negative(self):
        t = self.torch
        worker = self._worker()
        materialize_calls = []
        worker._materialize_main_hidden_to_draft_state = lambda **kwargs: (
            materialize_calls.append(kwargs)
        )
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=t.arange(16, dtype=t.int64).view(1, 16)
            )
        )
        draft_input = self.draft_input_cls(
            bonus_tokens=t.tensor([0], dtype=t.int64),
            new_seq_lens=t.tensor([0], dtype=t.int64),
            prefill_tail_hidden_states=t.ones((1, 4, 3)),
            prefill_tail_valid_mask=t.ones((1, 4), dtype=t.bool),
        )

        self.worker_cls._materialize_prefill_tail_hidden_to_draft_state(
            worker,
            draft_input=draft_input,
            batch=SimpleNamespace(req_pool_indices=t.tensor([0], dtype=t.int64)),
            prefix_lens=t.tensor([0], dtype=t.int64),
        )

        self.assertEqual(materialize_calls, [])

    def test_prefill_tail_replay_skips_when_cache_locs_are_invalid(self):
        t = self.torch
        worker = self._worker()
        materialize_calls = []
        worker._materialize_main_hidden_to_draft_state = lambda **kwargs: (
            materialize_calls.append(kwargs)
        )
        worker.model_runner = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=t.full((1, 8), -1, dtype=t.int64)
            )
        )
        draft_input = self.draft_input_cls(
            bonus_tokens=t.tensor([0], dtype=t.int64),
            new_seq_lens=t.tensor([4], dtype=t.int64),
            prefill_tail_hidden_states=t.ones((1, 4, 3)),
            prefill_tail_valid_mask=t.ones((1, 4), dtype=t.bool),
        )

        self.worker_cls._materialize_prefill_tail_hidden_to_draft_state(
            worker,
            draft_input=draft_input,
            batch=SimpleNamespace(req_pool_indices=t.tensor([0], dtype=t.int64)),
            prefix_lens=t.tensor([4], dtype=t.int64),
        )

        self.assertEqual(materialize_calls, [])

    def test_prefill_tail_replay_debug_records_executed_payload_and_history(self):
        t = self.torch
        worker = self._worker()
        worker.draft_model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(
                    translate_loc_from_full_to_swa=lambda locs: locs + 100
                )
            )
        )
        positions = t.tensor([3, 4], dtype=t.int64)
        cache_locs = t.tensor([13, 14], dtype=t.int64)
        hidden_rows = t.tensor([[1.0, 2.0], [3.0, 4.0]])

        self.worker_cls._record_prefill_tail_replay_executed_debug(
            worker,
            req_pool_idx=9,
            positions=positions,
            cache_locs=cache_locs,
            hidden_rows=hidden_rows,
            tail_len=4,
            tail_end_len=5,
            prefix_len=4,
        )

        payload = worker._prefill_tail_replay_debug_by_req_pool[9]
        self.assertTrue(payload["executed"])
        self.assertEqual(payload["valid_write_count"], 2)
        self.assertEqual(payload["tail_end_minus_prefix"], 1)
        self.assertEqual(payload["writes"]["positions"], [3, 4])
        self.assertEqual(payload["writes"]["full_locs"], [13, 14])
        self.assertEqual(payload["writes"]["swa_locs"], [113, 114])
        history = self.worker_cls._prefill_tail_replay_history_debug(worker, 9)
        self.assertEqual(history["valid_write_count"], 2)
        self.assertEqual(history["write_positions_first_last"], [3, 4])
        self.assertEqual(history["write_swa_first_last"], [113, 114])

    def test_materialize_state_uses_main_hidden_for_kv_by_default(self):
        t = self.torch
        worker = self._worker()
        calls = []
        worker._materialize_main_hidden_to_draft_compressors = lambda **kwargs: (
            calls.append(("compressor", kwargs["main_hidden"].clone()))
        )
        worker._materialize_main_hidden_to_draft_kv = lambda **kwargs: calls.append(
            (
                "kv",
                kwargs["main_hidden"].clone(),
                kwargs["cache_loc"].clone(),
                kwargs["positions"].clone(),
                kwargs["projected"],
            )
        )
        main_hidden = t.arange(6, dtype=t.float32).view(2, 3)
        cache_loc = t.tensor([31, 32], dtype=t.int64)
        positions = t.tensor([41, 42], dtype=t.int64)

        self.worker_cls._materialize_main_hidden_to_draft_state(
            worker,
            main_hidden=main_hidden,
            cache_loc=cache_loc,
            positions=positions,
            draft_forward_batch=SimpleNamespace(name="batch"),
            projected=False,
        )

        self.assertEqual(calls[0][0], "compressor")
        self.assertEqual(calls[1][0], "kv")
        self.assertTrue(t.equal(calls[1][1], main_hidden))
        self.assertTrue(t.equal(calls[1][2], cache_loc))
        self.assertTrue(t.equal(calls[1][3], positions))
        self.assertFalse(calls[1][4])

    def test_kv_materialization_per_layer_projects_and_casts_locs_and_positions(self):
        import contextlib

        t = self.torch
        worker = self._worker()
        calls = []

        class _DraftModel:
            def project_main_hidden(self, hidden):
                calls.append(("project", hidden.clone()))
                return hidden + 1.0

        class _Attn:
            def __init__(self, layer_id):
                self.layer_id = layer_id

            def kv_from_hidden(self, ctx_x, positions, cache_loc, attn_backend):
                calls.append(
                    (
                        "kv",
                        self.layer_id,
                        ctx_x.clone(),
                        positions.clone(),
                        cache_loc.clone(),
                        positions.dtype,
                        cache_loc.dtype,
                        attn_backend.name,
                    )
                )

        worker.draft_model = _DraftModel()
        worker.draft_model_runner = SimpleNamespace(
            tp_group="tp0",
            attn_backend=SimpleNamespace(name="attn_backend"),
        )
        worker.draft_tp_context = lambda tp_group: contextlib.nullcontext(tp_group)
        worker._stacked_wqkv_fp8_proj = None
        worker._draft_inner = SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=_Attn(layer_id=10)),
                SimpleNamespace(self_attn=_Attn(layer_id=11)),
            ]
        )
        hidden = t.arange(6, dtype=t.float32).view(2, 3)

        self.worker_cls._materialize_main_hidden_to_draft_kv(
            worker,
            main_hidden=hidden,
            cache_loc=t.tensor([3, 4], dtype=t.int32),
            positions=t.tensor([100, 101], dtype=t.int32),
            projected=False,
        )

        self.assertEqual(worker._last_context_kv_path_debug, "per_layer_kv_from_hidden")
        self.assertEqual(calls[0][0], "project")
        kv_calls = [call for call in calls if call[0] == "kv"]
        self.assertEqual([call[1] for call in kv_calls], [10, 11])
        self.assertTrue(t.equal(kv_calls[0][2], hidden + 1.0))
        self.assertEqual(kv_calls[0][3].dtype, t.int64)
        self.assertEqual(kv_calls[0][4].dtype, t.int64)
        self.assertEqual(kv_calls[0][5], t.int64)
        self.assertEqual(kv_calls[0][6], t.int64)
        self.assertEqual(kv_calls[0][7], "attn_backend")

    def test_kv_materialization_projected_hidden_skips_projection(self):
        import contextlib

        t = self.torch
        worker = self._worker()
        calls = []

        class _DraftModel:
            def project_main_hidden(self, _hidden):
                raise AssertionError("project_main_hidden should not be called")

        class _Attn:
            layer_id = 0

            def kv_from_hidden(self, ctx_x, positions, cache_loc, _attn_backend):
                calls.append((ctx_x.clone(), positions.clone(), cache_loc.clone()))

        worker.draft_model = _DraftModel()
        worker.draft_model_runner = SimpleNamespace(
            tp_group=None,
            attn_backend=SimpleNamespace(),
        )
        worker.draft_tp_context = lambda tp_group: contextlib.nullcontext()
        worker._stacked_wqkv_fp8_proj = None
        worker._draft_inner = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=_Attn())]
        )
        projected_hidden = t.full((2, 3), 7.0)

        self.worker_cls._materialize_main_hidden_to_draft_kv(
            worker,
            main_hidden=projected_hidden,
            cache_loc=t.tensor([5, 6], dtype=t.int64),
            positions=t.tensor([9, 10], dtype=t.int64),
            projected=True,
        )

        self.assertEqual(len(calls), 1)
        self.assertTrue(t.equal(calls[0][0], projected_hidden))

    def test_stacked_fp8_kv_materialization_splits_and_writes_projected_kv(self):
        import contextlib

        t = self.torch
        worker = self._worker()
        calls = []

        class _QuantMethod:
            def apply(self, proj, ctx_x, bias):
                calls.append(("stacked_apply", ctx_x.clone(), bias.clone()))
                return ctx_x @ proj.weight + bias

        class _DraftModel:
            def project_main_hidden(self, hidden):
                return hidden + 2.0

        worker.draft_model = _DraftModel()
        worker.draft_model_runner = SimpleNamespace(
            tp_group=None,
            attn_backend=SimpleNamespace(name="attn_backend"),
        )
        worker.draft_tp_context = lambda tp_group: contextlib.nullcontext()
        proj = SimpleNamespace(
            weight=t.arange(3 * 10, dtype=t.float32).view(3, 10) / 10.0,
            bias=t.arange(10, dtype=t.float32),
            quant_method=_QuantMethod(),
        )
        worker._stacked_wqkv_fp8_proj = proj
        worker._stacked_wqkv_out_sizes = [4, 6]
        worker._stacked_wqkv_kv_offsets = [(1, 4), (2, 6)]
        worker._draft_inner = SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(layer_id=0)),
                SimpleNamespace(self_attn=SimpleNamespace(layer_id=1)),
            ]
        )

        def record_projected_kv(**kwargs):
            calls.append(
                (
                    "write",
                    kwargs["attn"].layer_id,
                    kwargs["kv"].clone(),
                    kwargs["positions"].clone(),
                    kwargs["cache_loc"].clone(),
                    kwargs["attn_backend"].name,
                )
            )

        worker._write_draft_kv_from_projected_kv = record_projected_kv
        hidden = t.arange(6, dtype=t.float32).view(2, 3)
        positions = t.tensor([10, 11], dtype=t.int64)
        cache_loc = t.tensor([20, 21], dtype=t.int64)

        self.worker_cls._materialize_main_hidden_to_draft_kv(
            worker,
            main_hidden=hidden,
            cache_loc=cache_loc,
            positions=positions,
            projected=False,
        )

        self.assertEqual(worker._last_context_kv_path_debug, "stacked_fp8_wqkv")
        self.assertEqual(calls[0][0], "stacked_apply")
        stacked = (hidden + 2.0) @ proj.weight + proj.bias
        first_layer, second_layer = t.split(stacked, [4, 6], dim=-1)
        write_calls = [call for call in calls if call[0] == "write"]
        self.assertEqual([call[1] for call in write_calls], [0, 1])
        self.assertTrue(t.equal(write_calls[0][2], first_layer[:, 1:4]))
        self.assertTrue(t.equal(write_calls[1][2], second_layer[:, 2:6]))
        self.assertTrue(t.equal(write_calls[0][3], positions))
        self.assertTrue(t.equal(write_calls[0][4], cache_loc))
        self.assertEqual(write_calls[0][5], "attn_backend")

    def test_write_projected_kv_translates_full_locs_and_passes_rope_metadata(self):
        t = self.torch
        worker = self._worker()
        calls = []

        class _TokenToKVPool:
            def translate_loc_from_full_to_swa(self, cache_loc):
                calls.append(("translate", cache_loc.clone()))
                return cache_loc + 1000

            def set_swa_key_buffer_radix_fused_norm_rope(self, **kwargs):
                calls.append(("set_swa", kwargs))

        token_to_kv_pool = _TokenToKVPool()
        attn_backend = SimpleNamespace(token_to_kv_pool=token_to_kv_pool)
        attn = SimpleNamespace(
            layer_id=9,
            kv_norm=SimpleNamespace(weight=SimpleNamespace(data=t.tensor([1.0, 2.0]))),
            eps=1e-6,
            freqs_cis="freqs",
        )
        kv = t.arange(4, dtype=t.float32).view(2, 2)
        positions = t.tensor([7, 8], dtype=t.int64)
        cache_loc = t.tensor([3, 4], dtype=t.int64)

        self.worker_cls._write_draft_kv_from_projected_kv(
            worker,
            attn=attn,
            kv=kv,
            positions=positions,
            cache_loc=cache_loc,
            attn_backend=attn_backend,
        )

        self.assertEqual(calls[0][0], "translate")
        self.assertTrue(t.equal(calls[0][1], cache_loc))
        self.assertEqual(calls[1][0], "set_swa")
        payload = calls[1][1]
        self.assertEqual(payload["layer_id"], 9)
        self.assertEqual(payload["swa_loc"].dtype, t.int32)
        self.assertEqual(payload["swa_loc"].tolist(), [1003, 1004])
        self.assertTrue(t.equal(payload["kv"], kv))
        self.assertTrue(t.equal(payload["kv_weight"], t.tensor([1.0, 2.0])))
        self.assertEqual(payload["eps"], 1e-6)
        self.assertEqual(payload["freqs_cis"], "freqs")
        self.assertTrue(t.equal(payload["positions"], positions))

    def test_materialization_empty_hidden_returns_without_backend_calls(self):
        import contextlib

        t = self.torch
        worker = self._worker()
        calls = []

        class _DraftModel:
            def project_main_hidden(self, hidden):
                calls.append(("project", hidden.clone()))
                return hidden

        class _AttnBackend:
            forward_metadata = {"old": True}

            def init_forward_metadata(self, forward_batch):
                calls.append(("init_metadata", forward_batch))

            def forward_core_compressor(self, *args, **kwargs):
                calls.append(("compressor", args, kwargs))

        class _Attn:
            layer_id = 0
            compressor = "compressor"

            def kv_from_hidden(self, *args, **kwargs):
                calls.append(("kv", args, kwargs))

        worker.draft_model = _DraftModel()
        worker.draft_model_runner = SimpleNamespace(
            tp_group=None,
            attn_backend=_AttnBackend(),
        )
        worker.draft_tp_context = lambda tp_group: contextlib.nullcontext()
        worker._stacked_wqkv_fp8_proj = None
        worker._draft_inner = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=_Attn())]
        )
        empty_hidden = t.empty((0, 3))

        self.worker_cls._materialize_main_hidden_to_draft_kv(
            worker,
            main_hidden=empty_hidden,
            cache_loc=t.empty((0,), dtype=t.int64),
            positions=t.empty((0,), dtype=t.int64),
        )
        self.worker_cls._materialize_main_hidden_to_draft_compressors(
            worker,
            main_hidden=empty_hidden,
            draft_forward_batch=SimpleNamespace(name="batch"),
        )
        self.worker_cls._materialize_main_hidden_to_draft_compressors(
            worker,
            main_hidden=t.ones((1, 3)),
            draft_forward_batch=None,
        )

        self.assertEqual(calls, [])

    def test_materialization_raises_when_main_hidden_is_missing(self):
        t = self.torch
        worker = self._worker()

        with self.assertRaisesRegex(RuntimeError, "missing target main_hidden"):
            self.worker_cls._materialize_main_hidden_to_draft_kv(
                worker,
                main_hidden=None,
                cache_loc=t.tensor([1], dtype=t.int64),
                positions=t.tensor([2], dtype=t.int64),
            )
        with self.assertRaisesRegex(RuntimeError, "missing target main_hidden"):
            self.worker_cls._materialize_main_hidden_to_draft_compressors(
                worker,
                main_hidden=None,
                draft_forward_batch=SimpleNamespace(name="batch"),
            )

class TestDSparkDeepSpecSemanticReference(CustomTestCase):
    def setUp(self):
        try:
            import torch
        except Exception as e:  # pragma: no cover - torch unavailable on some runners
            self.skipTest(f"torch unavailable on this runner: {e}")
        self.torch = torch

    def test_materialized_context_matches_deepspec_concat_when_kv_is_equivalent(self):
        # This is the algebra DSpark relies on:
        #   DeepSpec: K/V = K/V(context hidden) + K/V(anchor/noise block)
        #   SGLang:   write K/V(context hidden), then run anchor/noise block
        # If the same projections, positions, masks and cache cleanup are used,
        # the block output should be identical.
        t = self.torch
        t.manual_seed(0)
        batch_size = 1
        context_len = 4
        block_size = 3
        hidden_size = 8
        head_dim = 8

        context_hidden = t.randn(batch_size, context_len, hidden_size)
        block_hidden = t.randn(batch_size, block_size, hidden_size)
        wq = t.randn(hidden_size, head_dim)
        wk = t.randn(hidden_size, head_dim)
        wv = t.randn(hidden_size, head_dim)

        def attention(q, k, v):
            scores = q @ k.transpose(-1, -2) / (head_dim**0.5)
            probs = t.softmax(scores, dim=-1)
            return probs @ v

        q_block = block_hidden @ wq
        k_context = context_hidden @ wk
        v_context = context_hidden @ wv
        k_block = block_hidden @ wk
        v_block = block_hidden @ wv

        deepspec_k = t.cat([k_context, k_block], dim=1)
        deepspec_v = t.cat([v_context, v_block], dim=1)
        deepspec_out = attention(q_block, deepspec_k, deepspec_v)

        materialized_k_cache = k_context.clone()
        materialized_v_cache = v_context.clone()
        sglang_k = t.cat([materialized_k_cache, k_block], dim=1)
        sglang_v = t.cat([materialized_v_cache, v_block], dim=1)
        sglang_out = attention(q_block, sglang_k, sglang_v)

        self.assertTrue(t.equal(deepspec_out, sglang_out))

    def test_materialized_context_matches_concat_with_batch_masked_windows(self):
        # A stronger local proxy for the runtime invariant: each request may
        # expose a different context window, but materialized K/V rows must be
        # exactly the same rows that DeepSpec would concatenate before block
        # attention.
        t = self.torch
        t.manual_seed(1)
        batch_size = 2
        context_len = 5
        block_size = 4
        hidden_size = 6
        head_dim = 6

        context_hidden = t.randn(batch_size, context_len, hidden_size)
        block_hidden = t.randn(batch_size, block_size, hidden_size)
        wq = t.randn(hidden_size, head_dim)
        wk = t.randn(hidden_size, head_dim)
        wv = t.randn(hidden_size, head_dim)
        visible_context = t.tensor(
            [
                [False, True, True, True, True],
                [False, False, True, True, True],
            ],
            dtype=t.bool,
        )
        causal = t.tril(t.ones((block_size, block_size), dtype=t.bool))

        def masked_attention(q, k, v, mask):
            scores = q @ k.transpose(-1, -2) / (head_dim**0.5)
            scores = scores.masked_fill(~mask, float("-inf"))
            return t.softmax(scores, dim=-1) @ v

        q_block = block_hidden @ wq
        k_context = context_hidden @ wk
        v_context = context_hidden @ wv
        k_block = block_hidden @ wk
        v_block = block_hidden @ wv

        deepspec_k = t.cat([k_context, k_block], dim=1)
        deepspec_v = t.cat([v_context, v_block], dim=1)
        full_mask = t.cat(
            [
                visible_context[:, None, :].expand(-1, block_size, -1),
                causal[None, :, :].expand(batch_size, -1, -1),
            ],
            dim=-1,
        )
        deepspec_out = masked_attention(q_block, deepspec_k, deepspec_v, full_mask)

        materialized_k_cache = k_context.clone()
        materialized_v_cache = v_context.clone()
        sglang_k = t.cat([materialized_k_cache, k_block], dim=1)
        sglang_v = t.cat([materialized_v_cache, v_block], dim=1)
        sglang_out = masked_attention(q_block, sglang_k, sglang_v, full_mask)

        self.assertTrue(t.equal(deepspec_out, sglang_out))

    def _apply_rope(self, x, positions):
        t = self.torch
        dim = x.shape[-1]
        self.assertEqual(dim % 2, 0)
        inv_freq = 1.0 / (
            10000
            ** (t.arange(0, dim, 2, dtype=x.dtype, device=x.device) / dim)
        )
        freqs = positions.to(dtype=x.dtype, device=x.device)[..., None] * inv_freq
        cos = freqs.cos()
        sin = freqs.sin()
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = t.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def test_materialized_context_matches_concat_with_rope_absolute_positions(self):
        # This covers the main precision risk that plain concat tests miss:
        # Q/K must be rotated with the same absolute positions before the
        # context K/V rows are materialized into the draft cache.
        t = self.torch
        t.manual_seed(2)
        batch_size = 2
        context_len = 6
        block_size = 3
        hidden_size = 8
        head_dim = 8

        context_hidden = t.randn(batch_size, context_len, hidden_size)
        block_hidden = t.randn(batch_size, block_size, hidden_size)
        wq = t.randn(hidden_size, head_dim)
        wk = t.randn(hidden_size, head_dim)
        wv = t.randn(hidden_size, head_dim)
        context_positions = t.tensor(
            [[101, 102, 103, 104, 105, 106], [205, 206, 207, 208, 209, 210]]
        )
        block_positions = t.tensor([[107, 108, 109], [211, 212, 213]])
        visible_context = t.tensor(
            [
                [False, True, True, True, True, True],
                [True, True, False, True, True, True],
            ],
            dtype=t.bool,
        )
        causal = t.tril(t.ones((block_size, block_size), dtype=t.bool))

        def masked_attention(q, k, v, mask):
            scores = q @ k.transpose(-1, -2) / (head_dim**0.5)
            scores = scores.masked_fill(~mask, float("-inf"))
            return t.softmax(scores, dim=-1) @ v

        q_block = self._apply_rope(block_hidden @ wq, block_positions)
        k_context = self._apply_rope(context_hidden @ wk, context_positions)
        k_block = self._apply_rope(block_hidden @ wk, block_positions)
        v_context = context_hidden @ wv
        v_block = block_hidden @ wv
        full_mask = t.cat(
            [
                visible_context[:, None, :].expand(-1, block_size, -1),
                causal[None, :, :].expand(batch_size, -1, -1),
            ],
            dim=-1,
        )

        deepspec_out = masked_attention(
            q_block,
            t.cat([k_context, k_block], dim=1),
            t.cat([v_context, v_block], dim=1),
            full_mask,
        )
        materialized_k_cache = k_context.clone()
        materialized_v_cache = v_context.clone()
        sglang_out = masked_attention(
            q_block,
            t.cat([materialized_k_cache, k_block], dim=1),
            t.cat([materialized_v_cache, v_block], dim=1),
            full_mask,
        )

        self.assertTrue(t.allclose(deepspec_out, sglang_out, atol=0.0, rtol=0.0))

    def test_materialized_context_precision_grid_matches_concat(self):
        t = self.torch
        configs = [
            (t.float32, 1, 2, 2, 4),
            (t.float32, 3, 5, 4, 8),
            (t.float64, 2, 7, 3, 10),
        ]
        for dtype, batch_size, context_len, block_size, hidden_size in configs:
            with self.subTest(
                dtype=str(dtype),
                batch_size=batch_size,
                context_len=context_len,
                block_size=block_size,
                hidden_size=hidden_size,
            ):
                t.manual_seed(100 + batch_size + context_len + block_size)
                context_hidden = t.randn(
                    batch_size, context_len, hidden_size, dtype=dtype
                )
                block_hidden = t.randn(
                    batch_size, block_size, hidden_size, dtype=dtype
                )
                wq = t.randn(hidden_size, hidden_size, dtype=dtype)
                wk = t.randn(hidden_size, hidden_size, dtype=dtype)
                wv = t.randn(hidden_size, hidden_size, dtype=dtype)
                q_block = block_hidden @ wq
                k_context = context_hidden @ wk
                v_context = context_hidden @ wv
                k_block = block_hidden @ wk
                v_block = block_hidden @ wv
                mask = t.ones(
                    (batch_size, block_size, context_len + block_size),
                    dtype=t.bool,
                )
                block_causal = t.tril(t.ones((block_size, block_size), dtype=t.bool))
                mask[:, :, context_len:] = block_causal

                def attention(q, k, v):
                    scores = q @ k.transpose(-1, -2) / (hidden_size**0.5)
                    scores = scores.masked_fill(~mask, float("-inf"))
                    return t.softmax(scores, dim=-1) @ v

                deepspec_out = attention(
                    q_block,
                    t.cat([k_context, k_block], dim=1),
                    t.cat([v_context, v_block], dim=1),
                )
                materialized_out = attention(
                    q_block,
                    t.cat([k_context.clone(), k_block], dim=1),
                    t.cat([v_context.clone(), v_block], dim=1),
                )
                self.assertTrue(
                    t.allclose(deepspec_out, materialized_out, atol=0.0, rtol=0.0)
                )

    def test_materialized_context_precision_test_detects_wrong_kv_order(self):
        t = self.torch
        batch_size = 1
        block_size = 1
        hidden_size = 4
        q_block = t.tensor([[[20.0, 0.0, 0.0, 0.0]]])
        k_context = t.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            ]
        )
        v_context = t.tensor(
            [
                [
                    [100.0, 0.0, 0.0, 0.0],
                    [-100.0, 0.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0, 0.0],
                ]
            ]
        )
        k_block = t.zeros((batch_size, block_size, hidden_size))
        v_block = t.zeros((batch_size, block_size, hidden_size))
        mask = t.tensor([[[True, False, True, True]]], dtype=t.bool)

        def attention(q, k, v):
            scores = q @ k.transpose(-1, -2) / (hidden_size**0.5)
            scores = scores.masked_fill(~mask, float("-inf"))
            return t.softmax(scores, dim=-1) @ v

        expected = attention(
            q_block,
            t.cat([k_context, k_block], dim=1),
            t.cat([v_context, v_block], dim=1),
        )
        wrong_k_context = k_context[:, [1, 0, 2], :]
        wrong_v_context = v_context[:, [1, 0, 2], :]
        wrong = attention(
            q_block,
            t.cat([wrong_k_context, k_block], dim=1),
            t.cat([wrong_v_context, v_block], dim=1),
        )

        self.assertFalse(t.allclose(expected, wrong, atol=1e-5, rtol=1e-5))
        self.assertGreater(float((expected - wrong).abs().max().item()), 1e-4)

    def _rms_norm(self, x, weight):
        return x * (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).rsqrt() * weight

    def _make_toy_decoder_layer_weights(self, hidden_size, mlp_size, seed):
        t = self.torch
        generator = t.Generator().manual_seed(seed)

        def randn(*shape):
            return t.randn(*shape, generator=generator) / (hidden_size**0.5)

        return {
            "attn_norm": randn(hidden_size),
            "wq": randn(hidden_size, hidden_size),
            "wk": randn(hidden_size, hidden_size),
            "wv": randn(hidden_size, hidden_size),
            "wo": randn(hidden_size, hidden_size),
            "mlp_norm": randn(hidden_size),
            "w_gate": randn(hidden_size, mlp_size),
            "w_up": randn(hidden_size, mlp_size),
            "w_down": randn(mlp_size, hidden_size),
        }

    def _toy_decoder_layer_full(self, x, positions, key_valid, weights):
        t = self.torch
        seq_len = x.shape[1]
        hidden_size = x.shape[-1]
        normed = self._rms_norm(x, weights["attn_norm"])
        q = self._apply_rope(normed @ weights["wq"], positions)
        k = self._apply_rope(normed @ weights["wk"], positions)
        v = normed @ weights["wv"]
        causal = t.tril(t.ones((seq_len, seq_len), dtype=t.bool, device=x.device))
        mask = key_valid[:, None, :] & causal[None, :, :]
        scores = q @ k.transpose(-1, -2) / (hidden_size**0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_out = t.softmax(scores, dim=-1) @ v
        h = x + attn_out @ weights["wo"]
        mlp_in = self._rms_norm(h, weights["mlp_norm"])
        mlp_out = t.nn.functional.silu(mlp_in @ weights["w_gate"])
        mlp_out = mlp_out * (mlp_in @ weights["w_up"])
        return h + mlp_out @ weights["w_down"]

    def _toy_decoder_layer_incremental(
        self,
        context_x,
        block_x,
        context_positions,
        block_positions,
        context_valid,
        weights,
    ):
        t = self.torch
        block_size = block_x.shape[1]
        hidden_size = block_x.shape[-1]
        context_normed = self._rms_norm(context_x, weights["attn_norm"])
        block_normed = self._rms_norm(block_x, weights["attn_norm"])
        cached_k = self._apply_rope(context_normed @ weights["wk"], context_positions)
        cached_v = context_normed @ weights["wv"]
        q = self._apply_rope(block_normed @ weights["wq"], block_positions)
        block_k = self._apply_rope(block_normed @ weights["wk"], block_positions)
        block_v = block_normed @ weights["wv"]
        causal = t.tril(
            t.ones((block_size, block_size), dtype=t.bool, device=block_x.device)
        )
        mask = t.cat(
            [
                context_valid[:, None, :].expand(-1, block_size, -1),
                causal[None, :, :].expand(block_x.shape[0], -1, -1),
            ],
            dim=-1,
        )
        k = t.cat([cached_k, block_k], dim=1)
        v = t.cat([cached_v, block_v], dim=1)
        scores = q @ k.transpose(-1, -2) / (hidden_size**0.5)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn_out = t.softmax(scores, dim=-1) @ v
        h = block_x + attn_out @ weights["wo"]
        mlp_in = self._rms_norm(h, weights["mlp_norm"])
        mlp_out = t.nn.functional.silu(mlp_in @ weights["w_gate"])
        mlp_out = mlp_out * (mlp_in @ weights["w_up"])
        return h + mlp_out @ weights["w_down"]

    def test_two_layer_toy_decoder_incremental_kv_matches_concat_block_hidden(self):
        # This is the closest CPU-only proxy in this file for DSpark's
        # block_hidden path: precomputed context K/V per layer, RoPE on Q/K,
        # causal block attention, RMSNorm, residuals and an MLP.
        t = self.torch
        t.manual_seed(4)
        batch_size = 2
        context_len = 5
        block_size = 3
        hidden_size = 8
        mlp_size = 16
        context_x = t.randn(batch_size, context_len, hidden_size)
        block_x = t.randn(batch_size, block_size, hidden_size)
        context_valid = t.tensor(
            [
                [True, False, True, True, True],
                [True, True, False, True, True],
            ],
            dtype=t.bool,
        )
        context_positions = t.tensor(
            [[50, 51, 52, 53, 54], [100, 101, 102, 103, 104]]
        )
        block_positions = t.tensor([[55, 56, 57], [105, 106, 107]])
        full_positions = t.cat([context_positions, block_positions], dim=1)
        full_valid = t.cat(
            [
                context_valid,
                t.ones((batch_size, block_size), dtype=t.bool),
            ],
            dim=1,
        )
        layer1 = self._make_toy_decoder_layer_weights(hidden_size, mlp_size, seed=10)
        layer2 = self._make_toy_decoder_layer_weights(hidden_size, mlp_size, seed=20)

        full_input = t.cat([context_x, block_x], dim=1)
        full_l1 = self._toy_decoder_layer_full(
            full_input, full_positions, full_valid, layer1
        )
        full_l2 = self._toy_decoder_layer_full(
            full_l1, full_positions, full_valid, layer2
        )
        deepspec_block_hidden = full_l2[:, context_len:, :]

        context_l1 = self._toy_decoder_layer_full(
            context_x, context_positions, context_valid, layer1
        )
        block_l1 = self._toy_decoder_layer_incremental(
            context_x,
            block_x,
            context_positions,
            block_positions,
            context_valid,
            layer1,
        )
        block_l2 = self._toy_decoder_layer_incremental(
            context_l1,
            block_l1,
            context_positions,
            block_positions,
            context_valid,
            layer2,
        )

        max_abs = float((deepspec_block_hidden - block_l2).abs().max().item())
        if os.getenv("SGLANG_DSPARK_TEST_VERBOSE", "0") == "1":
            print(
                "DSpark toy decoder parity debug: "
                + json.dumps(
                    {
                        "case": "two_layer_incremental_kv_matches_concat",
                        "batch_size": batch_size,
                        "context_len": context_len,
                        "block_size": block_size,
                        "hidden_size": hidden_size,
                        "max_abs_diff": max_abs,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        self.assertTrue(
            t.allclose(deepspec_block_hidden, block_l2, atol=1e-5, rtol=1e-5)
        )
        self.assertLess(max_abs, 1e-5)

    def test_toy_decoder_block_hidden_feeds_identical_markov_candidates(self):
        repo_root = Path(__file__).resolve().parents[5]
        deepspec_root = repo_root / "DeepSpec"
        if not deepspec_root.exists():
            self.skipTest(f"DeepSpec checkout not found: {deepspec_root}")
        sys.path.insert(0, str(deepspec_root))
        try:
            from deepspec.modeling.dspark.markov_head import VanillaMarkov
            from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
        except Exception as e:
            self.skipTest(f"DeepSpec/SGLang DSpark imports unavailable: {e}")

        t = self.torch
        t.manual_seed(5)
        batch_size = 2
        context_len = 4
        block_size = 3
        hidden_size = 8
        mlp_size = 16
        vocab_size = 31
        markov_rank = 6
        context_x = t.randn(batch_size, context_len, hidden_size)
        block_x = t.randn(batch_size, block_size, hidden_size)
        context_valid = t.tensor(
            [
                [True, True, False, True],
                [True, False, True, True],
            ],
            dtype=t.bool,
        )
        context_positions = t.tensor([[30, 31, 32, 33], [70, 71, 72, 73]])
        block_positions = t.tensor([[34, 35, 36], [74, 75, 76]])
        full_positions = t.cat([context_positions, block_positions], dim=1)
        full_valid = t.cat(
            [
                context_valid,
                t.ones((batch_size, block_size), dtype=t.bool),
            ],
            dim=1,
        )
        layer1 = self._make_toy_decoder_layer_weights(hidden_size, mlp_size, seed=30)
        layer2 = self._make_toy_decoder_layer_weights(hidden_size, mlp_size, seed=40)

        full_l1 = self._toy_decoder_layer_full(
            t.cat([context_x, block_x], dim=1),
            full_positions,
            full_valid,
            layer1,
        )
        full_l2 = self._toy_decoder_layer_full(
            full_l1,
            full_positions,
            full_valid,
            layer2,
        )
        concat_block_hidden = full_l2[:, context_len:, :]

        context_l1 = self._toy_decoder_layer_full(
            context_x, context_positions, context_valid, layer1
        )
        block_l1 = self._toy_decoder_layer_incremental(
            context_x,
            block_x,
            context_positions,
            block_positions,
            context_valid,
            layer1,
        )
        incremental_block_hidden = self._toy_decoder_layer_incremental(
            context_l1,
            block_l1,
            context_positions,
            block_positions,
            context_valid,
            layer2,
        )

        class _LogitsProcessor:
            def _compute_lm_head(self, hidden_states, head):
                return head(hidden_states)

        class _ConfidenceHead:
            def __call__(self, hidden_states, markov_embeds):
                return hidden_states.new_zeros(hidden_states.shape[:2])

        lm_head = t.nn.Linear(hidden_size, vocab_size, bias=False)
        markov_head = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.block_size = block_size
        worker.verify_stride = block_size + 1
        worker.markov_rank = markov_rank
        worker.noise_token_id = 0
        worker._markov_refine_buffer_cap = 0
        worker._markov_candidates_buf = None
        worker._markov_embeds_buf = None
        worker._vocab_shard_mapping_cache = {}
        worker._accept_anomaly_enabled = False
        worker._draft_inner = SimpleNamespace(
            vocab_size=vocab_size,
            markov_head=markov_head,
            confidence_head=_ConfidenceHead(),
        )
        worker.draft_model = SimpleNamespace(
            lm_head=lm_head,
            logits_processor=_LogitsProcessor(),
        )
        anchor_tokens = t.tensor([3, 17], dtype=t.int64)

        concat_candidates, concat_confidence = DSparkWorkerV2._refine_block_markov(
            worker,
            block_hidden=concat_block_hidden,
            anchor_tokens=anchor_tokens,
        )
        incremental_candidates, incremental_confidence = (
            DSparkWorkerV2._refine_block_markov(
                worker,
                block_hidden=incremental_block_hidden,
                anchor_tokens=anchor_tokens,
            )
        )

        hidden_max_abs = float(
            (concat_block_hidden - incremental_block_hidden).abs().max().item()
        )
        base_logits_max_abs = float(
            (lm_head(concat_block_hidden) - lm_head(incremental_block_hidden))
            .abs()
            .max()
            .item()
        )
        if os.getenv("SGLANG_DSPARK_TEST_VERBOSE", "0") == "1":
            print(
                "DSpark toy end-to-end parity debug: "
                + json.dumps(
                    {
                        "case": "toy_decoder_block_hidden_to_markov_candidates",
                        "candidate_match": bool(
                            t.equal(concat_candidates, incremental_candidates)
                        ),
                        "confidence_match": bool(
                            t.equal(concat_confidence, incremental_confidence)
                        ),
                        "hidden_max_abs_diff": hidden_max_abs,
                        "base_logits_max_abs_diff": base_logits_max_abs,
                        "concat_candidates": concat_candidates.tolist(),
                        "incremental_candidates": incremental_candidates.tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        self.assertTrue(t.allclose(concat_block_hidden, incremental_block_hidden))
        self.assertLess(hidden_max_abs, 1e-5)
        self.assertLess(base_logits_max_abs, 1e-5)
        self.assertEqual(concat_candidates.tolist(), incremental_candidates.tolist())
        self.assertTrue(t.equal(concat_confidence, incremental_confidence))

    def test_markov_refine_matches_deepspec_across_shape_grid(self):
        repo_root = Path(__file__).resolve().parents[5]
        deepspec_root = repo_root / "DeepSpec"
        if not deepspec_root.exists():
            self.skipTest(f"DeepSpec checkout not found: {deepspec_root}")
        sys.path.insert(0, str(deepspec_root))
        try:
            from deepspec.modeling.dspark.markov_head import VanillaMarkov
            from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
        except Exception as e:
            self.skipTest(f"DeepSpec/SGLang DSpark imports unavailable: {e}")

        t = self.torch

        class _LogitsProcessor:
            def _compute_lm_head(self, hidden_states, head):
                return head(hidden_states)

        class _ConfidenceHead:
            def __call__(self, hidden_states, markov_embeds):
                return hidden_states.new_zeros(hidden_states.shape[:2])

        configs = [
            (1, 1, 4, 11, 3, 11),
            (2, 4, 6, 19, 5, 0),
            (3, 5, 8, 23, 7, 17),
        ]
        for bs, block_size, hidden_size, vocab_size, markov_rank, seed in configs:
            with self.subTest(
                bs=bs,
                block_size=block_size,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                markov_rank=markov_rank,
            ):
                t.manual_seed(seed)
                block_hidden = t.randn(bs, block_size, hidden_size)
                anchor_tokens = t.randint(0, vocab_size, (bs,), dtype=t.int64)
                lm_head = t.nn.Linear(hidden_size, vocab_size, bias=False)
                markov_head = VanillaMarkov(
                    vocab_size=vocab_size,
                    markov_rank=markov_rank,
                )
                worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
                worker.block_size = block_size
                worker.verify_stride = block_size + 1
                worker.markov_rank = markov_rank
                worker.noise_token_id = 0
                worker._markov_refine_buffer_cap = 0
                worker._markov_candidates_buf = None
                worker._markov_embeds_buf = None
                worker._vocab_shard_mapping_cache = {}
                worker._accept_anomaly_enabled = False
                worker._draft_inner = SimpleNamespace(
                    vocab_size=vocab_size,
                    markov_head=markov_head,
                    confidence_head=_ConfidenceHead(),
                )
                worker.draft_model = SimpleNamespace(
                    lm_head=lm_head,
                    logits_processor=_LogitsProcessor(),
                )

                candidates, confidence = DSparkWorkerV2._refine_block_markov(
                    worker,
                    block_hidden=block_hidden,
                    anchor_tokens=anchor_tokens,
                )

                base_logits = lm_head(block_hidden)
                ref_candidates = t.empty(bs, block_size + 1, dtype=t.int64)
                ref_candidates[:, 0] = anchor_tokens
                prev_tokens = anchor_tokens
                for step in range(block_size):
                    bias = markov_head.markov_w2(
                        markov_head.get_prev_embeddings(prev_tokens)
                    )
                    next_tokens = t.argmax(base_logits[:, step, :] + bias, dim=-1)
                    ref_candidates[:, step + 1] = next_tokens
                    prev_tokens = next_tokens

                self.assertEqual(candidates.tolist(), ref_candidates.tolist())
                self.assertEqual(tuple(confidence.shape), (bs, block_size))
                self.assertTrue(t.equal(confidence, t.zeros_like(confidence)))

    def test_markov_refine_matches_deepspec_vanilla_markov_reference(self):
        repo_root = Path(__file__).resolve().parents[5]
        deepspec_root = repo_root / "DeepSpec"
        if not deepspec_root.exists():
            self.skipTest(f"DeepSpec checkout not found: {deepspec_root}")
        sys.path.insert(0, str(deepspec_root))
        try:
            from deepspec.modeling.dspark.markov_head import VanillaMarkov
            from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
        except Exception as e:
            self.skipTest(f"DeepSpec/SGLang DSpark imports unavailable: {e}")

        t = self.torch
        t.manual_seed(0)
        bs = 2
        block_size = 4
        verify_stride = block_size + 1
        hidden_size = 6
        vocab_size = 19
        markov_rank = 5

        block_hidden = t.randn(bs, block_size, hidden_size)
        anchor_tokens = t.tensor([3, 7], dtype=t.int64)
        lm_head = t.nn.Linear(hidden_size, vocab_size, bias=False)
        markov_head = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)

        class _LogitsProcessor:
            def _compute_lm_head(self, hidden_states, head):
                return head(hidden_states)

        class _ConfidenceHead:
            def __call__(self, hidden_states, markov_embeds):
                return hidden_states.new_zeros(hidden_states.shape[:2])

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.block_size = block_size
        worker.verify_stride = verify_stride
        worker.markov_rank = markov_rank
        worker.noise_token_id = 0
        worker._markov_refine_buffer_cap = 0
        worker._markov_candidates_buf = None
        worker._markov_embeds_buf = None
        worker._vocab_shard_mapping_cache = {}
        worker._accept_anomaly_enabled = False
        worker._draft_inner = SimpleNamespace(
            vocab_size=vocab_size,
            markov_head=markov_head,
            confidence_head=_ConfidenceHead(),
        )
        worker.draft_model = SimpleNamespace(
            lm_head=lm_head,
            logits_processor=_LogitsProcessor(),
        )

        candidates, confidence = DSparkWorkerV2._refine_block_markov(
            worker,
            block_hidden=block_hidden,
            anchor_tokens=anchor_tokens,
        )

        base_logits = lm_head(block_hidden)
        ref_candidates = t.empty(bs, verify_stride, dtype=t.int64)
        ref_candidates[:, 0] = anchor_tokens
        prev_tokens = anchor_tokens
        for step in range(block_size):
            bias = markov_head.markov_w2(markov_head.get_prev_embeddings(prev_tokens))
            next_tokens = t.argmax(base_logits[:, step, :] + bias, dim=-1)
            ref_candidates[:, step + 1] = next_tokens
            prev_tokens = next_tokens

        parity_match = t.equal(candidates, ref_candidates)
        if os.getenv("SGLANG_DSPARK_TEST_VERBOSE", "0") == "1":
            print(
                "DSpark Markov parity debug: "
                + json.dumps(
                    {
                        "bs": bs,
                        "block_size": block_size,
                        "verify_stride": verify_stride,
                        "hidden_size": hidden_size,
                        "vocab_size": vocab_size,
                        "markov_rank": markov_rank,
                        "anchor_tokens": anchor_tokens.tolist(),
                        "sglang_candidates": candidates.tolist(),
                        "deepspec_reference_candidates": ref_candidates.tolist(),
                        "candidate_match": bool(parity_match),
                        "confidence_shape": list(confidence.shape),
                        "confidence_abs_sum": float(confidence.abs().sum().item()),
                        "base_top1": t.argmax(base_logits, dim=-1).tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        self.assertEqual(candidates.tolist(), ref_candidates.tolist())
        self.assertEqual(tuple(confidence.shape), (bs, block_size))
        self.assertTrue(t.equal(confidence, t.zeros_like(confidence)))


if __name__ == "__main__":
    unittest.main(verbosity=3)
