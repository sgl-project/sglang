"""Unit tests for DSpark speculative-decoding registration and arg handling."""

import unittest
from types import SimpleNamespace

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

    def test_dp_attention_rejected(self):
        args = _make_server_args(enable_dp_attention=True)
        with self.assertRaisesRegex(ValueError, "dp attention"):
            _handle_dspark(args)

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

    def test_overlap_placeholders_inert_without_future_indices(self):
        a = self._make(2)
        self.assertTrue(a.direct_carry_valid)
        self.assertIsNone(a.future_indices)
        self.assertEqual(a.topk_p.numel(), 0)
        a.filter_batch(self.torch.tensor([1], dtype=self.torch.int64))
        self.assertEqual(len(a.bonus_tokens), 1)


class TestDSparkRequestValidation(CustomTestCase):
    """validate_dspark_request rejects features DSpark cannot serve and accepts a
    plain request. Kept import-light: requests are duck-typed namespaces."""

    def setUp(self):
        try:
            from sglang.srt.speculative.dspark_info import validate_dspark_request
        except Exception as e:  # pragma: no cover - GPU-only deps on some runners
            self.skipTest(f"dspark_info unavailable on this runner: {e}")
        self.validate = validate_dspark_request

    @staticmethod
    def _make_req(**overrides):
        sampling = SimpleNamespace(
            json_schema=None, regex=None, ebnf=None, structural_tag=None
        )
        base = dict(
            return_logprob=False,
            return_hidden_states=False,
            sampling_params=sampling,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_accepts_plain_request(self):
        self.assertIsNone(self.validate(self._make_req()))

    def test_rejects_return_logprob(self):
        msg = self.validate(self._make_req(return_logprob=True))
        self.assertIsNotNone(msg)
        self.assertIn("return_logprob", msg)

    def test_rejects_return_hidden_states(self):
        msg = self.validate(self._make_req(return_hidden_states=True))
        self.assertIsNotNone(msg)
        self.assertIn("return_hidden_states", msg)

    def test_rejects_grammar_constrained(self):
        for field_name in ("json_schema", "regex", "ebnf", "structural_tag"):
            sampling = SimpleNamespace(
                json_schema=None, regex=None, ebnf=None, structural_tag=None
            )
            setattr(sampling, field_name, "x")
            msg = self.validate(self._make_req(sampling_params=sampling))
            self.assertIsNotNone(msg, field_name)
            self.assertIn("grammar", msg)


if __name__ == "__main__":
    unittest.main(verbosity=3)
