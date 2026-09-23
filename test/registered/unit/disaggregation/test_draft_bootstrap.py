"""PD draft handoff capability and prefix replay contract tests."""

import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.draft_bootstrap import (
    bootstrap_decode_draft,
    bootstrap_prompt,
    validate_draft_handoff,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDraftHandoff(unittest.TestCase):
    def mode(self, p, d, prs=False, drs=False, bootstrap=False):
        return validate_draft_handoff(
            prefill_has_draft=p,
            prefill_rs=prs,
            decode_has_draft=d,
            decode_rs=drs,
            decode_bootstrap=bootstrap,
        )

    def test_rs_only_matters_when_both_roles_draft(self):
        for prs in (False, True, None):
            for drs in (False, True):
                self.assertEqual(self.mode(False, False, prs, drs), "none")
                self.assertEqual(self.mode(False, True, prs, drs, True), "decode")

    def test_seeded_requires_matching_sampling_distribution(self):
        for enabled in (False, True):
            self.assertEqual(self.mode(True, True, enabled, enabled), "prefill")
            with self.assertRaisesRegex(RuntimeError, "rejection-sampling mismatch"):
                self.mode(True, True, enabled, not enabled)
        with self.assertRaisesRegex(RuntimeError, "unknown setting"):
            self.mode(True, True, None, True)

    def test_unknown_capability_is_not_treated_as_target_only(self):
        for cap in (None, 0, 1, "false", "true"):
            for bootstrap in (False, True):
                with self.assertRaisesRegex(
                    RuntimeError, "capability missing or invalid"
                ):
                    self.mode(cap, True, bootstrap=bootstrap)

    def test_seedless_requires_explicit_initialization_mode(self):
        with self.assertRaisesRegex(
            RuntimeError, "requires --disaggregation-decode-draft-bootstrap"
        ):
            self.mode(False, True, False, True)

    def test_seeded_and_seedless_listeners_cannot_be_mixed(self):
        for p, d in ((True, True), (True, False), (False, False)):
            with self.assertRaisesRegex(RuntimeError, "requires target-only prefill"):
                self.mode(p, d, bootstrap=True)


class TestBootstrapPrefix(unittest.TestCase):
    def req(self, **kwargs):
        values = dict(
            origin_input_ids=[10, 11, 12],
            output_ids=[20],
            multimodal_inputs=None,
            input_embeds=None,
            grammar=None,
        )
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_first_handoff_excluded_and_not_mutated(self):
        req = self.req()
        self.assertEqual(bootstrap_prompt(req, 3), [10, 11, 12])
        self.assertEqual(req.output_ids, [20])

    def test_rebootstrap_includes_previously_emitted_prefix(self):
        req = self.req(output_ids=[20, 21, 22])
        self.assertEqual(bootstrap_prompt(req, 5), [10, 11, 12, 20, 21])
        self.assertEqual(req.output_ids, [20, 21, 22])

    def test_budget_fails_without_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "exceeds budget"):
            bootstrap_prompt(self.req(), 2)

    def test_missing_handoff_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "committed handoff"):
            bootstrap_prompt(self.req(output_ids=[]), 8)

    def test_non_text_and_grammar_rejected(self):
        for name in ("multimodal_inputs", "input_embeds", "grammar"):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                bootstrap_prompt(self.req(**{name: object()}), 8)


class TestDecodeBootstrapExecution(unittest.TestCase):
    """CPU tensors plus fake runners: orchestration, not model/GPU validation."""

    def fixture(self):
        req = SimpleNamespace(
            rid="bootstrap-test",
            origin_input_ids=[10, 11, 12],
            output_ids=[20],
            multimodal_inputs=None,
            input_embeds=None,
            grammar=None,
            kv=SimpleNamespace(req_pool_idx=1),
            pd_draft_bootstrap_pending=True,
            pd_draft_bootstrap_tokens=0,
            output_topk_p=None,
            output_topk_index=None,
            hidden_states_tensor=None,
            output_draft_probs=None,
        )
        target_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def target(batch, *, is_verify, capture_hidden_mode):
            self.assertTrue(is_verify, "must not resample the CTX handoff token")
            self.assertEqual(capture_hidden_mode, "FULL")
            self.assertEqual(batch.forward_mode, "EXTEND")
            self.assertEqual(batch.input_ids.tolist(), [10, 11, 12])
            self.assertEqual(batch.out_cache_loc.tolist(), [5, 6, 7])
            self.assertEqual(batch.prefix_lens, [0])
            self.assertEqual(batch.seq_lens.tolist(), [3])
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=target_hidden)
            )

        def draft_extend(batch, hidden, handoff):
            self.assertIs(hidden, target_hidden)
            self.assertEqual(handoff.tolist(), [20])
            q = torch.softmax(hidden[-1], dim=-1).unsqueeze(0)
            return SimpleNamespace(
                topk_p=q[:, :1],
                topk_index=torch.tensor([[0]]),
                hidden_states=hidden[-1:].clone(),
                draft_probs=q,
            )

        draft = SimpleNamespace(
            draft_tp_context=lambda *args: contextlib.nullcontext(),
            draft_runner=SimpleNamespace(tp_group=None),
            _draft_extend_for_prefill=Mock(side_effect=draft_extend),
        )
        scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor([[0, 0, 0], [5, 6, 7]])
            ),
            token_to_kv_pool_allocator=object(),
            tree_cache=object(),
            model_config=SimpleNamespace(vocab_size=2),
            spec_algorithm="EAGLE",
            model_worker=SimpleNamespace(
                target_worker=SimpleNamespace(
                    forward_batch_generation=Mock(side_effect=target)
                ),
                draft_worker=draft,
            ),
        )
        modules = {
            "sglang.srt.layers.moe.utils": SimpleNamespace(
                speculative_moe_a2a_backend_context=contextlib.nullcontext,
                speculative_moe_backend_context=contextlib.nullcontext,
            ),
            "sglang.srt.managers.schedule_batch": SimpleNamespace(
                ScheduleBatch=SimpleNamespace(
                    init_new=Mock(return_value=SimpleNamespace(device="cpu"))
                )
            ),
            "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(
                CaptureHiddenMode=SimpleNamespace(FULL="FULL"),
                ForwardMode=SimpleNamespace(EXTEND="EXTEND"),
            ),
            "sglang.srt.runtime_context": SimpleNamespace(
                get_disagg=lambda: SimpleNamespace(
                    disaggregation_decode_draft_bootstrap_max_tokens=3
                )
            ),
            "sglang.srt.sampling.sampling_batch_info": SimpleNamespace(
                SamplingBatchInfo=SimpleNamespace(
                    from_schedule_batch=Mock(return_value=object())
                )
            ),
        }
        return req, scheduler, modules

    def test_seed_state_without_resampling(self):
        req, scheduler, modules = self.fixture()
        with patch.dict(sys.modules, modules):
            bootstrap_decode_draft(scheduler, req)
        self.assertFalse(req.pd_draft_bootstrap_pending)
        self.assertEqual(req.pd_draft_bootstrap_tokens, 3)
        self.assertEqual(req.output_ids, [20])
        torch.testing.assert_close(
            req.output_draft_probs, torch.softmax(torch.tensor([5.0, 6.0]), dim=-1)
        )
        self.assertIsNone(req.output_dsa_topk_indices)
        scheduler.model_worker.target_worker.forward_batch_generation.assert_called_once()
        scheduler.model_worker.draft_worker._draft_extend_for_prefill.assert_called_once()

    def test_draft_failure_does_not_publish(self):
        req, scheduler, modules = self.fixture()
        scheduler.model_worker.draft_worker._draft_extend_for_prefill.side_effect = (
            RuntimeError("draft failed")
        )
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(RuntimeError, "draft failed"),
        ):
            bootstrap_decode_draft(scheduler, req)
        self.assertTrue(req.pd_draft_bootstrap_pending)
        self.assertIsNone(req.output_draft_probs)
        self.assertEqual(req.pd_draft_bootstrap_tokens, 0)
        self.assertEqual(req.output_ids, [20])

    def test_missing_kv_fails_before_forward(self):
        req, scheduler, modules = self.fixture()
        scheduler.req_to_token_pool.req_to_token[1, 1] = 0
        with (
            patch.dict(sys.modules, modules),
            self.assertRaisesRegex(RuntimeError, "incomplete committed KV"),
        ):
            bootstrap_decode_draft(scheduler, req)
        scheduler.model_worker.target_worker.forward_batch_generation.assert_not_called()
        self.assertTrue(req.pd_draft_bootstrap_pending)


if __name__ == "__main__":
    unittest.main()
