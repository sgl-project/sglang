"""Routed experts kept across retraction: which rows a retraction may keep, and
how the resumed prefill is forced to them."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.state_capturer import routed_experts_forcing as forcing_mod
from sglang.srt.state_capturer import routed_experts_retract as retract_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

NUM_SLOTS, NUM_LAYERS, TOPK = 64, 2, 3
PROMPT_LEN = 4


class TestKeepRowsBeforeRelease(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.buffer = torch.zeros(NUM_SLOTS, NUM_LAYERS, TOPK, dtype=torch.int32)
        self.pool = SimpleNamespace(
            req_to_token=torch.zeros(1, NUM_SLOTS, dtype=torch.int64)
        )
        capturer = SimpleNamespace(
            has_host_cache=True, host_cache=SimpleNamespace(buffer=self.buffer)
        )
        for patcher in (
            mock.patch.object(
                retract_mod,
                "get_resources",
                return_value=SimpleNamespace(experts_capturer=capturer),
            ),
            envs.SGLANG_ENABLE_ROUTED_EXPERTS_RETRACT_FORCING.override(True),
        ):
            patcher.__enter__()
            self.addCleanup(patcher.__exit__, None, None, None)
        self.req = SimpleNamespace(
            rid="r",
            return_routed_experts=True,
            input_embeds=None,
            origin_input_ids=list(range(PROMPT_LEN)),
            output_ids=[],
            routed_experts_kept=None,
            kv=SimpleNamespace(req_pool_idx=0, kv_committed_len=0, holds_kv=True),
        )

    def _forward(self, *, slots, first_position, value, processed=True):
        """Run positions [first_position, +len(slots)) into `slots`, each row = value."""
        end = first_position + len(slots)
        self.pool.req_to_token[0, first_position:end] = torch.tensor(slots)
        self.req.kv.kv_committed_len = end
        if processed:
            self.buffer[slots] = value
            self.req.output_ids = list(range(end + 1 - PROMPT_LEN))

    def _retract(self):
        retract_mod.keep_rows_before_release(req=self.req, req_to_token_pool=self.pool)
        self.req.kv.kv_committed_len = 0

    def _kept_values(self):
        return self.req.routed_experts_kept[:, 0, 0].tolist()

    def test_second_retract_keeps_first_rows_and_appends_the_rest(self):
        """A resumed request re-prefills its earlier positions into reused slots;
        those rows are not the ones its tokens were sampled with."""
        self._forward(slots=[10, 11, 12, 13, 14, 15], first_position=0, value=1)
        self._retract()
        self._forward(slots=[15, 14, 13, 12, 11, 10], first_position=0, value=2)
        self._forward(slots=[20, 21], first_position=6, value=3)
        self._retract()
        self.assertEqual(self._kept_values(), [1] * 6 + [3] * 2)

    def test_allocated_but_unprocessed_position_is_not_read(self):
        """KV is committed when a forward is scheduled, before its row is written."""
        self._forward(slots=[10, 11, 12, 13, 14], first_position=0, value=1)
        self._forward(slots=[30], first_position=5, value=9, processed=False)
        self._retract()
        self.assertEqual(self._kept_values(), [1] * 5)

    def test_retract_during_resume_prefill_does_not_shrink_or_overwrite(self):
        self._forward(slots=[10, 11, 12, 13, 14, 15], first_position=0, value=1)
        self._retract()
        output_ids = self.req.output_ids
        self._forward(slots=[40, 41], first_position=0, value=2)
        self.req.output_ids = output_ids
        self._retract()
        self.assertEqual(self._kept_values(), [1] * 6)

    def test_input_embeds_retract_drops_kept_rows(self):
        """Such a request restarts generation, so no kept row matches a token."""
        self._forward(slots=[10, 11, 12, 13, 14, 15], first_position=0, value=1)
        self._retract()
        self.req.input_embeds = [[0.0]]
        self._forward(slots=[20, 21, 22, 23, 24, 25], first_position=0, value=2)
        self._retract()
        self.assertIsNone(self.req.routed_experts_kept)

    def test_overlay_respects_start_len_and_shorter_returned_rows(self):
        self._forward(slots=[10, 11, 12, 13, 14, 15], first_position=0, value=1)
        self.buffer[[10, 11, 12, 13, 14, 15], 0, 0] = torch.arange(6, dtype=torch.int32)
        self._retract()
        returned = torch.full((3, NUM_LAYERS, TOPK), 7, dtype=torch.int32)
        out = retract_mod.overlay_kept_rows(req=self.req, rows=returned, start_len=2)
        self.assertEqual(out[:, 0, 0].tolist(), [2, 3, 4])
        returned = torch.full((8, NUM_LAYERS, TOPK), 7, dtype=torch.int32)
        out = retract_mod.overlay_kept_rows(req=self.req, rows=returned, start_len=4)
        self.assertEqual(out[:, 0, 0].tolist(), [4, 5] + [7] * 6)


class TestForcing(CustomTestCase):
    def test_batch_rows_follow_each_request_extend_offset(self):
        """Request i's position p sits at forward row sum(extend_lens[:i]) + p - prefix."""
        kept = torch.arange(6, dtype=torch.int32)[:, None, None].expand(6, 2, 3)
        batch = forcing_mod.build_forcing_batch(
            kept_rows=[None, kept, kept],
            prefix_lens=[0, 2, 5],
            extend_lens=[3, 2, 4],
        )
        self.assertEqual(
            batch.mask.tolist(), [False] * 3 + [True, True] + [True] + [False] * 3
        )
        self.assertEqual(batch.routes[:, 0, 0].tolist(), [0, 0, 0, 2, 3, 5, 0, 0, 0])
        self.assertEqual(batch.padded_to(11).mask.tolist()[9:], [False, False])

    def test_no_kept_position_recomputed_builds_nothing(self):
        kept = torch.zeros(4, 2, 3, dtype=torch.int32)
        self.assertIsNone(
            forcing_mod.build_forcing_batch(
                kept_rows=[kept, None], prefix_lens=[4, 0], extend_lens=[2, 5]
            )
        )

    def test_forced_weights_equal_the_gate_weights_for_the_gate_ids(self):
        """Forcing a row to the experts its gate picked must not change the MoE input."""
        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logits = torch.randn(32, 64, device=device, dtype=torch.float32)
        config = TopKConfig(top_k=8, renormalize=True, scoring_func="softmax")
        out = select_experts(
            hidden_states=logits, router_logits=logits, topk_config=config
        )
        weights = forcing_mod._forced_routing_weights(
            logits,
            out.topk_ids,
            scoring_func=config.scoring_func,
            renormalize=config.renormalize,
            routed_scaling_factor=config.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=config.apply_routed_scaling_factor_on_output,
        )
        torch.testing.assert_close(
            weights, out.topk_weights.float(), rtol=1e-5, atol=1e-6
        )

    def test_only_masked_rows_with_a_different_set_are_rerouted(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]] * 3)
        gate_ids = torch.tensor([[0, 1]] * 3)
        gate_weights = torch.tensor([[0.7, 0.3]] * 3)
        ids, weights = forcing_mod.apply_forced_routing(
            topk_ids=gate_ids,
            topk_weights=gate_weights,
            router_logits=logits,
            forced_ids=torch.tensor([[1, 0], [2, 3], [2, 3]], dtype=torch.int32),
            mask=torch.tensor([True, True, False]),
            scoring_func="softmax",
            renormalize=True,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
        )
        self.assertEqual(ids.tolist(), [[0, 1], [2, 3], [0, 1]])
        self.assertEqual(weights[[0, 2]].tolist(), gate_weights[[0, 2]].tolist())
        expected = torch.softmax(torch.tensor([2.0, 1.0]), dim=0)
        torch.testing.assert_close(weights[1], expected)

    def test_forced_extend_check_reports_rows_captured_with_other_experts(self):
        """The check reads the forward's own capture by row offset, not by slot."""
        kept = torch.arange(6, dtype=torch.int32)[:, None, None].expand(6, 2, 3)
        reqs = [
            SimpleNamespace(rid="a", routed_experts_kept=None),
            SimpleNamespace(rid="b", routed_experts_kept=kept),
        ]
        captured = torch.zeros(7, 2, 3, dtype=torch.int32)
        captured[3:7] = kept[2:6]
        kwargs = dict(reqs=reqs, prefix_lens=[0, 2], extend_lens=[3, 4])
        with self.assertNoLogs(retract_mod.logger, level="ERROR"):
            retract_mod.check_forced_extend(captured=captured, **kwargs)
        captured[4] = 9
        with self.assertLogs(retract_mod.logger, level="ERROR") as logs:
            retract_mod.check_forced_extend(captured=captured, **kwargs)
        self.assertIn("1 of the 4 rows of request b", logs.output[0])


if __name__ == "__main__":
    unittest.main()
