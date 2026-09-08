"""CPU contracts for the exact-request EAGLE numerical probe."""

import json
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.eagle_numerical_probe import (
    EagleNumericalProbe,
    maybe_record_eagle_numerical_stage,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEagleNumericalProbe(unittest.TestCase):
    def test_default_off_is_noop(self):
        probe = EagleNumericalProbe(None)
        batch = SimpleNamespace(reqs=[SimpleNamespace(rid="probe-rid")])

        self.assertFalse(probe.can_probe)
        self.assertFalse(probe.matches_schedule_batch(batch))

    def test_exact_rid_records_complete_decode_fingerprint(self):
        probe = EagleNumericalProbe("probe-rid")
        forward_batch = SimpleNamespace(
            rids=["probe-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        input_ids = torch.tensor([11, 12, 99], dtype=torch.int64)
        positions = torch.tensor([101, 102, 999], dtype=torch.int64)
        target_hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)

        with (
            probe.forward_scope(
                forward_batch,
                phase="decode",
                logical_rows=2,
                using_cuda_graph=False,
                input_ids=input_ids,
                target_hidden_states=target_hidden,
                positions=positions,
            ) as active,
        ):
            self.assertTrue(active)
            for stage, tensor in (
                ("nextn_embed", target_hidden + 1),
                ("nextn_decoder", target_hidden + 2),
                ("nextn_norm", target_hidden + 3),
            ):
                maybe_record_eagle_numerical_stage(
                    forward_batch, stage, hidden_states=tensor
                )
            maybe_record_eagle_numerical_stage(
                forward_batch,
                "nextn_logits",
                logits=torch.arange(18).reshape(3, 6).float(),
            )

        probe.record_proposal(
            phase="decode",
            logical_rows=1,
            topk_index=torch.tensor([[3], [4], [5]]),
            topk_probability=torch.tensor([[0.7], [0.8], [0.9]]),
        )
        self.assertIsNone(forward_batch._eagle_numerical_probe_callback)
        self.assertIsNone(forward_batch._eagle_numerical_probe_phase)

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.finish(rid="probe-rid", natural_stop=True, normal_completion=True)

        payload = json.loads(
            logs.output[-1].split("EAGLE_NUMERICAL_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(payload["status"], "complete")
        self.assertTrue(payload["natural_stop"])
        self.assertTrue(payload["normal_completion"])
        stages = payload["phases"]["decode"]
        self.assertEqual(
            set(stages),
            {
                "draft_extend_input",
                "nextn_embed",
                "nextn_decoder",
                "nextn_norm",
                "nextn_logits",
                "proposed_token",
            },
        )
        self.assertEqual(
            stages["draft_extend_input"]["tensors"]["input_ids"]["values"],
            [11, 12],
        )
        self.assertEqual(
            stages["proposed_token"]["tensors"]["topk_index"]["values"],
            [3],
        )
        self.assertEqual(stages["draft_extend_input"]["logical_rows"], 2)
        self.assertEqual(stages["proposed_token"]["logical_rows"], 1)
        self.assertEqual(
            stages["draft_extend_input"]["row_domain"],
            "dense_request_major_prefix",
        )
        self.assertEqual(stages["proposed_token"]["row_domain"], "request_terminal")
        self.assertEqual(
            stages["nextn_embed"]["tensors"]["hidden_states"]["shape"],
            [2, 4],
        )
        self.assertEqual(
            len(stages["nextn_embed"]["tensors"]["hidden_states"]["sha256"]),
            64,
        )

    def test_co_batched_rid_defers_without_forcing_eager_or_rejecting(self):
        probe = EagleNumericalProbe("probe-rid")
        co_batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="probe-rid"), SimpleNamespace(rid="other")]
        )
        self.assertFalse(probe.needs_eager_for_schedule_batch(co_batch))
        self.assertTrue(probe.can_probe)

        sole_batch = SimpleNamespace(reqs=[SimpleNamespace(rid="probe-rid")])
        self.assertTrue(probe.needs_eager_for_schedule_batch(sole_batch))

    def test_wrong_rid_and_second_decode_do_not_capture(self):
        probe = EagleNumericalProbe("probe-rid")
        wrong = SimpleNamespace(
            rids=["other-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        tensor = torch.ones((1, 2))
        with probe.forward_scope(
            wrong,
            phase="decode",
            logical_rows=1,
            using_cuda_graph=False,
            input_ids=torch.ones(1, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(1, dtype=torch.int64),
        ) as active:
            self.assertFalse(active)

        probe._records["decode"] = {}
        matching = SimpleNamespace(
            rids=["probe-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        with probe.forward_scope(
            matching,
            phase="decode",
            logical_rows=1,
            using_cuda_graph=False,
            input_ids=torch.ones(1, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(1, dtype=torch.int64),
        ) as active:
            self.assertFalse(active)

    def test_missing_required_tensor_fails_closed(self):
        probe = EagleNumericalProbe("probe-rid")
        forward_batch = SimpleNamespace(
            rids=["probe-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        tensor = torch.ones((1, 2))
        with probe.forward_scope(
            forward_batch,
            phase="decode",
            logical_rows=1,
            using_cuda_graph=False,
            input_ids=torch.ones(1, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(1, dtype=torch.int64),
        ):
            maybe_record_eagle_numerical_stage(
                forward_batch, "nextn_logits", logits=None
            )
        self.assertFalse(probe.can_probe)

    def test_proposal_row_domain_mismatch_fails_closed(self):
        probe = EagleNumericalProbe("probe-rid")
        forward_batch = SimpleNamespace(
            rids=["probe-rid"],
            _eagle_numerical_probe_callback=None,
            _eagle_numerical_probe_phase=None,
        )
        tensor = torch.ones((2, 2))
        with probe.forward_scope(
            forward_batch,
            phase="decode",
            logical_rows=2,
            using_cuda_graph=False,
            input_ids=torch.ones(2, dtype=torch.int64),
            target_hidden_states=tensor,
            positions=torch.ones(2, dtype=torch.int64),
        ):
            pass
        probe.record_proposal(
            phase="decode",
            logical_rows=2,
            topk_index=torch.ones((2, 1), dtype=torch.int64),
            topk_probability=torch.ones((2, 1)),
        )
        self.assertFalse(probe.can_probe)

    def test_missing_stage_fails_closed(self):
        probe = EagleNumericalProbe("probe-rid")
        probe._seen = True

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.finish(rid="probe-rid", natural_stop=False, normal_completion=True)

        payload = json.loads(
            logs.output[-1].split("EAGLE_NUMERICAL_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(payload["status"], "rejected")
        self.assertIn("phase decode missing stages", payload["rejection"])

    def test_abnormal_completion_fails_closed_and_preserves_stop_reason(self):
        probe = EagleNumericalProbe("probe-rid")
        probe._seen = True

        with self.assertLogs(
            "sglang.srt.speculative.eagle_numerical_probe", level="WARNING"
        ) as logs:
            probe.finish(rid="probe-rid", natural_stop=False, normal_completion=False)

        payload = json.loads(
            logs.output[-1].split("EAGLE_NUMERICAL_PROBE_RESULT ", 1)[1]
        )
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["rejection"], "request did not complete normally")
        self.assertFalse(payload["natural_stop"])
        self.assertFalse(payload["normal_completion"])


if __name__ == "__main__":
    unittest.main()
