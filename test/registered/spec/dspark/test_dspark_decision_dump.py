import json
import logging
import unittest
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dspark_components import dspark_decision_dump
from sglang.srt.speculative.dspark_components.dspark_decision_dump import (
    DsparkDecisionDumper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


def _dump_records(
    *,
    enabled: bool,
    tp_rank: int = 0,
    confidence: Optional[torch.Tensor] = None,
    verify_lens_cpu: Optional[list[int]] = None,
    bs: int = 2,
    gamma: int = 3,
) -> list[dict]:
    dumper = DsparkDecisionDumper(
        gamma=gamma, verify_num_draft_tokens=gamma + 1, tp_rank=tp_rank
    )
    handler = _CaptureHandler()
    dspark_decision_dump.logger.addHandler(handler)
    dspark_decision_dump.logger.setLevel(logging.INFO)
    try:
        with envs.SGLANG_DSPARK_DEBUG_MAIN_OUTPUT.override(enabled):
            dumper.maybe_dump(
                forward_ct=42,
                bs=bs,
                mode="cap-accept",
                budget=5,
                lag_steps=2,
                verify_lens_cpu=verify_lens_cpu,
                confidence=confidence,
                req_pool_indices=torch.tensor([4, 5][:bs]),
                prefix_lens=torch.tensor([100, 200][:bs]),
                draft_tokens=torch.tensor([[11, 12, 13], [21, 22, 23]][:bs]),
                bonus_tokens=torch.tensor([7, 8][:bs]),
                correct_len=torch.tensor([1, 3][:bs]),
                cap_trim_lens=torch.tensor([0, 1][:bs]),
                commit_lens=torch.tensor([2, 4][:bs]),
            )
    finally:
        dspark_decision_dump.logger.removeHandler(handler)
    return [
        json.loads(msg.split("DSPARK_DEBUG_MAIN_OUTPUT=", 1)[1])
        for msg in handler.records
        if msg.startswith("DSPARK_DEBUG_MAIN_OUTPUT=")
    ]


class TestDsparkDecisionDumper(CustomTestCase):
    def test_disabled_emits_nothing(self):
        self.assertEqual(_dump_records(enabled=False), [])

    def test_non_rank0_emits_nothing(self):
        self.assertEqual(_dump_records(enabled=True, tp_rank=1), [])

    def test_enabled_dumps_global_and_per_request_decision(self):
        confidence = torch.tensor([[0.9, 0.8, 0.5], [1.0, 0.0, 0.0]])
        records = _dump_records(
            enabled=True, confidence=confidence, verify_lens_cpu=[2, 4]
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["forward_ct"], 42)
        self.assertEqual(record["bs"], 2)
        self.assertEqual(record["gamma"], 3)
        self.assertEqual(record["mode"], "cap-accept")
        self.assertEqual(record["budget"], 5)
        self.assertEqual(record["lag_steps"], 2)
        self.assertEqual(record["num_verify_tokens"], 6)
        self.assertEqual(record["avg_verify_len"], 3.0)

        first, second = record["reqs"]
        self.assertEqual(first["req"], 4)
        self.assertEqual(first["prefix"], 100)
        self.assertEqual(first["verify_len"], 2)
        self.assertEqual(first["acc_len"], 2)
        self.assertEqual(first["correct_drafts"], 1)
        self.assertEqual(first["cap_trim"], 0)
        self.assertEqual(first["draft_tokens"], [11, 12, 13])
        self.assertEqual(first["bonus_token"], 7)
        self.assertEqual(second["cap_trim"], 1)

    def test_survival_is_prefix_product_of_confidence(self):
        confidence = torch.tensor([[0.9, 0.8, 0.5], [1.0, 0.5, 0.5]])
        record = _dump_records(
            enabled=True, confidence=confidence, verify_lens_cpu=[2, 4]
        )[0]
        for row, conf_row in enumerate(confidence.tolist()):
            survival = record["reqs"][row]["survival"]
            expected = []
            running = 1.0
            for p in conf_row:
                running *= p
                expected.append(round(running, 4))
            self.assertEqual(survival, expected)
            self.assertEqual(
                record["reqs"][row]["confidence"], [round(p, 4) for p in conf_row]
            )

    def test_none_confidence_omits_confidence_but_keeps_outcome(self):
        record = _dump_records(enabled=True, confidence=None, verify_lens_cpu=[2, 4])[0]
        for entry in record["reqs"]:
            self.assertNotIn("confidence", entry)
            self.assertNotIn("survival", entry)
            self.assertIn("acc_len", entry)

    def test_none_layout_falls_back_to_uniform_full_block(self):
        record = _dump_records(enabled=True, confidence=None, verify_lens_cpu=None)[0]
        self.assertTrue(all(entry["verify_len"] == 4 for entry in record["reqs"]))
        self.assertEqual(record["num_verify_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
