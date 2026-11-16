from __future__ import annotations

import time
import unittest

from sglang.srt.kv_canary.canary_info.checker import CANARY_INFO_MISMATCH_CODE
from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.kv_canary.perturb.detokenizer_swap import DetokenizerSwapRecord
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.canary_info_e2e_base import (
    CanaryInfoE2EBase,
    DetokenizerSwapE2EBase,
    first_mismatch_positions,
)

register_cuda_ci(est_time=520, stage="extra-a", runner_config="1-gpu-small")


class TestCanaryInfoPassthroughEagle(CanaryInfoE2EBase):
    kv_canary_mode = CanaryMode.RAISE

    def test_client_rechecks_every_delivered_token(self) -> None:
        for payload, body in self.run_requests().values():
            self.assertEqual(body["meta_info"]["finish_reason"]["type"], "length")
            self.assertEqual(len(body["output_ids"]), self.max_new_tokens)
            self.assertIsNone(self.recheck(payload, body))
        self.assert_no_violation(wait_seconds=2.0)


class TestPerturbDetokenizerSwapLogEagle(DetokenizerSwapE2EBase):
    kv_canary_mode = CanaryMode.LOG

    def test_server_logs_and_delivers_while_client_recheck_catches_swap(
        self,
    ) -> None:
        by_rid = self.run_requests()
        swaps = self.collect_swaps()

        time.sleep(2.0)
        logged_positions = first_mismatch_positions(self._captured_log_text())
        for rid, (payload, body) in by_rid.items():
            self.assertEqual(body["meta_info"]["finish_reason"]["type"], "length")
            self.assertEqual(len(body["output_ids"]), self.max_new_tokens)
            mismatch = self.recheck(payload, body)
            if (swap := swaps.get(rid)) is None:
                self.assertIsNone(mismatch, rid)
                self.assertNotIn(rid, logged_positions)
            else:
                expected_position = len(payload["input_ids"]) + swap.position
                self.assertEqual(logged_positions.get(rid), expected_position, rid)
                self.assertIsNotNone(mismatch, rid)
                self.assertEqual(mismatch.position, expected_position, rid)
                self.assertEqual(mismatch.token_id, swap.token, rid)
                self.assertEqual(body["output_ids"][swap.position], swap.token, rid)


class TestPerturbDetokenizerSwapCanaryDisabledEagle(DetokenizerSwapE2EBase):
    kv_canary_mode = CanaryMode.NONE

    def test_disabled_canary_skips_detokenizer_swaps(self) -> None:
        by_rid = self.run_requests()

        for _, body in by_rid.values():
            self.assertNotEqual(body["meta_info"]["finish_reason"]["type"], "abort")
            self.assertNotIn("canary_infos", body["meta_info"])
            assert len(body["output_ids"]) == body["meta_info"]["completion_tokens"]
        assert not DetokenizerSwapRecord.find_all(self._captured_log_text())
        self.assertNotIn(CANARY_INFO_MISMATCH_CODE, self._captured_log_text())


if __name__ == "__main__":
    unittest.main()
