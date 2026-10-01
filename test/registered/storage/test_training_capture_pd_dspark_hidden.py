"""PD capture also records the accepted path of existing hidden-input drafts."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-small")


class TestPDHiddenDraftCapture(PDCaptureRuntimeBase):
    def exercise_mode(self, *, replay, mode="static"):
        self.exercise(
            replay=replay,
            draft_kind="target_hidden",
            prefill_draft=True,
            ragged_mode=mode,
        )

    def test_static_eager(self):
        self.exercise_mode(replay=False)

    def test_static_graph(self):
        self.exercise_mode(replay=True)

    def test_cap_accept_eager(self):
        self.exercise_mode(replay=False, mode="cap-accept")

    def test_cap_accept_graph(self):
        self.exercise_mode(replay=True, mode="cap-accept")

    def test_compact_eager(self):
        self.exercise_mode(replay=False, mode="compact")

    def test_compact_graph(self):
        self.exercise_mode(replay=True, mode="compact")


if __name__ == "__main__":
    unittest.main()
