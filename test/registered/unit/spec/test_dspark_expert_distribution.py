import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderReal,
    _SelectExpertsSinglePassGatherer,
)
from sglang.srt.models import deepseek_v4_dspark
from sglang.srt.utils import Withable
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkExpertDistribution(CustomTestCase):
    def setUp(self):
        # The recorder belongs to the target; DSpark stages use their own
        # zero-based layer IDs and must not update these target-layer counts.
        self.gatherer = _SelectExpertsSinglePassGatherer.__new__(
            _SelectExpertsSinglePassGatherer
        )
        self.gatherer._data = torch.zeros((2, 4), dtype=torch.int)
        self.recorder = _ExpertDistributionRecorderReal.__new__(
            _ExpertDistributionRecorderReal
        )
        self.recorder._disable_all = False
        self.recorder._recording = True
        self.recorder._current_layer_idx = Withable()
        self.recorder._current_debug_name = Withable()
        self.recorder._accumulator = SimpleNamespace(
            get_single_pass_gatherer_key=lambda _: "default"
        )
        self.recorder._single_pass_gatherers = {"default": self.gatherer}
        self.topk_ids = torch.tensor([[0, 2], [2, -1]])
        self.stage = SimpleNamespace(dim=3, _run_moe_ffn_dp_sync=self.moe_forward)
        self.inputs = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)

    def moe_forward(self, x, forward_batch, *, input_ids, input_ids_global):
        self.assertIsNone(input_ids)
        self.assertIsNone(input_ids_global)
        self.recorder.on_select_experts(self.topk_ids)
        return x + 1

    def run_draft(self):
        with mock.patch.object(
            deepseek_v4_dspark,
            "get_global_expert_distribution_recorder",
            return_value=self.recorder,
        ):
            return deepseek_v4_dspark.DSparkV4Stage._run_ffn(
                self.stage, self.inputs, None
            )

    def test_draft_preserves_target_counts_and_output(self):
        for capturing in (False, True):
            with (
                self.subTest(capturing=capturing),
                mock.patch(
                    "torch.get_device_module",
                    return_value=SimpleNamespace(
                        is_current_stream_capturing=lambda: capturing
                    ),
                ),
            ):
                self.recorder._recording = not capturing
                self.gatherer._data.zero_()
                # Exercise the real stat gatherer before and after draft work.
                with self.recorder.with_current_layer(1):
                    self.recorder.on_select_experts(self.topk_ids)
                result = self.run_draft()
                self.assertTrue(torch.equal(result, self.inputs + 1))
                self.assertFalse(self.recorder._disable_all)
                with self.recorder.with_current_layer(1):
                    self.recorder.on_select_experts(self.topk_ids)
                self.assertTrue(
                    torch.equal(
                        self.gatherer._data,
                        torch.tensor([[0, 0, 0, 0], [2, 0, 4, 0]]),
                    )
                )

    def test_draft_restores_recorder_after_exception(self):
        self.stage._run_moe_ffn_dp_sync = mock.Mock(side_effect=ValueError("moe"))
        with self.assertRaisesRegex(ValueError, "moe"):
            self.run_draft()
        self.assertFalse(self.recorder._disable_all)

    def test_draft_preserves_enclosing_disabled_region(self):
        with self.recorder.disable_this_region():
            self.run_draft()
            self.assertTrue(self.recorder._disable_all)
        self.assertFalse(self.recorder._disable_all)


if __name__ == "__main__":
    unittest.main()
