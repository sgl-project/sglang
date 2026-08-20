import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.eplb_map_record_fused import eplb_map_and_record_fused
from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderReal,
    _SelectExpertsSinglePassGatherer,
    should_record_megamoe_prefill_pass,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import Withable
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch(mode: ForwardMode, *, extend_tokens: int = 0, any_prefill: bool = False):
    return SimpleNamespace(
        forward_mode=mode,
        extend_num_tokens=extend_tokens,
        is_extend_in_batch=any_prefill,
    )


class TestMegaMoEPrefillRecorder(unittest.TestCase):
    def setUp(self):
        self.server_args = SimpleNamespace(moe_a2a_backend="megamoe")
        self.env_patches = (
            patch(
                "sglang.srt.eplb.expert_distribution.envs."
                "SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.eplb.expert_distribution.envs."
                "SGLANG_AITER_MEGA_RANK_SYNC.get",
                return_value=True,
            ),
        )
        for env_patch in self.env_patches:
            env_patch.start()
            self.addCleanup(env_patch.stop)
        capture_patch = patch(
            "sglang.srt.eplb.expert_distribution._is_model_capture_mode",
            return_value=False,
        )
        capture_patch.start()
        self.addCleanup(capture_patch.stop)

    def test_pass_gate_records_only_real_prefill(self):
        cases = [
            (_batch(ForwardMode.EXTEND, extend_tokens=8192), True),
            (_batch(ForwardMode.SPLIT_PREFILL, extend_tokens=8192), True),
            (_batch(ForwardMode.EXTEND, extend_tokens=0), False),
            (_batch(ForwardMode.DECODE), False),
            (_batch(ForwardMode.MIXED, extend_tokens=8192), False),
            (_batch(ForwardMode.IDLE), False),
            (_batch(ForwardMode.TARGET_VERIFY, extend_tokens=8), False),
        ]
        for batch, expected in cases:
            with self.subTest(mode=batch.forward_mode, extend=batch.extend_num_tokens):
                self.assertEqual(
                    should_record_megamoe_prefill_pass(batch, self.server_args),
                    expected,
                )

    def test_capture_warmup_is_not_recorded(self):
        batch = _batch(ForwardMode.EXTEND, extend_tokens=8192)
        with patch(
            "sglang.srt.eplb.expert_distribution._is_model_capture_mode",
            return_value=True,
        ):
            self.assertFalse(
                should_record_megamoe_prefill_pass(batch, self.server_args)
            )

    def test_gate_off_preserves_legacy_decode_recording(self):
        with patch(
            "sglang.srt.eplb.expert_distribution.envs."
            "SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY.get",
            return_value=False,
        ):
            self.assertTrue(
                should_record_megamoe_prefill_pass(
                    _batch(ForwardMode.DECODE), self.server_args
                )
            )

    def test_recorder_skips_decode_and_commits_prefill(self):
        recorder = self._make_recorder()
        recorder._on_forward_pass_start(_batch(ForwardMode.DECODE))
        recorder._on_hook("on_select_experts", topk_ids=torch.tensor([[0]]))
        recorder._on_forward_pass_end(1, {})
        recorder._single_pass_gatherers["primary"].on_select_experts.assert_not_called()

        recorder._on_forward_pass_start(_batch(ForwardMode.EXTEND, extend_tokens=8192))
        recorder._on_forward_pass_end(2, {})
        recorder._accumulator.append.assert_called_once()

    def test_eplb_counter_advances_only_on_global_prefill(self):
        manager = EPLBManager.__new__(EPLBManager)
        manager._server_args = self.server_args
        advances = []

        def generator():
            while True:
                advances.append(1)
                yield

        manager._main_generator = generator()
        manager.on_forward_pass_end(_batch(ForwardMode.DECODE, any_prefill=False))
        self.assertEqual(advances, [])
        manager.on_forward_pass_end(_batch(ForwardMode.IDLE, any_prefill=True))
        self.assertEqual(advances, [1])

    def test_select_experts_filters_all_out_of_range_ids(self):
        gatherer = _SelectExpertsSinglePassGatherer.__new__(
            _SelectExpertsSinglePassGatherer
        )
        gatherer._expert_location_metadata = SimpleNamespace(num_physical_experts=4)
        gatherer._data = torch.zeros((1, 4), dtype=torch.int32)
        gatherer.on_select_experts(
            0, torch.tensor([[-1, 0, 3, 4, 9]], dtype=torch.int32)
        )
        self.assertTrue(torch.equal(gatherer._data, torch.tensor([[1, 0, 0, 1]])))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
    def test_fused_map_record_filters_padding(self):
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
        dispatch = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="cuda")
        info = SimpleNamespace(
            ep_dispatch_algorithm="static",
            partial_logical_to_rank_dispatch_physical_map=dispatch,
        )
        load = torch.zeros(4, dtype=torch.int32, device="cuda")
        valid_tokens = torch.tensor([1], dtype=torch.int32, device="cuda")
        output = eplb_map_and_record_fused(topk_ids, info, load, valid_tokens)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(output.cpu(), torch.tensor([[2, 0], [3, 1]], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(load.cpu(), torch.tensor([1, 0, 1, 0], dtype=torch.int32))
        )

    def _make_recorder(self):
        recorder = _ExpertDistributionRecorderReal.__new__(
            _ExpertDistributionRecorderReal
        )
        gatherer = Mock()
        gatherer.collect.return_value = {"global_physical_count": torch.zeros(1)}
        accumulator = Mock()
        accumulator.get_single_pass_gatherer_key.return_value = "primary"
        recorder._server_args = self.server_args
        recorder._recording = True
        recorder._disable_all = False
        recorder._record_current_pass = False
        recorder._single_pass_gatherers = {"primary": gatherer}
        recorder._accumulator = accumulator
        recorder._current_debug_name = Withable()
        recorder._current_layer_idx = SimpleNamespace(value=0)
        return recorder


if __name__ == "__main__":
    unittest.main()
