import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderReal,
    _InfiniteBuffer,
    _SelectExpertsSinglePassGatherer,
    draft_expert_distribution_scope,
    get_global_expert_distribution_recorder,
    should_advance_eplb_counter,
    should_record_forward_pass,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_resources, override_platform
from sglang.srt.utils import Withable
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _batch(mode: ForwardMode, *, extend_tokens: int = 0, any_prefill: bool = False):
    return SimpleNamespace(
        forward_mode=mode,
        extend_num_tokens=extend_tokens,
        is_extend_in_batch=any_prefill,
    )


class TestMegaMoEPrefillRecorder(unittest.TestCase):
    def setUp(self):
        self.env_patches = (
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_megamoe=lambda: True),
            ),
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
        # The prefill-only arm is ROCm-only, so say so rather than inheriting
        # whichever platform the test host happens to be.
        platform_override = override_platform(is_hip=True)
        platform_override.install()
        self.addCleanup(platform_override.restore)

    def test_pass_gate_records_only_real_prefill(self):
        cases = [
            (_batch(ForwardMode.EXTEND, extend_tokens=8192), True),
            (_batch(ForwardMode.EXTEND, extend_tokens=0), False),
            (_batch(ForwardMode.DECODE), False),
            (_batch(ForwardMode.MIXED, extend_tokens=8192), False),
            (_batch(ForwardMode.IDLE), False),
            (_batch(ForwardMode.TARGET_VERIFY, extend_tokens=8), False),
        ]
        for batch, expected in cases:
            with self.subTest(mode=batch.forward_mode, extend=batch.extend_num_tokens):
                self.assertEqual(
                    should_record_forward_pass(batch),
                    expected,
                )

    def test_capture_warmup_is_not_recorded(self):
        batch = _batch(ForwardMode.EXTEND, extend_tokens=8192)
        with patch(
            "sglang.srt.eplb.expert_distribution._is_model_capture_mode",
            return_value=True,
        ):
            self.assertFalse(should_record_forward_pass(batch))

    def test_gate_off_preserves_legacy_decode_recording(self):
        with patch(
            "sglang.srt.eplb.expert_distribution.envs."
            "SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY.get",
            return_value=False,
        ):
            self.assertTrue(should_record_forward_pass(_batch(ForwardMode.DECODE)))

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
        advances = []

        def generator():
            while True:
                advances.append(1)
                yield

        manager._main_generator = generator()

        decode = _batch(ForwardMode.DECODE, any_prefill=False)
        self.assertFalse(should_advance_eplb_counter(decode))
        self.assertEqual(advances, [])

        idle_with_global_prefill = _batch(ForwardMode.IDLE, any_prefill=True)
        self.assertTrue(should_advance_eplb_counter(idle_with_global_prefill))
        manager.on_forward_pass_end()
        self.assertEqual(advances, [1])

    def test_prefill_only_arm_is_inert_off_rocm(self):
        decode = _batch(ForwardMode.DECODE, any_prefill=False)
        with override_platform(is_hip=False):
            self.assertTrue(should_advance_eplb_counter(decode))
            self.assertTrue(should_record_forward_pass(decode))

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

    def test_local_count_buffer_is_this_ranks_slice_of_the_layer_row(self):
        recorder = _ExpertDistributionRecorderReal.__new__(
            _ExpertDistributionRecorderReal
        )
        gatherer = _SelectExpertsSinglePassGatherer.__new__(
            _SelectExpertsSinglePassGatherer
        )
        gatherer._data = torch.zeros((3, 8), dtype=torch.int32)
        recorder._single_pass_gatherers = {"primary": gatherer}
        recorder._current_layer_idx = SimpleNamespace(value=1)
        local = dict(num_experts=8, ep_rank=2, ep_size=4)
        with patch.object(recorder, "_hook_enabled", return_value=True):
            recorder.get_current_pass_local_count_buffer(**local).fill_(1)
            expected = torch.zeros((3, 8), dtype=torch.int32)
            expected[1, 4:6] = 1
            self.assertTrue(torch.equal(gatherer._data, expected))
            self.assertIsNone(
                recorder.get_current_pass_local_count_buffer(
                    num_experts=12, ep_rank=2, ep_size=4
                )
            )
            recorder._current_layer_idx = SimpleNamespace(value=None)
            self.assertIsNone(recorder.get_current_pass_local_count_buffer(**local))
        recorder._current_layer_idx = SimpleNamespace(value=1)
        with patch.object(recorder, "_hook_enabled", return_value=False):
            self.assertIsNone(recorder.get_current_pass_local_count_buffer(**local))

    def _make_recorder(self):
        recorder = _ExpertDistributionRecorderReal.__new__(
            _ExpertDistributionRecorderReal
        )
        gatherer = Mock()
        gatherer.collect.return_value = {"global_physical_count": torch.zeros(1)}
        accumulator = Mock()
        accumulator.get_single_pass_gatherer_key.return_value = "primary"
        recorder._recording = True
        recorder._disable_all = False
        recorder._record_current_pass = False
        recorder._single_pass_gatherers = {"primary": gatherer}
        recorder._accumulator = accumulator
        recorder._current_debug_name = Withable()
        recorder._current_layer_idx = SimpleNamespace(value=0)
        return recorder


def _counting_recorder(num_layers: int, num_physical_experts: int):
    gatherer = _SelectExpertsSinglePassGatherer.__new__(
        _SelectExpertsSinglePassGatherer
    )
    gatherer._expert_location_metadata = SimpleNamespace(
        num_physical_experts=num_physical_experts
    )
    gatherer._data = torch.zeros((num_layers, num_physical_experts), dtype=torch.int32)
    accumulator = Mock()
    accumulator.get_single_pass_gatherer_key.return_value = "primary"
    recorder = _ExpertDistributionRecorderReal.__new__(_ExpertDistributionRecorderReal)
    recorder._recording = True
    recorder._disable_all = False
    recorder._record_current_pass = True
    recorder._single_pass_gatherers = {"primary": gatherer}
    recorder._accumulator = accumulator
    recorder._current_debug_name = Withable()
    recorder._current_layer_idx = Withable()
    return recorder, gatherer._data


class TestDraftFamilyRecorder(unittest.TestCase):
    def setUp(self):
        platform_override = override_platform(is_hip=False)
        platform_override.install()
        self.addCleanup(platform_override.restore)
        self.target, self.target_counts = _counting_recorder(4, 8)
        self.draft, self.draft_counts = _counting_recorder(2, 6)

    def test_draft_routes_count_only_in_the_draft_buffers(self):
        """Draft hooks and the fused count buffer resolve to the draft recorder by
        stage index; the target's buffers never see a draft route."""
        picks = torch.tensor([[0, 5], [5, 2], [-1, -1]], dtype=torch.int32)
        with get_resources().override(
            expert_distribution_recorder=self.target,
            draft_expert_distribution_recorder=self.draft,
        ):
            with draft_expert_distribution_scope() as recorder:
                self.assertIs(recorder, self.draft)
                self.assertIs(
                    get_global_expert_distribution_recorder("target"), self.target
                )
                with recorder.with_current_layer(1):
                    get_global_expert_distribution_recorder().on_select_experts(picks)
                    buffer = get_global_expert_distribution_recorder().get_current_pass_count_buffer()
            self.assertIs(get_global_expert_distribution_recorder(), self.target)
            self.assertIs(get_global_expert_distribution_recorder("draft"), self.draft)
        self.assertEqual(buffer.data_ptr(), self.draft_counts[1].data_ptr())
        self.assertEqual(self.draft_counts[1].tolist(), [1, 0, 1, 0, 0, 2])
        self.assertEqual(self.draft_counts[0].tolist(), [0] * 6)
        self.assertEqual(int(self.target_counts.abs().sum()), 0)

    def test_without_draft_family_the_draft_records_nothing(self):
        picks = torch.tensor([[0, 5]], dtype=torch.int32)
        with get_resources().override(
            expert_distribution_recorder=self.target,
            draft_expert_distribution_recorder=None,
        ):
            with draft_expert_distribution_scope() as recorder:
                self.assertIsNone(recorder)
                with self.target.with_current_layer(0):
                    get_global_expert_distribution_recorder().on_select_experts(picks)
                    self.assertIsNone(
                        get_global_expert_distribution_recorder().get_current_pass_count_buffer()
                    )
        self.assertEqual(int(self.target_counts.abs().sum()), 0)


class TestInfiniteBuffer(unittest.TestCase):
    def test_growth_inside_inference_mode_accepts_later_appends(self):
        """A buffer grown under inference_mode must still accept appends outside it."""
        buffer = _InfiniteBuffer(item_shape=(2,), dtype=torch.int32, device="cpu")
        with torch.inference_mode():
            for i in range(129):
                buffer.append(torch.full((2,), i, dtype=torch.int32))
        with torch.no_grad():
            buffer.append(torch.full((2,), 129, dtype=torch.int32))
        self.assertEqual(buffer.get_all()[:, 0].tolist(), list(range(130)))


if __name__ == "__main__":
    unittest.main()
