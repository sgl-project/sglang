"""ROCm batches ViT across requests only when the DP encoder has ranks to split over.

CPU-only: the platform, the `mm` bag and the attention-TP topology are stated
through the runtime context; the vision tower and the encoder are fakes.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    maybe_precompile_model_kernels_after_loading,
)
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.multimodal.mm_utils import hip_dp_encoder_batches_vit
from sglang.srt.runtime_context import (
    get_context,
    get_parallel,
    override_platform,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PLATFORMS = ("cuda", "hip", "npu", "xpu")


class _RecordingVisionTower:
    device = torch.device("cpu")
    dtype = torch.float32
    patch_embed = SimpleNamespace(in_channels=3, temporal_patch_size=2, patch_size=16)

    def __init__(self):
        self.calls = []

    def __call__(self, pixel_values, *, grid_thw):
        self.calls.append(
            (pixel_values.shape[0], grid_thw.tolist(), torch.is_grad_enabled())
        )


class _RuntimeCase(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def _runtime(self, *, platform, dp_encoder, attn_tp_size):
        stack = ExitStack()
        stack.enter_context(
            override_platform(**{f"is_{name}": name == platform for name in _PLATFORMS})
        )
        stack.enter_context(
            get_context().override_server_args(mm_enable_dp_encoder=dp_encoder)
        )
        stack.enter_context(
            get_parallel().override(
                tp_size=attn_tp_size,
                attn_tp_size=attn_tp_size,
                moe_tp_size=attn_tp_size,
                attn_tp_rank=0,
            )
        )
        return stack


# (platform, mm_enable_dp_encoder, attn_tp_size, batch across requests)
_GATE_CASES = (
    ("hip", True, 2, True),
    ("hip", True, 1, False),
    ("hip", False, 2, False),
    ("hip", False, 1, False),
    ("npu", True, 2, False),
    ("npu", False, 1, False),
    ("xpu", True, 2, False),
    ("xpu", False, 1, False),
    ("cuda", True, 2, True),
    ("cuda", False, 1, True),
)


class TestVitBatchingGate(_RuntimeCase):
    def test_gate(self):
        for platform, dp_encoder, attn_tp_size, batched in _GATE_CASES:
            with (
                self.subTest(
                    platform=platform, dp_encoder=dp_encoder, attn_tp=attn_tp_size
                ),
                self._runtime(
                    platform=platform,
                    dp_encoder=dp_encoder,
                    attn_tp_size=attn_tp_size,
                ),
            ):
                self.assertEqual(mm_schedule._batch_vit_across_requests(), batched)
                self.assertEqual(
                    hip_dp_encoder_batches_vit(), platform == "hip" and batched
                )

    def test_chunked_prefill_encodes_requests_in_one_call_only_under_gate(self):
        offsets = [(2, 5)]
        seq_len = 8
        for dp_encoder, attn_tp_size, expected_calls in (
            (True, 2, [2]),
            (True, 1, [1, 1]),
            (False, 2, [1, 1]),
        ):
            with (
                self.subTest(dp_encoder=dp_encoder, attn_tp=attn_tp_size),
                self._runtime(
                    platform="hip", dp_encoder=dp_encoder, attn_tp_size=attn_tp_size
                ),
            ):
                mm_schedule.init_mm_embedding_cache(1 << 30)
                items = [
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        hash=2000 + req,
                        feature=torch.zeros(1),
                        offsets=offsets,
                    )
                    for req in range(2)
                ]
                encoder = Mock(side_effect=lambda batch: torch.ones(4 * len(batch), 8))

                chunk, _ = mm_schedule._get_chunked_prefill_embedding(
                    encoder,
                    items,
                    [0, 1, 2],
                    [0, 0],
                    [seq_len, seq_len],
                    [offsets, offsets],
                    torch.zeros(2 * seq_len, dtype=torch.long),
                )

                self.assertEqual(
                    [len(call.args[0]) for call in encoder.call_args_list],
                    expected_calls,
                )
                self.assertEqual(tuple(chunk.shape), (8, 8))


class TestQwen3VLVisionWarmUp(_RuntimeCase):
    @staticmethod
    def _model(visual):
        model = Qwen3VLForConditionalGeneration.__new__(Qwen3VLForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.visual = visual
        return model

    def test_post_load_hook_warms_the_tower_once_under_gate(self):
        tower = _RecordingVisionTower()
        with self._runtime(platform="hip", dp_encoder=True, attn_tp_size=2):
            maybe_precompile_model_kernels_after_loading(self._model(tower), "cpu")

        self.assertEqual(
            tower.calls,
            [
                (32 * 32, [[1, 32, 32]], False),
                (48 * 64, [[1, 48, 64]], False),
                (6 * 32 * 32, [[1, 32, 32]] * 6, False),
            ],
        )

    def test_no_warm_up_off_gate(self):
        for platform, dp_encoder, attn_tp_size, _ in _GATE_CASES:
            if platform == "hip" and dp_encoder and attn_tp_size > 1:
                continue
            tower = _RecordingVisionTower()
            with (
                self.subTest(
                    platform=platform, dp_encoder=dp_encoder, attn_tp=attn_tp_size
                ),
                self._runtime(
                    platform=platform,
                    dp_encoder=dp_encoder,
                    attn_tp_size=attn_tp_size,
                ),
            ):
                maybe_precompile_model_kernels_after_loading(self._model(tower), "cpu")
                self.assertEqual(tower.calls, [])

    def test_language_model_only_skips_warm_up(self):
        with self._runtime(platform="hip", dp_encoder=True, attn_tp_size=2):
            maybe_precompile_model_kernels_after_loading(self._model(None), "cpu")


class TestGlm5NextVisionWarmUp(TestQwen3VLVisionWarmUp):
    @staticmethod
    def _model(visual):
        model = Glm5NextForConditionalGeneration.__new__(
            Glm5NextForConditionalGeneration
        )
        torch.nn.Module.__init__(model)
        model.visual = visual
        return model

    def test_post_load_hook_warms_the_tower_once_under_gate(self):
        tower = _RecordingVisionTower()
        with self._runtime(platform="hip", dp_encoder=True, attn_tp_size=2):
            maybe_precompile_model_kernels_after_loading(self._model(tower), "cpu")

        self.assertEqual(
            tower.calls,
            [
                (32 * 32, [[1, 32, 32]], False),
                (48 * 64, [[1, 48, 64]], False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
