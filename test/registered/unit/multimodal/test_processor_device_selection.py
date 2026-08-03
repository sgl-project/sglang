"""The fast-image-processor device comes from the processor's own ServerArgs.

Regression: the device decision read the published global ServerArgs, which is
last-publish-wins. Two engines in one tokenizer process then shared whichever
config published last, so one engine's images were preprocessed on the other
engine's GPU.
"""

import unittest
from unittest.mock import patch

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BASE = "sglang.srt.multimodal.processors.base_processor"


class _Processor:
    pass


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _make(**fields):
    processor = _StubProcessor.__new__(_StubProcessor)
    processor.server_args = ServerArgs(model_path="dummy", **fields)
    return processor


class TestFastImageProcessorDevice(CustomTestCase):
    def _device(self, processor, **platform):
        flags = {"_is_cpu": False, "_is_xpu": False, "_is_npu": False}
        flags.update(platform)
        with patch.multiple(BASE, **flags):
            return processor._fast_image_processor_device(_Processor())

    def test_device_follows_the_instance_base_gpu_id(self):
        self.assertEqual(self._device(_make(base_gpu_id=3)), "cuda:3")

    def test_engines_in_one_process_keep_their_own_device(self):
        first, second = _make(base_gpu_id=0), _make(base_gpu_id=5)
        self.assertEqual(self._device(first), "cuda:0")
        self.assertEqual(self._device(second), "cuda:5")

    def test_publishing_another_config_does_not_move_the_device(self):
        from sglang.srt.runtime_context import get_context

        processor = _make(base_gpu_id=2)
        override = get_context().override_server_args(base_gpu_id=7)
        override.install()
        self.addCleanup(override.restore)
        self.assertEqual(self._device(processor), "cuda:2")

    def test_rl_on_policy_target_forces_cpu(self):
        processor = _make(base_gpu_id=3, rl_on_policy_target="fsdp")
        self.assertEqual(self._device(processor), "cpu")

    def test_cpu_and_xpu_platforms_win_over_base_gpu_id(self):
        processor = _make(base_gpu_id=3)
        self.assertEqual(self._device(processor, _is_cpu=True), "cpu")
        self.assertEqual(self._device(processor, _is_xpu=True), "xpu")

    def test_npu_glm4v_leaves_the_device_unset(self):
        class Glm4vProcessor:
            pass

        processor = _make(base_gpu_id=3)
        with patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=True):
            device = processor._fast_image_processor_device(Glm4vProcessor())
        self.assertIsNone(device)


if __name__ == "__main__":
    unittest.main()
