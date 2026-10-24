"""Qwen-VL image artifacts must not keep a block of the fast processor pool.

Regression: prepare_artifact_batch copied pixel_values to the CPU inside the
private CUDA pool but kept the processor output, which still held the device
tensor, until the function returned. The pool was released while that block
was live, so its segment stayed reserved in the tokenizer process. Every image
cache miss stranded one more segment on the serving GPU. A processor that
raised after allocating, for example on an extreme aspect ratio, left its
tensors in the traceback while the pool was released, with the same effect.
"""

import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.media_artifacts import MediaArtifactInput
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

BASE = "sglang.srt.multimodal.processors.base_processor"
QWEN = "sglang.srt.multimodal.processors.qwen_vl"


class _DeviceTensor(torch.Tensor):
    """A CPU tensor that reports a CUDA device, standing in for pool memory."""

    __torch_function__ = torch._C._disabled_torch_function_impl

    @property
    def device(self):
        return torch.device("cuda", 0)

    def cpu(self):
        return self.as_subclass(torch.Tensor).clone()


class _ImageProcessor:
    def __init__(self, outputs):
        self.outputs = outputs

    def __call__(self, images, return_tensors, device):
        rows = 4 * len(images)
        pixel_values = torch.arange(rows * 3, dtype=torch.float32)
        pixel_values = pixel_values.reshape(rows, 3).as_subclass(_DeviceTensor)
        grid = torch.tensor([[1, 2, 2]] * len(images)).as_subclass(_DeviceTensor)
        self.outputs.extend([weakref.ref(pixel_values), weakref.ref(grid)])
        return {"pixel_values": pixel_values, "image_grid_thw": grid}


class _RaisingImageProcessor(_ImageProcessor):
    def __call__(self, images, return_tensors, device):
        staged = torch.zeros(len(images), 3, 2, 500).as_subclass(_DeviceTensor)
        self.outputs.append(weakref.ref(staged))
        raise ValueError("absolute aspect ratio must be smaller than 200")


def _processor(outputs, image_processor_cls=_ImageProcessor):
    processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
    image_processor = image_processor_cls(outputs)
    processor._processor = SimpleNamespace(image_processor=image_processor)
    processor._tokenizer = None
    processor.image_config = {}
    processor.disable_fast_image_processor = False
    processor.mm_feature_transport = "cpu"
    processor.mm_preprocess_cache = SimpleNamespace(enabled=False)
    processor._fast_image_processor_device = lambda _: "cuda:0"
    return processor


class TestQwenVLArtifactPool(CustomTestCase):
    def test_no_processor_output_is_live_when_the_pool_is_released(self):
        outputs = []
        live_at_exit = []

        class PoolContext:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                live_at_exit.extend(ref() is not None for ref in outputs)

        entries = [
            MediaArtifactInput(
                content_digest=f"sha256:{index + 1:064x}",
                artifact_key=f"sha256:{index + 11:064x}",
                modality=Modality.IMAGE,
                media=object(),
            )
            for index in range(2)
        ]
        with (
            patch(f"{QWEN}.BaseImageProcessor", _ImageProcessor),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool"),
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=PoolContext()),
        ):
            artifacts = _processor(outputs).prepare_artifact_batch(entries)

        self.assertEqual(len(outputs), 2)
        self.assertEqual(live_at_exit, [False, False])
        self.assertEqual(len(artifacts), 2)
        expected = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        for index, artifact in enumerate(artifacts):
            grid = artifact.model_specific_data["image_grid_thw"]
            self.assertIs(type(artifact.feature), torch.Tensor)
            self.assertIs(type(grid), torch.Tensor)
            self.assertTrue(
                torch.equal(artifact.feature, expected[4 * index : 4 * index + 4])
            )
            self.assertEqual(grid.tolist(), [[1, 2, 2]])

    def test_a_raising_processor_frees_its_tensors_before_the_pool_is_released(
        self,
    ):
        outputs = []
        live_at_exit = []

        class PoolContext:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                live_at_exit.extend(ref() is not None for ref in outputs)

        entries = [
            MediaArtifactInput(
                content_digest=f"sha256:{1:064x}",
                artifact_key=f"sha256:{11:064x}",
                modality=Modality.IMAGE,
                media=object(),
            )
        ]
        processor = _processor(outputs, _RaisingImageProcessor)
        with (
            patch(f"{QWEN}.BaseImageProcessor", _ImageProcessor),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool"),
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=PoolContext()),
            self.assertRaisesRegex(ValueError, "aspect ratio"),
        ):
            processor.prepare_artifact_batch(entries)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(live_at_exit, [False])


if __name__ == "__main__":
    unittest.main()
