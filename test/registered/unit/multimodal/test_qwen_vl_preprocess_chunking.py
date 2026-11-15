"""Unit tests for byte-bounded GPU chunking of Qwen-VL image preprocessing."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from sglang.srt.multimodal.processors import qwen_vl
from sglang.srt.multimodal.processors.qwen_vl import (
    QwenVLImageProcessor,
    _chunk_by_byte_budget,
    _estimate_preprocess_bytes,
)
from sglang.test.test_utils import CustomTestCase


class TestChunkByByteBudget(CustomTestCase):
    def test_packs_in_order_under_budget(self):
        groups = _chunk_by_byte_budget([4, 4, 4, 4, 4], budget=10)
        self.assertEqual(groups, [([0, 1], False), ([2, 3], False), ([4], False)])

    def test_oversized_image_isolated(self):
        groups = _chunk_by_byte_budget([3, 50, 3], budget=10)
        self.assertEqual(groups, [([0], False), ([1], True), ([2], False)])

    def test_single_group_when_all_fit(self):
        self.assertEqual(_chunk_by_byte_budget([1, 2, 3], 100), [([0, 1, 2], False)])

    def test_empty(self):
        self.assertEqual(_chunk_by_byte_budget([], 10), [])


class TestEstimate(CustomTestCase):
    kw = dict(factor=28, min_pixels=4 * 28 * 28, max_pixels=1280 * 28 * 28)

    def test_cost_grows_with_source_size(self):
        small = _estimate_preprocess_bytes(Image.new("RGB", (1000, 1000)), **self.kw)
        big = _estimate_preprocess_bytes(Image.new("RGB", (4000, 4000)), **self.kw)
        self.assertGreater(big, small)

    def test_target_pixels_capped_by_max_pixels(self):
        kw = dict(self.kw, max_pixels=28 * 28 * 100)
        cost = _estimate_preprocess_bytes(Image.new("RGB", (4000, 4000)), **kw)
        src = 4000 * 4000 * qwen_vl._PREPROCESS_SOURCE_BYTES_PER_PIXEL
        self.assertLessEqual(
            cost - src, 28 * 28 * 100 * qwen_vl._PREPROCESS_TARGET_BYTES_PER_PIXEL
        )

    def test_unknown_media_isolated(self):
        self.assertGreaterEqual(
            _estimate_preprocess_bytes(object(), **self.kw),
            qwen_vl._PREPROCESS_GPU_BUDGET_BYTES,
        )


class TestReleasePreprocessCache(CustomTestCase):
    def _run(self, reserved, allocated):
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "memory_reserved", return_value=reserved),
            patch.object(torch.cuda, "memory_allocated", return_value=allocated),
            patch.object(torch.cuda, "empty_cache") as empty,
        ):
            qwen_vl._release_preprocess_cache()
        return empty

    def test_releases_when_idle_cache_large(self):
        self._run(3 << 30, 1 << 30).assert_called_once()

    def test_keeps_small_idle_cache(self):
        self._run(1 << 30, 1 << 29).assert_not_called()

    def test_noop_without_cuda(self):
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.cuda, "empty_cache") as empty,
        ):
            qwen_vl._release_preprocess_cache()
        empty.assert_not_called()


class TestOomFallbackEnabled(CustomTestCase):
    def test_qwen_vl_retries_on_cpu_after_gpu_oom(self):
        self.assertTrue(QwenVLImageProcessor.retry_on_cpu_after_gpu_oom)


class TestChunkedRun(CustomTestCase):
    def _make(self, fail_on_gpu=False):
        proc = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        proc._spatial_merge_size = 2
        proc.mm_preprocess_cache = SimpleNamespace(enabled=True)
        proc.mm_feature_transport = "default"
        proc.precompute_hash_before_cpu_transfer = False
        proc._temporary_fast_processor_cuda_pool = lambda d: __import__(
            "contextlib"
        ).nullcontext()
        calls = []

        def fake_run(processor, entries, kwargs, device):
            calls.append((len(entries), device))
            if fail_on_gpu and device != "cpu":
                raise torch.cuda.OutOfMemoryError("oom")
            n = len(entries)
            return torch.arange(n, dtype=torch.float32).repeat_interleave(4).view(
                -1, 1
            ), torch.tensor([[1, 2, 2]] * n)

        proc._run_image_processor = fake_run
        return proc, calls

    def _entries(self, n):
        return [SimpleNamespace(media=Image.new("RGB", (64, 64))) for _ in range(n)]

    def _processor(self):
        return SimpleNamespace(image_processor=SimpleNamespace(size={}))

    def test_oversized_images_run_on_cpu_individually(self):
        proc, calls = self._make()
        with patch.object(qwen_vl, "_PREPROCESS_GPU_BUDGET_BYTES", 1):
            # budget 1 => every image oversized => each runs on CPU individually
            f, g = proc._run_image_processor_chunked(
                self._processor(), self._entries(3), {}, "cuda:0"
            )
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(d == "cpu" for _, d in calls))
        self.assertEqual(g.shape, (3, 3))
        self.assertEqual(f.shape[0], 12)

    def test_multi_chunk_on_gpu_joined_on_cpu(self):
        proc, calls = self._make()
        per = qwen_vl._estimate_preprocess_bytes(
            Image.new("RGB", (64, 64)),
            factor=28,
            min_pixels=4 * 28 * 28,
            max_pixels=qwen_vl.MAX_PIXELS,
        )
        with patch.object(qwen_vl, "_PREPROCESS_GPU_BUDGET_BYTES", per * 2):
            f, g = proc._run_image_processor_chunked(
                self._processor(), self._entries(5), {}, "cuda:0"
            )
        self.assertEqual([n for n, _ in calls], [2, 2, 1])
        self.assertEqual(g.shape, (5, 3))
        self.assertEqual(f.device.type, "cpu")

    def test_oom_falls_back_to_cpu(self):
        proc, calls = self._make(fail_on_gpu=True)
        f, g = proc._run_image_processor_chunked(
            self._processor(), self._entries(2), {}, "cuda:0"
        )
        self.assertEqual([d for _, d in calls], ["cuda:0", "cpu"])
        self.assertEqual(g.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
