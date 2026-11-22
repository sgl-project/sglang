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

    def test_request_estimate_sums_its_images(self):
        """The free-memory check needs the whole request's need, which grows
        with both the image count and the image size."""
        proc = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        proc._spatial_merge_size = 2
        proc._processor = SimpleNamespace(image_processor=SimpleNamespace(size={}))
        small, big = Image.new("RGB", (500, 500)), Image.new("RGB", (3000, 3000))
        one_small = proc._estimate_gpu_preprocess_bytes([small], {})
        self.assertGreater(proc._estimate_gpu_preprocess_bytes([big], {}), one_small)
        self.assertEqual(
            proc._estimate_gpu_preprocess_bytes([small, small], {}), 2 * one_small
        )


class TestOomFallbackEnabled(CustomTestCase):
    def test_qwen_vl_retries_on_cpu_after_gpu_oom(self):
        self.assertTrue(QwenVLImageProcessor.retry_on_cpu_after_gpu_oom)


class TestChunkedRun(CustomTestCase):
    def _make(self, fail_on_gpu=False, gpu_error=None):
        proc = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        proc._spatial_merge_size = 2
        proc.mm_preprocess_cache = SimpleNamespace(enabled=True)
        proc.mm_feature_transport = "default"
        proc.precompute_hash_before_cpu_transfer = False
        proc._temporary_fast_processor_cuda_pool = lambda d: __import__(
            "contextlib"
        ).nullcontext()
        calls = []
        if fail_on_gpu and gpu_error is None:
            gpu_error = lambda: torch.cuda.OutOfMemoryError("oom")  # noqa: E731

        def fake_run(processor, entries, kwargs, device):
            calls.append((len(entries), device))
            if gpu_error is not None and device != "cpu":
                raise gpu_error()
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

    def test_cuda_call_oom_falls_back_to_cpu_per_chunk(self):
        """An OOM from a CUDA call, not the allocator, is a RuntimeError; both
        the single-call and the per-chunk path must fall back on it."""
        per = qwen_vl._estimate_preprocess_bytes(
            Image.new("RGB", (64, 64)),
            factor=28,
            min_pixels=4 * 28 * 28,
            max_pixels=qwen_vl.MAX_PIXELS,
        )
        error = lambda: RuntimeError("CUDA error: out of memory")  # noqa: E731
        for name, budget, expected in [
            ("single call", None, ["cuda:0", "cpu"]),
            ("per chunk", per * 2, ["cuda:0", "cpu", "cuda:0", "cpu"]),
        ]:
            with self.subTest(name):
                proc, calls = self._make(gpu_error=error)
                with patch.object(
                    qwen_vl,
                    "_PREPROCESS_GPU_BUDGET_BYTES",
                    budget or qwen_vl._PREPROCESS_GPU_BUDGET_BYTES,
                ):
                    f, g = proc._run_image_processor_chunked(
                        self._processor(), self._entries(4), {}, "cuda:0"
                    )
                self.assertEqual([d for _, d in calls], expected)
                self.assertEqual(g.shape, (4, 3))

    def test_non_oom_cuda_error_is_raised(self):
        proc, calls = self._make(
            gpu_error=lambda: RuntimeError("an illegal memory access")
        )
        with self.assertRaises(RuntimeError):
            proc._run_image_processor_chunked(
                self._processor(), self._entries(2), {}, "cuda:0"
            )
        self.assertEqual([d for _, d in calls], ["cuda:0"])


if __name__ == "__main__":
    unittest.main()
