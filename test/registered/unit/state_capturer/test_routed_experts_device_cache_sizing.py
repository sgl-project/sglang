"""RoutedExpertsCapturer: device-cache row budget for an unchunked prefill."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.srt.state_capturer.base import BaseDeviceCache, BaseTopkCapturer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

MAX_PREFILL_TOKENS = 16384


def _rows(**override):
    bounds = dict(
        chunked_prefill_size=2048,
        max_prefill_tokens=MAX_PREFILL_TOKENS,
        max_running_requests=8,
        num_dp_ranks=1,
    )
    bounds.update(override)
    return re_mod._device_cache_rows(**bounds)


def _previous_bound(chunked_prefill_size, max_running_requests, num_dp_ranks):
    """The expression on main, kept for the no-shrink and no-growth checks."""
    return max(chunked_prefill_size * num_dp_ranks, max_running_requests * num_dp_ranks)


class TestDeviceCacheRows(CustomTestCase):
    def test_unchunked_prefill_uses_the_prefill_budget(self):
        """#26860: with chunking off, nothing covered an extend batch."""
        self.assertEqual(_rows(chunked_prefill_size=-1), MAX_PREFILL_TOKENS)

    def test_unset_chunked_prefill_also_falls_back(self):
        self.assertEqual(_rows(chunked_prefill_size=None), MAX_PREFILL_TOKENS)

    def test_chunked_prefill_keeps_the_smaller_budget(self):
        """With chunking on, rows stay at the chunk instead of jumping to the ceiling."""
        self.assertEqual(_rows(chunked_prefill_size=2048), 2048)

    def test_dp_ranks_scale_the_prefill_budget(self):
        """The buffer is shared across DP ranks and indexed by dp_rank."""
        self.assertEqual(
            _rows(chunked_prefill_size=-1, num_dp_ranks=2), 2 * MAX_PREFILL_TOKENS
        )

    def test_chunked_budget_is_unchanged(self):
        """Chunked configs must come out byte-for-byte as before, at any DP width."""
        for chunked in (64, 2048, 8192):
            for requests in (1, 32, 512, 4096):
                for dp in (1, 2, 4):
                    with self.subTest(chunked=chunked, requests=requests, dp=dp):
                        self.assertEqual(
                            _rows(
                                chunked_prefill_size=chunked,
                                max_running_requests=requests,
                                num_dp_ranks=dp,
                            ),
                            _previous_bound(chunked, requests, dp),
                        )

    def test_never_smaller_than_the_previous_bound(self):
        """The fix may only grow the buffer, so no shape can start crashing."""
        for chunked in (-1, 0, 256, 8192):
            for requests in (1, 32, 512):
                for dp in (1, 4):
                    with self.subTest(chunked=chunked, requests=requests, dp=dp):
                        self.assertGreaterEqual(
                            _rows(
                                chunked_prefill_size=chunked,
                                max_running_requests=requests,
                                num_dp_ranks=dp,
                            ),
                            _previous_bound(chunked, requests, dp),
                        )


class TestCapturerFeedsTheBudget(CustomTestCase):
    """The constructor must pass the live config into the row budget."""

    MODEL_CONFIG = SimpleNamespace(
        hf_text_config=SimpleNamespace(num_experts_per_tok=4, num_hidden_layers=3)
    )

    def _device_rows(self, *, chunked_prefill_size, max_running_requests, dp_ranks=1):
        seen = {}

        def record(self, **kwargs):
            seen.update(kwargs)

        with (
            mock.patch.object(
                re_mod,
                "get_schedule",
                return_value=SimpleNamespace(
                    chunked_prefill_size=chunked_prefill_size,
                    max_prefill_tokens=MAX_PREFILL_TOKENS,
                ),
            ),
            mock.patch.object(
                re_mod,
                "get_parallel",
                return_value=SimpleNamespace(num_dp_ranks=dp_ranks),
            ),
            mock.patch.object(
                re_mod, "get_moe_a2a_backend", return_value=MoeA2ABackend("none")
            ),
            mock.patch.object(BaseTopkCapturer, "__init__", record),
        ):
            re_mod.RoutedExpertsCapturer(
                model_config=self.MODEL_CONFIG,
                num_tokens=8,
                max_running_requests=max_running_requests,
                num_fused_shared_experts=0,
                device="cpu",
            )
        return seen["max_batch_size"]

    def test_unchunked_prefill_reaches_the_device_cache(self):
        self.assertEqual(
            self._device_rows(chunked_prefill_size=-1, max_running_requests=8),
            MAX_PREFILL_TOKENS,
        )

    def test_chunked_prefill_reaches_the_device_cache(self):
        self.assertEqual(
            self._device_rows(chunked_prefill_size=4096, max_running_requests=8),
            4096,
        )

    def test_decode_heavier_than_prefill_keeps_the_decode_budget(self):
        self.assertEqual(
            self._device_rows(chunked_prefill_size=-1, max_running_requests=32768),
            32768,
        )

    def test_dp_scaling_reaches_the_device_cache(self):
        self.assertEqual(
            self._device_rows(
                chunked_prefill_size=2048, max_running_requests=8, dp_ranks=4
            ),
            8192,
        )


class TestReporterShape(CustomTestCase):
    """#26860's own numbers: an 8192-token unchunked prefill, 48 layers, top-8."""

    LAYERS = 48
    TOPK = 8
    PREFILL_ROWS = 8192
    MAX_RUNNING_REQUESTS = 2048

    def _cache(self, max_batch_size):
        return BaseDeviceCache(
            max_batch_size=max_batch_size,
            num_layers=self.LAYERS,
            topk_size=self.TOPK,
            device="cpu",
            name="routed_experts",
        )

    def test_reporter_configuration_grows_the_buffer(self):
        rows = _rows(
            chunked_prefill_size=-1,
            max_running_requests=self.MAX_RUNNING_REQUESTS,
        )
        self.assertGreaterEqual(rows, self.PREFILL_ROWS)

    def test_capture_succeeds_for_the_unchunked_prefill(self):
        cache = self._cache(
            _rows(
                chunked_prefill_size=-1,
                max_running_requests=self.MAX_RUNNING_REQUESTS,
            )
        )
        topk = torch.arange(self.PREFILL_ROWS * self.TOPK, dtype=torch.int32).reshape(
            self.PREFILL_ROWS, self.TOPK
        )
        cache.capture(layer_id=0, topk_indices=topk)
        self.assertTrue(torch.equal(cache.buffer[: self.PREFILL_ROWS, 0, :], topk))


class TestCaptureBeyondRowsIsFatal(CustomTestCase):
    """Why an omitted term is a crash rather than dropped data."""

    def test_capture_beyond_the_buffer_raises(self):
        cache = BaseDeviceCache(
            max_batch_size=4, num_layers=2, topk_size=3, device="cpu", name="test"
        )
        topk = torch.ones((8, 3), dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            cache.capture(layer_id=0, topk_indices=topk)

    def test_capture_at_the_limit_writes(self):
        cache = BaseDeviceCache(
            max_batch_size=8, num_layers=2, topk_size=3, device="cpu", name="test"
        )
        topk = torch.full((8, 3), 7, dtype=torch.int32)
        cache.capture(layer_id=1, topk_indices=topk)
        self.assertTrue(torch.equal(cache.buffer[:, 1, :], topk))


if __name__ == "__main__":
    unittest.main()
