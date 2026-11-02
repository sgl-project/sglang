"""Device-cache sizing: how many rows one forward can route.

Both top-k capturers share ``device_cache_rows``; the routed-experts one scales
it by DP ranks and the indexer one does not, since its capture is per-rank-local.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.state_capturer import indexer_topk as ix_mod
from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.srt.state_capturer.base import (
    BaseDeviceCache,
    BaseTopkCapturer,
    device_cache_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# #26860's routing shape, and its budgets: chunked prefill off, 16384 input
# tokens, 2048 running requests.
LAYERS = 48
TOPK = 8
MAX_PREFILL_TOKENS = 16384
MAX_RUNNING_REQUESTS = 2048


def _rows(**override):
    bounds = dict(
        chunked_prefill_size=2048,
        max_prefill_tokens=MAX_PREFILL_TOKENS,
        max_req_len=131071,
        max_running_requests=8,
        num_dp_ranks=1,
    )
    bounds.update(override)
    return device_cache_rows(**bounds)


def _bound_on_main(chunked_prefill_size, max_running_requests, num_dp_ranks):
    """The expression on main, to prove chunked configs are untouched."""
    return max(chunked_prefill_size * num_dp_ranks, max_running_requests * num_dp_ranks)


class TestDeviceCacheRows(CustomTestCase):
    # chunked, max_req_len, running, dp -> rows
    CASES = [
        (4096, 131071, 8, 1, 4096),
        (4096, 131071, 8192, 1, 8192),
        (-1, 131071, 2048, 1, 131071),
        (-1, 8192, 2048, 1, MAX_PREFILL_TOKENS),
        (-1, 131071, 2048, 4, 4 * 131071),
        (None, 131071, 2048, 1, 131071),
    ]

    def test_row_budget(self):
        for chunked, max_req_len, running, dp, rows in self.CASES:
            with self.subTest(chunked=chunked, max_req_len=max_req_len, dp=dp):
                self.assertEqual(
                    _rows(
                        chunked_prefill_size=chunked,
                        max_req_len=max_req_len,
                        max_running_requests=running,
                        num_dp_ranks=dp,
                    ),
                    rows,
                )

    def test_chunked_configs_are_untouched(self):
        for chunked in (64, 2048, 8192):
            for running in (1, 512, 4096):
                for dp in (1, 4):
                    with self.subTest(chunked=chunked, running=running, dp=dp):
                        self.assertEqual(
                            _rows(
                                chunked_prefill_size=chunked,
                                max_running_requests=running,
                                num_dp_ranks=dp,
                            ),
                            _bound_on_main(chunked, running, dp),
                        )


class TestRoutedExpertsConstructorWiring(CustomTestCase):
    """The live config must reach the budget through the constructor."""

    MODEL_CONFIG = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            num_experts_per_tok=TOPK, num_hidden_layers=LAYERS
        )
    )

    def _rows(self, *, chunked_prefill_size, max_req_len, dp_ranks=1):
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
                max_req_len=max_req_len,
                max_running_requests=MAX_RUNNING_REQUESTS,
                num_fused_shared_experts=0,
                device="cpu",
            )
        return seen["max_batch_size"]

    def test_unchunked_prefill_scales_with_the_request_cap(self):
        self.assertEqual(self._rows(chunked_prefill_size=-1, max_req_len=24576), 24576)

    def test_chunked_prefill_scales_with_the_chunk(self):
        self.assertEqual(
            self._rows(chunked_prefill_size=4096, max_req_len=24576, dp_ranks=2), 8192
        )


class TestIndexerTopkConstructorWiring(CustomTestCase):
    """IndexerTopkCapturer shares the budget and never scales it by DP."""

    def _rows(self, *, chunked_prefill_size, max_req_len):
        seen = {}

        def record(self, **kwargs):
            seen.update(kwargs)

        with (
            mock.patch.object(
                ix_mod,
                "get_schedule",
                return_value=SimpleNamespace(
                    chunked_prefill_size=chunked_prefill_size,
                    max_prefill_tokens=MAX_PREFILL_TOKENS,
                ),
            ),
            mock.patch.object(
                ix_mod, "get_parallel", return_value=SimpleNamespace(attn_tp_size=1)
            ),
            mock.patch.object(BaseTopkCapturer, "__init__", record),
        ):
            ix_mod.IndexerTopkCapturer(
                num_tokens=8,
                num_indexer_layers=LAYERS,
                index_topk=TOPK,
                max_req_len=max_req_len,
                max_running_requests=MAX_RUNNING_REQUESTS,
                device="cpu",
            )
        return seen["max_batch_size"]

    def test_unchunked_dsa_prefill_scales_with_the_request_cap(self):
        self.assertEqual(self._rows(chunked_prefill_size=-1, max_req_len=24576), 24576)

    def test_chunked_dsa_prefill_scales_with_the_chunk(self):
        self.assertEqual(self._rows(chunked_prefill_size=4096, max_req_len=24576), 4096)


class TestCaptureAtTheBudget(CustomTestCase):
    def test_unchunked_prefill_captures_past_the_input_budget(self):
        """A forward may route more rows than max_prefill_tokens."""
        rows = _rows(
            chunked_prefill_size=-1,
            max_req_len=24576,
            max_running_requests=MAX_RUNNING_REQUESTS,
        )
        cache = BaseDeviceCache(
            max_batch_size=rows,
            num_layers=LAYERS,
            topk_size=TOPK,
            device="cpu",
            name="routed_experts",
        )
        for batch in (8192, MAX_PREFILL_TOKENS, rows):
            with self.subTest(batch=batch):
                topk = torch.arange(batch * TOPK, dtype=torch.int32).reshape(
                    batch, TOPK
                )
                cache.capture(layer_id=0, topk_indices=topk)
                self.assertTrue(torch.equal(cache.buffer[:batch, 0, :], topk))


if __name__ == "__main__":
    unittest.main()
