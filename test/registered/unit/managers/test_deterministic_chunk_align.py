"""A prompt that no chunk can admit must never wait forever:
a chunk below the prefill alignment fails at startup, a bidir prompt at intake."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_MOD = "sglang.srt.managers.scheduler"


def _check(align, chunk, attn_dp_size=1, mode=DisaggregationMode.NULL):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = mode
    scheduler.truncation_align_size = align
    scheduler.chunked_prefill_size = chunk
    parallel = SimpleNamespace(attn_dp_size=attn_dp_size)
    with patch(f"{_MOD}.get_parallel", return_value=parallel):
        scheduler.check_truncation_align_fits_chunk()


class TestDeterministicChunkAlign(CustomTestCase):
    def test_chunk_shorter_than_alignment_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "at least 8192"):
            _check(align=4096, chunk=2048, attn_dp_size=2)


def _bidir_scheduler(window_pool_names=()):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        window_pool_names=partial(tuple, window_pool_names)
    )
    return scheduler


class TestBidirPromptCut(CustomTestCase):
    def test_window_pools_refuse_a_bidir_prompt(self):
        # Every window pool, a windowless target's fused draft included,
        # prices re-prefills by chunk, so the largest fitting cut can split the prompt.
        for windows in ((), ("window_512",)):
            scheduler = _bidir_scheduler(window_pool_names=windows)
            for boundary in (-1, 0):
                with self.subTest(windows=windows, boundary=boundary):
                    req = SimpleNamespace(
                        origin_input_ids=list(range(8)), bidir_prefix_len=boundary
                    )
                    error = scheduler._validate_bidir_prompt_cut(req)
                    if windows and boundary >= 0:
                        self.assertIsNotNone(error)
                        self.assertIn("sliding-window KV pools", error)
                    else:
                        self.assertIsNone(error)

    def test_prompt_must_fit_the_largest_aligned_cut(self):
        # A retracted request re-prefills in cuts of chunk // align * align tokens,
        # and admission never cuts inside a block-bidirectional prompt.
        scheduler = _bidir_scheduler()
        for align, chunk, prompt, boundary, max_len in (
            (64, 100, 64, 0, None),
            (64, 100, 65, 0, 64),
            (64, 100, 65, -1, None),
            (64, 130, 128, 0, None),
            (64, 130, 129, 0, 128),
            (None, 130, 129, 0, None),
        ):
            with self.subTest(
                align=align, chunk=chunk, prompt=prompt, boundary=boundary
            ):
                scheduler.truncation_align_size = align
                scheduler.chunked_prefill_size = chunk
                req = SimpleNamespace(
                    origin_input_ids=list(range(prompt)), bidir_prefix_len=boundary
                )
                error = scheduler._validate_bidir_prompt_cut(req)
                if max_len is None:
                    self.assertIsNone(error)
                else:
                    self.assertIn(f"at most {max_len} tokens", error)


if __name__ == "__main__":
    unittest.main()
