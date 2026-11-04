"""Deterministic prefill alignment: a per-rank chunk shorter than the alignment must
fail at startup (it truncates every chunk to zero tokens and the prompt waits
forever), and DSA k-pool models put chunk ends on the chunked-prefill grid."""

import unittest
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


def _kpool_align(deterministic, chunk, kpool=4, align=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model_config = SimpleNamespace(hf_config=object())
    scheduler.truncation_align_size = align
    scheduler.chunked_prefill_size = chunk
    exec_view = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )
    with (
        patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
        patch(
            "sglang.srt.configs.model_config.get_dsa_index_kpool", return_value=kpool
        ),
        patch(f"{_MOD}.get_exec", return_value=exec_view),
    ):
        scheduler.init_dsa_kpool_truncation_align()
    return scheduler.truncation_align_size


class TestDeterministicChunkAlign(CustomTestCase):
    def test_chunk_shorter_than_alignment_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "at least 8192"):
            _check(align=4096, chunk=2048, attn_dp_size=2)

    def test_dsa_kpool_chunk_ends_on_the_chunked_prefill_grid(self):
        # Deterministic inference: chunk ends on multiples of the chunk size.
        self.assertEqual(_kpool_align(True, chunk=16384), 16384)
        # The grid stays a multiple of the k-pool size.
        self.assertEqual(_kpool_align(True, chunk=1000, kpool=64), 960)
        # Without chunked prefill, or outside deterministic inference, only
        # the k-pool alignment applies.
        self.assertEqual(_kpool_align(True, chunk=None), 4)
        self.assertEqual(_kpool_align(False, chunk=16384), 4)


if __name__ == "__main__":
    unittest.main()
