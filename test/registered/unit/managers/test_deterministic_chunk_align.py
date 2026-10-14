"""A per-rank chunk shorter than the deterministic prefill alignment must fail at
startup: it truncates every chunk to zero tokens and the prompt waits forever."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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

    def test_chunk_covering_alignment_starts(self):
        _check(align=4096, chunk=4096)
        _check(align=4096, chunk=2048, mode=DisaggregationMode.DECODE)


if __name__ == "__main__":
    unittest.main()
