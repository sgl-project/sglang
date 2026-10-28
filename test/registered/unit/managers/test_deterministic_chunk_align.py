"""A per-rank chunk shorter than the deterministic prefill alignment must fail at
startup: it truncates every chunk to zero tokens and the prompt waits forever."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.prefill_chunking import (
    check_truncation_align_fits_chunk,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_MOD = "sglang.srt.managers.scheduler_components.prefill_chunking"


def _check(align, chunk, attn_dp_size=1, mode=DisaggregationMode.NULL):
    parallel = SimpleNamespace(attn_dp_size=attn_dp_size)
    with patch(f"{_MOD}.get_parallel", return_value=parallel):
        check_truncation_align_fits_chunk(align, chunk, mode)


class TestDeterministicChunkAlign(CustomTestCase):
    def test_chunk_shorter_than_alignment_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "at least 8192"):
            _check(align=4096, chunk=2048, attn_dp_size=2)


if __name__ == "__main__":
    unittest.main()
