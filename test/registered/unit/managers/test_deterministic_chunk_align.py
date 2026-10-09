"""Deterministic inference must never leave a prompt unschedulable.

A chunk shorter than the prefill alignment truncates every chunk to zero
tokens, so a prompt longer than the chunk used to wait in the queue forever.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MOD = "sglang.srt.managers.scheduler"


def _align_for(prefill_backend):
    scheduler = Scheduler.__new__(Scheduler)
    exec_ctx = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=True)
    )
    with (
        patch(f"{_MOD}.get_exec", return_value=exec_ctx),
        patch(f"{_MOD}.attention_backends", return_value=(prefill_backend, None)),
    ):
        scheduler.init_deterministic_inference_config()
    return scheduler.truncation_align_size


def _check(align, chunk, attn_dp_size=1, mode=DisaggregationMode.NULL):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = mode
    scheduler.truncation_align_size = align
    scheduler.chunked_prefill_size = chunk
    parallel = SimpleNamespace(attn_dp_size=attn_dp_size)
    with patch(f"{_MOD}.get_parallel", return_value=parallel):
        scheduler.check_truncation_align_fits_chunk()


class TestDeterministicChunkAlign(CustomTestCase):
    def test_triton_does_not_align_prefill_chunks(self):
        self.assertIsNone(_align_for("triton"))
        self.assertEqual(_align_for("flashinfer"), 4096)

    def test_chunk_shorter_than_alignment_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "at least 8192"):
            _check(align=4096, chunk=2048, attn_dp_size=2)

    def test_chunk_covering_alignment_starts(self):
        _check(align=4096, chunk=4096)
        _check(align=4096, chunk=None)
        _check(align=None, chunk=2048)
        _check(align=4096, chunk=2048, mode=DisaggregationMode.DECODE)


if __name__ == "__main__":
    unittest.main()
