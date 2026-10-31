"""A per-rank chunk shorter than the deterministic prefill alignment must fail at
startup: it truncates every chunk to zero tokens and the prompt waits forever."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
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


def _align(prefill_backend, gdn, deterministic=True):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model_config = object()
    exec_cfg = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )
    with (
        patch(f"{_MOD}.get_exec", return_value=exec_cfg),
        patch(f"{_MOD}.attention_backends", return_value=(prefill_backend, "triton")),
        patch(f"{_MOD}.hybrid_gdn_config", return_value=object() if gdn else None),
    ):
        scheduler.init_deterministic_inference_config()
    return scheduler.truncation_align_size


class TestGdnChunkAlign(CustomTestCase):
    """Gated DeltaNet prefill results depend on where a prompt is split, so deterministic
    hybrid GDN models must split prompts on the kernel's 64-token block grid."""

    def test_gdn_model_aligns_to_its_block(self):
        self.assertEqual(_align("triton", gdn=True), 64)
        self.assertEqual(_align("flashinfer", gdn=True), 4096)

    def test_other_models_and_normal_mode_unchanged(self):
        self.assertIsNone(_align("triton", gdn=False))
        self.assertEqual(_align("flashinfer", gdn=False), 4096)
        self.assertIsNone(_align("triton", gdn=True, deterministic=False))


class TestDeterministicChunkAlign(CustomTestCase):
    def test_chunk_shorter_than_alignment_fails_at_startup(self):
        with self.assertRaisesRegex(ValueError, "at least 8192"):
            _check(align=4096, chunk=2048, attn_dp_size=2)


if __name__ == "__main__":
    unittest.main()
