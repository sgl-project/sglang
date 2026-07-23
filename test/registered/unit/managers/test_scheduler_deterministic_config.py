"""Unit tests for Scheduler.init_deterministic_inference_config — no server,
no model loading.

Regression test for the fa3 backend missing from the chunked-prefill
truncation alignment table under deterministic inference: the table only
covered flashinfer and triton, so fa3 ran with truncation_align_size=None
and chunked prefill could split requests at batch-dependent, unaligned
boundaries.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _truncation_align_size(*, enable: bool, backend: str):
    """Run init_deterministic_inference_config on a minimal fake scheduler."""
    fake_scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_deterministic_inference=enable,
            attention_backend=backend,
        )
    )
    Scheduler.init_deterministic_inference_config(fake_scheduler)
    return fake_scheduler.truncation_align_size


class TestInitDeterministicInferenceConfig(CustomTestCase):
    def test_disabled_yields_none_for_all_backends(self):
        for backend in ("flashinfer", "fa3", "triton", "torch_native"):
            with self.subTest(backend=backend):
                self.assertIsNone(_truncation_align_size(enable=False, backend=backend))

    def test_supported_backends_get_default_alignment(self):
        # fa3 is the regression case: it is listed in
        # DETERMINISTIC_ATTENTION_BACKEND_CHOICES (and is the default fallback
        # on Hopper) but was missing from the backend table, so chunked
        # prefill could truncate at unaligned boundaries and break
        # determinism.
        for backend in ("flashinfer", "fa3", "triton"):
            with self.subTest(backend=backend):
                self.assertEqual(
                    _truncation_align_size(enable=True, backend=backend), 4096
                )

    def test_fa3_alignment_env_override(self):
        # Also cross-checks that the name registered in environ.py matches
        # the name the scheduler reads via get_int_env_var.
        with envs.SGLANG_FA3_PREFILL_TRUNCATION_ALIGN_SIZE.override(8192):
            self.assertEqual(_truncation_align_size(enable=True, backend="fa3"), 8192)

    def test_unsupported_backend_yields_none(self):
        self.assertIsNone(_truncation_align_size(enable=True, backend="trtllm_mha"))


if __name__ == "__main__":
    unittest.main()
