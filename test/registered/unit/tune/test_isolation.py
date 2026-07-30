import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.tune.isolation import (
    CRASH,
    IMPORT_ERROR,
    NO_KERNEL_IMAGE,
    OOM,
    classify_exception,
    run_candidate_isolated,
)
from sglang.tune.shapes import AttnProfile

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PROF = AttnProfile(40, 8, 128, "bfloat16")
KEYS = ["1:1024", "64:4096"]


class TestFailureClassification(unittest.TestCase):
    def test_classify_exception(self):
        self.assertEqual(classify_exception(ImportError("x")), IMPORT_ERROR)
        self.assertEqual(
            classify_exception(RuntimeError("no kernel image is available")),
            NO_KERNEL_IMAGE,
        )
        self.assertEqual(classify_exception(RuntimeError("CUDA out of memory")), OOM)
        self.assertEqual(classify_exception(ValueError("weird")), CRASH)


class TestSubprocessIsolation(unittest.TestCase):
    def test_healthy_candidate_returns_latencies(self):
        r = run_candidate_isolated(
            "flashinfer", "decode", KEYS, PROF, False, mock=True, isolate=True
        )
        self.assertIsNone(r.failure)
        self.assertEqual(set(r.latencies), set(KEYS))

    def test_uncatchable_abort_is_isolated_not_fatal(self):
        # A signal-killed child (SIGKILL here; the live path uses a true SIGABRT) must be
        # caught by the parent as a crash, NOT bring down the tuner — the whole point of
        # subprocess-per-candidate.
        r = run_candidate_isolated(
            "crash-fa3", "decode", KEYS, PROF, False, mock=True, isolate=True
        )
        self.assertEqual(r.failure, CRASH)
        self.assertEqual(r.latencies, {})

    def test_oom_classified(self):
        r = run_candidate_isolated(
            "oom-backend", "decode", KEYS, PROF, False, mock=True, isolate=True
        )
        self.assertEqual(r.failure, OOM)

    def test_no_kernel_image_classified(self):
        r = run_candidate_isolated(
            "nokernel-backend", "decode", KEYS, PROF, False, mock=True, isolate=True
        )
        self.assertEqual(r.failure, NO_KERNEL_IMAGE)

    def test_tuner_survives_a_crashing_candidate_then_continues(self):
        # crash first, then a healthy candidate in the same run
        bad = run_candidate_isolated(
            "crash-x", "decode", KEYS, PROF, False, mock=True, isolate=True
        )
        good = run_candidate_isolated(
            "fa4", "decode", KEYS, PROF, False, mock=True, isolate=True
        )
        self.assertEqual(bad.failure, CRASH)
        self.assertIsNone(good.failure)  # tuner kept going


if __name__ == "__main__":
    unittest.main()
