import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.dsa import mqa_logits_backend
from sglang.srt.layers.attention.dsa.mqa_logits_backend import DSAMQALogitsBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSAMQALogitsBackend(unittest.TestCase):
    def test_auto_preserves_triton_default(self):
        self.assertEqual(
            DSAMQALogitsBackend.resolve("auto"), DSAMQALogitsBackend.TRITON
        )
        self.assertEqual(
            DSAMQALogitsBackend.resolve("triton"), DSAMQALogitsBackend.TRITON
        )

    def test_flydsl_requires_rocm_gfx950(self):
        with (
            patch.object(mqa_logits_backend, "is_hip", return_value=False),
            self.assertRaisesRegex(ValueError, "requires ROCm gfx950"),
        ):
            DSAMQALogitsBackend.resolve("flydsl")

        with (
            patch.object(mqa_logits_backend, "is_hip", return_value=True),
            patch.object(mqa_logits_backend, "is_gfx95_supported", return_value=False),
            self.assertRaisesRegex(ValueError, "requires ROCm gfx950"),
        ):
            DSAMQALogitsBackend.resolve("flydsl")

    def test_flydsl_resolves_on_gfx950(self):
        with (
            patch.object(mqa_logits_backend, "is_hip", return_value=True),
            patch.object(mqa_logits_backend, "is_gfx95_supported", return_value=True),
        ):
            self.assertEqual(
                DSAMQALogitsBackend.resolve("flydsl"),
                DSAMQALogitsBackend.FLYDSL,
            )

    def test_unknown_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown dsa_mqa_logits_backend"):
            DSAMQALogitsBackend.resolve("unknown")


if __name__ == "__main__":
    unittest.main()
