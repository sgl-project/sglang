import importlib.metadata
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints import startup_checks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMXFP4JITCacheWarning(CustomTestCase):
    def test_missing_cache_reports_installed_versions_and_cold_start_risk(self):
        with (
            patch.object(startup_checks.importlib.util, "find_spec", return_value=None),
            patch.object(
                startup_checks.importlib.metadata, "version", return_value="0.6.17"
            ),
            self.assertLogs(startup_checks.logger, level="WARNING") as logs,
        ):
            startup_checks.warn_missing_mxfp4_jit_cache("flashinfer_mxfp4", "13.0")
        self.assertEqual(len(logs.output), 1)
        message = logs.output[0]
        self.assertIn("optional flashinfer-jit-cache", message)
        self.assertIn("cold cache", message)
        self.assertIn("flashinfer-python 0.6.17 and CUDA 13.0", message)
        self.assertIn("https://docs.flashinfer.ai/installation.html", message)

    def test_installed_cache_leaves_compatibility_checks_to_flashinfer(self):
        with (
            patch.object(
                startup_checks.importlib.util, "find_spec", return_value=object()
            ),
            patch.object(startup_checks.importlib.metadata, "version") as version,
            self.assertNoLogs(startup_checks.logger, level="WARNING"),
        ):
            startup_checks.warn_missing_mxfp4_jit_cache("flashinfer_mxfp4", "13.0")
        version.assert_not_called()

    def test_other_backends_and_non_cuda_skip_dependency_inspection(self):
        for backend, cuda in [
            ("triton", "13.0"),
            ("auto", "13.0"),
            ("flashinfer_mxfp4", None),
        ]:
            with (
                self.subTest(backend=backend, cuda=cuda),
                patch.object(startup_checks.importlib.util, "find_spec") as find_spec,
                self.assertNoLogs(startup_checks.logger, level="WARNING"),
            ):
                startup_checks.warn_missing_mxfp4_jit_cache(backend, cuda)
                find_spec.assert_not_called()

    def test_missing_flashinfer_does_not_mask_dependency_error(self):
        with (
            patch.object(startup_checks.importlib.util, "find_spec", return_value=None),
            patch.object(
                startup_checks.importlib.metadata,
                "version",
                side_effect=importlib.metadata.PackageNotFoundError,
            ),
            self.assertNoLogs(startup_checks.logger, level="WARNING"),
        ):
            startup_checks.warn_missing_mxfp4_jit_cache("flashinfer_mxfp4", "13.0")


if __name__ == "__main__":
    unittest.main()
