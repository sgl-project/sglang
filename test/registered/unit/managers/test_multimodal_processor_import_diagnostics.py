import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers import multimodal_processor as mm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMultimodalProcessorImportDiagnostics(CustomTestCase):
    def test_import_failure_is_reported_and_cleared_after_retry(self):
        package_name = "test_processors"
        processor_module = f"{package_name}.broken"
        package = SimpleNamespace(__path__=[])
        module = SimpleNamespace(__name__=processor_module)
        failures = mm.PROCESSOR_IMPORT_FAILURES
        failures.clear()
        self.addCleanup(failures.clear)

        with (
            patch.object(
                mm.importlib,
                "import_module",
                side_effect=[
                    package,
                    ImportError("libavcodec.so.61: cannot open shared object file"),
                ],
            ),
            patch.object(
                mm.pkgutil,
                "iter_modules",
                return_value=[(None, processor_module, False)],
            ),
        ):
            mm.import_processors(package_name)

        hf_config = SimpleNamespace(architectures=["ExampleForConditionalGeneration"])
        with (
            patch.object(mm, "get_mm_processor_cls", return_value=None),
            self.assertRaises(ValueError) as context,
        ):
            mm.get_mm_processor(
                hf_config,
                server_args=None,
                processor=None,
                transport_mode=None,
            )

        message = str(context.exception)
        self.assertIn("No processor registered for architecture", message)
        self.assertIn("Processor modules that failed to import:", message)
        self.assertIn(processor_module, message)
        self.assertIn("ImportError: libavcodec.so.61", message)

        with (
            patch.object(
                mm.importlib,
                "import_module",
                side_effect=[package, module],
            ),
            patch.object(
                mm.pkgutil,
                "iter_modules",
                return_value=[(None, processor_module, False)],
            ),
        ):
            mm.import_processors(package_name)

        self.assertNotIn(processor_module, failures)


if __name__ == "__main__":
    unittest.main()
