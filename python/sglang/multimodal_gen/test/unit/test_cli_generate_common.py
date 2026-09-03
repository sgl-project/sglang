import os
import unittest
from unittest.mock import patch

from PIL import Image

from sglang.multimodal_gen.test.single_test_file import cli_generate_common


class TestCLIBaseHelpers(unittest.TestCase):
    def _make_case(self):
        class _ImageDataType:
            @staticmethod
            def get_default_extension():
                return "png"

        class _ImageCLI(cli_generate_common.CLIBase):
            model_path = "dummy/model"
            extra_args = ("--dummy-extra",)
            data_type = _ImageDataType()
            width = 32
            height = 16

        case = _ImageCLI(methodName="test_single_gpu")
        case.setUp()
        self.addCleanup(case.tearDown)
        return case

    def test_run_command_builds_generate_command(self):
        case = self._make_case()
        captured = []

        def fake_run_command(command):
            captured.append(command)
            return True

        with patch(
            "sglang.multimodal_gen.test.single_test_file.cli_generate_common.run_command",
            side_effect=fake_run_command,
        ):
            name, status = case._run_command(
                "sample",
                model_path="dummy/model",
                args='--negative-prompt "low quality"',
            )

        self.assertEqual(name, "sample")
        self.assertEqual(status, "Success")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0:2], ["sglang", "generate"])
        self.assertIn("--model-path=dummy/model", captured[0])
        self.assertIn("--negative-prompt", captured[0])
        self.assertIn("low quality", captured[0])
        self.assertEqual(
            captured[0][-3:], ["--output-file-name", "sample", "--dummy-extra"]
        )

    def test_verify_accepts_expected_image_output(self):
        case = self._make_case()
        output_path = os.path.join(case.output_path, "sample.png")
        Image.new("RGB", (case.width, case.height), color=(255, 0, 0)).save(output_path)

        case.verify("Success", "sample")

    def test_verify_fails_when_output_missing(self):
        case = self._make_case()

        with self.assertRaises(AssertionError):
            case.verify("Success", "missing")
