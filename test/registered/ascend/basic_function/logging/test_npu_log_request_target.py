import os
import tempfile
import unittest
from pathlib import Path

from sglang.test.ascend.output_capturer import OutputCapturer
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPULogRequestTarget(TestNPULoggingBase):
    """Testcase: Verify that logs are stored in the target path by setting --log-requests-target

    [Test Category] Parameter
    [Test Target] --log-requests-target;
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = cls._temp_dir_obj.name
        cls.temp_multi_level_dir = os.path.join(
            cls.temp_dir, "level1", "level2", "level3"
        )
        os.makedirs(cls.temp_multi_level_dir, exist_ok=True)
        cls.other_args.extend(
            ["--log-requests-target", "stdout", cls.temp_dir, cls.temp_multi_level_dir]
        )
        cls.output_capturer = OutputCapturer()
        cls.output_capturer.start()
        cls.launch_server()

    def test_log_requests_target(self):
        """Validate that request logs are correctly output to the target files configured via --log-requests-target."""
        self.inference_once()

        # Standard output should include log information.
        content = self.output_capturer.get_all()
        self.assertIn("Receive:", content)
        self.assertIn("Finish:", content)

        # The target log file in a single-level directory should contain log information.
        log_files = list(Path(self.temp_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)
        file_content = log_files[0].read_text()
        self.assertIn("Receive:", file_content)
        self.assertIn("Finish:", file_content)

        # The target log file in a multi-level directory should contain log information.
        log_files = list(Path(self.temp_multi_level_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)
        file_content = log_files[0].read_text()
        self.assertIn("Receive:", file_content)
        self.assertIn("Finish:", file_content)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.output_capturer.stop()
        cls._temp_dir_obj.cleanup()


if __name__ == "__main__":
    unittest.main()
