import io
import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.utils import logging_utils as diffusion_logging_utils
from sglang.srt.utils import common as srt_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")


class TestLoggingConfig(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._root_logger = logging.getLogger()
        self._original_level = self._root_logger.level
        self._original_handlers = self._root_logger.handlers[:]

    def tearDown(self):
        for handler in self._root_logger.handlers[:]:
            self._root_logger.removeHandler(handler)
            handler.close()
        for handler in self._original_handlers:
            self._root_logger.addHandler(handler)
        self._root_logger.setLevel(self._original_level)
        super().tearDown()

    def _assert_stream_split(self, configure_logger):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            configure_logger(SimpleNamespace(log_level="info"))
            logger = logging.getLogger("test.logger")
            logger.info("info-msg")
            logger.warning("warn-msg")
            logger.error("error-msg")
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                logger.exception("exception-msg")

        self.assertIn("info-msg", stdout.getvalue())
        self.assertIn("warn-msg", stdout.getvalue())
        self.assertNotIn("error-msg", stdout.getvalue())
        self.assertNotIn("exception-msg", stdout.getvalue())

        self.assertIn("error-msg", stderr.getvalue())
        self.assertIn("exception-msg", stderr.getvalue())
        self.assertIn("RuntimeError: boom", stderr.getvalue())
        self.assertNotIn("info-msg", stderr.getvalue())
        self.assertNotIn("warn-msg", stderr.getvalue())

    def test_srt_configure_logger_splits_streams_by_error_level(self):
        self._assert_stream_split(srt_common.configure_logger)

    def test_diffusion_configure_logger_splits_streams_by_error_level(self):
        self._assert_stream_split(diffusion_logging_utils.configure_logger)


if __name__ == "__main__":
    unittest.main()
