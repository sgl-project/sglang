# SPDX-License-Identifier: Apache-2.0

import logging
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.utils.logging_utils import (
    globally_suppress_loggers,
    log_generation_timer,
)


class TestSuppressNoisyDependencyLogs(unittest.TestCase):
    def test_filters_only_pytree_enum_registration_deprecation(self):
        logger = logging.getLogger("torch.utils._pytree")
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            globally_suppress_loggers()
            logger.warning(
                "<enum 'KernelPreference'> is an Enum subclass and is now "
                "natively supported by torch.compile as an opaque value type. "
                "Calling register_constant() on Enum subclasses is deprecated "
                "and will be an error in a future release."
            )
            logger.warning("unrelated pytree warning")

        self.assertEqual(
            captured.output,
            ["WARNING:torch.utils._pytree:unrelated pytree warning"],
        )


class TestGenerationTimer(unittest.TestCase):
    @patch(
        "sglang.multimodal_gen.runtime.utils.logging_utils.time.perf_counter",
        side_effect=[10.0, 12.5, 14.0],
    )
    def test_duration_is_available_inside_context(self, _perf_counter):
        logger = logging.getLogger("test_generation_timer")

        with log_generation_timer(logger, "test prompt") as timer:
            self.assertEqual(timer.duration, 2.5)

        self.assertEqual(timer.duration, 4.0)


if __name__ == "__main__":
    unittest.main()
