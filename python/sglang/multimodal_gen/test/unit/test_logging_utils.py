# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import unittest
import weakref

from sglang.multimodal_gen.runtime.utils.logging_utils import (
    globally_suppress_loggers,
    init_logger,
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


class TestLogOnceTakesFormatArgs(unittest.TestCase):
    """`warning_once(msg, *args)` must format, not raise.

    The helpers took only the message, so every caller that formatted lazily --
    the way logger.warning wants -- raised TypeError instead of logging. Three
    call sites did, and each sat on a branch that rarely runs, so the bug was
    invisible: cfg_parallel_utils only reaches its call when a CFG-parallel
    group has more ranks than branches, which input validation used to reject
    outright. Removing that rejection turned the latent TypeError into a crash
    on the first single-branch request.
    """

    def test_warning_once_formats_lazy_args(self):
        logger = init_logger("sglang.test.logonce.warning")
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            logger.warning_once("cfg_parallel_size=%d > n_branches=%d", 2, 1)
        self.assertIn("cfg_parallel_size=2 > n_branches=1", captured.output[0])

    def test_info_once_formats_lazy_args(self):
        logger = init_logger("sglang.test.logonce.info")
        with self.assertLogs(logger, level=logging.INFO) as captured:
            logger.info_once("degree %d on %d GPUs", 2, 2)
        self.assertIn("degree 2 on 2 GPUs", captured.output[0])

    def test_record_does_not_name_the_caller(self):
        """Documents a wart, so nobody "fixes" it and breaks the bcg assertion.

        init_logger also replaces `logger.warning` with a patched method that
        forwards to `logger.log`, so there is one more frame than the stacklevel
        accounts for and the record names this module rather than the caller.
        That predates these helpers -- the original passed stacklevel=2 through
        the same patched method -- and raising the number would contradict
        test_diffusion_bcg_padding, which asserts the literal `stacklevel=2`.
        """
        logger = init_logger("sglang.test.logonce.stacklevel")
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            logger.warning_once("from the caller %d", 1)
        self.assertEqual(captured.records[0].filename, "logging_utils.py")

    def test_arguments_are_not_retained(self):
        """The once-cache must key on text, not on the arguments.

        An lru_cache keyed on the arguments holds a strong reference to each of
        them for the life of the process, and callers in this package pass
        tensors. Formatting first and caching the result keeps only strings.
        """
        logger = init_logger("sglang.test.logonce.retain")

        class _Heavy:
            def __repr__(self):
                return "<heavy>"

        obj = _Heavy()
        ref = weakref.ref(obj)
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            logger.warning_once("holding %s", obj)
        self.assertIn("<heavy>", captured.output[0])
        del obj
        gc.collect()
        self.assertIsNone(ref(), "the once-cache kept the argument alive")

    def test_same_message_and_args_logs_once(self):
        logger = init_logger("sglang.test.logonce.dedup")
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            logger.warning_once("idle ranks: %d", 1)
            logger.warning_once("idle ranks: %d", 1)
            logger.warning_once("idle ranks: %d", 2)  # different args, new line
        self.assertEqual(len(captured.output), 2, captured.output)


if __name__ == "__main__":
    unittest.main()
