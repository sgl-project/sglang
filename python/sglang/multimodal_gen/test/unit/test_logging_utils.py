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

    def test_record_points_at_the_caller(self):
        """The record must name the caller's file, not this helper's.

        The helpers exist so `warning_once` reads like `warning`, and that
        includes where the line came from -- the original code set stacklevel
        explicitly for it. Nesting the helper deeper moves that frame, and a
        wrong stacklevel silently relabels every warning_once in the package as
        coming from logging_utils.py.
        """
        logger = init_logger("sglang.test.logonce.stacklevel")
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            logger.warning_once("from the caller %d", 1)
        self.assertEqual(captured.records[0].filename, "test_logging_utils.py")

    def test_it_calls_warning_with_stacklevel_2(self):
        """The exact call is the contract, not just the text that comes out.

        test_diffusion_bcg_padding asserts `warning.assert_called_once_with(msg,
        stacklevel=2)`, so both the method and the stacklevel are observable.
        Routing through `logger.log(WARNING, ...)` broke the first; putting a
        helper between the wrapper and `logger.warning` broke the second, because
        stacklevel counts frames and the entry point is bound one frame from the
        caller. Every test I wrote before this one read the emitted record, which
        cannot see either mistake.
        """
        from unittest import mock

        logger = init_logger("sglang.test.logonce.contract")
        with mock.patch.object(logger, "warning") as warning:
            logger.warning_once("cfg_parallel_size=%d > n_branches=%d", 2, 1)
            logger.warning_once("cfg_parallel_size=%d > n_branches=%d", 2, 1)
        warning.assert_called_once_with(
            "cfg_parallel_size=2 > n_branches=1", stacklevel=2
        )

        info_logger = init_logger("sglang.test.logonce.contract.info")
        with mock.patch.object(info_logger, "info") as info:
            info_logger.info_once("degree %d", 2)
        info.assert_called_once_with("degree 2", stacklevel=2)

    def test_cache_clear_is_still_on_the_helpers(self):
        """`.cache_clear()` is part of these helpers' surface, and is used.

        They were lru_cache objects; moving the cache one level down silently
        removed the attribute, and test_diffusion_bcg_padding -- which resets the
        dedup that way so its warning fires -- died with AttributeError.
        """
        from sglang.multimodal_gen.runtime.utils.logging_utils import (
            _print_info_once,
            _print_warning_once,
        )

        logger = init_logger("sglang.test.logonce.clear")
        logger.warning_once("clear me %d", 1)
        _print_warning_once.cache_clear()
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            logger.warning_once("clear me %d", 1)
        self.assertIn("clear me 1", captured.output[0])
        self.assertTrue(callable(_print_info_once.cache_clear))

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
