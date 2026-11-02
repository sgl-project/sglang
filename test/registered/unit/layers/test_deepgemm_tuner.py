"""CPU regression tests for the DeepGEMM tuning cache lifecycle and exit signals."""

import json
import os
import signal
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import mock_open, patch

from sglang.srt.layers.deep_gemm_tuner import utils as tuner
from sglang.srt.runtime_context import get_parallel, get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDeepGemmTuner(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(get_resources().override(buffers={}))

    def test_missing_lut_is_cached_and_context_reset_discards_it(self):
        """A missing LUT is cached too; rebuilding the runtime must not reuse the
        previous round's config or collected results."""
        vendor = ModuleType("deep_gemm.deep_gemm_tuner")
        results = iter((None, (16, 32, 64)))
        vendor.get_deep_gemm_luts = lambda *args, **kwargs: next(results)
        with patch.dict(sys.modules, {vendor.__name__: vendor}):
            self.assertIsNone(tuner.get_deep_gemm_config(8, 128, 256, 4))
            self.assertIsNone(tuner.get_deep_gemm_config(8, 128, 256, 4))
            self.assertEqual(
                tuner._state().non_tuned_mnk_set,
                {(8, 256, 128, 4, False, "int8")},
            )
            with get_resources().override(buffers={}):
                self.assertEqual(
                    tuner.get_deep_gemm_config(8, 128, 256, 4), (16, 32, 64)
                )
                self.assertEqual(tuner._state().non_tuned_mnk_set, set())

    def test_sigint_dump_preserves_shutdown_even_when_write_fails(self):
        """The tuner must not swallow SIGINT; the original exit handler must run even
        when the file write fails."""
        for fail_write in (False, True):
            with self.subTest(fail_write=fail_write):
                events = []

                def shutdown(signum, frame):
                    events.append("shutdown")
                    raise KeyboardInterrupt

                handlers = {signal.SIGINT: shutdown}
                file = mock_open()
                if fail_write:
                    file.side_effect = OSError("disk is not writable")
                tuner.append_non_tuned_case(8, 128, 256, 4, False, "int8")
                tuner.get_default_user_dir.cache_clear()
                tuner.get_default_deep_gemm_dump_file.cache_clear()
                self.addCleanup(tuner.get_default_user_dir.cache_clear)
                self.addCleanup(tuner.get_default_deep_gemm_dump_file.cache_clear)
                with (
                    patch.dict(os.environ, {"DG_CACHE_DIR": "unused-tuner-cache"}),
                    patch.object(tuner.os, "makedirs"),
                    patch("builtins.open", file),
                    patch.object(tuner.signal, "getsignal", side_effect=handlers.get),
                    patch.object(
                        tuner.signal,
                        "signal",
                        side_effect=lambda sig, handler: handlers.__setitem__(
                            sig, handler
                        ),
                    ),
                    get_parallel().override(
                        moe_ep_group=SimpleNamespace(rank=0, world_size=2)
                    ),
                ):
                    tuner.install_signal_handler()
                    handler = handlers[signal.SIGINT]
                    tuner.install_signal_handler()
                    self.assertIs(handlers[signal.SIGINT], handler)
                    with self.assertRaises(KeyboardInterrupt):
                        handler(signal.SIGINT, None)
                    self.assertEqual(events, ["shutdown"])
                    if not fail_write:
                        payload = "".join(
                            args[0] for args, _ in file().write.call_args_list
                        )
                        self.assertEqual(
                            json.loads(payload), [[8, 256, 128, 4, False, "int8"]]
                        )


if __name__ == "__main__":
    unittest.main()
