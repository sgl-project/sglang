"""Explicit draft window/sink options must be applied by the selected draft path.

Both options used to be accepted and then silently dropped by EAGLE-3 MLA
drafters (window and sink) and by Llama EAGLE-3 / DFLASH (sink).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.speculative_hook import _validate_draft_window_options
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_utils import check_draft_window_honored
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _args(algorithm, window=None, sink=None):
    return ServerArgs(
        model_path="dummy",
        speculative_algorithm=algorithm,
        speculative_draft_window_size=window,
        speculative_draft_sink_size=sink,
    )


def _check(window, backend, sink=None, native_window=None):
    runner = SimpleNamespace(sliding_window_size=native_window, model=SimpleNamespace())
    with get_context().override_server_args(
        speculative_draft_window_size=window, speculative_draft_sink_size=sink
    ):
        check_draft_window_honored(runner, backend)


class TestDraftWindowOptionsByAlgorithm(unittest.TestCase):
    def test_algorithms_without_a_consumer_reject_the_window(self):
        for algorithm in ("DSPARK", "NGRAM"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, "window-size is not supported"):
                    _validate_draft_window_options(_args(algorithm, window=1024))

    def test_dflash_takes_the_window_but_rejects_sinks(self):
        _validate_draft_window_options(_args("DFLASH", window=1024))
        _validate_draft_window_options(_args("DFLASH", window=1024, sink=0))
        with self.assertRaisesRegex(ValueError, "sink-size is not supported"):
            _validate_draft_window_options(_args("DFLASH", window=1024, sink=64))

    def test_eagle_family_defers_to_the_built_draft_path(self):
        for algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
            with self.subTest(algorithm=algorithm):
                _validate_draft_window_options(_args(algorithm, window=1024, sink=64))


class TestDraftWindowHonoredByDraftPath(unittest.TestCase):
    def test_unset_window_needs_no_consumer(self):
        _check(None, None)

    def test_mha_draft_decode_backend_applies_window_and_sink(self):
        _check(1024, SimpleNamespace(draft_window_size=1024, draft_sink_size=64), 64)

    def test_backend_without_draft_windowing_rejects_the_window(self):
        with self.assertRaisesRegex(ValueError, "window-size=1024 is not applied"):
            _check(1024, SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "num-steps <= 1"):
            _check(1024, None)

    def test_native_window_draft_applies_the_window_but_not_sinks(self):
        # LlamaForCausalLMEagle3 routes the flag into its own sliding window.
        backend = SimpleNamespace(draft_window_size=0, draft_sink_size=0)
        _check(1024, backend, native_window=1024)
        with self.assertRaisesRegex(ValueError, "sink-size=64 is not applied"):
            _check(1024, backend, 64, native_window=1024)

    def test_a_different_native_window_overrides_the_window_but_rejects_sinks(self):
        _check(1024, None, native_window=4096)
        for backend in (None, SimpleNamespace(draft_window_size=0, draft_sink_size=0)):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "sink-size=64 is not applied"):
                    _check(1024, backend, 64, native_window=4096)


if __name__ == "__main__":
    unittest.main()
