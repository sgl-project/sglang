# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.cli.serve import (
    _enable_omni_srt_runtime_defaults,
    _extract_model_type_override,
)


class TestOmniCLIDispatch(unittest.TestCase):
    def test_model_type_override_controls_dispatch_argv(self):
        cases = [
            (
                ["--model-type", "omni", "--model-path", "sensenova-u1"],
                "omni",
                ["--model-path", "sensenova-u1"],
            ),
            (["--model-path", "qwen"], "auto", ["--model-path", "qwen"]),
        ]

        for argv, expected_model_type, expected_argv in cases:
            with self.subTest(argv=argv):
                model_type, remaining_argv = _extract_model_type_override(argv)

                self.assertEqual(expected_model_type, model_type)
                self.assertEqual(expected_argv, remaining_argv)

    def test_omni_model_type_enables_streaming_session_default(self):
        for model_type, expected_enabled in [("omni", True), ("llm", False)]:
            with self.subTest(model_type=model_type):
                server_args = SimpleNamespace(
                    enable_streaming_session=False,
                    max_running_requests=None,
                    pp_max_micro_batch_size=None,
                )

                _enable_omni_srt_runtime_defaults(server_args, model_type)

                self.assertEqual(expected_enabled, server_args.enable_streaming_session)

    def test_omni_preserves_serial_admission_with_internal_srt_slots(self):
        server_args = SimpleNamespace(
            enable_streaming_session=False,
            max_running_requests=1,
            pp_max_micro_batch_size=None,
        )

        _enable_omni_srt_runtime_defaults(server_args, "omni")

        self.assertTrue(server_args.enable_streaming_session)
        self.assertEqual(4, server_args.max_running_requests)
        self.assertEqual(1, server_args.pp_max_micro_batch_size)

    def test_omni_keeps_explicit_pp_micro_batch_size(self):
        server_args = SimpleNamespace(
            enable_streaming_session=False,
            max_running_requests=1,
            pp_max_micro_batch_size=2,
        )

        _enable_omni_srt_runtime_defaults(server_args, "omni")

        self.assertEqual(4, server_args.max_running_requests)
        self.assertEqual(2, server_args.pp_max_micro_batch_size)


if __name__ == "__main__":
    unittest.main()
