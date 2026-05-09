# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from sglang.cli.serve import (
    _enable_omni_srt_runtime_defaults,
    _extract_model_type_override,
)


class TestOmniCLIDispatch(unittest.TestCase):
    def test_model_type_omni_is_explicit_srt_deployment(self):
        model_type, argv = _extract_model_type_override(
            ["--model-type", "omni", "--model-path", "sensenova-u1"]
        )

        self.assertEqual("omni", model_type)
        self.assertEqual(["--model-path", "sensenova-u1"], argv)

    def test_model_type_auto_stays_default_dispatch(self):
        model_type, argv = _extract_model_type_override(["--model-path", "qwen"])

        self.assertEqual("auto", model_type)
        self.assertEqual(["--model-path", "qwen"], argv)

    def test_model_type_omni_enables_streaming_session(self):
        server_args = SimpleNamespace(enable_streaming_session=False)

        _enable_omni_srt_runtime_defaults(server_args, "omni")

        self.assertTrue(server_args.enable_streaming_session)

    def test_model_type_llm_does_not_enable_streaming_session(self):
        server_args = SimpleNamespace(enable_streaming_session=False)

        _enable_omni_srt_runtime_defaults(server_args, "llm")

        self.assertFalse(server_args.enable_streaming_session)


if __name__ == "__main__":
    unittest.main()
