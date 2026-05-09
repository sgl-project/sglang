# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.cli.serve import _extract_model_type_override


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


if __name__ == "__main__":
    unittest.main()
