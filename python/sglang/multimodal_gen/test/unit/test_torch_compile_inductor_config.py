"""Per-model inductor overrides, and what they must switch off alongside."""

import os
import unittest
from unittest.mock import patch

import torch._inductor.config as inductor_config

from sglang.multimodal_gen.runtime.utils.torch_compile import apply_inductor_config
from sglang.test.test_utils import CustomTestCase

_NO_PICK_CACHE = "TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE"


class TestApplyInductorConfig(CustomTestCase):
    def test_multi_kernel_hints_drop_the_saved_pick(self):
        with (
            inductor_config.patch(multi_kernel_hints=[]),
            patch.dict(os.environ, clear=True),
        ):
            apply_inductor_config({"multi_kernel_hints": [64, 4096]})

            self.assertEqual(inductor_config.multi_kernel_hints, [64, 4096])
            self.assertEqual(os.environ.get(_NO_PICK_CACHE), "1")

    def test_other_overrides_keep_the_saved_pick(self):
        with (
            inductor_config.patch(max_autotune=False),
            patch.dict(os.environ, clear=True),
        ):
            apply_inductor_config({"max_autotune": True})

            self.assertNotIn(_NO_PICK_CACHE, os.environ)

    def test_an_explicit_setting_wins(self):
        with (
            inductor_config.patch(multi_kernel_hints=[]),
            patch.dict(os.environ, {_NO_PICK_CACHE: "0"}, clear=True),
        ):
            apply_inductor_config({"multi_kernel_hints": [64, 4096]})

            self.assertEqual(os.environ[_NO_PICK_CACHE], "0")


if __name__ == "__main__":
    unittest.main()
