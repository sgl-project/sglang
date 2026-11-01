# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module


class TestRocmDefaultAttentionBackend(unittest.TestCase):
    def test_default_backend_follows_arch(self):
        platform = server_args_module.current_platform
        for arch, expected in (
            ("gfx942", "aiter"),
            ("gfx950", "aiter"),
            ("gfx1250", "aiter"),
            ("gfx1100", "torch_sdpa"),
            ("gfx1151", "torch_sdpa"),
            ("gfx1201", "torch_sdpa"),
        ):
            args = SimpleNamespace(attention_backend=None)
            with (
                self.subTest(arch=arch),
                patch.object(platform, "is_rocm", return_value=True),
                patch.object(
                    platform, "get_gcn_arch_name", return_value=arch, create=True
                ),
            ):
                server_args_module.ServerArgs._set_default_attention_backend(args)
            self.assertEqual(args.attention_backend, expected)


if __name__ == "__main__":
    unittest.main()
