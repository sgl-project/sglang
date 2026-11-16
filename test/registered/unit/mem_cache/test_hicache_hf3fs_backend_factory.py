"""StorageBackendFactory builds the hf3fs backend only for host layouts it can size a page for."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.storage import StorageBackendFactory
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _host_pool(layout):
    return SimpleNamespace(
        layout=layout,
        page_size=4,
        dtype=torch.bfloat16,
        get_ksize_per_token=lambda: 3,
        get_size_per_token=lambda: 6,
    )


class TestHf3fsBackendLayouts(CustomTestCase):
    def test_unsupported_host_layout_raises_a_value_error(self):
        """A host layout hf3fs cannot size a page for must fail with an error that
        names the supported layouts, not an UnboundLocalError."""
        with patch.object(
            HiCacheHF3FS, "from_env_config", return_value=MagicMock()
        ) as from_env_config:
            for layout in ("page_first", "page_first_direct", "layer_first"):
                with self.subTest(layout=layout):
                    StorageBackendFactory.create_backend(
                        "hf3fs", None, _host_pool(layout)
                    )
            self.assertEqual(from_env_config.call_count, 3)

            for layout in ("page_first_kv_split", "page_head", "page_unified"):
                with (
                    self.subTest(layout=layout),
                    self.assertRaisesRegex(ValueError, "supported layouts"),
                ):
                    StorageBackendFactory.create_backend(
                        "hf3fs", None, _host_pool(layout)
                    )
            self.assertEqual(from_env_config.call_count, 3)


if __name__ == "__main__":
    unittest.main()
