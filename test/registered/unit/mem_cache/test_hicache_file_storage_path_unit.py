"""
Unit test for --file-storage-path reaching the file HiCache backend through
HiCacheController._generate_storage_config, which every storage attach uses.

Pure CPU test; no server, no CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_file_storage_path_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.test_utils import CustomTestCase


def _resolve_file_backend_dir(flag, env, extra):
    fields = {"file_storage_path": flag} if flag else {}
    with (
        get_context().override_server_args(**fields),
        get_parallel().override(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1),
        envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(env),
    ):
        controller = object.__new__(HiCacheController)
        controller.storage_backend_type = "file"
        controller.mem_pool_device = object()
        controller.mem_pool_host = SimpleNamespace(layout="page_first")
        controller.storage_host_pool = SimpleNamespace(storage_format_tag=None)
        # A nonzero CP rank so HiCacheFile does not create the directory.
        controller.get_attn_cp_rank_and_size = lambda: (1, 2)
        controller.enable_storage_metrics = False
        config = controller._generate_storage_config(
            "model", {"file_storage_path": extra} if extra else None
        )
        return HiCacheFile(config).file_path


class TestFileStoragePath(CustomTestCase):
    def test_directory_precedence(self):
        # (--file-storage-path, env var, extra-config key) -> directory used.
        cases = [
            (None, None, None, "/tmp/hicache"),
            ("/flag", None, None, "/flag"),
            (None, "/env", None, "/env"),
            ("/flag", "/env", None, "/flag"),
            ("/flag", "/env", "/extra", "/extra"),
        ]
        for flag, env, extra, expected in cases:
            with self.subTest(flag=flag, env=env, extra=extra):
                self.assertEqual(_resolve_file_backend_dir(flag, env, extra), expected)


if __name__ == "__main__":
    unittest.main()
