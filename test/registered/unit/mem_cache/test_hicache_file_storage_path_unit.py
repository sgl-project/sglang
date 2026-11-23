"""
Unit test for --file-storage-path reaching the file and fast_file HiCache
backends through HiCacheController._generate_storage_config, which every
storage attach uses.

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


def _storage_config(backend, flag, env, extra):
    fields = {"file_storage_path": flag} if flag else {}
    with (
        get_context().override_server_args(**fields),
        get_parallel().override(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_dcp_rank=0,
            attn_dcp_size=1,
        ),
        envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(env),
    ):
        controller = object.__new__(HiCacheController)
        controller.storage_backend_type = backend
        controller.mem_pool_device = object()
        controller.mem_pool_host = SimpleNamespace(layout="page_first")
        controller.storage_host_pool = SimpleNamespace(storage_format_tag=None)
        # A nonzero CP rank so HiCacheFile does not create the directory.
        controller.get_attn_cp_rank_and_size = lambda: (1, 2)
        controller.enable_storage_metrics = False
        controller.page_size = 1
        return controller._generate_storage_config(
            "model", {"file_storage_path": extra} if extra else None
        )


# (--file-storage-path, env var, extra-config key) -> directory used.
_CASES = [
    (None, None, None, "/tmp/hicache"),
    ("/flag", None, None, "/flag"),
    (None, "/env", None, "/env"),
    ("/flag", "/env", None, "/env"),
    ("/flag", None, "/extra", "/extra"),
    ("/flag", "/env", "/extra", "/env"),
]


class TestFileStoragePath(CustomTestCase):
    def test_file_directory_precedence(self):
        for flag, env, extra, expected in _CASES:
            with self.subTest(flag=flag, env=env, extra=extra):
                config = _storage_config("file", flag, env, extra)
                with envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(env):
                    self.assertEqual(HiCacheFile(config).file_path, expected)

    def test_fast_file_gets_the_flag(self):
        # fast_file reads the same key; its root resolution is covered in
        # test_hicache_fast_file_unit.py.
        config = _storage_config("fast_file", "/flag", None, None)
        self.assertEqual(config.extra_config["file_storage_path"], "/flag")
        config = _storage_config("fast_file", "/flag", None, "/extra")
        self.assertEqual(config.extra_config["file_storage_path"], "/extra")

    def test_other_backends_do_not_get_the_flag(self):
        config = _storage_config("mooncake", "/flag", None, None)
        self.assertNotIn("file_storage_path", config.extra_config)


if __name__ == "__main__":
    unittest.main()
