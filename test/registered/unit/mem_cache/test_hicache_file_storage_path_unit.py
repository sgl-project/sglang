"""
Unit test for --file-storage-path routing to the file HiCache backend, plus the
regression guard that an UNSET flag keeps HiCacheFile on its /tmp/hicache default
instead of routing the arg's default value into the backend.

Pure CPU test; no server, no CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_file_storage_path_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import dataclasses
import os
import unittest

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_ENV = "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"


def _make_config(extra_config):
    # tp_rank=1 so HiCacheFile does not create the storage dir as a side effect.
    return HiCacheStorageConfig(
        tp_rank=1,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="testmodel",
        extra_config=extra_config,
    )


def _inject(file_storage_path, extra_config=None):
    # Mirrors the HiRadixCache / UnifiedRadixCache routing: only route the arg
    # when it is set, so an unset (default) value is left out of extra_config.
    extra_config = dict(extra_config or {})
    if file_storage_path:
        extra_config.setdefault("file_storage_path", file_storage_path)
    return extra_config


class TestFileStoragePathRouting(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.pop(_ENV, None)

    def tearDown(self):
        if self._saved is not None:
            os.environ[_ENV] = self._saved
        else:
            os.environ.pop(_ENV, None)

    def test_arg_default_is_unset(self):
        # If the default were a real path, an unset flag would silently reroute the
        # backend off its /tmp/hicache default, so keep it None.
        field = {f.name: f for f in dataclasses.fields(ServerArgs)}["file_storage_path"]
        self.assertIsNone(field.default)

    def test_unset_flag_keeps_tmp_hicache(self):
        extra = _inject(None)
        self.assertNotIn("file_storage_path", extra)
        self.assertEqual(HiCacheFile(_make_config(extra)).file_path, "/tmp/hicache")

    def test_set_flag_is_routed(self):
        extra = _inject("/mnt/nvme/hicache")
        self.assertEqual(
            HiCacheFile(_make_config(extra)).file_path, "/mnt/nvme/hicache"
        )

    def test_env_var_wins_over_flag(self):
        os.environ[_ENV] = "/env/hicache"
        extra = _inject("/mnt/nvme/hicache")
        self.assertEqual(HiCacheFile(_make_config(extra)).file_path, "/env/hicache")


if __name__ == "__main__":
    unittest.main()
