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

import os
import unittest

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
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


class TestFileBackendMHADraftBackup(CustomTestCase):
    def test_file_backend_mha_draft_is_backed_up_per_rank(self):
        # Regression for an MLA target with an MHA draft (e.g. Kimi-K2-Instruct
        # + Kimi-K2-Instruct-eagle3) on the file backend. A nonzero TP rank has
        # backup_skip=True, and should_backup() used to allow MHA draft/draft-SWA
        # only for mooncake, so the file backend never wrote rank>0's draft
        # shard. Because HiCacheFile keys that shard per rank, the missing file
        # zeroed rank>0's storage hit and the cross-rank MIN collapsed the whole
        # prefix hit to zero.
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from sglang.srt.mem_cache.hicache_storage import PoolName
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        # should_backup now lets every rank back up an MHA draft on the file
        # backend (matching mooncake), while keeping the TP0-only rules for
        # non-MHA draft hosts and for backends that do not namespace draft
        # keys per rank.
        def controller_for(backend_type, host_pool):
            controller = HybridCacheController.__new__(HybridCacheController)
            # Nonzero MLA TP rank: backup_skip is derived from storage_config.
            controller.storage_config = SimpleNamespace(is_storage_writer=False)
            controller.storage_backend_type = backend_type
            controller.mem_pool_host = SimpleNamespace(
                entry_map={PoolName.DRAFT: SimpleNamespace(host_pool=host_pool)}
            )
            return controller

        mha_draft = MagicMock(spec=MHATokenToKVPoolHost)
        draft_transfer = SimpleNamespace(name=PoolName.DRAFT)
        self.assertTrue(controller_for("file", mha_draft).should_backup(draft_transfer))
        self.assertTrue(
            controller_for("mooncake", mha_draft).should_backup(draft_transfer)
        )
        self.assertFalse(
            controller_for("file", SimpleNamespace()).should_backup(draft_transfer)
        )
        self.assertFalse(
            controller_for("hf3fs", mha_draft).should_backup(draft_transfer)
        )


if __name__ == "__main__":
    unittest.main()
