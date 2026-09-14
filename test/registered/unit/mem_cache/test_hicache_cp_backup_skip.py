import unittest

from sglang.srt.managers.cache_controller import should_skip_mla_storage_backup
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_config(*, is_mla_model=True, tp_rank=1, attn_cp_size=1):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=8,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=tp_rank if attn_cp_size > 1 else 0,
        attn_cp_size=attn_cp_size,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test",
    )


class TestHiCacheCPBackupSkip(CustomTestCase):
    def test_nonzero_mla_tp_rank_skips_without_cp(self):
        self.assertTrue(should_skip_mla_storage_backup(_make_config()))

    def test_mla_cp_ranks_all_write_storage(self):
        for rank in range(8):
            config = _make_config(tp_rank=rank, attn_cp_size=8)
            self.assertFalse(should_skip_mla_storage_backup(config))

    def test_rank_zero_and_non_mla_never_skip(self):
        self.assertFalse(should_skip_mla_storage_backup(_make_config(tp_rank=0)))
        self.assertFalse(
            should_skip_mla_storage_backup(_make_config(is_mla_model=False))
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
