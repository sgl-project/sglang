"""Unit tests for CP KV LayerSplit HiCache storage key suffixes."""

import tempfile
import unittest

from sglang.srt.mem_cache.hicache_storage import (
    CP_KV_LAYER_SPLIT_STORAGE_SUFFIX,
    HiCacheFile,
    HiCacheStorageConfig,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    _with_cp_kv_layer_split_suffixes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cfg(
    *,
    attn_cp_rank=0,
    attn_cp_size=1,
    cp_kv_layer_split=True,
    is_mla_model=True,
    pp_size=1,
    pp_rank=0,
    tp_rank=0,
    tp_size=1,
):
    return HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="org/model",
        cp_kv_layer_split=cp_kv_layer_split,
    )


def _file_suffix(cfg):
    with tempfile.TemporaryDirectory() as tmp:
        return HiCacheFile(cfg, file_path=tmp).config_suffix


class TestCpKvLayerSplitHiCacheStorage(CustomTestCase):
    def test_file_layer_split_suffix(self):
        regular_cp_suffix = _file_suffix(
            _cfg(attn_cp_rank=1, attn_cp_size=2, cp_kv_layer_split=False)
        )
        layer_split_suffix = _file_suffix(_cfg(attn_cp_rank=1, attn_cp_size=2))

        self.assertEqual(regular_cp_suffix, "_org-model_cp1_2")
        self.assertNotIn(CP_KV_LAYER_SPLIT_STORAGE_SUFFIX, regular_cp_suffix)
        self.assertEqual(
            layer_split_suffix,
            f"_org-model_{CP_KV_LAYER_SPLIT_STORAGE_SUFFIX}_cp1_2",
        )

    def test_mooncake_layer_split_suffix(self):
        mha_suffix, mla_suffix = _with_cp_kv_layer_split_suffixes("0", "", 1, 2)
        self.assertEqual(mha_suffix, f"0_{CP_KV_LAYER_SPLIT_STORAGE_SUFFIX}_cp1_2")
        self.assertEqual(mla_suffix, f"{CP_KV_LAYER_SPLIT_STORAGE_SUFFIX}_cp1_2")

        mha_suffixes, _ = _with_cp_kv_layer_split_suffixes(["0", "1"], "", 1, 2)
        self.assertEqual(
            mha_suffixes,
            [
                f"0_{CP_KV_LAYER_SPLIT_STORAGE_SUFFIX}_cp1_2",
                f"1_{CP_KV_LAYER_SPLIT_STORAGE_SUFFIX}_cp1_2",
            ],
        )


if __name__ == "__main__":
    unittest.main()
