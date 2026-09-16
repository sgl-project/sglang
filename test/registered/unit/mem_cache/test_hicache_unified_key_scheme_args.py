import unittest

from sglang.srt.arg_groups.hicache_hook import (
    handle_hicache,
    handle_hicache_ratio_default,
)
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _resolve(**overrides):
    """Run the real hicache resolution; the rewrites land in the resolution record."""
    fields = dict(
        model_path="dummy",
        enable_hierarchical_cache=True,
        hicache_storage_backend="file",
        hicache_storage_key_scheme="unified",
    )
    fields.update(overrides)
    args = ServerArgs(**fields)
    # The ratio default is a separate hook; handle_hicache refuses an unsized
    # host pool without it.
    handle_hicache_ratio_default(args)
    handle_hicache(args)
    return args


class TestUnifiedKeySchemeArgs(CustomTestCase):
    def test_unified_pins_the_layout_and_io_backend(self):
        """The layout is object identity, so it is pinned rather than checked.

        A unified deployment that silently kept another layout would publish
        byte-permuted KV under keys a reader trusts.
        """
        args = _resolve(hicache_mem_layout="page_first", hicache_io_backend="direct")
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_unified")
        self.assertEqual(resolution_result(args, "hicache_io_backend"), "kernel")

    def test_rank_suffix_leaves_the_layout_alone(self):
        args = _resolve(
            hicache_storage_key_scheme="rank-suffix", hicache_mem_layout="page_first"
        )
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_first")

    def test_partition_configs_require_the_unified_scheme(self):
        with self.assertRaisesRegex(ValueError, "require --hicache-storage-key-scheme"):
            _resolve(
                hicache_storage_key_scheme="rank-suffix",
                hicache_storage_head_group=2,
            )

    def test_non_positive_partition_config_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _resolve(hicache_storage_layer_partition=0)

    def test_unified_needs_a_supported_storage_backend(self):
        with self.assertRaisesRegex(ValueError, "needs"):
            _resolve(hicache_storage_backend=None)
        with self.assertRaises(NotImplementedError):
            _resolve(hicache_storage_backend="hf3fs")

    def test_unified_refuses_speculative_decoding(self):
        # Draft layers sit on the host pool's layer axis but outside the grid.
        with self.assertRaises(NotImplementedError):
            _resolve(speculative_algorithm="EAGLE")


class TestFileBackendUnifiedKeys(CustomTestCase):
    def _config(self, unified_suffixes):
        return HiCacheStorageConfig(
            tp_rank=1,
            tp_size=4,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="org/model",
            unified_suffixes=unified_suffixes,
        )

    def test_unified_suffix_replaces_the_whole_rank_suffix(self):
        """No tp/pp/cp coordinate may survive, or the key stops being portable."""
        backend = HiCacheFile.__new__(HiCacheFile)
        HiCacheFile._init_config_suffix(backend, self._config(["ukv1-abc_L0-80_H1"]))
        self.assertEqual(backend.config_suffix, "_ukv1-abc_L0-80_H1")

    def test_rank_suffix_path_is_unchanged(self):
        backend = HiCacheFile.__new__(HiCacheFile)
        HiCacheFile._init_config_suffix(backend, self._config(None))
        self.assertEqual(backend.config_suffix, "_org-model_1_4")

    def test_multi_chunk_grid_is_refused(self):
        """One object per page here, so a fan-out would store only one chunk."""
        backend = HiCacheFile.__new__(HiCacheFile)
        with self.assertRaises(NotImplementedError):
            HiCacheFile._init_config_suffix(backend, self._config(["a", "b"]))


if __name__ == "__main__":
    unittest.main()
