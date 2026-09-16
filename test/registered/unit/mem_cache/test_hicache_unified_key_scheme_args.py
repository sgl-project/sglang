import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.hicache_hook import (
    handle_hicache,
    handle_hicache_ratio_default,
)
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_key_scheme import derive_namespace, namespace_digest
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
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
        for backend in ("hf3fs", "mooncake"):
            with self.subTest(backend=backend), self.assertRaises(NotImplementedError):
                _resolve(hicache_storage_backend=backend)

    def test_unified_refuses_speculative_decoding(self):
        # Draft layers sit on the host pool's layer axis but outside the grid.
        with self.assertRaises(NotImplementedError):
            _resolve(speculative_algorithm="EAGLE")

    def test_unified_refuses_shared_decode_host_receive_pool(self):
        with self.assertRaisesRegex(ValueError, "layer_first"):
            _resolve(
                disaggregation_mode="decode",
                disaggregation_decode_host_receive_threshold=0.5,
            )


class TestControllerUnifiedKeys(CustomTestCase):
    def _controller(self, *, mla=False):
        device_pool = object.__new__(MLATokenToKVPool) if mla else SimpleNamespace()
        # FP8 variants share a uint8 storage view but require distinct keys.
        device_pool.dtype = torch.float8_e4m3fn
        device_pool.store_dtype = torch.uint8
        host_pool = SimpleNamespace(
            device_pool=device_pool,
            head_num=0 if mla else 4,
            head_group_num=1,
            start_layer=4,
            end_layer=8,
            layer_num=4,
            page_unified_layout=PageUnifiedLayout(
                page_size=16,
                layer_num=4,
                head_num=1 if mla else 4,
                head_group_num=1,
                head_dim=128,
                itemsize=1,
                is_mla=mla,
            ),
        )
        controller = object.__new__(HiCacheController)
        controller.page_size = 16
        controller.storage_host_pool = host_pool
        controller.mem_pool_host = SimpleNamespace(entries=[host_pool])
        # A composite device pool need not describe its anchor's physical KV.
        controller.mem_pool_device = SimpleNamespace(dtype=torch.float16)
        return controller

    def _suffixes(self, controller):
        memory = SimpleNamespace(
            hicache_storage_key_scheme="unified",
            hicache_storage_head_group=4,
            hicache_storage_layer_partition=4,
        )
        parallel = SimpleNamespace(
            tp_rank=5,
            tp_size=8,
            attn_tp_rank=1,
            attn_tp_size=2,
            pp_rank=1,
            pp_size=2,
        )
        with (
            patch(
                "sglang.srt.managers.cache_controller.get_memory", return_value=memory
            ),
            patch(
                "sglang.srt.managers.cache_controller.get_parallel",
                return_value=parallel,
            ),
        ):
            return controller._build_unified_suffixes("org/model", attn_cp_size=1)

    def test_anchor_dtype_and_attention_topology_define_mha_keys(self):
        namespace = derive_namespace(
            model_id="org/model",
            dtype="float8_e4m3fn",
            page_size=16,
            rank_replicated=False,
            total_kv_heads=8,
            head_group=4,
            layer_partition=4,
        )
        self.assertEqual(
            self._suffixes(self._controller()),
            [f"{namespace_digest(namespace)}_L4-8_H1"],
        )

    def test_anchor_mla_pool_has_no_head_coordinate(self):
        namespace = derive_namespace(
            model_id="org/model",
            dtype="float8_e4m3fn",
            page_size=16,
            rank_replicated=True,
            total_kv_heads=0,
            head_group=0,
            layer_partition=4,
        )
        self.assertEqual(
            self._suffixes(self._controller(mla=True)),
            [f"{namespace_digest(namespace)}_L4-8"],
        )

    def test_incompatible_host_grids_are_rejected(self):
        for unsupported in ("sidecar", "missing_geometry", "packed_draft"):
            with self.subTest(unsupported=unsupported):
                controller = self._controller()
                if unsupported == "sidecar":
                    controller.mem_pool_host.entries.append(SimpleNamespace())
                elif unsupported == "missing_geometry":
                    del controller.storage_host_pool.page_unified_layout
                else:
                    controller.storage_host_pool.layer_num += 1
                with self.assertRaises(NotImplementedError):
                    self._suffixes(controller)


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
