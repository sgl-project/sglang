import threading
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.hicache_hook import (
    handle_hicache,
    handle_hicache_ratio_default,
)
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.hicache_key_scheme import derive_namespace, namespace_digest
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
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

    def test_page_unified_storage_requires_distinct_keys(self):
        with self.assertRaisesRegex(ValueError, "requires.*unified"):
            _resolve(
                hicache_storage_key_scheme="rank-suffix",
                hicache_mem_layout="page_unified",
            )
        args = _resolve(
            hicache_storage_key_scheme="rank-suffix",
            hicache_mem_layout="page_unified",
            hicache_storage_backend=None,
        )
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_unified")

    def test_partition_configs_require_the_unified_scheme(self):
        with self.assertRaisesRegex(ValueError, "require --hicache-storage-key-scheme"):
            _resolve(
                hicache_storage_key_scheme="rank-suffix",
                hicache_storage_head_partition=2,
            )

    def test_non_positive_partition_config_is_refused(self):
        for field in (
            "hicache_storage_head_partition",
            "hicache_storage_layer_partition",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "must be positive"),
            ):
                _resolve(**{field: 0})

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

    def test_unified_supports_full_page_attention_cp(self):
        args = _resolve(attn_cp_size=2, enable_prefill_cp=True)
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_unified")

    def test_unified_refuses_decode_context_parallelism(self):
        with self.assertRaisesRegex(NotImplementedError, "decode context parallelism"):
            _resolve(dcp_size=2)

    def test_unified_refuses_shared_decode_host_receive_pool(self):
        with self.assertRaisesRegex(ValueError, "layer_first"):
            _resolve(
                disaggregation_mode="decode",
                disaggregation_decode_host_receive_threshold=0.5,
            )


class TestControllerUnifiedKeys(CustomTestCase):
    def _controller(self, *, mla=False, grouped=True):
        device_pool = object.__new__(MLATokenToKVPool) if mla else SimpleNamespace()
        # FP8 variants share a uint8 storage view but require distinct keys.
        device_pool.dtype = torch.float8_e4m3fn
        device_pool.store_dtype = torch.uint8
        # Buffer-info host pools expose the physical dtype and no device_pool.
        host_pool = SimpleNamespace(
            dtype=torch.uint8,
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
        if grouped:
            anchor = SimpleNamespace(host_pool=host_pool, device_pool=device_pool)
            controller.mem_pool_host = SimpleNamespace(
                entries=[anchor], anchor_entry=anchor
            )
            # A composite device pool need not describe its anchor's physical KV.
            controller.mem_pool_device = SimpleNamespace(dtype=torch.float16)
        else:
            controller.mem_pool_host = host_pool
            controller.mem_pool_device = device_pool
        return controller

    def _suffixes(self, controller, *, attn_cp_size=1):
        memory = SimpleNamespace(
            hicache_storage_key_scheme="unified",
            hicache_storage_head_partition=4,
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
            return controller._build_unified_plan(
                "org/model", attn_cp_size=attn_cp_size
            ).suffixes

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

    def test_standalone_and_grouped_pools_share_logical_identity(self):
        for mla in (False, True):
            with self.subTest(mla=mla):
                self.assertEqual(
                    self._suffixes(self._controller(mla=mla, grouped=False)),
                    self._suffixes(self._controller(mla=mla)),
                )

    def test_full_page_cp_keeps_attention_tp_shard_keys(self):
        for mla in (False, True):
            with self.subTest(mla=mla):
                controller = self._controller(mla=mla)
                self.assertEqual(
                    self._suffixes(controller, attn_cp_size=4),
                    self._suffixes(controller, attn_cp_size=1),
                )

    def test_runtime_attach_refuses_dcp_pool(self):
        controller = self._controller(mla=True)
        controller.storage_host_pool.dcp_size = 2
        with self.assertRaisesRegex(NotImplementedError, "decode context parallelism"):
            self._suffixes(controller)

    def test_cp_writer_election_preserves_each_attention_tp_shard(self):
        # With attention DP, the storage TP rank is attention-group-local.
        # Every MHA head shard needs a writer, while MLA needs a single writer.
        expected_writers = {
            (False, False): [(0, 0), (1, 0), (0, 1), (1, 1)],
            (False, True): [(0, 0), (0, 1)],
            (True, False): [(0, 0), (1, 0)],
            (True, True): [(0, 0)],
        }
        for (unified, mla), expected in expected_writers.items():
            with self.subTest(unified=unified, mla=mla):
                writers = []
                for cp_rank in range(2):
                    for tp_rank in range(2):
                        controller = self._controller(mla=mla)
                        controller.enable_storage = False
                        controller.storage_host_pool.layout = (
                            "page_unified" if unified else "page_first"
                        )
                        controller.mem_pool_host.size = 256
                        controller.mem_pool_host.logical_size = 256
                        controller.host_memory_mode = "cache"
                        controller.storage_stop_event = threading.Event()
                        config = HiCacheStorageConfig(
                            tp_rank=tp_rank,
                            tp_size=2,
                            pp_rank=0,
                            pp_size=1,
                            attn_cp_rank=cp_rank,
                            attn_cp_size=2,
                            is_mla_model=mla,
                            enable_storage_metrics=False,
                            is_page_first_layout=not unified,
                            model_name="org/model",
                            unified_suffixes=["shared"] if unified else None,
                        )
                        self.assertEqual(
                            config.is_storage_writer, (tp_rank, cp_rank) in expected
                        )
                        memory = SimpleNamespace(
                            hicache_storage_key_scheme=(
                                "unified" if unified else "rank-suffix"
                            )
                        )
                        with (
                            patch(
                                "sglang.srt.managers.cache_controller.get_memory",
                                return_value=memory,
                            ),
                            patch(
                                "sglang.srt.managers.cache_controller.get_parallel",
                                return_value=SimpleNamespace(attn_dcp_size=1),
                            ),
                            patch.object(controller, "_stop_storage_threads"),
                            patch.object(controller, "_start_storage_threads"),
                            patch.object(
                                controller,
                                "_generate_storage_config",
                                return_value=config,
                            ),
                            patch.object(
                                controller, "_create_sync_groups", return_value=[]
                            ),
                            patch(
                                "sglang.srt.mem_cache.storage.StorageBackendFactory.create_backend",
                                return_value=MagicMock(),
                            ),
                        ):
                            controller.attach_storage_backend("file")
                        if not controller.backup_skip:
                            writers.append((tp_rank, cp_rank))
                self.assertEqual(writers, expected)

    def test_attention_dp_config_uses_local_tp_and_cp_ranks(self):
        controller = self._controller(mla=True)
        controller.mem_pool_device = controller.mem_pool_host.anchor_entry.device_pool
        controller.mem_pool_host.layout = "page_unified"
        controller.storage_host_pool.storage_format_tag = None
        controller.enable_storage_metrics = False
        parallel = SimpleNamespace(
            tp_rank=6,
            tp_size=8,
            attn_tp_rank=0,
            attn_tp_size=2,
            attn_cp_rank=1,
            attn_cp_size=2,
            attn_dp_rank=1,
            attn_dcp_size=1,
            attn_dcp_rank=0,
            pp_rank=0,
            pp_size=1,
        )
        with (
            patch(
                "sglang.srt.managers.cache_controller.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
                return_value=True,
            ),
            patch.object(controller, "get_attn_cp_rank_and_size", return_value=(1, 2)),
            patch.object(
                controller,
                "_build_unified_plan",
                return_value=SimpleNamespace(
                    suffixes=["shared"], layer_ranges=[(0, 4)], head_group_num=1
                ),
            ),
        ):
            config = controller._generate_storage_config("org/model")
        self.assertEqual((config.tp_rank, config.tp_size, config.dp_rank), (0, 2, 1))
        self.assertEqual((config.attn_cp_rank, config.attn_cp_size), (1, 2))
        self.assertTrue(config.is_mla_model)

    def test_config_pairs_grid_keys_with_physical_page_spans(self):
        # MHA key order alternates head groups, while each group's layers are
        # contiguous in memory. Include a short final PP layer chunk.
        cases = (
            (False, 4, [(0, 16384), (32768, 16384), (16384, 16384), (49152, 16384)]),
            (False, 3, [(0, 16384), (24576, 16384), (16384, 8192), (40960, 8192)]),
            (True, 4, [(0, 4096), (4096, 4096)]),
            (True, 3, [(0, 4096), (4096, 2048)]),
        )
        for mla, layer_num, expected_spans in cases:
            with self.subTest(mla=mla, layer_num=layer_num):
                controller = self._controller(mla=mla, grouped=False)
                host_pool = controller.storage_host_pool
                host_pool.layout = "page_unified"
                host_pool.storage_format_tag = None
                host_pool.layer_num = layer_num
                host_pool.end_layer = 4 + layer_num
                host_pool.head_group_num = 1 if mla else 2
                host_pool.page_unified_layout = PageUnifiedLayout(
                    page_size=16,
                    layer_num=layer_num,
                    head_num=1 if mla else 4,
                    head_group_num=1 if mla else 2,
                    head_dim=128,
                    itemsize=1,
                    is_mla=mla,
                )
                controller.enable_storage_metrics = False
                parallel = SimpleNamespace(
                    tp_rank=1,
                    tp_size=2,
                    attn_tp_rank=1,
                    attn_tp_size=2,
                    attn_dcp_size=1,
                    attn_dcp_rank=0,
                    pp_rank=1,
                    pp_size=2,
                )
                memory = SimpleNamespace(
                    hicache_storage_key_scheme="unified",
                    hicache_storage_head_partition=2,
                    hicache_storage_layer_partition=2,
                )
                with (
                    patch(
                        "sglang.srt.managers.cache_controller.get_memory",
                        return_value=memory,
                    ),
                    patch(
                        "sglang.srt.managers.cache_controller.get_parallel",
                        return_value=parallel,
                    ),
                    patch(
                        "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
                        return_value=False,
                    ),
                    patch.object(
                        controller, "get_attn_cp_rank_and_size", return_value=(0, 1)
                    ),
                ):
                    config = controller._generate_storage_config("org/model")

                namespace = derive_namespace(
                    model_id="org/model",
                    dtype="float8_e4m3fn",
                    page_size=16,
                    rank_replicated=mla,
                    total_kv_heads=0 if mla else 8,
                    head_group=0 if mla else 2,
                    layer_partition=2,
                )
                digest = namespace_digest(namespace)
                head_suffixes = ("",) if mla else ("_H2", "_H3")
                self.assertEqual(
                    config.unified_suffixes,
                    [
                        f"{digest}_{layers}{heads}"
                        for layers in ("L4-6", f"L6-{4 + layer_num}")
                        for heads in head_suffixes
                    ],
                )
                self.assertEqual(config.unified_chunk_spans, expected_spans)

    def test_hybrid_write_back_preserves_cpu_host_indices(self):
        controller = object.__new__(HybridCacheController)
        controller.io_backend = "kernel"
        controller.mem_pool_host = SimpleNamespace(
            layout="page_unified",
            supports_per_pool_backup_indices=True,
            anchor_entry=SimpleNamespace(
                host_pool=SimpleNamespace(can_use_write_back_jit=True)
            ),
        )
        operation = CacheOperation(torch.arange(16), torch.arange(16, 32), node_id=1)
        with patch.object(
            controller,
            "move_indices",
            side_effect=AssertionError("Staged write-back needs CPU host indices"),
        ):
            host_indices, device_indices, transfers = controller._move_write_operation(
                operation
            )
        self.assertIs(host_indices, operation.host_indices)
        self.assertIs(device_indices, operation.device_indices)
        self.assertIsNone(transfers)

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
        config = self._config(["ukv1-abc_L0-80_H1"])
        config.attn_cp_rank = 3
        config.attn_cp_size = 4
        HiCacheFile._init_config_suffix(backend, config)
        self.assertEqual(backend.config_suffix, "_ukv1-abc_L0-80_H1")
        self.assertEqual(backend.tp_config_suffix, backend.config_suffix)

    def test_file_round_trip_and_eviction_ownership_follow_writer_election(self):
        # Full-page CP replicas share files; MLA also shares across TP ranks.
        cases = (
            (False, 1, 0, True),
            (False, 1, 1, False),
            (True, 0, 0, True),
            (True, 1, 0, False),
            (True, 0, 1, False),
        )
        value = torch.arange(16, dtype=torch.uint8)
        for mla, tp_rank, cp_rank, owns_files in cases:
            with (
                self.subTest(mla=mla, tp_rank=tp_rank, cp_rank=cp_rank),
                TemporaryDirectory() as directory,
                envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(directory),
            ):
                owner_config = replace(
                    self._config(["ukv1-abc_L0-80" + ("" if mla else "_H1")]),
                    is_mla_model=mla,
                    tp_rank=0 if mla else tp_rank,
                    attn_cp_size=2,
                    extra_config={
                        "enable_metadata_cache": True,
                        "metadata_ttl": 60,
                        "max_size": "32",
                        "min_free_space": "0",
                        "eviction_ratio": 1.0,
                    },
                )
                writer = HiCacheFile(owner_config)
                self.assertTrue(writer.set("existing", value))

                # Initializing another rank scans shared files for metadata,
                # while its evictor must track only the elected writer's files.
                backend = HiCacheFile(
                    replace(owner_config, tp_rank=tp_rank, attn_cp_rank=cp_rank)
                )
                self.assertTrue(backend.exists("existing"))
                restored = torch.empty_like(value)
                self.assertIs(backend.get("existing", restored), restored)
                torch.testing.assert_close(restored, value)
                self.assertEqual(backend.set("new", value), owns_files)
                self.assertTrue(writer.exists("existing"))

    def test_rank_suffix_path_is_unchanged(self):
        backend = HiCacheFile.__new__(HiCacheFile)
        HiCacheFile._init_config_suffix(backend, self._config(None))
        self.assertEqual(backend.config_suffix, "_org-model_1_4")

    def test_multi_chunk_grid_requires_byte_spans(self):
        """Coordinates alone cannot identify a chunk's position in a page."""
        backend = HiCacheFile.__new__(HiCacheFile)
        with self.assertRaises(ValueError):
            HiCacheFile._init_config_suffix(backend, self._config(["a", "b"]))


if __name__ == "__main__":
    unittest.main()
