"""Legacy HiCache MLA keys keep a writer on every context-parallel rank.

Issue 51608: ``--tp-size 8 --attn-cp-size 8`` stores the global TP rank, which
is the CP rank, in ``HiCacheStorageConfig.tp_rank``. Electing the MLA writer
with ``tp_rank == dcp_rank`` then lets only rank 0 write ``_cp{i}_8`` files.
Prefetch takes the minimum hit length across CP ranks, so the hit stays 0.

Pure CPU: no process group. Parallel state is overridden the same way as the
other HiCache storage-config unit tests.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.kv.memory.views import MLAKVView
from sglang.srt.mem_cache.storage.fast_file.fast_file_store import HiCacheFastFile
from sglang.test.test_utils import CustomTestCase


class _MlaDevice(MLAKVView):
    """MLA view stand-in. ``_generate_storage_config`` only checks the type."""

    def __init__(self):
        self.dtype = torch.float16


def _attention_coords(tp_rank: int, attn_tp_size: int, attn_cp_size: int):
    """(attn_tp_rank, attn_cp_rank) for one DP-attention-off global rank.

    Layout is ``(dp, cp, tp)`` with TP changing fastest and attention DP off:
    ``tp_rank = attn_cp_rank * attn_tp_size + attn_tp_rank``.
    """
    return tp_rank % attn_tp_size, (tp_rank // attn_tp_size) % attn_cp_size


def _generate_mla_config(
    *,
    tp_rank: int,
    tp_size: int,
    attn_tp_rank: int,
    attn_tp_size: int,
    attn_cp_rank: int,
    attn_cp_size: int,
    dcp_size: int = 1,
    dcp_rank: int = 0,
):
    """Build one config the way a rank does, with parallel state overridden."""
    controller = HiCacheController.__new__(HiCacheController)
    controller.mem_pool_device = _MlaDevice()
    controller.mem_pool_host = SimpleNamespace(
        layout="page_first", storage_format_tag=None
    )
    controller.storage_host_pool = controller.mem_pool_host
    controller.enable_storage_metrics = False
    controller.page_size = 16
    controller.get_attn_cp_rank_and_size = lambda: (attn_cp_rank, attn_cp_size)
    parallel = SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_cp_rank=attn_cp_rank,
        attn_cp_size=attn_cp_size,
        attn_dp_rank=0,
        attn_dcp_size=dcp_size,
        attn_dcp_rank=dcp_rank,
        pp_rank=0,
        pp_size=1,
    )
    with (
        patch(
            "sglang.srt.managers.cache_controller.get_memory",
            return_value=SimpleNamespace(hicache_storage_key_scheme="rank-suffix"),
        ),
        patch(
            "sglang.srt.managers.cache_controller.get_parallel",
            return_value=parallel,
        ),
        patch(
            "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
            return_value=False,
        ),
    ):
        return controller._generate_storage_config("mla-cp")


def _hand_config(**overrides) -> HiCacheStorageConfig:
    fields = dict(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="mla-cp",
    )
    fields.update(overrides)
    return HiCacheStorageConfig(**fields)


class TestHiCacheCpMlaWriter(CustomTestCase):
    def test_issue_51608_every_cp_rank_writes_its_own_suffix(self):
        """tp 8 / attn CP 8 / attn TP 1: attn_tp_rank is 0 on every GPU."""
        suffixes = []
        for tp_rank in range(8):
            attn_tp_rank, attn_cp_rank = _attention_coords(tp_rank, 1, 8)
            self.assertEqual((attn_tp_rank, attn_cp_rank), (0, tp_rank))
            config = _generate_mla_config(
                tp_rank=tp_rank,
                tp_size=8,
                attn_tp_rank=attn_tp_rank,
                attn_tp_size=1,
                attn_cp_rank=attn_cp_rank,
                attn_cp_size=8,
            )
            self.assertTrue(config.is_mla_model)
            self.assertTrue(config.is_rank_replicated)
            self.assertIsNone(config.unified_suffixes)
            self.assertEqual(config.dcp_size, 1)
            self.assertEqual(config.tp_rank, tp_rank)
            self.assertEqual(config.attn_tp_rank, 0)
            self.assertEqual(config.attn_cp_rank, tp_rank)
            self.assertTrue(config.is_storage_writer)
            suffix = _suffix_of(config)
            self.assertIn(f"_cp{tp_rank}_8", suffix)
            suffixes.append(suffix)
        self.assertEqual(len(set(suffixes)), 8)

    def test_wider_cp_group_writes_only_attn_tp_rank_zero(self):
        """tp 4 / attn CP 2 / attn TP 2: writers are global ranks 0 and 2."""
        writers = []
        suffixes = {}
        for tp_rank in range(4):
            attn_tp_rank, attn_cp_rank = _attention_coords(tp_rank, 2, 2)
            config = _generate_mla_config(
                tp_rank=tp_rank,
                tp_size=4,
                attn_tp_rank=attn_tp_rank,
                attn_tp_size=2,
                attn_cp_rank=attn_cp_rank,
                attn_cp_size=2,
            )
            self.assertEqual(config.tp_rank, tp_rank)
            self.assertEqual(config.attn_tp_rank, attn_tp_rank)
            self.assertEqual(config.attn_cp_rank, attn_cp_rank)
            suffix = _suffix_of(config)
            self.assertIn(f"_cp{attn_cp_rank}_2", suffix)
            suffixes[tp_rank] = suffix
            if config.is_storage_writer:
                writers.append(tp_rank)
        self.assertEqual(writers, [0, 2])
        self.assertNotIn(1, writers)
        self.assertNotIn(3, writers)
        self.assertEqual(suffixes[0], suffixes[1])
        self.assertEqual(suffixes[2], suffixes[3])
        self.assertNotEqual(suffixes[0], suffixes[2])

    def test_mla_without_cp_still_writes_only_on_rank_zero(self):
        for tp_rank in range(4):
            config = _generate_mla_config(
                tp_rank=tp_rank,
                tp_size=4,
                attn_tp_rank=tp_rank,
                attn_tp_size=4,
                attn_cp_rank=0,
                attn_cp_size=1,
            )
            self.assertEqual(config.is_storage_writer, tp_rank == 0)
        # attn_tp_rank 0 on a non-zero global rank must not widen the writer
        # set when the legacy suffix has no CP component.
        for tp_rank in range(4):
            config = _hand_config(
                tp_rank=tp_rank,
                tp_size=4,
                attn_cp_rank=0,
                attn_cp_size=1,
                attn_tp_rank=0,
            )
            self.assertEqual(config.is_storage_writer, tp_rank == 0)

    def test_unset_attn_tp_rank_keeps_the_global_tp_formula(self):
        for cp_rank in range(2):
            for tp_rank in range(2):
                config = _hand_config(
                    tp_rank=tp_rank,
                    tp_size=2,
                    attn_cp_rank=cp_rank,
                    attn_cp_size=2,
                )
                self.assertIsNone(config.attn_tp_rank)
                self.assertEqual(config.is_storage_writer, tp_rank == 0)

    def test_unified_suffixes_still_write_only_on_cp_rank_zero(self):
        for cp_rank in range(8):
            config = _hand_config(
                tp_rank=cp_rank,
                tp_size=8,
                attn_cp_rank=cp_rank,
                attn_cp_size=8,
                attn_tp_rank=0,
                unified_suffixes=["shared"],
            )
            self.assertEqual(config.is_storage_writer, cp_rank == 0)
        # CP rank 0 still uses the stored tp_rank, not attn_tp_rank.
        self.assertFalse(
            _hand_config(
                tp_rank=1,
                tp_size=8,
                attn_cp_rank=0,
                attn_cp_size=8,
                attn_tp_rank=0,
                unified_suffixes=["shared"],
            ).is_storage_writer
        )
        for tp_rank in range(2):
            self.assertTrue(
                _hand_config(
                    tp_rank=tp_rank,
                    tp_size=2,
                    attn_cp_rank=0,
                    attn_cp_size=2,
                    attn_tp_rank=0,
                    is_mla_model=False,
                    unified_suffixes=["shared"],
                ).is_storage_writer
            )
            self.assertFalse(
                _hand_config(
                    tp_rank=tp_rank,
                    tp_size=2,
                    attn_cp_rank=1,
                    attn_cp_size=2,
                    attn_tp_rank=0,
                    is_mla_model=False,
                    unified_suffixes=["shared"],
                ).is_storage_writer
            )

    def test_dcp_election_ignores_attn_tp_rank(self):
        # Today's formula: MLA writes iff tp_rank == dcp_rank. attn_tp_rank 0
        # must not make a later DCP group a writer, and a non-zero
        # attn_tp_rank must not silence the first DCP group.
        cases = (
            (0, 0, 0, True),
            (1, 1, 1, True),
            (2, 0, 0, False),
            (3, 1, 1, False),
            (0, 0, 1, True),
            (2, 0, 1, False),
        )
        for tp_rank, dcp_rank, attn_tp_rank, expected in cases:
            config = _hand_config(
                tp_rank=tp_rank,
                tp_size=4,
                attn_cp_rank=tp_rank // 2,
                attn_cp_size=2,
                attn_tp_rank=attn_tp_rank,
                dcp_size=2,
                dcp_rank=dcp_rank,
                logical_page_size=4,
            )
            self.assertEqual(config.is_storage_writer, expected)

    def test_non_mla_legacy_cp_writes_on_every_rank(self):
        for tp_rank, cp_rank in ((0, 0), (1, 0), (0, 1), (1, 1)):
            config = _hand_config(
                tp_rank=tp_rank,
                tp_size=2,
                attn_cp_rank=cp_rank,
                attn_cp_size=2,
                attn_tp_rank=0,
                is_mla_model=False,
            )
            self.assertTrue(config.is_storage_writer)

    def test_fast_file_round_trip_min_hit_is_one(self):
        """Each CP rank writes one page, then reads its own key.

        The scheduler's storage hit is the minimum of those hit lengths.
        """
        directory = tempfile.mkdtemp(prefix="hicache_cp_mla_writer_")
        backends = []
        try:
            with envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(None):
                hits = []
                suffixes = []
                for tp_rank in range(8):
                    config = _generate_mla_config(
                        tp_rank=tp_rank,
                        tp_size=8,
                        attn_tp_rank=0,
                        attn_tp_size=1,
                        attn_cp_rank=tp_rank,
                        attn_cp_size=8,
                    )
                    config.extra_config = {
                        "max_size": 0,
                        "min_free_space": 0,
                        "evict_high_watermark": 1.0,
                    }
                    backend = HiCacheFastFile(config, file_path=directory)
                    backends.append(backend)
                    suffix = backend.config_suffix
                    self.assertIn(f"_cp{tp_rank}_8", suffix)
                    suffixes.append(suffix)
                    page = torch.full((64,), tp_rank + 1, dtype=torch.uint8)
                    if config.is_storage_writer:
                        self.assertTrue(backend.set("page", page))
                    hits.append(backend.batch_exists(["page"]))
                    if config.is_storage_writer:
                        loaded = torch.empty(64, dtype=torch.uint8)
                        self.assertIsNotNone(backend.get("page", loaded))
                        torch.testing.assert_close(loaded, page)
                self.assertEqual(len(set(suffixes)), 8)
                self.assertEqual(min(hits), 1)
                self.assertEqual(hits, [1] * 8)
        finally:
            for backend in backends:
                backend.close()
            shutil.rmtree(directory, ignore_errors=True)


def _suffix_of(config: HiCacheStorageConfig) -> str:
    """The legacy file suffix, without opening a storage directory."""
    probe = HiCacheFile.__new__(HiCacheFile)
    probe._init_config_suffix(config)
    return probe.config_suffix


if __name__ == "__main__":
    import unittest

    unittest.main()
