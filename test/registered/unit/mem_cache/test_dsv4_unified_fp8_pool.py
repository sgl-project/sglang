import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_FP8_NOPE_ROW_BYTES,
    DSV4_FP8_QUANT_TILE,
    DeepSeekV4UnifiedKVPool,
    dsv4_unified_row_bytes,
    resolve_unified_kv_fp8,
)
from sglang.srt.mem_cache.kv_cache_configurator import unified_fp8_for_dsv4_pool
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# DeepSeek-V4-Pro geometry.
NOPE_DIM = 448
ROPE_DIM = 64


class _StubMemorySaver:
    def region(self, _tag):
        return contextlib.nullcontext()


class TestDSV4UnifiedRowBytes(CustomTestCase):
    """Row width drives both `bytes_per_full_token` and `_fixed_swa_bytes`, so the
    capacity claim for the fp8 pool is only as good as this arithmetic."""

    def test_bf16_row_is_the_whole_latent(self):
        self.assertEqual(
            dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=False),
            (NOPE_DIM + ROPE_DIM) * 2,
        )

    def test_fp8_row_is_padded_nope_plus_bf16_rope(self):
        self.assertEqual(
            dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=True),
            DSV4_FP8_NOPE_ROW_BYTES + ROPE_DIM * 2,
        )

    def test_fp8_saves_exactly_three_eighths(self):
        """0.625x is where the >=1.40x capacity target comes from; the remaining
        dilution is the fixed SWA/c4-state bias, not the row."""
        bf16 = dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=False)
        fp8 = dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=True)
        self.assertEqual((bf16, fp8), (1024, 640))
        self.assertAlmostEqual(fp8 / bf16, 0.625)

    def test_scales_and_latent_fit_the_asm_stride(self):
        """7 tiles written twice = 14 B; 448 + 14 leaves 50 B the reader never
        touches. If a future head_dim broke this the pack would silently overlap."""
        num_tiles = NOPE_DIM // DSV4_FP8_QUANT_TILE
        self.assertEqual(num_tiles, 7)
        self.assertLessEqual(NOPE_DIM + 2 * num_tiles, DSV4_FP8_NOPE_ROW_BYTES)

    def test_oversized_latent_is_rejected(self):
        # ValueError, not assert: sizing has to keep checking under python -O
        with self.assertRaises(ValueError):
            dsv4_unified_row_bytes(DSV4_FP8_NOPE_ROW_BYTES, ROPE_DIM, fp8=True)


class TestDSV4UnifiedFp8PoolAllocation(CustomTestCase):
    """The sizing formula and the allocation are two separate code paths; this pins
    them to the same row width so a change to one cannot silently outrun the other."""

    STAGE_RATIOS = [4, 128]
    NUM_SLOTS = 3
    NUM_BLOCKS = 5
    PAGE_SIZE = 256
    SWA_RING = 8

    def _pool(self, fp8):
        return DeepSeekV4UnifiedKVPool(
            stage_ratios=self.STAGE_RATIOS,
            num_slots=self.NUM_SLOTS,
            num_blocks=self.NUM_BLOCKS,
            page_size=self.PAGE_SIZE,
            qk_nope_head_dim=NOPE_DIM,
            qk_rope_head_dim=ROPE_DIM,
            device="cpu",
            memory_saver_adapter=_StubMemorySaver(),
            custom_mem_pool=None,
            swa_ring_size=self.SWA_RING,
            fp8=fp8,
        )

    def test_bf16_pool_is_unchanged(self):
        """fp8 defaults off, so the bf16 arm must keep one pool of bf16 latents."""
        pool = self._pool(fp8=False)
        for buf, rope in zip(pool.kv_buffer, pool.kv_buffer_rope):
            self.assertEqual(buf.dtype, torch.bfloat16)
            self.assertEqual(buf.shape[1], NOPE_DIM + ROPE_DIM)
            self.assertIsNone(rope)

    def test_fp8_pool_row_counts_match_across_both_pools(self):
        """A row index addresses the SWA ring and the compressed region in both
        pools, so the two must have identical row counts."""
        pool = self._pool(fp8=True)
        for buf, rope in zip(pool.kv_buffer, pool.kv_buffer_rope):
            self.assertEqual(buf.dtype, torch.float8_e4m3fn)
            self.assertEqual(rope.dtype, torch.bfloat16)
            self.assertEqual(buf.shape[0], rope.shape[0])
            self.assertEqual(buf.shape[1], DSV4_FP8_NOPE_ROW_BYTES)
            self.assertEqual(rope.shape[1], ROPE_DIM)

    def test_fp8_pool_bytes_match_the_sizing_row_width(self):
        bf16, fp8 = self._pool(fp8=False), self._pool(fp8=True)
        for layer, buf in enumerate(bf16.kv_buffer):
            rows = buf.shape[0]
            self.assertEqual(fp8.kv_buffer[layer].shape[0], rows)
            self.assertEqual(
                buf.nbytes,
                rows * dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=False),
            )
            self.assertEqual(
                fp8.kv_buffer[layer].nbytes + fp8.kv_buffer_rope[layer].nbytes,
                rows * dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=True),
            )

    def test_rope_accessor_rejects_the_bf16_pool(self):
        with self.assertRaises(AssertionError):
            self._pool(fp8=False).get_unified_kv_rope(0)


class TestResolveUnifiedKvFp8(CustomTestCase):
    def test_override_false_wins_over_env(self):
        env_mod = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
        with patch(f"{env_mod}.is_unified_kv_fp8", return_value=True):
            self.assertFalse(resolve_unified_kv_fp8(False))
            self.assertTrue(resolve_unified_kv_fp8(True))
            self.assertTrue(resolve_unified_kv_fp8(None))

    def test_none_follows_env_off(self):
        env_mod = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
        with patch(f"{env_mod}.is_unified_kv_fp8", return_value=False):
            self.assertFalse(resolve_unified_kv_fp8(None))
            self.assertFalse(resolve_unified_kv_fp8(False))
            self.assertTrue(resolve_unified_kv_fp8(True))


class TestDsv4PoolFp8Gate(CustomTestCase):
    _ENV = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"

    def _layout(self, *, is_draft_worker, algo, env_on=True):
        with patch(f"{self._ENV}.is_unified_kv_fp8", return_value=env_on):
            return unified_fp8_for_dsv4_pool(
                is_draft_worker=is_draft_worker, spec_algorithm=algo
            )

    def test_dspark_draft_stays_bf16_when_env_on(self):
        self.assertFalse(
            self._layout(
                is_draft_worker=True, algo=SpeculativeAlgorithm.DSPARK, env_on=True
            )
        )

    def test_eagle_draft_stays_two_pool_when_env_on(self):
        self.assertTrue(
            self._layout(
                is_draft_worker=True, algo=SpeculativeAlgorithm.EAGLE, env_on=True
            )
        )

    def test_target_stays_two_pool_under_dspark_and_eagle(self):
        for algo in (SpeculativeAlgorithm.DSPARK, SpeculativeAlgorithm.EAGLE):
            with self.subTest(algo=algo):
                self.assertTrue(
                    self._layout(is_draft_worker=False, algo=algo, env_on=True)
                )

    def test_env_off_is_bf16_for_every_worker(self):
        for draft, algo in (
            (True, SpeculativeAlgorithm.DSPARK),
            (True, SpeculativeAlgorithm.EAGLE),
            (False, SpeculativeAlgorithm.DSPARK),
        ):
            with self.subTest(draft=draft, algo=algo):
                self.assertFalse(
                    self._layout(is_draft_worker=draft, algo=algo, env_on=False)
                )


class TestBuildDsv4KvPoolPassesGate(CustomTestCase):
    class _RecPool:
        last = None

        def __init__(self, **kwargs):
            type(self).last = kwargs
            self._unified_kv = False
            self._unified_kv_fp8 = kwargs.get("unified_fp8")

    def _kvc(self, *, is_draft_worker, spec_algorithm):
        from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

        kvc = object.__new__(KVCacheConfigurator)
        kvc.is_draft_worker = is_draft_worker
        kvc.spec_algorithm = spec_algorithm
        kvc.layer_info = SimpleNamespace(
            num_effective_layers=1, start_layer=0, end_layer=1
        )
        kvc.model_config = SimpleNamespace(
            compress_ratios=[0],
            window_size=256,
            qk_nope_head_dim=NOPE_DIM,
            qk_rope_head_dim=ROPE_DIM,
            index_head_dim=128,
        )
        kvc.kv_cache_dtype = torch.bfloat16
        kvc.device = "cpu"
        return kvc

    def _build(self, *, is_draft_worker, spec_algorithm):
        kvc = self._kvc(is_draft_worker=is_draft_worker, spec_algorithm=spec_algorithm)
        sched = MagicMock()
        sched.page_size = 256
        exec_cfg = MagicMock()
        exec_cfg.features.enable_memory_saver = False
        mem = MagicMock()
        mem.enable_hisparse = False
        par = MagicMock()
        par.attn_dcp_size = 1
        req = SimpleNamespace(req_to_token=torch.zeros(4, 1))
        rec = self._RecPool
        rec.last = None
        env = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.DeepSeekV4TokenToKVPool",
                rec,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_schedule",
                return_value=sched,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
                return_value=exec_cfg,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_memory",
                return_value=mem,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.get_parallel",
                return_value=par,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_configurator.max_speculative_num_draft_tokens",
                return_value=0,
            ),
            patch(f"{env}.is_unified_kv_fp8", return_value=True),
        ):
            kvc._build_dsv4_kv_pool(
                max_running_requests=2,
                swa_max_total_num_tokens=256,
                c4_max_total_num_tokens=0,
                c128_max_total_num_tokens=1,
                c4_state_pool_size=0,
                c128_state_pool_size=0,
                c4_state_dtype=None,
                c128_state_dtype=None,
                req_to_token_pool=req,
            )
        return rec.last

    def test_dspark_draft_ctor_gets_unified_fp8_false(self):
        kw = self._build(
            is_draft_worker=True, spec_algorithm=SpeculativeAlgorithm.DSPARK
        )
        self.assertIsNotNone(kw)
        self.assertFalse(kw["unified_fp8"])

    def test_eagle_draft_ctor_gets_unified_fp8_true(self):
        kw = self._build(
            is_draft_worker=True, spec_algorithm=SpeculativeAlgorithm.EAGLE
        )
        self.assertIsNotNone(kw)
        self.assertTrue(kw["unified_fp8"])

    def test_target_ctor_gets_unified_fp8_true(self):
        kw = self._build(
            is_draft_worker=False, spec_algorithm=SpeculativeAlgorithm.DSPARK
        )
        self.assertIsNotNone(kw)
        self.assertTrue(kw["unified_fp8"])


class TestUnifiedKvPoolFollowsCtorFp8(CustomTestCase):
    def _pool(self, fp8):
        return DeepSeekV4UnifiedKVPool(
            stage_ratios=[0],
            num_slots=2,
            num_blocks=1,
            page_size=256,
            qk_nope_head_dim=NOPE_DIM,
            qk_rope_head_dim=ROPE_DIM,
            device="cpu",
            memory_saver_adapter=_StubMemorySaver(),
            custom_mem_pool=None,
            swa_ring_size=8,
            fp8=fp8,
        )

    def test_dspark_draft_layout_has_no_rope_pool(self):
        env = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
        with patch(f"{env}.is_unified_kv_fp8", return_value=True):
            fp8 = unified_fp8_for_dsv4_pool(
                is_draft_worker=True, spec_algorithm=SpeculativeAlgorithm.DSPARK
            )
        self.assertFalse(fp8)
        pool = self._pool(fp8)
        buf = pool.kv_buffer[0]
        self.assertEqual(buf.dtype, torch.bfloat16)
        self.assertEqual(buf.shape[1], NOPE_DIM + ROPE_DIM)
        self.assertIsNone(pool.kv_buffer_rope[0])

    def test_eagle_draft_layout_has_rope_pool(self):
        env = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"
        with patch(f"{env}.is_unified_kv_fp8", return_value=True):
            fp8 = unified_fp8_for_dsv4_pool(
                is_draft_worker=True, spec_algorithm=SpeculativeAlgorithm.EAGLE
            )
        self.assertTrue(fp8)
        pool = self._pool(fp8)
        buf, rope = pool.kv_buffer[0], pool.kv_buffer_rope[0]
        self.assertEqual(buf.dtype, torch.float8_e4m3fn)
        self.assertEqual(rope.dtype, torch.bfloat16)
        self.assertEqual(rope.shape[1], ROPE_DIM)


if __name__ == "__main__":
    unittest.main()
