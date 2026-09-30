"""CPU-only tests for the Ascend DSA KV cache capability and its consumers.

Covers the capability accept/reject matrix, the MLA KV cache dim selector, the
NPU MLA pool's PD layer ids and indexer dtypes, and the indexer sizer. NPU
extension modules are stubbed; no operator runs.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

sys.modules.setdefault("sgl_kernel_npu", types.ModuleType("sgl_kernel_npu"))
sys.modules.setdefault("torch_npu", MagicMock(name="torch_npu"))
import sglang.srt.utils as _utils
from sglang.srt.utils import common as _common_utils

_real_is_npu = _common_utils.is_npu
_real_utils_is_npu = _utils.is_npu
_common_utils.is_npu = lambda: False
_utils.is_npu = _common_utils.is_npu
try:
    from sglang.srt.hardware_backend.npu import kv_capability, memory_pool_npu
    from sglang.srt.hardware_backend.npu.kv_capability import resolve_kv_capability
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.mem_cache import kv_cache_configurator, kv_cache_dtype
    from sglang.srt.mem_cache.kv_cache_configurator import (
        calculate_mla_kv_cache_dim,
        is_packed_mla_kv_cache,
    )
    from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
    from sglang.srt.model_executor import pool_configurator
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
    from sglang.srt.runtime_context import get_context
finally:
    _common_utils.is_npu = _real_is_npu
    _utils.is_npu = _real_utils_is_npu

A2_SOC = "Ascend910B3"
FP8 = torch.float8_e4m3fn


def _patch_npu(*, arch35: bool):
    """Patch the SoC probes behind resolve_npu_kv_capability."""
    return (
        patch(
            "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
            return_value=arch35,
        ),
        patch.object(kv_capability, "_get_npu_soc_name", return_value=A2_SOC),
    )


class _NpuPatchedTestCase(unittest.TestCase):
    def _use_npu(self, *, arch35: bool):
        for p in _patch_npu(arch35=arch35):
            p.start()
            self.addCleanup(p.stop)


class TestResolveKvCapability(unittest.TestCase):
    def test_fp8_on_arch35_is_packed_with_quantized_indexer(self):
        cap = resolve_kv_capability(kv_cache_dtype=FP8, is_arch35=True, soc_name="x")
        self.assertTrue(cap.supported)
        self.assertTrue(cap.main_kv_packed)
        self.assertEqual(cap.main_kv_scale_dtype, torch.float32)
        self.assertTrue(cap.indexer_quant)
        self.assertEqual(cap.indexer_kv_dtype, FP8)
        self.assertEqual(cap.indexer_scale_dtype, torch.float32)

    def test_fp8_on_non_arch35_is_rejected_with_soc_and_alternatives(self):
        cap = resolve_kv_capability(
            kv_cache_dtype=FP8, is_arch35=False, soc_name=A2_SOC
        )
        self.assertFalse(cap.supported)
        self.assertFalse(cap.main_kv_packed)
        self.assertIn(A2_SOC, cap.unsupported_reason)
        self.assertIn("--kv-cache-dtype fp8_e4m3", cap.unsupported_reason)
        self.assertIn("arch35", cap.unsupported_reason)
        self.assertIn("bf16", cap.unsupported_reason)

    def test_rejection_names_the_requested_value(self):
        cap = resolve_kv_capability(
            kv_cache_dtype=FP8, is_arch35=False, soc_name=A2_SOC, requested="mxfp8"
        )
        self.assertIn("--kv-cache-dtype mxfp8", cap.unsupported_reason)

    def test_unquantized_and_e5m2_are_unpacked_on_every_soc(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float8_e5m2):
            for is_arch35 in (True, False):
                with self.subTest(dtype=dtype, is_arch35=is_arch35):
                    cap = resolve_kv_capability(
                        kv_cache_dtype=dtype, is_arch35=is_arch35, soc_name="x"
                    )
                    self.assertTrue(cap.supported)
                    self.assertFalse(cap.main_kv_packed)
                    self.assertIsNone(cap.main_kv_scale_dtype)
                    self.assertFalse(cap.indexer_quant)
                    self.assertEqual(cap.indexer_kv_dtype, dtype)
                    self.assertIsNone(cap.indexer_scale_dtype)

    def test_capability_is_frozen(self):
        cap = resolve_kv_capability(kv_cache_dtype=FP8, is_arch35=True, soc_name="x")
        with self.assertRaises(Exception):
            cap.main_kv_packed = False


class TestConfigureKvCacheDtypeOnNpu(_NpuPatchedTestCase):
    def _configure(self, server_args_kv_cache_dtype, *, model=None, is_npu=True):
        platform = SimpleNamespace(is_npu=lambda: is_npu, is_cpu=lambda: False)
        with patch.object(kv_cache_dtype, "current_platform", platform):
            return configure_kv_cache_dtype(
                server_args_kv_cache_dtype=server_args_kv_cache_dtype,
                model=model,
                model_dtype=torch.bfloat16,
                is_draft_worker=False,
                is_dflash=False,
                speculative_draft_attention_backend="fa3",
            )

    def test_fp8_on_a2_is_a_startup_error(self):
        self._use_npu(arch35=False)
        for requested in ("fp8_e4m3", "mxfp8"):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError) as ctx:
                    self._configure(requested)
                message = str(ctx.exception)
                self.assertIn(f"--kv-cache-dtype {requested}", message)
                self.assertIn(A2_SOC, message)
                self.assertIn("bf16", message)

    def test_auto_with_fp8_checkpoint_on_a2_is_a_startup_error(self):
        self._use_npu(arch35=False)
        model = SimpleNamespace(quant_config=SimpleNamespace(kv_cache_quant_algo="FP8"))
        with self.assertRaisesRegex(ValueError, "--kv-cache-dtype auto"):
            self._configure("auto", model=model)

    def test_bf16_and_auto_are_accepted_on_a2(self):
        self._use_npu(arch35=False)
        self.assertEqual(self._configure("bf16"), (None, torch.bfloat16))
        self.assertEqual(self._configure("auto"), (None, torch.bfloat16))

    def test_fp8_is_accepted_on_arch35(self):
        self._use_npu(arch35=True)
        self.assertEqual(self._configure("fp8_e4m3"), (None, FP8))

    def test_non_npu_platforms_are_not_checked(self):
        self._use_npu(arch35=False)
        self.assertEqual(self._configure("fp8_e4m3", is_npu=False), (None, FP8))


class TestMlaKvCacheDim(_NpuPatchedTestCase):
    def setUp(self):
        kernel = SimpleNamespace(
            dsa_prefill_backend="flashmla_kv", dsa_decode_backend="flashmla_kv"
        )
        for p in (
            patch.object(kv_cache_configurator, "is_deepseek_dsa", return_value=True),
            patch.object(
                kv_cache_configurator,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch.object(
                kv_cache_configurator,
                "get_exec",
                return_value=SimpleNamespace(kernel=kernel),
            ),
            patch.object(kv_cache_configurator, "_is_hip", False),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.model_config = SimpleNamespace(
            hf_config=None, kv_lora_rank=512, qk_rope_head_dim=64
        )

    def _dim(self, dtype, packed):
        return calculate_mla_kv_cache_dim(
            model_config=self.model_config, kv_cache_dtype=dtype, packed=packed
        )

    def test_packed_selector_is_caller_owned(self):
        self.assertEqual(self._dim(FP8, packed=True), 512 + 512 // 128 * 4 + 64 * 2)
        self.assertEqual(self._dim(FP8, packed=False), 576)
        self.assertEqual(self._dim(torch.bfloat16, packed=False), 576)

    def test_packed_requires_a_one_byte_dtype(self):
        with self.assertRaises(AssertionError):
            self._dim(torch.bfloat16, packed=True)

    def test_is_packed_off_npu_keeps_the_dtype_rule(self):
        with patch.object(kv_cache_configurator, "_is_npu", False):
            self.assertTrue(is_packed_mla_kv_cache(FP8))
            self.assertFalse(is_packed_mla_kv_cache(torch.float8_e5m2))
            self.assertFalse(is_packed_mla_kv_cache(torch.bfloat16))

    def test_is_packed_on_npu_follows_the_capability(self):
        with patch.object(kv_cache_configurator, "_is_npu", True):
            self._use_npu(arch35=True)
            self.assertTrue(is_packed_mla_kv_cache(FP8))
            self.assertFalse(is_packed_mla_kv_cache(torch.float8_e5m2))
            self.assertFalse(is_packed_mla_kv_cache(torch.bfloat16))

    def test_is_packed_on_a2_is_false_for_fp8(self):
        with patch.object(kv_cache_configurator, "_is_npu", True):
            self._use_npu(arch35=False)
            self.assertFalse(is_packed_mla_kv_cache(FP8))


def _build_pool(
    dtype,
    *,
    arch35=True,
    start_layer=0,
    layer_num=2,
    indexer_layer_ids=None,
    dcp_size=1,
):
    capability = resolve_kv_capability(
        kv_cache_dtype=dtype, is_arch35=arch35, soc_name="x"
    )
    parallel = SimpleNamespace(attn_dcp_size=dcp_size, attn_dcp_rank=0)
    with patch.object(memory_pool_npu, "get_parallel", return_value=parallel):
        return NPUMLATokenToKVPool(
            16,
            page_size=4,
            dtype=dtype,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            layer_num=layer_num,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=128,
            index_size=16 * dcp_size,
            start_layer=start_layer,
            end_layer=start_layer + layer_num,
            indexer_layer_ids=indexer_layer_ids,
            kv_cache_dim=656 if capability.main_kv_packed else None,
            kv_capability=capability,
        )


class TestNpuMlaPoolLayout(unittest.TestCase):
    def assertLayerIdsAlign(self, pool, expected):
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        layer_ids = pool.get_kv_layer_ids()
        self.assertEqual(layer_ids, expected)
        self.assertEqual(len(layer_ids), len(ptrs))
        self.assertEqual(len(layer_ids), len(item_lens))
        self.assertEqual(len(layer_ids), len(pool.get_dcp_remote_decode_layout()))

    def test_packed_pool_entries_are_k_index_scale(self):
        pool = _build_pool(FP8)
        self.assertTrue(pool.dsa_kv_cache_store_fp8)
        self.assertEqual(pool.v_buffer.shape[-1], 0)
        self.assertLayerIdsAlign(pool, [0, 1] * 3)

    def test_unpacked_pool_entries_are_k_v_index(self):
        pool = _build_pool(torch.bfloat16, start_layer=4)
        self.assertFalse(pool.dsa_kv_cache_store_fp8)
        self.assertIsNone(pool.index_k_scale_buffer)
        self.assertLayerIdsAlign(pool, [4, 5] * 3)

    def test_compact_indexer_uses_indexer_layer_ids(self):
        packed = _build_pool(FP8, start_layer=4, indexer_layer_ids=(5,))
        self.assertLayerIdsAlign(packed, [4, 5, 5, 5])
        unpacked = _build_pool(torch.bfloat16, start_layer=4, indexer_layer_ids=(5,))
        self.assertLayerIdsAlign(unpacked, [4, 5, 4, 5, 5])

    def test_dcp_scales_only_indexer_item_lens(self):
        pool = _build_pool(FP8, dcp_size=2)
        self.assertLayerIdsAlign(pool, [0, 1] * 3)
        _, _, item_lens = pool.get_contiguous_buf_infos()
        local = pool.k_buffer[0][0].nbytes
        index = pool.index_k_buffer[0][0].nbytes * 2
        scale = pool.index_k_scale_buffer[0][0].nbytes * 2
        self.assertEqual(item_lens, [local] * 2 + [index] * 2 + [scale] * 2)

    def test_indexer_dtypes_live_on_the_pool(self):
        packed = _build_pool(FP8)
        self.assertEqual(packed.indexer_kv_dtype, FP8)
        self.assertEqual(packed.indexer_store_dtype, FP8)
        self.assertEqual(packed.indexer_scale_dtype, torch.float32)
        self.assertEqual(packed.index_k_buffer.dtype, FP8)
        self.assertEqual(packed.index_k_scale_buffer.dtype, torch.float32)
        unpacked = _build_pool(torch.bfloat16)
        self.assertEqual(unpacked.indexer_kv_dtype, torch.bfloat16)
        self.assertEqual(unpacked.indexer_store_dtype, torch.bfloat16)
        self.assertIsNone(unpacked.indexer_scale_dtype)
        self.assertEqual(unpacked.index_k_buffer.dtype, torch.bfloat16)

    def test_pool_rejects_fp8_on_a2(self):
        with self.assertRaisesRegex(ValueError, "arch35"):
            _build_pool(FP8, arch35=False)


class TestNpuIndexerSizing(_NpuPatchedTestCase):
    NUM_LAYERS = 3

    def _indexer_bytes(self, dtype):
        configurator = object.__new__(DefaultPoolConfigurator)
        kvc = SimpleNamespace(
            kv_cache_dtype=dtype,
            model_config=SimpleNamespace(hf_config=None),
            server_args=SimpleNamespace(enable_hisparse=False),
            is_draft_worker=False,
        )
        with (
            get_context().override_server_args(enable_hisparse=False),
            patch.object(pool_configurator, "_is_npu", True),
            patch.object(pool_configurator, "get_dsa_index_head_dim", return_value=128),
        ):
            return configurator._compute_dsa_indexer_cell_size(
                kvc=kvc, num_layers=self.NUM_LAYERS, allocate_all_layers=True
            )

    @staticmethod
    def _pool_indexer_bytes_per_token(pool):
        per_token = pool.index_k_buffer[0][0].nbytes // pool.index_page_size
        if pool.index_k_scale_buffer is not None:
            per_token += pool.index_k_scale_buffer[0][0].nbytes // pool.index_page_size
        return per_token

    def test_fp8_on_arch35_counts_scale_bytes(self):
        self._use_npu(arch35=True)
        self.assertEqual(self._indexer_bytes(FP8), 132 * self.NUM_LAYERS)
        self.assertEqual(self._pool_indexer_bytes_per_token(_build_pool(FP8)), 132)

    def test_bf16_has_no_scale_bytes(self):
        self._use_npu(arch35=False)
        self.assertEqual(self._indexer_bytes(torch.bfloat16), 256 * self.NUM_LAYERS)
        self.assertEqual(
            self._pool_indexer_bytes_per_token(_build_pool(torch.bfloat16)), 256
        )


if __name__ == "__main__":
    unittest.main()
