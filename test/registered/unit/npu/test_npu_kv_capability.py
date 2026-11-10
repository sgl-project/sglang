"""CPU-only tests for the Ascend DSA KV cache capability and its consumers.

Covers the startup rejection, the MLA KV cache dim selector, the NPU MLA pool
layout and the indexer sizer. NPU extension modules are stubbed; no operator
runs.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Loaded before the torch_npu mock below: importing transformers runs
# accelerate's find_spec("torch_npu"), which fails on a mock without __spec__.
import sglang.srt.hf_transformers  # noqa: F401
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

sys.modules.setdefault("sgl_kernel_npu", types.ModuleType("sgl_kernel_npu"))
sys.modules.setdefault("torch_npu", MagicMock(name="torch_npu"))
import sglang.srt.utils as _utils
from sglang.srt.utils import device as _device_utils

_real_is_npu = _device_utils.is_npu
_real_utils_is_npu = _utils.is_npu
_device_utils.is_npu = lambda: False
_utils.is_npu = _device_utils.is_npu
try:
    from sglang.srt.hardware_backend.npu import kv_capability, memory_pool_npu
    from sglang.srt.hardware_backend.npu.kv_capability import resolve_kv_capability
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.mem_cache import kv_cache_configurator
    from sglang.srt.mem_cache.kv_cache_configurator import (
        KVCacheConfigurator,
        calculate_mla_kv_cache_dim,
    )
    from sglang.srt.model_executor import pool_configurator
    from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
    from sglang.srt.runtime_context import get_context
finally:
    _device_utils.is_npu = _real_is_npu
    _utils.is_npu = _real_utils_is_npu

A2_SOC = "Ascend910B3"
FP8 = torch.float8_e4m3fn
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
# 512 FP8 latent bytes + 4 FP32 tile scales + 64 BF16 rope values.
PACKED_KV_CACHE_DIM = 656


class _NpuPatchedTestCase(CustomTestCase):
    def _use_npu(self, *, arch35: bool):
        """Patch the SoC probes behind resolve_npu_kv_capability."""
        for p in (
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=arch35,
            ),
            patch.object(kv_capability, "_get_npu_soc_name", return_value=A2_SOC),
        ):
            p.start()
            self.addCleanup(p.stop)


class TestResolveKvCapability(CustomTestCase):
    def test_only_e4m3_on_arch35_is_packed(self):
        """Other dtypes, one-byte e5m2 included, keep the unpacked layout."""
        for dtype in (torch.bfloat16, torch.float16, torch.float8_e5m2):
            for is_arch35 in (True, False):
                with self.subTest(dtype=dtype, is_arch35=is_arch35):
                    cap = resolve_kv_capability(
                        kv_cache_dtype=dtype, is_arch35=is_arch35, soc_name="x"
                    )
                    self.assertTrue(cap.supported)
                    self.assertFalse(cap.main_kv_packed)
                    self.assertFalse(cap.indexer_quant)
                    self.assertEqual(cap.indexer_kv_dtype, dtype)


class TestMlaKvCacheDim(CustomTestCase):
    def setUp(self):
        super().setUp()
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
            hf_config=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        )

    def _dim(self, dtype, **kwargs):
        return calculate_mla_kv_cache_dim(
            model_config=self.model_config, kv_cache_dtype=dtype, **kwargs
        )

    def test_packed_flag_overrides_the_dtype_rule(self):
        """The NPU builder passes ``packed``; callers that omit it keep the
        float8_e4m3fn rule."""
        self.assertEqual(self._dim(FP8, packed=True), PACKED_KV_CACHE_DIM)
        self.assertEqual(self._dim(FP8, packed=False), 576)
        self.assertEqual(self._dim(FP8), PACKED_KV_CACHE_DIM)
        self.assertEqual(self._dim(torch.float8_e5m2), 576)
        self.assertEqual(self._dim(torch.bfloat16), 576)


def _build_pool(
    dtype,
    *,
    arch35=True,
    start_layer=0,
    layer_num=2,
    index_head_dim=INDEX_HEAD_DIM,
    indexer_layer_ids=None,
    dcp_size=1,
    **kwargs,
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
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            layer_num=layer_num,
            device="cpu",
            enable_memory_saver=False,
            index_head_dim=index_head_dim,
            index_size=16 * dcp_size,
            start_layer=start_layer,
            end_layer=start_layer + layer_num,
            indexer_layer_ids=indexer_layer_ids,
            kv_cache_dim=PACKED_KV_CACHE_DIM if capability.main_kv_packed else None,
            kv_capability=capability,
            **kwargs,
        )


class TestNpuMlaPoolLayout(CustomTestCase):
    def test_dsa_pool_rejects_fp8_on_a2(self):
        """A pool built without the configurator's check still refuses the
        unpacked FP8 layout that sparse attention would read as packed."""
        with self.assertRaisesRegex(ValueError, "arch35"):
            _build_pool(FP8, arch35=False)

    def test_fp8_pool_without_an_indexer_is_not_packed(self):
        """Dense MLA and ModelSlim FP8 KV pools on arch35 store K and rope
        separately; only the DSA layout has the packed record."""
        layout = SimpleNamespace(
            layout="mla_fp8", key_dtype=FP8, value_dtype=torch.bfloat16
        )
        for kwargs in ({}, {"modelslim_cache_layout": layout}):
            with self.subTest(kwargs=kwargs):
                pool = _build_pool(FP8, index_head_dim=None, **kwargs)
                self.assertFalse(pool.dsa_kv_cache_store_fp8)
                self.assertEqual(pool.k_buffer.shape[-1], KV_LORA_RANK)
                self.assertEqual(pool.v_buffer.shape[-1], QK_ROPE_HEAD_DIM)
                self.assertIsNone(pool.index_k_buffer)


class TestAscendMlaPoolBuilder(_NpuPatchedTestCase):
    def _build_kwargs(self, dtype, *, is_dsa_model, requested="auto", resolved=None):
        kvc = SimpleNamespace(
            modelslim_cache_layout=None,
            model_config=SimpleNamespace(
                hf_config=None,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                index_head_dim=INDEX_HEAD_DIM,
            ),
            layer_info=SimpleNamespace(
                start_layer=0, end_layer=2, num_effective_layers=2
            ),
            is_draft_worker=False,
            use_combined_kv_buffer=False,
            kv_cache_dtype=dtype,
            kv_cache_dtype_str=resolved,
            server_args=SimpleNamespace(kv_cache_dtype=requested),
            device="cpu",
        )
        kernel = SimpleNamespace(
            dsa_prefill_backend="ascend", dsa_decode_backend="ascend"
        )
        with (
            get_context().override_server_args(page_size=4),
            patch.object(kv_cache_configurator, "_is_npu", True),
            patch.object(kv_cache_configurator, "_is_hip", False),
            patch.object(
                kv_cache_configurator, "is_deepseek_dsa", return_value=is_dsa_model
            ),
            patch.object(
                kv_cache_configurator, "_should_elide_dsa_index_k", return_value=False
            ),
            patch.object(
                kv_cache_configurator,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=kernel, features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch.object(
                kv_cache_configurator,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch.object(memory_pool_npu, "NPUMLATokenToKVPool") as pool_cls,
        ):
            KVCacheConfigurator._build_ascend_mla_kv_pool(
                kvc, max_total_num_tokens=16, is_dsa_model=is_dsa_model
            )
        return pool_cls.call_args.kwargs

    def test_dsa_fp8_on_arch35_builds_the_packed_record(self):
        self._use_npu(arch35=True)
        kwargs = self._build_kwargs(FP8, is_dsa_model=True)
        self.assertEqual(kwargs["kv_cache_dim"], PACKED_KV_CACHE_DIM)
        self.assertTrue(kwargs["kv_capability"].main_kv_packed)

    def test_dsa_fp8_on_a2_is_a_startup_error(self):
        """FP8 KV on a non-arch35 NPU used to start a DSA model with an unpacked
        FP8 pool while sparse attention read it as the packed record."""
        self._use_npu(arch35=False)
        # `auto` with an FP8 checkpoint resolves to fp8_e4m3; the message names
        # the resolved value.
        for requested, resolved in (
            ("fp8_e4m3", None),
            ("mxfp8", "mxfp8"),
            ("auto", "fp8_e4m3"),
        ):
            with self.subTest(requested=requested):
                with self.assertRaises(ValueError) as ctx:
                    self._build_kwargs(
                        FP8, is_dsa_model=True, requested=requested, resolved=resolved
                    )
                message = str(ctx.exception)
                self.assertIn(f"--kv-cache-dtype {resolved or requested}", message)
                self.assertIn(A2_SOC, message)
                self.assertIn("bf16", message)

    def test_non_dsa_fp8_keeps_its_own_layout_on_every_soc(self):
        for arch35 in (True, False):
            with self.subTest(arch35=arch35):
                self._use_npu(arch35=arch35)
                kwargs = self._build_kwargs(
                    FP8, is_dsa_model=False, requested="fp8_e4m3"
                )
                self.assertIsNone(kwargs["kv_cache_dim"])
                self.assertIsNone(kwargs["kv_capability"])
                self.assertIsNone(kwargs["index_head_dim"])


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
            patch.object(
                pool_configurator, "get_dsa_index_head_dim", return_value=INDEX_HEAD_DIM
            ),
            patch.object(pool_configurator, "get_dsa_index_kpool", return_value=1),
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

    def test_sizer_matches_the_pool(self):
        """The sizer and the pool are written separately; a pool that outgrows
        its budget overruns device memory."""
        # FP8 on arch35: FP8 index K plus one FP32 scale; BF16: no scale.
        for dtype, arch35, per_token in (
            (FP8, True, 132),
            (torch.bfloat16, False, 256),
        ):
            with self.subTest(dtype=dtype):
                self._use_npu(arch35=arch35)
                self.assertEqual(
                    self._indexer_bytes(dtype), per_token * self.NUM_LAYERS
                )
                pool = _build_pool(dtype, arch35=arch35)
                self.assertEqual(self._pool_indexer_bytes_per_token(pool), per_token)


if __name__ == "__main__":
    unittest.main()
