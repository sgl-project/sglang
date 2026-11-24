"""CPU-only tests for the Ascend DSA KV cache capability and its consumers.

Covers the startup rejection, the MLA KV cache dim selector, the NPU MLA pool
layout and PD layer ids, the indexer sizer, and the INT8 KV write/read and
indexer paths. NPU extension modules are stubbed; no operator runs.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Loaded before the torch_npu mock below: importing transformers runs
# accelerate's find_spec("torch_npu"), which fails on a mock without __spec__.
import sglang.srt.hf_transformers  # noqa: F401
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

for _name in (
    "torch_npu",
    "sgl_kernel_npu",
    "sgl_kernel_npu.operators",
    "sgl_kernel_npu.operators.attention",
):
    sys.modules.setdefault(_name, MagicMock(name=_name))
import sglang.srt.utils as _utils
from sglang.srt.utils import device as _device_utils

_real_is_npu = _device_utils.is_npu
_real_utils_is_npu = _utils.is_npu
_device_utils.is_npu = lambda: False
_utils.is_npu = _device_utils.is_npu
try:
    from sglang.srt.hardware_backend.npu import kv_capability, memory_pool_npu
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
        ForwardMetadata,
    )
    from sglang.srt.hardware_backend.npu.kv_capability import resolve_kv_capability
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
    from sglang.srt.layers.attention.dsa import dsa_npu_indexer
    from sglang.srt.layers.attention.dsa.dsa_npu_indexer import DSANPUIndexerMixin
    from sglang.srt.mem_cache import kv_cache_configurator
    from sglang.srt.mem_cache.kv_cache_configurator import (
        KVCacheConfigurator,
        calculate_mla_kv_cache_dim,
    )
    from sglang.srt.mem_cache.kv_cache_dtype import (
        QUANT_MODE_TOKEN_INT8,
        configure_kv_cache_dtype,
        resolve_indexer_quant_mode,
    )
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
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
# 512 one-byte latent values + 4 FP32 tile scales + 64 BF16 rope values.
PACKED_KV_CACHE_DIM = 656
INT8 = torch.int8


def _model_config(arch):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch], index_topk=2048)
    )


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

    def _dim(self, dtype):
        return calculate_mla_kv_cache_dim(
            model_config=self.model_config, kv_cache_dtype=dtype
        )

    def test_packed_row_width_follows_the_dtype(self):
        """The packed record is the one-byte quantized dtype's row width."""
        self.assertEqual(self._dim(FP8), PACKED_KV_CACHE_DIM)
        self.assertEqual(self._dim(INT8), PACKED_KV_CACHE_DIM)
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
    with (
        # The INT8 indexer recipe checks the memory config for HiCache.
        get_context().override_server_args(enable_hierarchical_cache=False),
        patch.object(memory_pool_npu, "get_parallel", return_value=parallel),
    ):
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
    def assertLayerIdsAlign(self, pool, expected):
        ptrs, _, item_lens = pool.get_contiguous_buf_infos()
        layer_ids = pool.get_kv_layer_ids()
        self.assertEqual(layer_ids, expected)
        self.assertEqual(len(layer_ids), len(ptrs))
        self.assertEqual(len(layer_ids), len(item_lens))
        self.assertEqual(len(layer_ids), len(pool.get_dcp_remote_decode_layout()))

    def test_packed_pool_layer_ids_match_the_pd_entries(self):
        """The packed record transfers no V buffers; the layer ids used to list
        K and V per layer and so outnumbered the PD entries (8 ids, 6 buffers)."""
        pool = _build_pool(FP8)
        self.assertEqual(pool.v_buffer.shape[-1], 0)
        self.assertLayerIdsAlign(pool, [0, 1] * 3)

    def test_int8_pool_transfers_packed_k_and_bf16_index_k(self):
        """INT8 main KV is the packed record; the indexer stays BF16 with no
        scale buffer, so PD moves K (656 B rows) and index K (256 B rows)."""
        pool = _build_pool(INT8, arch35=False)
        self.assertEqual(pool.k_buffer.dtype, INT8)
        self.assertEqual(pool.v_buffer.shape[-1], 0)
        self.assertEqual(pool.index_k_buffer.dtype, torch.bfloat16)
        self.assertIsNone(pool.index_k_scale_buffer)
        self.assertLayerIdsAlign(pool, [0, 1] * 2)
        _, _, item_lens = pool.get_contiguous_buf_infos()
        page_size = pool.page_size
        self.assertEqual(
            item_lens,
            [PACKED_KV_CACHE_DIM * page_size] * 2
            + [INDEX_HEAD_DIM * 2 * page_size] * 2,
        )

    def test_int8_main_kv_composes_with_the_int8_indexer(self):
        """The INT8 indexer recipe stacks on the packed INT8 record: PD moves
        K (656 B rows), INT8 index K (128 B rows) and FP16 scales (2 B rows)."""
        pool = _build_pool(INT8, arch35=False, indexer_quant_mode=QUANT_MODE_TOKEN_INT8)
        self.assertTrue(pool.dsa_kv_cache_store_c8)
        self.assertEqual(pool.k_buffer.dtype, INT8)
        self.assertEqual(pool.index_k_buffer.dtype, INT8)
        self.assertEqual(pool.index_k_scale_buffer.dtype, torch.float16)
        self.assertLayerIdsAlign(pool, [0, 1] * 3)
        _, _, item_lens = pool.get_contiguous_buf_infos()
        page_size = pool.page_size
        self.assertEqual(
            item_lens,
            [PACKED_KV_CACHE_DIM * page_size] * 2
            + [INDEX_HEAD_DIM * page_size] * 2
            + [2 * page_size] * 2,
        )

    def test_fp8_record_keeps_its_own_quantized_indexer(self):
        """The arch35 FP8 record already quantizes the indexer (FP8 + FP32
        scale); the INT8 recipe does not stack on it."""
        pool = _build_pool(FP8, indexer_quant_mode=QUANT_MODE_TOKEN_INT8)
        self.assertFalse(pool.indexer_kv_store_int8)
        self.assertEqual(pool.index_k_buffer.dtype, FP8)
        self.assertEqual(pool.index_k_scale_buffer.dtype, torch.float32)

    def test_layer_ids_follow_each_entry_group(self):
        """Index K entries carry indexer_layer_ids (compact layout), unpacked
        pools keep their V entries, and a combined buffer has one per layer."""
        cases = (
            (FP8, {"indexer_layer_ids": (5,)}, [4, 5, 5, 5]),
            (torch.bfloat16, {"indexer_layer_ids": (5,)}, [4, 5, 4, 5, 5]),
            (
                torch.bfloat16,
                {"index_head_dim": None, "use_combined_kv_buffer": True},
                [4, 5],
            ),
        )
        for dtype, kwargs, expected in cases:
            with self.subTest(dtype=dtype, kwargs=kwargs):
                pool = _build_pool(dtype, start_layer=4, **kwargs)
                self.assertLayerIdsAlign(pool, expected)

    def test_combined_packed_buffer_gives_a_bf16_rope_placeholder(self):
        """A combined buffer holds only the packed record. As in the split
        layout, the rope cache handed to MLAProlog is an empty BF16 tensor,
        not a zero-width view of the one-byte record."""
        for dtype, arch35 in ((INT8, False), (FP8, True)):
            with self.subTest(dtype=dtype):
                pool = _build_pool(dtype, arch35=arch35, use_combined_kv_buffer=True)
                # Shared readers query the C8 flag under its FP8-era name.
                self.assertTrue(pool.dsa_kv_cache_store_c8)
                self.assertTrue(pool.dsa_kv_cache_store_fp8)
                self.assertEqual(pool.kv_buffer.shape[-1], PACKED_KV_CACHE_DIM)
                k_cache, v_cache = pool.get_kv_buffer(0)
                self.assertEqual(k_cache.dtype, dtype)
                self.assertEqual(v_cache.dtype, torch.bfloat16)
                self.assertEqual(v_cache.shape, (*k_cache.shape[:-1], 0))
                # PD still moves only the combined record, one entry per layer.
                self.assertLayerIdsAlign(pool, [0, 1])

    def test_combined_unpacked_buffer_keeps_its_rope_slice(self):
        """Without the packed record, the combined buffer's rope cache stays
        a slice of the shared allocation."""
        pool = _build_pool(torch.bfloat16, use_combined_kv_buffer=True)
        self.assertEqual(pool.k_buffer.shape[-1], KV_LORA_RANK)
        self.assertEqual(pool.v_buffer.shape[-1], QK_ROPE_HEAD_DIM)
        self.assertEqual(
            pool.v_buffer.data_ptr(), pool.kv_buffer[..., KV_LORA_RANK:].data_ptr()
        )

    def test_index_k_reads_and_writes_in_the_indexer_dtype(self):
        """The indexer K cache is read and written in its own dtype; only an
        indexer kept in the main KV's byte store (FP8 E5M2 as uint8) is viewed
        back to the KV dtype."""
        bf16, e5m2 = torch.bfloat16, torch.float8_e5m2
        cases = (
            # (kv dtype, arch35, indexer recipe, read dtype, stored dtype)
            (bf16, False, None, bf16, bf16),
            (bf16, False, QUANT_MODE_TOKEN_INT8, INT8, INT8),
            (INT8, False, None, bf16, bf16),
            (INT8, False, QUANT_MODE_TOKEN_INT8, INT8, INT8),
            (FP8, True, None, FP8, FP8),
            (e5m2, False, None, e5m2, torch.uint8),
            (e5m2, False, QUANT_MODE_TOKEN_INT8, INT8, INT8),
        )
        for dtype, arch35, recipe, read_dtype, stored_dtype in cases:
            with self.subTest(dtype=dtype, recipe=recipe):
                pool = _build_pool(dtype, arch35=arch35, indexer_quant_mode=recipe)
                self.assertEqual(pool.index_k_buffer.dtype, stored_dtype)
                self.assertEqual(pool.get_index_k_buffer(0).dtype, read_dtype)
                torch_npu = MagicMock(name="torch_npu")
                with patch.object(memory_pool_npu, "torch_npu", torch_npu, create=True):
                    pool.set_index_k_buffer(
                        0,
                        torch.tensor([1, 2]),
                        torch.zeros(
                            2, INDEX_HEAD_DIM, dtype=bf16 if recipe is None else INT8
                        ),
                    )
                written = torch_npu.npu_scatter_nd_update_.call_args.args[2]
                self.assertEqual(written.dtype, stored_dtype)

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
                self.assertFalse(pool.dsa_kv_cache_store_c8)
                self.assertFalse(pool.dsa_kv_cache_store_fp8)
                self.assertEqual(pool.k_buffer.shape[-1], KV_LORA_RANK)
                self.assertEqual(pool.v_buffer.shape[-1], QK_ROPE_HEAD_DIM)
                self.assertIsNone(pool.index_k_buffer)


class TestHostPoolMirrorsTheDevicePool(CustomTestCase):
    """HiCache, decode KV offload and the retraction backup all mirror the
    device pool through MLATokenToKVPoolHost; each mirror takes its dtype and
    width from the device buffer it copies."""

    def _host(self, pool, layout):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        host = MLATokenToKVPoolHost(
            pool._mla_buffer_info(),
            host_to_device_ratio=0,
            host_size=0,
            page_size=pool.page_size,
            layout=layout,
            pin_memory=False,
            device_capacity_tokens=pool.size,
            start_layer=pool.start_layer,
            end_layer=pool.start_layer + pool.layer_num,
            target_layer_num=pool.layer_num,
            host_token_capacity=pool.size,
        )
        self.addCleanup(host.destroy)
        return host

    def test_int8_main_kv_mirrors_with_and_without_the_int8_indexer(self):
        # (main dtype, arch35, indexer recipe, index K dtype, scale dtype)
        cases = (
            (INT8, False, None, torch.bfloat16, None),
            (INT8, False, QUANT_MODE_TOKEN_INT8, INT8, torch.float16),
            (FP8, True, None, FP8, torch.float32),
        )
        for dtype, arch35, recipe, index_dtype, scale_dtype in cases:
            pool = _build_pool(dtype, arch35=arch35, indexer_quant_mode=recipe)
            # NPU MLA host pools always use a kv_split layout.
            for layout in ("layer_first_kv_split", "page_first_kv_split"):
                with self.subTest(dtype=dtype, recipe=recipe, layout=layout):
                    host = self._host(pool, layout)
                    self.assertEqual(host.index_k_buffer.dtype, index_dtype)
                    if scale_dtype is None:
                        self.assertIsNone(host.index_k_scale_buffer)
                    else:
                        self.assertEqual(host.index_k_scale_buffer.dtype, scale_dtype)
                    # The packed record has no V; K rows are the 656-byte record.
                    self.assertEqual(
                        host.kv_buffer.element_size() * host.kv_buffer.shape[-1],
                        PACKED_KV_CACHE_DIM,
                    )
                    index_bytes = INDEX_HEAD_DIM * index_dtype.itemsize + (
                        scale_dtype.itemsize if scale_dtype is not None else 0
                    )
                    host_index_bytes = host.index_k_buffer.numel() * (
                        host.index_k_buffer.element_size()
                    )
                    if host.index_k_scale_buffer is not None:
                        host_index_bytes += host.index_k_scale_buffer.numel() * (
                            host.index_k_scale_buffer.element_size()
                        )
                    self.assertEqual(
                        host_index_bytes,
                        index_bytes * len(pool.indexer_layer_ids) * pool.size,
                    )


class TestAscendMlaPoolBuilder(_NpuPatchedTestCase):
    def _build_kwargs(
        self,
        dtype,
        *,
        is_dsa_model,
        requested="auto",
        resolved=None,
        combined=False,
        is_draft_worker=False,
        dcp_size=1,
        indexer_kv_cache_dtype=None,
    ):
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
            is_draft_worker=is_draft_worker,
            use_combined_kv_buffer=combined,
            kv_cache_dtype=dtype,
            kv_cache_dtype_str=resolved,
            server_args=SimpleNamespace(
                kv_cache_dtype=requested, indexer_kv_cache_dtype=indexer_kv_cache_dtype
            ),
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
            patch.object(
                kv_cache_configurator,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=dcp_size),
            ),
            patch.object(memory_pool_npu, "NPUMLATokenToKVPool") as pool_cls,
        ):
            KVCacheConfigurator._build_ascend_mla_kv_pool(
                kvc,
                max_total_num_tokens=16,
                max_running_requests=4,
                is_dsa_model=is_dsa_model,
            )
        return pool_cls.call_args.kwargs

    def test_dsa_fp8_on_arch35_builds_the_packed_record(self):
        self._use_npu(arch35=True)
        kwargs = self._build_kwargs(FP8, is_dsa_model=True)
        self.assertEqual(kwargs["kv_cache_dim"], PACKED_KV_CACHE_DIM)
        self.assertTrue(kwargs["kv_capability"].main_kv_packed)

    def test_packed_record_rejects_the_sparse_kv_offload(self):
        """The sparse KV offload reads separate unquantized latent and rope
        buffers, whatever the one-byte dtype of the record."""
        for dtype, requested, arch35 in (
            (FP8, "fp8_e4m3", True),
            (INT8, "int8", False),
        ):
            self._use_npu(arch35=arch35)
            with (
                self.subTest(dtype=dtype),
                envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.override(True),
            ):
                with self.assertRaisesRegex(
                    ValueError, "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD"
                ):
                    self._build_kwargs(dtype, is_dsa_model=True, requested=requested)
                # An unpacked pool keeps the offload.
                self._build_kwargs(torch.bfloat16, is_dsa_model=True)

    def test_dcp_reads_the_fp8_record_but_not_the_int8_one(self):
        """DCP sparse attention reads the packed record through
        kv_quant_sparse_flash_attention, which is wired for fp8_e4m3 on
        arch35 only."""
        self._use_npu(arch35=False)
        with self.assertRaisesRegex(ValueError, "--dcp-size"):
            self._build_kwargs(INT8, is_dsa_model=True, requested="int8", dcp_size=2)
        # A draft worker never runs DCP attention.
        self._build_kwargs(
            INT8, is_dsa_model=True, requested="int8", dcp_size=2, is_draft_worker=True
        )
        self._use_npu(arch35=True)
        kwargs = self._build_kwargs(
            FP8, is_dsa_model=True, requested="fp8_e4m3", dcp_size=2
        )
        self.assertEqual(kwargs["kv_cache_dim"], PACKED_KV_CACHE_DIM)

    def test_int8_record_rejects_the_dsa_cache_layer_split(self):
        """The CP layer-split pool allocates its indexer cache in the main KV
        store dtype, so it cannot hold the INT8 record's BF16 indexer."""
        self._use_npu(arch35=False)
        with (
            patch(
                "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
                return_value=(0, 2),
            ),
            self.assertRaisesRegex(ValueError, "--enable-dsa-cache-layer-split"),
        ):
            self._build_kwargs(INT8, is_dsa_model=True, requested="int8")
        # Without the layer split the INT8 record builds.
        kwargs = self._build_kwargs(INT8, is_dsa_model=True, requested="int8")
        self.assertTrue(kwargs["kv_capability"].main_kv_packed)

    def test_int8_main_kv_takes_the_int8_indexer_recipe(self):
        """--indexer-kv-cache-dtype int8 reaches the pool on top of an INT8
        main KV; an FP8 main KV quantizes its indexer itself and rejects it."""
        self._use_npu(arch35=False)
        kwargs = self._build_kwargs(
            INT8, is_dsa_model=True, requested="int8", indexer_kv_cache_dtype="int8"
        )
        self.assertEqual(kwargs["indexer_quant_mode"], QUANT_MODE_TOKEN_INT8)
        self.assertTrue(kwargs["kv_capability"].main_kv_packed)
        with self.assertRaisesRegex(ValueError, "non-FP8 main KV cache"):
            resolve_indexer_quant_mode("int8", FP8)

    def test_packed_record_keeps_the_combined_kv_buffer(self):
        """DSA attention reads the record through forward_sparse from the K
        view of the combined buffer, not through the flash_mla reader."""
        for dtype, requested, arch35 in (
            (FP8, "fp8_e4m3", True),
            (INT8, "int8", False),
        ):
            with self.subTest(dtype=dtype):
                self._use_npu(arch35=arch35)
                kwargs = self._build_kwargs(
                    dtype, is_dsa_model=True, requested=requested, combined=True
                )
                self.assertTrue(kwargs["use_combined_kv_buffer"])
                self.assertEqual(kwargs["kv_cache_dim"], PACKED_KV_CACHE_DIM)

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

    def test_dsa_int8_on_a2_builds_the_packed_record(self):
        self._use_npu(arch35=False)
        kwargs = self._build_kwargs(INT8, is_dsa_model=True, requested="int8")
        self.assertEqual(kwargs["kv_cache_dim"], PACKED_KV_CACHE_DIM)
        self.assertTrue(kwargs["kv_capability"].main_kv_packed)

    def test_int8_draft_pool_is_packed(self):
        """The draft inherits --kv-cache-dtype int8 and builds the same packed
        record as the target, as an FP8 draft does on arch35."""
        self._use_npu(arch35=False)
        resolved, dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype="int8",
            quant_config=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=True,
            is_dflash=False,
            speculative_draft_attention_backend="ascend",
            model_config=_model_config("GlmMoeDsaForCausalLMNextN"),
        )
        kwargs = self._build_kwargs(
            dtype,
            is_dsa_model=True,
            requested="int8",
            resolved=resolved,
            is_draft_worker=True,
        )
        self.assertEqual(kwargs["dtype"], INT8)
        self.assertTrue(kwargs["kv_capability"].main_kv_packed)
        self.assertEqual(kwargs["kv_cache_dim"], PACKED_KV_CACHE_DIM)

    def test_a_draft_that_cannot_read_the_record_keeps_its_compute_dtype(self):
        """INT8 is the packed DSA record only: a dense draft, a DSA draft on
        another backend or an unknown draft would cast K/V to INT8 without
        scales, so each resolves to its compute dtype. The target is untouched."""

        def resolve(arch, backend, *, is_draft_worker=True, is_dflash=True):
            return configure_kv_cache_dtype(
                server_args_kv_cache_dtype="int8",
                quant_config=None,
                model_dtype=torch.bfloat16,
                is_draft_worker=is_draft_worker,
                is_dflash=is_dflash,
                speculative_draft_attention_backend=backend,
                model_config=None if arch is None else _model_config(arch),
            )

        for arch, backend in (
            ("Qwen3ForCausalLM", "ascend"),
            ("Qwen3ForCausalLM", None),
            ("GlmMoeDsaForCausalLM", "triton"),
            (None, "ascend"),
        ):
            with self.subTest(arch=arch, backend=backend):
                self.assertEqual(resolve(arch, backend), ("auto", torch.bfloat16))
        for backend in ("ascend", None):
            with self.subTest(backend=backend):
                self.assertEqual(
                    resolve("GlmMoeDsaForCausalLM", backend), ("int8", INT8)
                )
        self.assertEqual(
            resolve("Qwen3ForCausalLM", None, is_draft_worker=False, is_dflash=False),
            ("int8", INT8),
        )

    def test_dsa_int8_on_arch35_is_a_startup_error(self):
        """arch35 has no INT8 packed path; it must not fall back to a plain
        one-byte cache."""
        self._use_npu(arch35=True)
        with self.assertRaises(ValueError) as ctx:
            self._build_kwargs(INT8, is_dsa_model=True, requested="int8")
        self.assertIn("--kv-cache-dtype int8", str(ctx.exception))
        self.assertIn("fp8_e4m3", str(ctx.exception))

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

    def _indexer_bytes(self, dtype, indexer_kv_cache_dtype=None):
        configurator = object.__new__(DefaultPoolConfigurator)
        kvc = SimpleNamespace(
            kv_cache_dtype=dtype,
            model_config=SimpleNamespace(hf_config=None),
            server_args=SimpleNamespace(
                enable_hisparse=False, indexer_kv_cache_dtype=indexer_kv_cache_dtype
            ),
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
        # FP8 on arch35: FP8 index K plus one FP32 scale; BF16: no scale; an
        # INT8 main KV keeps the BF16 indexer, not a one-byte one, unless the
        # INT8 indexer recipe asks for INT8 index K plus one FP16 scale.
        for dtype, arch35, indexer_kv_cache_dtype, per_token in (
            (FP8, True, None, 132),
            (torch.bfloat16, False, None, 256),
            (INT8, False, None, 256),
            (torch.bfloat16, False, "int8", 130),
            (INT8, False, "int8", 130),
        ):
            with self.subTest(dtype=dtype, indexer=indexer_kv_cache_dtype):
                self._use_npu(arch35=arch35)
                self.assertEqual(
                    self._indexer_bytes(dtype, indexer_kv_cache_dtype),
                    per_token * self.NUM_LAYERS,
                )
                pool = _build_pool(
                    dtype,
                    arch35=arch35,
                    indexer_quant_mode=resolve_indexer_quant_mode(
                        indexer_kv_cache_dtype, dtype
                    ),
                )
                self.assertEqual(self._pool_indexer_bytes_per_token(pool), per_token)


class TestConfigureInt8KvCacheDtype(CustomTestCase):
    def _configure(self, *, is_draft_worker):
        return configure_kv_cache_dtype(
            server_args_kv_cache_dtype="int8",
            quant_config=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=is_draft_worker,
            is_dflash=False,
            speculative_draft_attention_backend="ascend",
            model_config=_model_config("GlmMoeDsaForCausalLM"),
        )

    def test_a_draft_inherits_the_int8_record(self):
        """The target and its DSA draft both resolve --kv-cache-dtype int8."""
        self.assertEqual(self._configure(is_draft_worker=False), ("int8", INT8))
        self.assertEqual(self._configure(is_draft_worker=True), ("int8", INT8))


def _decode_forward_batch(out_cache_loc):
    mode = SimpleNamespace(
        is_extend=lambda: False,
        is_draft_extend_v2=lambda: False,
        is_target_verify=lambda: False,
    )
    return SimpleNamespace(
        forward_mode=mode,
        out_cache_loc=out_cache_loc,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        attn_dcp_metadata=None,
    )


class _MockedNpuOpsTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.torch_npu = MagicMock(name="torch_npu")
        self.torch_npu.npu_dynamic_quant.side_effect = lambda x, dst_type: (
            torch.zeros(x.shape, dtype=dst_type),
            torch.ones(x.shape[0]),
        )
        for module in (memory_pool_npu, dsa_npu_indexer):
            p = patch.object(module, "torch_npu", self.torch_npu, create=True)
            p.start()
            self.addCleanup(p.stop)
        self.pool = _build_pool(INT8, arch35=False)
        self.out_cache_loc = torch.tensor([5, 6])


class TestPackedKvWriteAndSparseRead(_MockedNpuOpsTestCase):
    def test_decode_writes_and_reads_the_packed_record(self):
        """Sparse attention reads the record by the pool's packed flag, not the
        FP8 dtype: INT8 on A2 and FP8 on arch35."""
        for dtype, arch35 in ((INT8, False), (FP8, True)):
            with self.subTest(dtype=dtype):
                self.torch_npu.reset_mock()
                pool = _build_pool(dtype, arch35=arch35)
                quant_attn, bf16_attn = self._decode(pool)
                bf16_attn.assert_not_called()
                self.assertEqual(
                    self.torch_npu.npu_dynamic_quant.call_args.kwargs["dst_type"],
                    dtype,
                )
                written = self.torch_npu.npu_scatter_nd_update_.call_args.args[2]
                self.assertEqual(written.dtype, pool.store_dtype)
                self.assertEqual(written.shape[-1], PACKED_KV_CACHE_DIM)
                key = quant_attn.call_args.kwargs["key"]
                self.assertEqual(key.dtype, dtype)
                self.assertEqual(key.shape[-1], PACKED_KV_CACHE_DIM)

    def test_the_packed_record_is_read_through_the_hybrid_wrapper(self):
        """HybridLinearKVPool forwards the packed flag and the K buffer but
        has no ``store_dtype``: the reader takes the K tensor as it is handed
        back, for INT8 and for FP8."""
        for dtype, arch35 in ((INT8, False), (FP8, True)):
            with self.subTest(dtype=dtype):
                self.torch_npu.reset_mock()
                full_kv_pool = _build_pool(dtype, arch35=arch35)
                pool = object.__new__(HybridLinearKVPool)
                pool.full_kv_pool = full_kv_pool
                pool.full_attention_layer_id_mapping = {0: 0, 1: 1}
                pool.layer_transfer_counter = None
                pool.start_layer = 0
                pool.use_mla = True
                self.assertFalse(hasattr(pool, "store_dtype"))
                quant_attn, bf16_attn = self._decode(pool)
                bf16_attn.assert_not_called()
                key = quant_attn.call_args.kwargs["key"]
                self.assertEqual(key.dtype, dtype)
                self.assertEqual(key.shape[-1], PACKED_KV_CACHE_DIM)
                self.assertIs(quant_attn.call_args.kwargs["value"], key)
                self.assertEqual(
                    key.data_ptr(), full_kv_pool.get_kv_buffer(1)[0].data_ptr()
                )

    def test_write_and_read_derive_from_the_pool_quant_block_size(self):
        """The record-width check, the packer's tile and scale columns and the
        sparse read's tile_size all derive from ``quant_block_size`` (default
        128): a pool with another tile packs and reads a wider record."""
        self.assertEqual(self.pool.quant_block_size, 128)

        class _Tile64Pool(memory_pool_npu.NPUMLATokenToKVPool):
            quant_block_size = 64

        dim64 = KV_LORA_RANK + KV_LORA_RANK // 64 * 4 + QK_ROPE_HEAD_DIM * 2
        capability = resolve_kv_capability(
            kv_cache_dtype=INT8, is_arch35=False, soc_name="x"
        )
        parallel = SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0)
        with (
            get_context().override_server_args(enable_hierarchical_cache=False),
            patch.object(memory_pool_npu, "get_parallel", return_value=parallel),
        ):
            pool = _Tile64Pool(
                16,
                page_size=4,
                dtype=INT8,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                index_head_dim=INDEX_HEAD_DIM,
                index_size=16,
                start_layer=0,
                end_layer=2,
                indexer_layer_ids=None,
                kv_cache_dim=dim64,
                kv_capability=capability,
            )
        quant_attn, _ = self._decode(pool)
        latent = self.torch_npu.npu_dynamic_quant.call_args.args[0]
        self.assertEqual(latent.shape[-1], 64)
        written = self.torch_npu.npu_scatter_nd_update_.call_args.args[2]
        self.assertEqual(written.shape[-1], dim64)
        self.assertEqual(quant_attn.call_args.kwargs["tile_size"], 64)

    def _decode(self, pool):
        backend = object.__new__(AscendAttnBackend)
        backend.kv_lora_rank = KV_LORA_RANK
        backend.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        backend.token_to_kv_pool = pool
        backend.dsa_dcp_target = False
        backend.speculative_num_draft_tokens = None
        backend.forward_metadata = ForwardMetadata(
            block_tables=torch.zeros(2, 4, dtype=torch.int32),
            actual_seq_lengths_kv=torch.tensor([5, 6], dtype=torch.int32),
        )
        layer = SimpleNamespace(layer_id=1, tp_k_head_num=1, scaling=0.1)
        tokens, heads = 2, 4
        bf16 = torch.bfloat16
        quant_attn = MagicMock(name="npu_kv_quant_sparse_flash_attention")
        bf16_attn = MagicMock(name="npu_sparse_flash_attention")
        with (
            patch.multiple(
                "sglang.srt.hardware_backend.npu.attention.ascend_backend.torch_npu",
                npu_kv_quant_sparse_flash_attention=quant_attn,
                npu_sparse_flash_attention=bf16_attn,
            ),
            # Decode never takes a token-shard plan.
            patch(
                "sglang.srt.hardware_backend.npu.attention.ascend_backend."
                "get_dsa_token_shard_plan",
                return_value=None,
            ),
        ):
            backend.forward_sparse(
                torch.zeros(tokens, heads, KV_LORA_RANK, dtype=bf16),
                torch.zeros(tokens, KV_LORA_RANK, dtype=bf16),
                None,
                layer,
                _decode_forward_batch(self.out_cache_loc),
                save_kv_cache=True,
                q_rope=torch.zeros(tokens, heads, QK_ROPE_HEAD_DIM, dtype=bf16),
                k_rope=torch.zeros(tokens, QK_ROPE_HEAD_DIM, dtype=bf16),
                topk_indices=torch.zeros(tokens, 8, dtype=torch.int32),
            )
        return quant_attn, bf16_attn


class _FakeRope:
    is_neox_style = False
    cos_sin_cache = torch.zeros(16, 64)

    def __call__(self, positions, q_pe, k_pe):
        return q_pe, k_pe


class _FakeWeightsProj:
    weight = torch.zeros(1, dtype=torch.bfloat16)

    def __init__(self, tokens, heads):
        self.out = torch.zeros(tokens, heads)

    def __call__(self, x):
        return (self.out,)


class TestInt8MainKvIndexer(_MockedNpuOpsTestCase):
    def _run_indexer(self, pool):
        tokens, heads = 2, 4
        bf16 = torch.bfloat16
        indexer = SimpleNamespace(
            rotary_emb=_FakeRope(),
            wq_b=lambda q: (torch.zeros(tokens, heads * INDEX_HEAD_DIM, dtype=bf16),),
            wk=lambda x: (torch.zeros(tokens, INDEX_HEAD_DIM, dtype=bf16),),
            k_norm=lambda k: k,
            weights_proj=_FakeWeightsProj(tokens, heads),
            n_heads=heads,
            head_dim=INDEX_HEAD_DIM,
            rope_head_dim=64,
            hidden_size=8,
            index_topk=4,
            num_init_tokens=0,
            num_local_tokens=0,
            dsa_enable_prefill_cp=False,
            alt_stream=None,
        )
        attn_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(
                seq_lens_cpu_int=None,
                seq_lens=torch.tensor([5, 6], dtype=torch.int32),
                actual_seq_lengths_q=None,
                block_tables=torch.zeros(2, 4, dtype=torch.int32),
            ),
            is_draft_worker=False,
        )
        with (
            patch.object(
                dsa_npu_indexer, "get_attn_backend", return_value=attn_backend
            ),
            patch.object(dsa_npu_indexer, "get_token_to_kv_pool", return_value=pool),
            patch.object(
                dsa_npu_indexer,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False, attn_dcp_size=1),
            ),
        ):
            DSANPUIndexerMixin.forward_npu(
                indexer,
                x=torch.zeros(tokens, 8, dtype=bf16),
                q_lora=torch.zeros(tokens, 16, dtype=bf16),
                positions=torch.zeros(tokens, dtype=torch.long),
                forward_batch=_decode_forward_batch(self.out_cache_loc),
                layer_id=0,
            )

    def test_indexer_writes_and_reads_bf16_without_quantization(self):
        """The indexer quantizes whenever the pool has a scale buffer; an INT8
        main KV alone must leave it BF16 and off the quantized lightning
        indexer."""
        bf16 = torch.bfloat16
        self._run_indexer(self.pool)
        self.torch_npu.npu_quant_lightning_indexer.assert_not_called()
        self.torch_npu.npu_dynamic_quant.assert_not_called()
        written = self.torch_npu.npu_scatter_nd_update_.call_args.args[2]
        self.assertEqual(written.dtype, bf16)
        key = self.torch_npu.npu_lightning_indexer.call_args.kwargs["key"]
        self.assertEqual(key.dtype, bf16)

    def test_int8_indexer_recipe_quantizes_on_an_int8_main_kv(self):
        """With the INT8 indexer recipe on top, the indexer writes INT8 keys
        and FP16 scales and selects through the quantized lightning indexer."""
        pool = _build_pool(INT8, arch35=False, indexer_quant_mode=QUANT_MODE_TOKEN_INT8)
        npu_ops = MagicMock(name="torch.ops.npu")
        with patch.object(torch.ops, "npu", npu_ops, create=True):
            self._run_indexer(pool)
        self.torch_npu.npu_lightning_indexer.assert_not_called()
        self.assertEqual(
            self.torch_npu.npu_dynamic_quant.call_args.kwargs["dst_type"], INT8
        )
        written = [
            call.args[2]
            for call in self.torch_npu.npu_scatter_nd_update_.call_args_list
        ]
        self.assertEqual(
            sorted(str(w.dtype) for w in written),
            sorted([str(INT8), str(torch.float16)]),
        )
        read = npu_ops.npu_quant_lightning_indexer.call_args.kwargs
        self.assertEqual(read["key"].dtype, INT8)
        self.assertEqual(read["key_dequant_scale"].dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
