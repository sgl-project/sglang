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
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

for _name in (
    "torch_npu",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
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
    from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
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
        # The pool sizer omits ``packed``; an INT8 cache is only ever packed.
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
    def _build_kwargs(
        self,
        dtype,
        *,
        is_dsa_model,
        requested="auto",
        resolved=None,
        combined=False,
        is_draft_worker=False,
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

    def test_dsa_int8_on_a2_builds_the_packed_record(self):
        self._use_npu(arch35=False)
        kwargs = self._build_kwargs(INT8, is_dsa_model=True, requested="int8")
        self.assertEqual(kwargs["kv_cache_dim"], PACKED_KV_CACHE_DIM)
        self.assertTrue(kwargs["kv_capability"].main_kv_packed)

    def test_int8_draft_pool_is_unpacked(self):
        """The draft inherits --kv-cache-dtype int8 but builds an unpacked BF16
        pool, which the target's pool sizer reserves at that width."""
        self._use_npu(arch35=False)
        resolved, dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype="int8",
            model=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=True,
            is_dflash=False,
            speculative_draft_attention_backend="ascend",
        )
        kwargs = self._build_kwargs(
            dtype,
            is_dsa_model=True,
            requested="int8",
            resolved=resolved,
            is_draft_worker=True,
        )
        self.assertEqual(kwargs["dtype"], torch.bfloat16)
        self.assertFalse(kwargs["kv_capability"].main_kv_packed)
        # No packed row width: the pool keeps separate latent and rope buffers.
        self.assertIsNone(kwargs["kv_cache_dim"])

    def test_dsa_int8_on_arch35_is_a_startup_error(self):
        """arch35 has no INT8 packed path; it must not fall back to a plain
        one-byte cache."""
        self._use_npu(arch35=True)
        with self.assertRaises(ValueError) as ctx:
            self._build_kwargs(INT8, is_dsa_model=True, requested="int8")
        self.assertIn("--kv-cache-dtype int8", str(ctx.exception))
        self.assertIn("fp8_e4m3", str(ctx.exception))

    def test_int8_with_the_combined_kv_buffer_is_a_startup_error(self):
        """The combined MLA buffer is read as an unquantized cache."""
        self._use_npu(arch35=False)
        with self.assertRaisesRegex(ValueError, "SGLANG_NPU_USE_FLASH_MLA"):
            self._build_kwargs(INT8, is_dsa_model=True, requested="int8", combined=True)

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
        # FP8 on arch35: FP8 index K plus one FP32 scale; BF16: no scale; an
        # INT8 main KV keeps the BF16 indexer, not a one-byte one.
        for dtype, arch35, per_token in (
            (FP8, True, 132),
            (torch.bfloat16, False, 256),
            (INT8, False, 256),
        ):
            with self.subTest(dtype=dtype):
                self._use_npu(arch35=arch35)
                self.assertEqual(
                    self._indexer_bytes(dtype), per_token * self.NUM_LAYERS
                )
                pool = _build_pool(dtype, arch35=arch35)
                self.assertEqual(self._pool_indexer_bytes_per_token(pool), per_token)


class TestConfigureInt8KvCacheDtype(CustomTestCase):
    def _configure(self, *, is_draft_worker):
        return configure_kv_cache_dtype(
            server_args_kv_cache_dtype="int8",
            model=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=is_draft_worker,
            is_dflash=False,
            speculative_draft_attention_backend="ascend",
        )

    def test_a_draft_does_not_inherit_the_int8_record(self):
        """A draft inherits --kv-cache-dtype; its pool falls back to the model
        dtype while the target keeps the packed INT8 record."""
        self.assertEqual(self._configure(is_draft_worker=False), (None, INT8))
        self.assertEqual(
            self._configure(is_draft_worker=True), ("auto", torch.bfloat16)
        )


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
        with patch.multiple(
            "sglang.srt.hardware_backend.npu.attention.ascend_backend.torch_npu",
            npu_kv_quant_sparse_flash_attention=quant_attn,
            npu_sparse_flash_attention=bf16_attn,
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


class TestInt8MainKvKeepsBf16Indexer(_MockedNpuOpsTestCase):
    def test_indexer_writes_and_reads_bf16_without_quantization(self):
        """The indexer quantizes whenever the pool has a scale buffer; an INT8
        main KV must leave it BF16 and off the quantized lightning indexer."""
        tokens, heads = 2, 4
        bf16 = torch.bfloat16
        indexer = SimpleNamespace(
            rotary_emb=_FakeRope(),
            wq_b=lambda q: (torch.zeros(tokens, heads * INDEX_HEAD_DIM, dtype=bf16),),
            wk=lambda x: (torch.zeros(tokens, INDEX_HEAD_DIM, dtype=bf16),),
            k_norm=lambda k: k,
            weights_proj=lambda x: (torch.zeros(tokens, heads),),
            n_heads=heads,
            head_dim=INDEX_HEAD_DIM,
            rope_head_dim=64,
            hidden_size=8,
            index_topk=4,
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
            patch.object(
                dsa_npu_indexer, "get_token_to_kv_pool", return_value=self.pool
            ),
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
        self.torch_npu.npu_quant_lightning_indexer.assert_not_called()
        self.torch_npu.npu_dynamic_quant.assert_not_called()
        written = self.torch_npu.npu_scatter_nd_update_.call_args.args[2]
        self.assertEqual(written.dtype, bf16)
        key = self.torch_npu.npu_lightning_indexer.call_args.kwargs["key"]
        self.assertEqual(key.dtype, bf16)


if __name__ == "__main__":
    unittest.main()
