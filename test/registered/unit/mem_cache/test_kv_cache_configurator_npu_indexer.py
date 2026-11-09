"""CPU regressions for NPU DSA indexer pool layer ownership."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.npu import memory_pool_npu
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _quantize_npu_indexer_activation,
)
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.kv_cache_dtype import (
    QUANT_MODE_TOKEN_INT8,
    resolve_indexer_quant_mode,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestNPUIndexerLayerLayout(CustomTestCase):
    def test_disabled_int8_indexer_skips_metadata_and_mode_validation(self):
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.server_args = SimpleNamespace(indexer_kv_cache_dtype=None)
        configurator.is_draft_worker = False
        configurator.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV32ForCausalLM"], index_topk=2048
            )
        )
        configurator.layer_info = SimpleNamespace(
            start_layer=0, end_layer=3, num_effective_layers=3
        )
        with (
            patch.object(kcc, "_is_npu", True),
            patch.object(kcc, "_should_elide_dsa_index_k", return_value=False),
            patch.object(kcc, "get_memory", side_effect=AssertionError("get_memory")),
            patch.object(kcc, "get_disagg", side_effect=AssertionError("get_disagg")),
            patch.object(
                kcc, "get_parallel", side_effect=AssertionError("get_parallel")
            ),
        ):
            configurator._resolve_npu_indexer_layers()
        self.assertEqual(configurator.npu_indexer_layer_ids, (0, 1, 2))
        self.assertEqual(configurator.npu_int8_indexer_layer_ids, frozenset())

    def _resolve_int8_layers(
        self,
        description=None,
        *,
        local_description=None,
        cache_layout=None,
        **disagg_overrides,
    ):
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.model = SimpleNamespace(
            quant_config=SimpleNamespace(quant_description=description)
        )
        configurator.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV32ForCausalLM"], index_topk=2048
            ),
            index_head_dim=128,
            model_path="org/remote-checkpoint",
            _find_quant_modelslim_config=Mock(return_value=local_description),
        )
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.modelslim_cache_layout = cache_layout
        configurator.server_args = SimpleNamespace(indexer_kv_cache_dtype="int8")
        configurator.is_draft_worker = False
        configurator.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        configurator.pp_size = 1
        configurator.layer_info = SimpleNamespace(
            start_layer=0, end_layer=3, num_effective_layers=3
        )
        disagg = dict(
            disaggregation_mode="decode",
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_retraction_backup="none",
        )
        disagg.update(disagg_overrides)
        with (
            patch.object(kcc, "_is_npu", True),
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(
                    enable_hierarchical_cache=False, enable_hisparse=False
                ),
            ),
            patch.object(kcc, "get_disagg", return_value=SimpleNamespace(**disagg)),
            patch.object(
                kcc, "get_parallel", return_value=SimpleNamespace(attn_cp_size=1)
            ),
            patch.object(
                kcc.envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD, "get", return_value=False
            ),
        ):
            configurator._resolve_npu_indexer_layers()
        return (
            configurator.npu_int8_indexer_layer_ids,
            configurator.model_config._find_quant_modelslim_config,
        )

    def test_remote_checkpoint_uses_loaded_modelslim_metadata(self):
        result, local_lookup = self._resolve_int8_layers(
            {
                "model.layers.0.self_attn.indexer.quant_type": "BF16",
                "model.layers.1.self_attn.indexer.quant_type": "INT8_DYNAMIC",
                "model.layers.2.self_attn.indexer.quant_type": "BF16",
            }
        )
        self.assertEqual(result, frozenset({1}))
        local_lookup.assert_not_called()

    def test_missing_loaded_metadata_falls_back_to_local_lookup(self):
        result, local_lookup = self._resolve_int8_layers()
        self.assertEqual(result, frozenset({0, 1, 2}))
        local_lookup.assert_called_once_with()

    def test_local_metadata_selects_only_calibrated_layers(self):
        result, local_lookup = self._resolve_int8_layers(
            local_description={
                "model.layers.2.self_attn.indexer.quant_type": "INT8_DYNAMIC"
            }
        )
        self.assertEqual(result, frozenset({2}))
        local_lookup.assert_called_once_with()

    def test_modelslim_fp8_storage_is_rejected(self):
        for key_dtype, value_dtype in (
            (torch.float8_e4m3fn, torch.bfloat16),
            (torch.bfloat16, torch.float8_e4m3fn),
        ):
            with self.subTest(key_dtype=key_dtype, value_dtype=value_dtype):
                with self.assertRaisesRegex(ValueError, "FP8 KV cache"):
                    self._resolve_int8_layers(
                        cache_layout=SimpleNamespace(
                            key_dtype=key_dtype, value_dtype=value_dtype
                        )
                    )

    def test_decode_host_cache_options_are_rejected_without_hicache(self):
        for options, message in (
            (
                {"disaggregation_decode_enable_offload_kvcache": True},
                "--disaggregation-decode-enable-offload-kvcache",
            ),
            (
                {"disaggregation_decode_retraction_backup": "host_pool"},
                "--disaggregation-decode-retraction-backup=host_pool",
            ),
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._resolve_int8_layers(**options)

    def test_non_arch35_dsa_pool_uses_only_active_indexer_layers(self):
        """Shared layers must not consume indexer slots on non-arch35 NPU."""
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.modelslim_cache_layout = None
        configurator.device = "cpu"
        configurator.is_draft_worker = False
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.layer_info = SimpleNamespace(
            start_layer=10,
            end_layer=14,
            num_effective_layers=4,
        )
        configurator.model_config = SimpleNamespace(
            hf_config=object(),
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            index_head_dim=32,
        )
        configurator.server_args = SimpleNamespace(indexer_kv_cache_dtype=None)
        pool = object()

        with (
            patch.object(kcc, "_is_npu", True),
            patch.object(kcc, "is_deepseek_dsa", return_value=True),
            patch.object(
                kcc,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=2),
            ),
            patch.object(
                kcc,
                "get_schedule",
                return_value=SimpleNamespace(page_size=16),
            ),
            patch.object(
                kcc,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch.object(kcc, "_should_elide_dsa_index_k", return_value=True),
            patch.object(
                kcc,
                "dsa_layer_skips_topk",
                side_effect=lambda _config, layer_id: layer_id in (11, 13),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.memory_pool_npu.NPUMLATokenToKVPool",
                return_value=pool,
            ) as pool_class,
        ):
            configurator._resolve_npu_indexer_layers()
            with patch.object(
                kcc.KVCacheConfigurator,
                "_resolve_npu_indexer_layers",
                side_effect=AssertionError("layer selection must be reused"),
            ):
                result = configurator._build_ascend_mla_kv_pool(
                    max_total_num_tokens=64,
                    is_dsa_model=True,
                )

        self.assertIs(result, pool)
        self.assertEqual(pool_class.call_args.args, (64,))
        self.assertEqual(pool_class.call_args.kwargs["index_size"], 128)
        self.assertEqual(pool_class.call_args.kwargs["indexer_layer_ids"], (10, 12))
        self.assertEqual(pool_class.call_args.kwargs["int8_indexer_layer_ids"], ())
        self.assertEqual(pool_class.call_args.kwargs["indexer_quant_mode"], None)


class TestNPUIndexerMemorySizing(CustomTestCase):
    def test_npu_indexer_cell_size_matches_storage(self):
        from sglang.srt.model_executor import pool_configurator as pc

        configurator = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
        cases = [
            (None, torch.bfloat16, 256),
            ("int8", torch.bfloat16, 130),
            (None, torch.float8_e4m3fn, 132),
        ]
        for indexer_dtype, kv_dtype, expected in cases:
            with self.subTest(indexer_dtype=indexer_dtype, kv_dtype=kv_dtype):
                kvc = SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(
                            architectures=["DeepseekV32ForCausalLM"],
                            index_topk=2048,
                            index_head_dim=128,
                        )
                    ),
                    kv_cache_dtype=kv_dtype,
                    npu_indexer_layer_ids=(0,),
                    npu_int8_indexer_layer_ids=frozenset({0}),
                    server_args=SimpleNamespace(
                        indexer_kv_cache_dtype=indexer_dtype,
                        enable_hisparse=True,
                    ),
                    is_draft_worker=False,
                )
                with (
                    patch.object(pc, "_is_npu", True),
                    patch.object(
                        pc,
                        "get_exec",
                        return_value=SimpleNamespace(
                            kernel=SimpleNamespace(enable_fp4_indexer=False)
                        ),
                    ),
                    patch.object(
                        pc,
                        "get_memory",
                        return_value=SimpleNamespace(enable_hisparse=False),
                    ),
                ):
                    actual = configurator._compute_dsa_indexer_cell_size(
                        kvc=kvc, num_layers=1
                    )
                self.assertEqual(actual, expected)

    def test_mixed_sizing_matches_cpu_allocations_with_dcp_padding(self):
        from sglang.srt.model_executor import pool_configurator as pc

        sizer = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
        page_size = 8
        tokens = 32
        cases = (
            # selected global IDs, allocated global IDs, bytes/token
            ((10, 12), (10, 12), 260),
            ((10,), (10, 12), 386),
            ((0,), (10, 12), 512),  # INT8 layers live on another PP rank.
            ((10, 11, 12, 13), (10, 11, 12, 13), 520),
            ((10,), (), 0),
        )
        for selected, allocated, expected in cases:
            for dcp_size in (1, 2, 4):
                with self.subTest(selected=selected, allocated=allocated, dcp=dcp_size):
                    description = {
                        f"model.layers.{i}.self_attn.indexer.quant_type": "INT8_DYNAMIC"
                        for i in selected
                    }
                    kvc = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
                    kvc.device = "cpu"
                    kvc.is_draft_worker = False
                    kvc.spec_algorithm = SimpleNamespace(is_none=lambda: True)
                    kvc.pp_size = 2
                    kvc.use_mla_backend = True
                    kvc.mambaish_config = None
                    kvc.kv_cache_dtype = torch.bfloat16
                    kvc.modelslim_cache_layout = None
                    kvc.model = SimpleNamespace(
                        quant_config=SimpleNamespace(quant_description=description)
                    )
                    kvc.model_config = SimpleNamespace(
                        hf_config=SimpleNamespace(
                            architectures=["DeepseekV32ForCausalLM"],
                            index_topk=2048,
                            index_head_dim=128,
                            indexer_types=[
                                "full" if i in allocated else "shared"
                                for i in range(14)
                            ],
                        ),
                        index_head_dim=128,
                        kv_lora_rank=16,
                        qk_rope_head_dim=8,
                    )
                    kvc.layer_info = SimpleNamespace(
                        start_layer=10, end_layer=14, num_effective_layers=4
                    )
                    kvc.server_args = SimpleNamespace(
                        indexer_kv_cache_dtype="int8", enable_hisparse=False
                    )
                    with (
                        get_context().override_server_args(
                            page_size=page_size,
                            enable_hierarchical_cache=False,
                            enable_hisparse=False,
                            disaggregation_mode="null",
                            enable_dsa_cache_layer_split=False,
                        ),
                        get_parallel().override(
                            tp_size=dcp_size,
                            moe_tp_size=dcp_size,
                            attn_dcp_size=dcp_size,
                            attn_dcp_rank=0,
                            attn_cp_size=1,
                            attn_tp_size=dcp_size,
                        ),
                        patch.object(kcc, "_is_npu", True),
                        patch.object(pc, "_is_npu", True),
                        patch.object(
                            kcc.envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD,
                            "get",
                            return_value=False,
                        ),
                        patch(
                            "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                            return_value=False,
                        ),
                        patch.object(pc, "dsa_shards_index_k", return_value=False),
                    ):
                        kvc._resolve_npu_indexer_layers()
                        self.assertEqual(kvc.npu_indexer_layer_ids, allocated)
                        self.assertEqual(
                            kvc.npu_int8_indexer_layer_ids,
                            frozenset(selected) & set(allocated),
                        )
                        # Metadata is resolved once; sizing/allocation use the saved IDs.
                        description.clear()
                        cell_size = (
                            pc.DefaultPoolConfigurator._compute_dsa_indexer_cell_size(
                                None, kvc=kvc, num_layers=4
                            )
                        )
                        self.assertEqual(cell_size, expected)
                        total_cell_size = sizer._compute_cell_size(
                            kvc=kvc, num_layers=4
                        )
                        self.assertEqual(
                            total_cell_size, 4 * 24 * 2 + expected * dcp_size
                        )
                        pool = kvc._build_ascend_mla_kv_pool(
                            max_total_num_tokens=tokens, is_dsa_model=True
                        )
                    padding = page_size * dcp_size
                    index_bytes = sum(buf.nbytes for buf in pool.index_k_buffer)
                    if pool.index_k_scale_buffer is not None:
                        index_bytes += sum(
                            buf.nbytes
                            for buf in pool.index_k_scale_buffer
                            if buf is not None
                        )
                    self.assertEqual(
                        index_bytes, (tokens * dcp_size + padding) * expected
                    )
                    self.assertEqual(
                        pool.get_kv_size_bytes(),
                        (tokens + page_size) * 4 * 24 * 2 + index_bytes,
                    )
                    for layer_id in allocated:
                        is_int8 = layer_id in selected
                        self.assertEqual(
                            pool.get_index_k_buffer(layer_id).dtype,
                            torch.int8 if is_int8 else torch.bfloat16,
                        )
                        self.assertEqual(
                            pool.get_index_k_quant_dtype(layer_id),
                            torch.int8 if is_int8 else None,
                        )
                        if pool.index_k_scale_buffer is not None:
                            scale = pool.get_index_k_scale_buffer(layer_id)
                            self.assertEqual(
                                None if scale is None else scale.dtype,
                                torch.float16 if is_int8 else None,
                            )

    def test_unsupported_modes_fail_during_configurator_initialization(self):
        for mode, layout, message in (
            (False, None, "speculative decoding"),
            (
                True,
                SimpleNamespace(
                    key_dtype=torch.float8_e4m3fn, value_dtype=torch.bfloat16
                ),
                "FP8 KV cache",
            ),
        ):
            with self.subTest(message=message):
                kvc = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
                kvc.is_draft_worker = False
                kvc.spec_algorithm = SimpleNamespace(
                    is_none=lambda: mode, is_dflash_family=lambda: False
                )
                kvc.spec_aux_config = SimpleNamespace()
                kvc.layer_info = SimpleNamespace(
                    start_layer=0,
                    end_layer=1,
                    num_effective_layers=1,
                    draft_owns_its_depth=False,
                )
                kvc.pp_size = 1
                kvc.use_mla_backend = True
                kvc.kv_cache_dtype = torch.bfloat16
                kvc.modelslim_cache_layout = None
                kvc.server_args = SimpleNamespace(indexer_kv_cache_dtype="int8")
                kvc.model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["DeepseekV32ForCausalLM"], index_topk=2048
                    ),
                    index_head_dim=128,
                )
                kvc.model = SimpleNamespace(
                    quant_config=SimpleNamespace(
                        get_name=lambda: "modelslim",
                        get_modelslim_kv_cache_layout=lambda *args, **kwargs: layout,
                        quant_description={"indexer_quant_type": "INT8_DYNAMIC"},
                    )
                )
                with (
                    get_context().override_server_args(),
                    patch.object(kcc, "_is_npu", True),
                    patch.object(kcc, "mambaish_config", return_value=None),
                    patch.object(kcc, "hybrid_gdn_config", return_value=None),
                    patch.object(kcc, "hybrid_kda_config", return_value=None),
                    patch.object(kcc.KVCacheConfigurator, "_resolve_kv_shard"),
                    patch.object(
                        kcc.KVCacheConfigurator, "_profile_available_bytes"
                    ) as profile,
                    patch.object(kcc.KVCacheConfigurator, "_init_pools") as allocate,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    kvc.__post_init__()
                profile.assert_not_called()
                allocate.assert_not_called()


class TestNPUIndexerInt8Cache(CustomTestCase):
    """--indexer-kv-cache-dtype=int8 may replace only the BF16 indexer cache."""

    def _build_pool(
        self, dtype, kv_cache_dim=None, indexer_quant_mode=None, hierarchical=False
    ):
        with (
            get_context().override_server_args(enable_hierarchical_cache=hierarchical),
            patch.object(
                memory_pool_npu,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0),
            ),
        ):
            return memory_pool_npu.NPUMLATokenToKVPool(
                256,
                page_size=128,
                dtype=dtype,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
                index_head_dim=128,
                kv_cache_dim=kv_cache_dim,
                indexer_quant_mode=indexer_quant_mode,
            )

    def test_int8_mode_rejects_hierarchical_cache_in_server_args(self):
        from sglang.srt.arg_groups import kv_cache_hook

        cfg = SimpleNamespace(
            indexer_kv_cache_dtype="int8", enable_hierarchical_cache=True
        )
        with (
            patch.object(kv_cache_hook, "resolving_view", return_value=cfg),
            self.assertRaisesRegex(ValueError, "enable-hierarchical-cache"),
        ):
            kv_cache_hook.handle_indexer_kv_cache_compatibility(object())

    def test_int8_mode_rejects_hierarchical_cache_in_pool(self):
        with self.assertRaisesRegex(ValueError, "enable-hierarchical-cache"):
            self._build_pool(
                torch.bfloat16,
                indexer_quant_mode=QUANT_MODE_TOKEN_INT8,
                hierarchical=True,
            )

    def test_int8_mode_swaps_only_the_bf16_indexer_cache(self):
        fp8_kv_cache_dim = 512 + 512 // 128 * 4 + 64 * 2
        cases = [
            # quant mode, kv dtype, kv_cache_dim, index K dtype, scale dtype
            (None, torch.bfloat16, None, torch.bfloat16, None),
            (QUANT_MODE_TOKEN_INT8, torch.bfloat16, None, torch.int8, torch.float16),
            # The FP8 KV cache already has an FP8 indexer; resolve rejects
            # int8 on top of it, and the pool must keep the FP8 indexer even
            # if the mode is passed in directly.
            (
                QUANT_MODE_TOKEN_INT8,
                torch.float8_e4m3fn,
                fp8_kv_cache_dim,
                torch.float8_e4m3fn,
                torch.float32,
            ),
        ]
        for quant_mode, dtype, kv_cache_dim, key_dtype, scale_dtype in cases:
            with self.subTest(quant_mode=quant_mode, dtype=dtype):
                pool = self._build_pool(dtype, kv_cache_dim, quant_mode)
                scale = pool.index_k_scale_buffer
                self.assertTrue(
                    all(buf.dtype == key_dtype for buf in pool.index_k_buffer)
                )
                self.assertEqual(None if scale is None else scale[0].dtype, scale_dtype)

    def test_resolve_indexer_quant_mode(self):
        self.assertIsNone(resolve_indexer_quant_mode(None, torch.bfloat16))
        self.assertIsNone(resolve_indexer_quant_mode(None, torch.float8_e4m3fn))
        self.assertEqual(
            resolve_indexer_quant_mode("int8", torch.bfloat16), QUANT_MODE_TOKEN_INT8
        )
        with self.assertRaises(ValueError):
            resolve_indexer_quant_mode("int8", torch.float8_e4m3fn)
        with self.assertRaises(ValueError):
            resolve_indexer_quant_mode("fp8_e4m3", torch.bfloat16)

    def test_empty_input_scale_matches_int8_cache(self):
        """A batch with no tokens still writes its (empty) scale to the pool,
        whose INT8 scale buffer rejects anything but FP16."""
        x = torch.empty(0, 128, dtype=torch.bfloat16)
        k, scale = _quantize_npu_indexer_activation(x, None, torch.int8, torch.float16)
        self.assertEqual((k.dtype, scale.dtype), (torch.int8, torch.float16))


if __name__ == "__main__":
    import unittest

    unittest.main()
