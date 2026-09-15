"""CPU tests for the DeepSeek-V4 ROCm C4 compressor GEMM fusion."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import sglang.srt.layers.attention.dsv4.compressor as compressor_module
import sglang.srt.layers.attention.dsv4.compressor_v2 as compressor_v2_module
import sglang.srt.models.deepseek_v4 as deepseek_v4
import sglang.srt.utils.offloader as offloader
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.compressor_v2 import (
    CompressorBackendMixin as CompressorBackendMixinV2,
)
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _WeightOnlyLinear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)


class _Compressor(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.wkv_gate = _WeightOnlyLinear(weight)

    def _compute_wkv_gate(
        self, x: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        return Compressor._compute_wkv_gate(self, x, weight)


class _Indexer(nn.Module):
    def __init__(self, compressor_weight: torch.Tensor) -> None:
        super().__init__()
        self.compressor = _Compressor(compressor_weight)


def _make_layer(
    main_weight: torch.Tensor, indexer_weight: torch.Tensor
) -> deepseek_v4.MQALayer:
    layer = deepseek_v4.MQALayer.__new__(deepseek_v4.MQALayer)
    nn.Module.__init__(layer)
    layer.compressor = _Compressor(main_weight)
    layer.indexer = _Indexer(indexer_weight)
    layer.compress_ratio = 4
    layer._fused_compressor_weight = None
    layer._fused_compressor_split_sizes = None
    layer.register_load_state_dict_post_hook(
        deepseek_v4.MQALayer._rebuild_compressor_gemm_fusion_after_state_load
    )
    return layer


def _make_post_load_model(
    layer: deepseek_v4.MQALayer,
) -> deepseek_v4.DeepseekV4ForCausalLM:
    layer.compressor.ape_converted = True
    layer.indexer.compressor.ape_converted = True
    model = deepseek_v4.DeepseekV4ForCausalLM.__new__(deepseek_v4.DeepseekV4ForCausalLM)
    nn.Module.__init__(model)
    model.model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[
            SimpleNamespace(self_attn=layer, refresh_mhc_norm_weight_cache=lambda: None)
        ],
    )
    return model


class TestDeepseekV4ROCmCompressorGemmFusion(CustomTestCase):
    def test_prepare_aliases_weights_and_survives_apply(self):
        main_value = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        indexer_value = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 100
        layer = _make_layer(main_value, indexer_value)

        self.assertTrue(layer.prepare_compressor_gemm_fusion())

        fused_weight = layer._fused_compressor_weight
        self.assertIsNotNone(fused_weight)
        self.assertEqual(layer._fused_compressor_split_sizes, (4, 2))
        torch.testing.assert_close(fused_weight, torch.cat((main_value, indexer_value)))
        self.assertEqual(
            set(layer.state_dict()),
            {"compressor.wkv_gate.weight", "indexer.compressor.wkv_gate.weight"},
        )

        layer.to(dtype=torch.float64)
        fused_weight = layer._fused_compressor_weight
        main_weight = layer.compressor.wkv_gate.weight
        indexer_weight = layer.indexer.compressor.wkv_gate.weight
        self.assertEqual(
            {
                main_weight.untyped_storage().data_ptr(),
                indexer_weight.untyped_storage().data_ptr(),
            },
            {fused_weight.untyped_storage().data_ptr()},
        )

    def test_state_dict_assign_rebuilds_fused_weight_aliases(self):
        layer = _make_layer(torch.ones(2, 3), torch.full((1, 3), 2.0))
        self.assertTrue(layer.prepare_compressor_gemm_fusion())
        state_dict = {
            "compressor.wkv_gate.weight": torch.full((2, 3), 3.0),
            "indexer.compressor.wkv_gate.weight": torch.full((1, 3), 5.0),
        }

        layer.load_state_dict(state_dict, assign=True)

        fused_weight = layer._fused_compressor_weight
        main_weight = layer.compressor.wkv_gate.weight
        indexer_weight = layer.indexer.compressor.wkv_gate.weight
        self.assertIsNotNone(fused_weight)
        self.assertEqual(
            {
                main_weight.untyped_storage().data_ptr(),
                indexer_weight.untyped_storage().data_ptr(),
            },
            {fused_weight.untyped_storage().data_ptr()},
        )
        torch.testing.assert_close(fused_weight, torch.cat(tuple(state_dict.values())))

    def test_post_load_fusion_is_default_off_and_rocm_opt_in(self):
        env_name = "SGLANG_OPT_DSV4_C4_COMPRESSOR_GEMM_FUSION"
        layer = _make_layer(torch.ones(2, 3), torch.full((1, 3), 2.0))
        model = _make_post_load_model(layer)

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(deepseek_v4, "_is_hip", True),
            patch.object(deepseek_v4, "_FP8_WO_A_GEMM", False),
        ):
            os.environ.pop(env_name, None)
            model.post_load_weights()
        self.assertIsNone(layer._fused_compressor_weight)

        with (
            envs.SGLANG_OPT_DSV4_C4_COMPRESSOR_GEMM_FUSION.override(True),
            patch.object(deepseek_v4, "_is_hip", True),
            patch.object(deepseek_v4, "_FP8_WO_A_GEMM", False),
        ):
            model.post_load_weights()
        self.assertIsNotNone(layer._fused_compressor_weight)

    def test_ipc_weight_cache_keeps_unfused_weights_without_allocation(self):
        layer = _make_layer(torch.ones(2, 3), torch.full((1, 3), 2.0))

        with patch.object(
            torch,
            "cat",
            side_effect=AssertionError("IPC fallback must not allocate packed storage"),
        ):
            IpcModelLoader._rebuild_stale_views(layer)

        self.assertIsNone(layer._fused_compressor_weight)

    def test_prepare_fails_closed_for_unsupported_setup(self):
        with patch.object(
            torch,
            "cat",
            side_effect=AssertionError("unsupported setup must not pack weights"),
        ):
            layer = _make_layer(torch.ones(4, 3), torch.ones(2, 3))
            layer.compress_ratio = 128
            self.assertFalse(layer.prepare_compressor_gemm_fusion())

            layer = _make_layer(torch.ones(4, 3), torch.ones(2, 3))
            layer.compressor.wkv_gate.weight = None
            self.assertFalse(layer.prepare_compressor_gemm_fusion())

            layer = _make_layer(torch.ones(4, 3), torch.ones(2, 3).to_sparse())
            self.assertFalse(layer.prepare_compressor_gemm_fusion())

            layer = _make_layer(torch.ones(4, 3), torch.ones(2, 3))
            with patch.object(offloader, "get_offloader", return_value=object()):
                self.assertFalse(layer.prepare_compressor_gemm_fusion())

    def test_fused_projection_uses_one_gemm_and_preserves_order(self):
        main_weight = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        )
        indexer_weight = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        layer = _make_layer(main_weight, indexer_weight)
        self.assertTrue(layer.prepare_compressor_gemm_fusion())
        hidden_states = torch.tensor(
            [[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]], dtype=torch.float32
        )
        gemm_calls = []

        def cpu_linear_bf16_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            gemm_calls.append((x, weight))
            return torch.matmul(x, weight.T).to(torch.float32)

        with patch.object(
            compressor_module,
            "linear_bf16_fp32",
            side_effect=cpu_linear_bf16_fp32,
        ):
            main_score, indexer_score = layer._compute_fused_compressor_kv_scores(
                hidden_states, SimpleNamespace(attn_cp_metadata=None)
            )

        torch.testing.assert_close(main_score, hidden_states @ main_weight.T)
        torch.testing.assert_close(indexer_score, hidden_states @ indexer_weight.T)
        self.assertEqual(len(gemm_calls), 1)
        self.assertEqual(gemm_calls[0][1].shape, (3, 3))

    def test_fused_projection_falls_back_for_dsa_prefill_cp(self):
        layer = _make_layer(torch.ones(2, 3), torch.ones(1, 3))
        self.assertTrue(layer.prepare_compressor_gemm_fusion())

        with (
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=True),
            patch.object(
                layer.compressor,
                "_compute_wkv_gate",
                side_effect=AssertionError("CP must use the original projections"),
            ),
        ):
            scores = layer._compute_fused_compressor_kv_scores(
                torch.empty(2, 3), object()
            )

        self.assertEqual(scores, (None, None))

    def test_v2_backend_consumes_precomputed_score(self):
        backend = CompressorBackendMixinV2()
        backend.enable_deepseek_v4_fp4_indexer = False
        backend.forward_metadata = SimpleNamespace(
            core_metadata=SimpleNamespace(c4_out_loc=torch.tensor([0]))
        )
        backend.token_to_kv_pool = SimpleNamespace(
            uniform_fp8=False,
            get_index_k_page_size=lambda: 1,
            get_index_k_with_scale_buffer=lambda _layer_id: torch.empty(4),
        )
        captured = {}
        backend._forward_compress_all_in_one = lambda **kwargs: captured.update(kwargs)

        precomputed_score = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        compressor = SimpleNamespace(
            compute_kv_score=lambda *_args: (_ for _ in ()).throw(
                AssertionError("projection must be skipped")
            ),
            _materialize_kv_score=lambda score, _batch: score,
            get_state_pool=lambda _backend: SimpleNamespace(
                kv_score_buffer=SimpleNamespace(kv_score=torch.empty(0))
            ),
            ratio=4,
            is_in_indexer=True,
            head_dim=2,
            ape=torch.empty(4, 4),
            norm=nn.Identity(),
            freqs_cis=torch.empty(0),
            rotate=True,
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False)
        )

        with patch(
            "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
            return_value=False,
        ):
            compressor_v2_module.CompressorBackendMixin.forward_unified(
                backend,
                torch.empty(1, 3),
                forward_batch,
                0,
                compressor,
                kv_score_input=precomputed_score,
            )

        self.assertIs(captured["kv_score_input"], precomputed_score)


if __name__ == "__main__":
    unittest.main()
