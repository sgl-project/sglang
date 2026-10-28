"""CPU regressions for dynamic INT8 MLAProlog dispatch and tensor contracts.

Only vendor operators are replaced; weight preparation and indexer adaptation
run through the production code. These tests do not validate NPU numerics.
"""

import importlib
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

# Initialize quantization before importing its NPU kernels (avoids a cycle).
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.attention import mla_preprocess as mla
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8DynamicLinearMethod,
    NPUW8A8Int8LinearMethod,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Projection(torch.nn.Module):
    """Small CPU projection with the post-load ModelSlim weight layout."""

    def __init__(self, input_size, output_size, kernel):
        super().__init__()
        self.input_size = input_size
        self.weight = torch.arange(input_size * output_size, dtype=torch.int8).view(
            input_size, output_size
        )
        self.weight_scale = torch.arange(1, output_size + 1).bfloat16() / 16
        self.scheme = SimpleNamespace(kernel=kernel)
        self.quant_method = SimpleNamespace(
            quantization_config=SimpleNamespace(get_name=lambda: "modelslim")
        )

    def forward(self, x):
        # Fail cleanly if preprocessing incorrectly frees a shared weight buffer.
        if self.weight.untyped_storage().nbytes() == 0:
            raise AssertionError("Source projection weight was released")
        weight = self.weight.float() * self.weight_scale
        return (x.float() @ weight).to(x.dtype), None


class TestDynamicInt8MLAProlog(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The attention wrapper imports vendor packages even on a CPU host.
        vendor_norm = ModuleType("sgl_kernel_npu.norm.fused_split_qk_norm")
        vendor_norm.fused_split_qk_norm = Mock(
            side_effect=AssertionError("Unexpected unfused NPU operator")
        )
        with patch.dict(
            "sys.modules",
            {
                "torch_npu": ModuleType("torch_npu"),
                "sgl_kernel_npu.norm.fused_split_qk_norm": vendor_norm,
            },
        ):
            cls.attention = importlib.import_module(
                "sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu"
            )

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.k_cache = torch.zeros(1, 4, 1, 3, dtype=torch.bfloat16)
        self.kr_cache = torch.zeros(1, 4, 1, 2, dtype=torch.bfloat16)
        self.pool = SimpleNamespace(
            index_head_dim=4,
            dsa_kv_cache_store_fp8=False,
            get_kv_buffer=lambda _: (self.k_cache, self.kr_cache),
        )
        self.disagg = SimpleNamespace(disaggregation_mode="decode")
        for module in (mla, self.attention):
            self.stack.enter_context(
                patch.object(module, "get_disagg", return_value=self.disagg)
            )
        self.stack.enter_context(
            patch.object(mla, "get_token_to_kv_pool", return_value=self.pool)
        )
        self.stack.enter_context(patch.object(mla, "is_npu_arch35", return_value=False))
        self.stack.enter_context(patch.object(mla, "is_fia_nz", return_value=False))
        self.stack.enter_context(
            patch.object(mla, "npu_format_cast", side_effect=lambda x: x.clone())
        )
        self.x = torch.tensor([[1, -2, 3, -4], [2, 1, -1, 3]], dtype=torch.bfloat16)
        self.positions = torch.arange(2)
        self.batch = SimpleNamespace(out_cache_loc=torch.tensor([0, 1]))
        self.quant_x = torch.tensor([[2, -4, 6, -8], [8, 4, -4, 12]], dtype=torch.int8)
        self.x_scale = torch.tensor([0.5, 0.25], dtype=torch.float64)
        self.query = torch.ones(2, 2, 3, dtype=torch.bfloat16)
        self.query_rope = torch.full((2, 2, 2), 2, dtype=torch.bfloat16)
        self.latent = torch.tensor([[1, -3], [4, 2]], dtype=torch.int8)
        self.latent_scale = torch.tensor([[0.125], [0.25]])
        self.vendor = ModuleType("torch_npu")
        self.vendor.npu_mla_prolog_v3 = Mock(
            return_value=(
                self.query,
                self.query_rope,
                None,
                self.latent,
                self.latent_scale,
            )
        )
        self.stack.enter_context(patch.dict("sys.modules", {"torch_npu": self.vendor}))
        self.stack.enter_context(
            patch.object(
                torch.ops,
                "npu",
                SimpleNamespace(
                    npu_dynamic_quant=Mock(return_value=(self.quant_x, self.x_scale))
                ),
            )
        )

    def make_model(self, kernel_type=NPUW8A8Int8DynamicLinearMethod):
        q_norm = RMSNorm(2, weight_dtype=torch.bfloat16, force_native=True)
        kv_norm = RMSNorm(3, weight_dtype=torch.bfloat16, force_native=True)
        # A present but zero bias is supported.
        q_norm.bias = torch.zeros(2, dtype=torch.bfloat16)
        return SimpleNamespace(
            fused_qkv_a_proj_with_mqa=_Projection(4, 7, kernel_type()),
            q_a_layernorm=q_norm,
            kv_a_layernorm=kv_norm,
            q_b_proj=_Projection(2, 8, kernel_type()),
            w_kc=torch.arange(12, dtype=torch.bfloat16).view(2, 2, 3),
            rotary_emb=SimpleNamespace(
                is_neox_style=False,
                cos_sin_cache=torch.tensor([[1, 0]] * 4, dtype=torch.bfloat16),
            ),
            layer_id=0,
            q_lora_rank=2,
            num_local_heads=2,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=3,
            quant_config=SimpleNamespace(ignore=["model.layers.0.self_attn.kv_b_proj"]),
            indexer=None,
        )

    def make_preprocess(self, model):
        return mla.NPUFusedMLAPreprocess(
            model.fused_qkv_a_proj_with_mqa,
            model.q_a_layernorm,
            model.kv_a_layernorm,
            model.q_b_proj,
            model.w_kc,
            model.rotary_emb,
            model.layer_id,
            model.num_local_heads,
            model.qk_nope_head_dim,
            model.qk_rope_head_dim,
            model.v_head_dim,
            model.quant_config,
        )

    def test_dynamic_dispatch_supplies_int8_operator_contract(self):
        """Dynamic ModelSlim must not access static input_offset parameters."""
        model = self.make_model()
        preprocess = self.make_preprocess(model)
        result = preprocess(self.positions, self.x, self.batch, None)
        args = self.vendor.npu_mla_prolog_v3.call_args.kwargs
        self.assertEqual(args["weight_quant_mode"], 2)
        self.assertEqual(args["kv_cache_quant_mode"], 0)
        self.assertEqual(args["query_quant_mode"], 0)
        self.assertEqual(args["token_x"].dtype, torch.int8)
        self.assertEqual(args["dequant_scale_x"].dtype, torch.float32)
        self.assertEqual(args["dequant_scale_x"].shape, (2, 1))
        for scale, channels in (
            ("dequant_scale_w_dq", 2),
            ("dequant_scale_w_dkv_kr", 5),
            ("dequant_scale_w_uq_qr", 8),
        ):
            self.assertEqual(args[scale].dtype, torch.float32)
            self.assertEqual(args[scale].shape, (1, channels))
        for weight, scale, expected in (
            (
                "weight_dq",
                "dequant_scale_w_dq",
                model.fused_qkv_a_proj_with_mqa(self.x)[0][:, :2],
            ),
            (
                "weight_dkv_kr",
                "dequant_scale_w_dkv_kr",
                model.fused_qkv_a_proj_with_mqa(self.x)[0][:, 2:],
            ),
        ):
            actual = (
                (args["token_x"].float() @ args[weight].float())
                * args["dequant_scale_x"]
                * args[scale]
            )
            torch.testing.assert_close(actual.to(torch.bfloat16), expected)
        torch.testing.assert_close(
            args["dequant_scale_w_uq_qr"],
            model.q_b_proj.weight_scale.float().view(1, -1),
        )
        self.assertIs(result[0], self.query_rope)
        self.assertIs(result[1], self.kr_cache)
        self.assertIs(result[2], self.query)
        self.assertIs(result[3], self.k_cache)
        self.assertIs(result[4], self.latent)
        torch.testing.assert_close(result[-1], self.latent_scale.flatten())

    def test_static_and_mixed_int8_do_not_select_dynamic_mlaprolog(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                model = self.make_model(NPUW8A8Int8LinearMethod)
                if mixed:
                    model.q_b_proj.scheme.kernel = NPUW8A8Int8DynamicLinearMethod()
                preprocess = self.make_preprocess(model)
                self.assertFalse(preprocess.uses_mlaprolog())
                with self.assertRaisesRegex(RuntimeError, "Unsupported MLAProlog"):
                    preprocess.mlaprolog_preprocess_weight()

    def test_weight_split_preserves_dequantized_projections_and_source_storage(self):
        model = self.make_model()
        qkv = model.fused_qkv_a_proj_with_mqa
        original = qkv.weight.clone()
        original_bytes = qkv.weight.untyped_storage().nbytes()
        preprocess = self.make_preprocess(model)
        preprocess.mlaprolog_preprocess_weight()
        self.assertEqual(qkv.weight.untyped_storage().nbytes(), original_bytes)
        torch.testing.assert_close(qkv.weight, original)
        restored = torch.cat(
            (
                preprocess.q_a_proj_weight.float() * preprocess.qkv_a_proj_scale_q,
                preprocess.kv_a_proj_weight.float() * preprocess.qkv_a_proj_scale_kv,
            ),
            dim=1,
        )
        torch.testing.assert_close(restored, original.float() * qkv.weight_scale)
        torch.testing.assert_close(
            preprocess.q_b_proj_weight.float() * preprocess.q_b_proj_scale,
            model.q_b_proj.weight.float() * model.q_b_proj.weight_scale,
        )

    def test_unsupported_weight_contracts_fail_before_operator_execution(self):
        for invalid in ("neox", "bias", "shape", "dtype", "scale"):
            with self.subTest(invalid=invalid):
                model = self.make_model()
                preprocess = self.make_preprocess(model)
                if invalid == "neox":
                    model.rotary_emb.is_neox_style = True
                    error = "interleaved RoPE"
                elif invalid == "bias":
                    model.q_a_layernorm.bias[0] = 1
                    error = "RMSNorm bias"
                elif invalid == "shape":
                    model.q_b_proj.weight = model.q_b_proj.weight.t()
                    error = "weight layout"
                elif invalid == "dtype":
                    model.fused_qkv_a_proj_with_mqa.weight = (
                        model.fused_qkv_a_proj_with_mqa.weight.float()
                    )
                    error = "weight layout"
                else:
                    model.q_b_proj.weight_scale = model.q_b_proj.weight_scale[:1]
                    error = "per-channel scales"
                with self.assertRaisesRegex(RuntimeError, error):
                    preprocess(self.positions, self.x, self.batch, None)

    def test_requires_bf16_activations_cache_and_final_query(self):
        for invalid in ("input", "kv", "kr", "w_kc", "query"):
            with self.subTest(invalid=invalid):
                model = self.make_model()
                preprocess = self.make_preprocess(model)
                x = self.x.float() if invalid == "input" else self.x
                k_cache = (
                    self.k_cache.to(torch.int8) if invalid == "kv" else self.k_cache
                )
                kr_cache = (
                    self.kr_cache.to(torch.int8) if invalid == "kr" else self.kr_cache
                )
                if invalid == "w_kc":
                    preprocess.w_kc = preprocess.w_kc.float()
                outputs = (
                    self.query.to(torch.int8) if invalid == "query" else self.query,
                    self.query_rope,
                    None,
                    self.latent,
                    self.latent_scale,
                )
                with (
                    patch.object(
                        self.pool, "get_kv_buffer", return_value=(k_cache, kr_cache)
                    ),
                    patch.object(
                        self.vendor, "npu_mla_prolog_v3", Mock(return_value=outputs)
                    ),
                    self.assertRaisesRegex(RuntimeError, "BF16"),
                ):
                    preprocess(self.positions, x, self.batch, None)

    def test_indexer_latent_and_decode_weight_lifetime(self):
        """BF16 indexers need the pre-quantization latent; shared-topk has none."""
        for kind in ("dynamic", "bf16", "none"):
            with self.subTest(indexer=kind):
                model = self.make_model()
                if kind != "none":
                    scheme = (
                        SimpleNamespace(kernel=NPUW8A8Int8DynamicLinearMethod())
                        if kind == "dynamic"
                        else None
                    )
                    model.indexer = SimpleNamespace(wq_b=SimpleNamespace(scheme=scheme))
                expected = model.q_a_layernorm(
                    model.fused_qkv_a_proj_with_mqa(self.x)[0][:, :2]
                )
                original_w_kc = model.w_kc.clone()
                original_bytes = model.w_kc.untyped_storage().nbytes()
                result = self.attention.npu_mla_preprocess(
                    model, self.x, self.positions, self.batch, None
                )
                self.assertEqual(model.w_kc.untyped_storage().nbytes(), original_bytes)
                torch.testing.assert_close(model.w_kc, original_w_kc)
                if kind == "bf16":
                    self.assertEqual(result[4].dtype, torch.bfloat16)
                    torch.testing.assert_close(result[4], expected)
                    self.assertIsNone(result[-1])
                else:
                    self.assertIs(result[4], self.latent)
                    torch.testing.assert_close(result[-1], self.latent_scale.flatten())


if __name__ == "__main__":
    unittest.main()
