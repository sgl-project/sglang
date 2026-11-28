"""Unit tests for Ascend 950 (arch35) online MXFP8 draft & LM-head acceleration."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=15, suite="extra-a-test-npu")


class TestNpuMxfp8LmHeadModeParsing(unittest.TestCase):
    def test_mxfp8_lm_head_mode_parsing(self):
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            npu_arch35_mxfp8_lm_head_enabled,
        )

        cases = [
            ("draft", True, False),
            ("DRAFT", True, False),
            ("all", True, True),
            ("1", True, False),
            ("true", True, False),
            ("0", False, False),
            ("off", False, False),
            ("", False, False),
        ]
        for mode, exp_draft, exp_target in cases:
            with envs.SGLANG_NPU_ARCH35_MXFP8_LM_HEAD.override(mode):
                self.assertEqual(
                    npu_arch35_mxfp8_lm_head_enabled(is_draft=True),
                    exp_draft,
                    f"mode={mode!r} is_draft=True",
                )
                self.assertEqual(
                    npu_arch35_mxfp8_lm_head_enabled(is_draft=False),
                    exp_target,
                    f"mode={mode!r} is_draft=False",
                )


@unittest.skipUnless(
    is_npu() and torch.npu.is_available() and is_npu_arch35(),
    "Requires Ascend 950 (arch35) NPU",
)
class TestNpuSpecMxfp8Draft950(unittest.TestCase):
    def test_npu_mxfp8_linear_idempotency_and_fused_tuple_inputs(self):
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            NPUMXFP8LinearMethod,
        )

        torch.manual_seed(11)
        M, K, N = 32, 512, 256
        layer = nn.Module()
        w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="npu") * 0.1
        layer.weight = nn.Parameter(w_bf16.clone(), requires_grad=False)
        layer.bias = None

        qm = NPUMXFP8LinearMethod()
        qm.process_weights_after_loading(layer)
        w_ptr = layer.weight.data_ptr()
        s_ptr = layer.weight_scale_inv.data_ptr()
        # Second call must be a clean idempotent no-op.
        qm.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.data_ptr(), w_ptr)
        self.assertEqual(layer.weight_scale_inv.data_ptr(), s_ptr)

        # 1. Plain BF16 activation
        x = torch.randn(M, K, dtype=torch.bfloat16, device="npu") * 0.5
        y_ref = F.linear(x, w_bf16)
        y_mx = qm.apply(layer, x)
        self.assertGreater(
            F.cosine_similarity(
                y_mx.float().flatten(), y_ref.float().flatten(), dim=0
            ).item(),
            0.998,
        )

        # 2. Pre-quantized tuple from npu_swiglu_mx_quant
        gate_up = torch.randn(M, 2 * K, dtype=torch.bfloat16, device="npu") * 0.5
        sw_ref = F.silu(gate_up[:, :K].float()) * gate_up[:, K:].float()
        y_sw_ref = F.linear(sw_ref.to(torch.bfloat16), w_bf16)
        qx, xs = torch.ops.npu.npu_swiglu_mx_quant(
            gate_up, activate_dim=-1, activate_left=True, dst_type=torch.float8_e4m3fn
        )
        y_sw_mx = qm.apply(layer, (qx, xs))
        self.assertGreater(
            F.cosine_similarity(
                y_sw_mx.float().flatten(), y_sw_ref.float().flatten(), dim=0
            ).item(),
            0.998,
        )

        # 3. Pre-quantized tuple from npu_rms_norm_dynamic_mx_quant
        norm_w = torch.randn(K, dtype=torch.bfloat16, device="npu") * 0.2 + 1.0
        eps = 1e-6
        x32 = x.float()
        norm_ref = (
            x32
            * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
            * norm_w.float()
        ).to(torch.bfloat16)
        y_norm_ref = F.linear(norm_ref, w_bf16)
        qx_n, xs_n, _ = torch.ops.npu.npu_rms_norm_dynamic_mx_quant(
            x, norm_w, epsilon=eps, dst_type=torch.float8_e4m3fn
        )
        y_norm_mx = qm.apply(layer, (qx_n, xs_n))
        self.assertGreater(
            F.cosine_similarity(
                y_norm_mx.float().flatten(), y_norm_ref.float().flatten(), dim=0
            ).item(),
            0.998,
        )

        # 4. Pre-quantized tuple from npu_add_rms_norm_dynamic_mx_quant
        residual = torch.randn(M, K, dtype=torch.bfloat16, device="npu") * 0.5
        sum_ref = (residual.float() + x.float()).to(torch.bfloat16)
        s32 = sum_ref.float()
        add_norm_ref = (
            s32
            * torch.rsqrt(s32.pow(2).mean(dim=-1, keepdim=True) + eps)
            * norm_w.float()
        ).to(torch.bfloat16)
        y_add_ref = F.linear(add_norm_ref, w_bf16)
        qx_a, res_out, xs_a, _ = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant(
            residual, x, norm_w, epsilon=eps, dst_type=torch.float8_e4m3fn
        )
        y_add_mx = qm.apply(layer, (qx_a, xs_a))
        self.assertLess((res_out.float() - sum_ref.float()).abs().max().item(), 1e-2)
        self.assertGreater(
            F.cosine_similarity(
                y_add_mx.float().flatten(), y_add_ref.float().flatten(), dim=0
            ).item(),
            0.998,
        )

    def test_online_mxfp8_matmul_cache_sharing_and_invalidation(self):
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            _NPU_ARCH35_MXFP8_WEIGHT_CACHE,
            npu_arch35_online_mxfp8_matmul,
            npu_arch35_prepare_online_mxfp8_cache,
        )

        torch.manual_seed(22)
        holder = nn.Linear(512, 1024, bias=False, dtype=torch.bfloat16, device="npu")
        self.assertTrue(npu_arch35_prepare_online_mxfp8_cache(holder))
        entry1 = _NPU_ARCH35_MXFP8_WEIGHT_CACHE[holder.weight.data_ptr()]

        # A second holder sharing the same weight tensor (e.g., draft lm_head sharing
        # target lm_head.weight) must reuse the exact same cached quantized tensors.
        shared_holder = SimpleNamespace(weight=holder.weight)
        x = torch.randn(16, 512, dtype=torch.bfloat16, device="npu") * 0.5
        y_ref = F.linear(x, holder.weight)
        y_mx = npu_arch35_online_mxfp8_matmul(x, shared_holder)
        self.assertIs(_NPU_ARCH35_MXFP8_WEIGHT_CACHE[holder.weight.data_ptr()], entry1)
        self.assertGreater(
            F.cosine_similarity(
                y_mx.float().flatten(), y_ref.float().flatten(), dim=0
            ).item(),
            0.998,
        )

        # Replacing weight must automatically invalidate and rebuild cache.
        new_w = torch.randn(1024, 512, dtype=torch.bfloat16, device="npu") * 0.1
        holder.weight = nn.Parameter(new_w, requires_grad=False)
        y_ref2 = F.linear(x, holder.weight)
        y_mx2 = npu_arch35_online_mxfp8_matmul(x, holder)
        self.assertIsNot(
            _NPU_ARCH35_MXFP8_WEIGHT_CACHE[holder.weight.data_ptr()], entry1
        )
        self.assertGreater(
            F.cosine_similarity(
                y_mx2.float().flatten(), y_ref2.float().flatten(), dim=0
            ).item(),
            0.998,
        )

    def test_quantize_dspark_draft_model_preserves_qkv_bf16(self):
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            NPUMXFP8LinearMethod,
            npu_arch35_quantize_dspark_draft_model,
        )
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        class _DummyLinear(LinearBase):
            def __init__(self, in_f, out_f):
                nn.Module.__init__(self)
                self.quant_method = UnquantizedLinearMethod()
                self.weight = nn.Parameter(
                    torch.randn(out_f, in_f, dtype=torch.bfloat16, device="npu") * 0.1,
                    requires_grad=False,
                )
                self.bias = None

        layer = SimpleNamespace(
            mlp=SimpleNamespace(
                gate_up_proj=_DummyLinear(512, 1024),
                down_proj=_DummyLinear(512, 512),
            ),
            self_attn=SimpleNamespace(
                qkv_proj=_DummyLinear(512, 768),
                o_proj=_DummyLinear(512, 512),
            ),
        )
        model = SimpleNamespace(
            layers=[layer],
            fc=nn.Linear(1024, 512, bias=False, dtype=torch.bfloat16, device="npu"),
        )
        converted = npu_arch35_quantize_dspark_draft_model(model)
        self.assertEqual(converted, 4)
        self.assertIsInstance(layer.mlp.gate_up_proj.quant_method, NPUMXFP8LinearMethod)
        self.assertIsInstance(layer.mlp.down_proj.quant_method, NPUMXFP8LinearMethod)
        self.assertIsInstance(layer.self_attn.o_proj.quant_method, NPUMXFP8LinearMethod)
        # qkv_proj must stay BF16 UnquantizedLinearMethod to preserve draft KV precision.
        self.assertIs(
            type(layer.self_attn.qkv_proj.quant_method), UnquantizedLinearMethod
        )
        self.assertEqual(layer.self_attn.qkv_proj.weight.dtype, torch.bfloat16)
        self.assertIsNotNone(getattr(model.fc, "_npu_arch35_mxfp8_cache", None))
        self.assertEqual(npu_arch35_quantize_dspark_draft_model(model), 0)

        # MiniMax-style nested layer.decoder_layer.mlp layout
        mm_layer = SimpleNamespace(
            decoder_layer=SimpleNamespace(
                mlp=SimpleNamespace(
                    gate_up_proj=_DummyLinear(512, 1024),
                    down_proj=_DummyLinear(512, 512),
                )
            ),
            self_attn=SimpleNamespace(
                qkv_proj=_DummyLinear(512, 768),
                o_proj=_DummyLinear(512, 512),
            ),
        )
        mm_model = SimpleNamespace(layers=[mm_layer])
        self.assertEqual(npu_arch35_quantize_dspark_draft_model(mm_model), 3)
        self.assertIsInstance(
            mm_layer.decoder_layer.mlp.gate_up_proj.quant_method, NPUMXFP8LinearMethod
        )
        self.assertIs(
            type(mm_layer.self_attn.qkv_proj.quant_method), UnquantizedLinearMethod
        )

    def test_platform_hooks_and_dflash_candidate_logits(self):
        from sglang.srt.models.dflash import _project_candidate_logits
        from sglang.srt.platforms import current_platform

        torch.manual_seed(55)
        lm_head = nn.Linear(512, 1024, bias=False, dtype=torch.bfloat16, device="npu")
        lm_head.quant_method = None
        draft = SimpleNamespace(logits_processor=SimpleNamespace(), lm_head=lm_head)

        with envs.SGLANG_NPU_ARCH35_MXFP8_LM_HEAD.override("draft"):
            current_platform.post_load_draft_model(draft)
            self.assertTrue(draft.logits_processor._is_draft_lm_head)
            self.assertIsNotNone(getattr(lm_head, "_npu_arch35_mxfp8_cache", None))

            # DFlash _project_candidate_logits with padded vocab tail (num_org=1000 < 1024)
            h = torch.randn(8, 512, dtype=torch.bfloat16, device="npu") * 0.5
            logits = _project_candidate_logits(
                h, lm_head, num_org=1000, use_quant_head=False
            )
            self.assertEqual(tuple(logits.shape), (8, 1024))
            self.assertTrue(torch.isneginf(logits[:, 1000:]).all())
            ref = F.linear(h, lm_head.weight[:1000])
            self.assertGreater(
                F.cosine_similarity(
                    logits[:, :1000].float().flatten(), ref.float().flatten(), dim=0
                ).item(),
                0.998,
            )

        lm_head2 = nn.Linear(512, 1024, bias=False, dtype=torch.bfloat16, device="npu")
        lm_head2.quant_method = None
        draft2 = SimpleNamespace(logits_processor=SimpleNamespace(), lm_head=lm_head2)
        with envs.SGLANG_NPU_ARCH35_MXFP8_LM_HEAD.override("off"):
            current_platform.post_load_draft_model(draft2)
            self.assertIsNone(getattr(lm_head2, "_npu_arch35_mxfp8_cache", None))
            self.assertIsNone(
                current_platform.maybe_project_mxfp8_lm_head(h, lm_head2, is_draft=True)
            )


if __name__ == "__main__":
    unittest.main()
