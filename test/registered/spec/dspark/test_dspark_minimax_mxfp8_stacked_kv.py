import unittest
from types import MethodType, SimpleNamespace

import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.dspark_minimax import (
    DSparkMiniMaxAttention,
    DSparkMiniMaxDraftModel,
)
from sglang.srt.runtime_context import get_platform
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="4-gpu-b200")


class Projection:
    def __init__(self, method, weight, scales, bias):
        self.quant_method = method
        self.weight = weight
        self.weight_scale_inv_swizzled = scales
        self.bias = bias

    def __call__(self, x):
        return self.quant_method.apply(self, x, self.bias), None


@unittest.skipUnless(
    torch.cuda.is_available() and get_platform().is_blackwell,
    "MXFP8 FlashInfer requires Blackwell",
)
class TestMiniMaxStackedKv(CustomTestCase):
    def test_cutedsl(self):
        self._check_backend("flashinfer_cutedsl")

    def test_cutlass(self):
        self._check_backend("flashinfer_cutlass")

    @torch.inference_mode()
    def _check_backend(self, backend):
        """Aligned swizzled KV tiles preserve layer order and partial-RoPE K/V."""
        from flashinfer import block_scale_interleave

        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", fp8_gemm_runner_backend=backend)
        )
        method = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                use_mxfp8=True,
                weight_block_size=[1, 32],
            )
        )
        g = torch.Generator(device="cuda").manual_seed(42)
        for q_size, hidden in ((1024, 6144), (128, 160)):
            layers = []
            for _ in range(5):
                n = q_size + 256
                weight = torch.randn(n, hidden, device="cuda", generator=g).to(
                    torch.float8_e4m3fn
                )
                scales = torch.randint(
                    121,
                    129,
                    (n, hidden // 32),
                    device="cuda",
                    generator=g,
                    dtype=torch.uint8,
                )
                norm = GemmaRMSNorm(128, eps=1e-6).to(
                    device="cuda", dtype=torch.bfloat16
                )
                norm.weight.copy_(torch.randn(128, device="cuda", generator=g))
                attn = SimpleNamespace(
                    qkv_proj=Projection(
                        method, weight, block_scale_interleave(scales), None
                    ),
                    q_size=q_size,
                    kv_size=128,
                    num_kv_heads=1,
                    head_dim=128,
                    k_norm=norm,
                    rotary_emb=get_rope(
                        128,
                        rotary_dim=64,
                        max_position=4096,
                        base=10000.0,
                        is_neox_style=True,
                    ).cuda(),
                )
                for name in ("kv_proj_only", "apply_k_norm", "apply_k_rope"):
                    setattr(
                        attn,
                        name,
                        MethodType(getattr(DSparkMiniMaxAttention, name), attn),
                    )
                layers.append(SimpleNamespace(self_attn=attn))
            model = SimpleNamespace(layers=layers)
            stacked = DSparkMiniMaxDraftModel._stacked_ctx_kv_params(model)
            self.assertIsNotNone(stacked)
            for rows in (1, 8, 64, 512):
                with self.subTest(hidden=hidden, rows=rows):
                    x = torch.randn(
                        rows, hidden, device="cuda", dtype=torch.bfloat16, generator=g
                    )
                    positions = torch.arange(rows, device="cuda") + 17
                    ks, vs = DSparkMiniMaxDraftModel._project_ctx_kv_stacked(
                        model, ctx_hidden=x, positions=positions, stacked=stacked
                    )
                    for layer, k, v in zip(layers, ks, vs):
                        attn = layer.self_attn
                        rk, rv = attn.kv_proj_only(x)
                        rk = attn.apply_k_rope(positions, attn.apply_k_norm(rk))
                        torch.testing.assert_close(k.flatten(1), rk, rtol=0, atol=0)
                        torch.testing.assert_close(v.flatten(1), rv, rtol=0, atol=0)
            for attr, value in (("q_size", 64), ("kv_size", 32)):
                original = getattr(layers[0].self_attn, attr)
                setattr(layers[0].self_attn, attr, value)
                model._mxfp8_ctx_kv_cache = False
                self.assertIsNone(DSparkMiniMaxDraftModel._stacked_ctx_kv_params(model))
                setattr(layers[0].self_attn, attr, original)
            layers[0].self_attn.qkv_proj.bias = torch.zeros(n, device="cuda")
            model._mxfp8_ctx_kv_cache = False
            self.assertIsNone(DSparkMiniMaxDraftModel._stacked_ctx_kv_params(model))


if __name__ == "__main__":
    unittest.main()
