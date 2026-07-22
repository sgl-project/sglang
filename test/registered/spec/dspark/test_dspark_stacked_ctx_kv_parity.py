"""Parity: dspark stacked ctx-KV write vs the per-layer loop.

Accuracy tests cannot catch a broken ctx-KV write (spec decoding stays correct
regardless of draft KV; only accept length drops), so compare the two paths
directly and check the fallbacks return None.
"""

import types
import unittest

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.dflash import DFlashAttention
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

DEVICE = torch.device("cuda")
HEAD_DIM = 64
NUM_KV_HEADS = 2
NUM_Q_HEADS = 4
HIDDEN = 128
EPS = 1e-6


class _MockQKV:
    """Fused-QKV stand-in that satisfies can_dflash_slice_qkv_weight."""

    def __init__(self, weight, bias, quantized=False):
        self.quant_method = object() if quantized else UnquantizedLinearMethod()
        self.weight = weight
        self.bias = bias


def _make_attn(rope, *, eps=EPS, has_bias=False, quantized=False, g=None):
    attn = types.SimpleNamespace()
    attn.num_kv_heads = NUM_KV_HEADS
    attn.head_dim = HEAD_DIM
    attn.q_size = NUM_Q_HEADS * HEAD_DIM
    attn.kv_size = NUM_KV_HEADS * HEAD_DIM
    out = attn.q_size + 2 * attn.kv_size
    weight = torch.randn(out, HIDDEN, device=DEVICE, dtype=torch.float32, generator=g)
    bias = (
        torch.randn(out, device=DEVICE, dtype=torch.float32, generator=g)
        if has_bias
        else None
    )
    attn.qkv_proj = _MockQKV(weight, bias, quantized=quantized)
    k_norm = RMSNorm(HEAD_DIM, eps=eps).to(DEVICE)
    with torch.no_grad():
        # Distinct per layer so a wrong layer order fails parity.
        k_norm.weight.copy_(torch.randn(HEAD_DIM, device=DEVICE, generator=g))
    attn.k_norm = k_norm
    attn.rotary_emb = rope
    for name in ("kv_proj_only", "apply_k_norm", "apply_k_rope"):
        setattr(attn, name, types.MethodType(getattr(DFlashAttention, name), attn))
    return attn


def _make_model(rope, num_layers, **kw):
    layers = [
        types.SimpleNamespace(self_attn=_make_attn(rope, **kw))
        for _ in range(num_layers)
    ]
    model = types.SimpleNamespace(layers=layers)
    for name in ("_stacked_ctx_kv_params", "_project_ctx_kv_stacked"):
        setattr(model, name, types.MethodType(getattr(DSparkDraftMixin, name), model))
    return model


def _per_layer_reference(model, ctx_hidden, positions):
    ks, vs = [], []
    for layer in model.layers:
        attn = layer.self_attn
        k, v = attn.kv_proj_only(ctx_hidden)
        k = attn.apply_k_norm(k)
        k = attn.apply_k_rope(positions, k)
        ks.append(k.view(-1, attn.num_kv_heads, attn.head_dim))
        vs.append(v.view(-1, attn.num_kv_heads, attn.head_dim))
    return ks, vs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDSparkStackedCtxKvParity(CustomTestCase):
    def setUp(self):
        super().setUp()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.rope = get_rope(
            HEAD_DIM,
            rotary_dim=HEAD_DIM,
            max_position=4096,
            base=10000.0,
            is_neox_style=True,
        ).to(DEVICE)

    def _check_parity(self, *, num_layers=4, tokens=5, has_bias=False, dtype):
        g = torch.Generator(device=DEVICE).manual_seed(0)
        model = _make_model(self.rope, num_layers, has_bias=has_bias, g=g)
        for layer in model.layers:
            attn = layer.self_attn
            attn.qkv_proj.weight = attn.qkv_proj.weight.to(dtype)
            if attn.qkv_proj.bias is not None:
                attn.qkv_proj.bias = attn.qkv_proj.bias.to(dtype)
            attn.k_norm.to(dtype)
        ctx_hidden = torch.randn(
            tokens, HIDDEN, device=DEVICE, dtype=dtype, generator=g
        )
        positions = torch.arange(tokens, device=DEVICE)

        ref_k, ref_v = _per_layer_reference(model, ctx_hidden, positions)
        stacked = model._stacked_ctx_kv_params()
        self.assertIsNotNone(stacked)
        k_all, v_all = model._project_ctx_kv_stacked(
            ctx_hidden=ctx_hidden, positions=positions, stacked=stacked
        )

        # fp32 tol covers only fused-vs-manual RMSNorm rounding; bf16 is looser.
        rtol, atol = (2e-4, 2e-4) if dtype == torch.float32 else (2e-2, 2e-2)
        for i in range(num_layers):
            torch.testing.assert_close(k_all[i], ref_k[i], rtol=rtol, atol=atol)
            torch.testing.assert_close(v_all[i], ref_v[i], rtol=rtol, atol=atol)

    def test_parity_fp32(self):
        self._check_parity(dtype=torch.float32)

    def test_parity_bf16(self):
        self._check_parity(dtype=torch.bfloat16)

    def test_parity_with_bias(self):
        self._check_parity(dtype=torch.float32, has_bias=True)

    def test_fallback_quantized_layer(self):
        g = torch.Generator(device=DEVICE).manual_seed(0)
        model = _make_model(self.rope, 3, g=g)
        model.layers[1].self_attn.qkv_proj.quant_method = object()
        self.assertIsNone(model._stacked_ctx_kv_params())

    def test_fallback_eps_mismatch(self):
        g = torch.Generator(device=DEVICE).manual_seed(0)
        model = types.SimpleNamespace(
            layers=[
                types.SimpleNamespace(self_attn=_make_attn(self.rope, eps=1e-6, g=g)),
                types.SimpleNamespace(self_attn=_make_attn(self.rope, eps=1e-5, g=g)),
            ]
        )
        model._stacked_ctx_kv_params = types.MethodType(
            DSparkDraftMixin._stacked_ctx_kv_params, model
        )
        self.assertIsNone(model._stacked_ctx_kv_params())

    def test_fallback_inconsistent_bias(self):
        g = torch.Generator(device=DEVICE).manual_seed(0)
        model = types.SimpleNamespace(
            layers=[
                types.SimpleNamespace(
                    self_attn=_make_attn(self.rope, has_bias=True, g=g)
                ),
                types.SimpleNamespace(
                    self_attn=_make_attn(self.rope, has_bias=False, g=g)
                ),
            ]
        )
        model._stacked_ctx_kv_params = types.MethodType(
            DSparkDraftMixin._stacked_ctx_kv_params, model
        )
        self.assertIsNone(model._stacked_ctx_kv_params())


if __name__ == "__main__":
    unittest.main()
