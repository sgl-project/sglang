"""Cross-block MiniMax-H3 FP8 decoder dataflow on the JIT producers.

Two real ``ViT3DDecoder`` blocks with FP8 linears run once through the eager
block path and once through ``fused_decoder`` (rmsnorm_fp8 /
residual_rmsnorm_fp8 / in-place fused Q/K norm+RoPE / silu_mul_quant_fp8 /
residual_layernorm); the shared ``OnlineFP8Linear`` and the load-time installer
on the released 36-block topology are checked separately. Weight-update
rejection is covered by the CPU tests.
"""

import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.layers.quantization.online_fp8_linear import (
    OnlineFP8Linear,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    fused_decoder,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fp8 import (
    install_fp8_block_linears,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    ViT3DDecoder,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    prepare_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_cuda_sm_at_least((10, 0)),
    reason="MiniMax-H3 VAE FP8 producers are gated to SM100+",
)


@pytest.fixture(autouse=True)
def inference():
    torch.manual_seed(20260920)
    with torch.inference_mode():
        yield


@pytest.fixture
def stub_server_args(monkeypatch):
    """Only server configuration is stubbed; every attention, quantizer, GEMM,
    norm and activation under test is the production implementation."""
    args = type("Args", (), {})()
    args.attention_backend = None
    args.attention_backend_config = None
    args.enable_attention_backend_autotune = False
    args.enable_breakable_cuda_graph = False
    args.enable_layerwise_nvtx_marker = False
    args.kv_gather_degree = 1
    args.sp_split_auto = False
    monkeypatch.setattr(server_args_module, "_global_server_args", args)


def record_error(record_property, actual, expected, prefix="output"):
    difference = (actual.float() - expected.float()).abs()
    finite = difference[torch.isfinite(difference)]
    if finite.numel():
        record_property(f"{prefix}_max_abs", finite.max().item())
        record_property(f"{prefix}_rms_abs", finite.square().mean().sqrt().item())


def _released_decoder(num_layers: int) -> ViT3DDecoder:
    """The MiniMax-H3 ViT decoder topology with ``num_layers`` blocks."""
    return ViT3DDecoder(
        num_layers=num_layers,
        heads=32,
        dim_head=64,
        in_channels=24,
        patch_size=1,
        patch_size_t=1,
        norm_type="rms_norm",
        qk_norm_type="rms_norm",
        qk_norm_affine=False,
        ffn_activation_fn="silu",
        ffn_use_gated=True,
        rope_dim_ratio=0.75,
    )


def _install_by_hand(decoder: ViT3DDecoder) -> None:
    """What install_fp8_block_linears does, on a decoder with fewer blocks."""
    decoder.prepare_autocast_linear_weights(torch.float16)
    for block in decoder.transformer_blocks:
        block.attn.to_qkv = OnlineFP8Linear.from_linear(block.attn.to_qkv)
        block.attn.to_out = OnlineFP8Linear.from_linear(block.attn.to_out)
        block.ff.w1 = OnlineFP8Linear.from_linear(block.ff.w1)
        block.ff.w2 = OnlineFP8Linear.from_linear(block.ff.w2)
    decoder.fp8_installed = True
    assert fused_decoder.structure_reject_reason(decoder) is None
    decoder.fused_decoder_ready = True


def test_online_fp8_linear_matches_the_dequantized_math(record_property):
    """The shared linear quantizes weights per output channel and activations
    per token; its output matches the fp32 product of the dequantized operands,
    and the pre-quantized entry point is the same GEMM."""
    source = torch.nn.Linear(2048, 256, device="cuda", dtype=torch.float16).eval()
    linear = OnlineFP8Linear.from_linear(source)
    linear.to(dtype=torch.bfloat16)
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert linear.weight.shape == (256, 2048)
    assert linear.weight_scale.shape == (1, 256)
    assert linear.bias.dtype == torch.float16
    x = torch.randn(3, 5, 2048, device="cuda", dtype=torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        actual = linear(x)
    assert actual.dtype == torch.float16 and actual.shape == (3, 5, 256)
    q, scales = sglang_per_token_quant_fp8(x.reshape(-1, 2048).contiguous())
    dequantized_x = q.float() * scales
    dequantized_w = linear.weight.float() * linear.weight_scale.reshape(-1, 1)
    # Check conversion against the original FP16 weights, independently of the
    # GEMM oracle below. E4M3 rounding is bounded by half an ULP, including subnormals.
    source_weight = source.weight.float()
    source_scale = source_weight.abs().amax(dim=-1, keepdim=True) / 448
    torch.testing.assert_close(linear.weight_scale.t(), source_scale, atol=0, rtol=1e-6)
    weight_error_bound = 0.063 * source_weight.abs() + source_scale / 1024
    assert torch.all((dequantized_w - source_weight).abs() <= weight_error_bound).item()
    expected = (dequantized_x @ dequantized_w.t() + linear.bias.float()).reshape(
        3, 5, 256
    )
    record_error(record_property, actual, expected)
    torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=1e-2)
    with torch.autocast("cuda", dtype=torch.float16):
        prequantized = linear.forward_quantized(q, scales, (3, 5, 2048))
    torch.testing.assert_close(prequantized, actual, atol=0, rtol=0)
    with pytest.raises(ValueError):
        linear.forward_quantized(q, scales.half(), (3, 5, 2048))


@pytest.mark.parametrize("backend", ["TORCH_SDPA", "TORCH_CUDNN_SDPA"])
def test_two_real_decoder_blocks(backend, stub_server_args, record_property):
    """Same FP8 weights through eager blocks versus cross-block JIT dataflow."""
    with global_force_attn_backend_context_manager(AttentionBackendEnum[backend]):
        decoder = _released_decoder(num_layers=2).cuda().eval()
    for block in decoder.transformer_blocks:
        # Constructor scales are zero: leaving them untouched would hide all
        # attention/FFN errors behind the residual identity.
        block.scale1.uniform_(0.2, 0.8)
        block.scale2.uniform_(0.2, 0.8)
        block.norm1.weight.uniform_(0.8, 1.2)
        block.norm2.weight.uniform_(0.8, 1.2)
    _install_by_hand(decoder)
    value = torch.randn(2, 65, 2048, device="cuda")
    token_ids = torch.randn(2, 65, 3, device="cuda")
    # Stacked tiles share one rotary table; positions cover both batch rows.
    rope = prepare_rotary_pos_emb(
        decoder.pos_embed(token_ids[:1]), dtype=torch.float16, batch=2
    )

    with set_forward_context(0, None), torch.autocast("cuda", dtype=torch.float16):
        # Admission looks at the tensors only; no request-level gate is involved.
        assert fused_decoder.can_use_fused_decoder(decoder, value, rope)
        assert not fused_decoder.can_use_fused_decoder(decoder, value.half(), rope)
        with torch.enable_grad():
            assert not fused_decoder.can_use_fused_decoder(decoder, value, rope)
        reference = value
        for block in decoder.transformer_blocks:
            reference = block(reference, rope)
        reference = decoder.norm_out(reference)
        qk_weight = decoder._qk_unit_weight(value.device)
        assert qk_weight.dtype == torch.float16 and qk_weight.shape == (64,)
        assert decoder._qk_unit_weight(value.device) is qk_weight
        with patch.object(
            decoder,
            "forward_transformer_blocks",
            side_effect=AssertionError("unexpected eager fallback"),
        ):
            actual = decoder._run_blocks(value, rope)
    assert actual.dtype == torch.float32 and actual.shape == reference.shape
    record_error(record_property, actual, reference)
    relative_rms = (
        ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
    )
    cosine = F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item()
    record_property("relative_rms", relative_rms)
    record_property("cosine", cosine)
    # Fused producers remove intermediate FP16 rounding before FP8. This is
    # a close-contract comparison with identical deployed FP8 weights.
    assert relative_rms < 0.01
    assert cosine > 0.99995
    torch.testing.assert_close(actual, reference, atol=0.075, rtol=0.02)


def test_install_fp8_block_linears_swaps_the_released_144_linears(stub_server_args):
    """Install once, preserve non-target weights, and reject dtype changes."""
    with torch.device("cuda"):
        decoder = _released_decoder(num_layers=36).eval()
    assert install_fp8_block_linears(decoder) == 144
    assert decoder.fp8_installed and decoder.fused_decoder_ready
    swapped = [m for m in decoder.modules() if isinstance(m, OnlineFP8Linear)]
    assert len(swapped) == 144
    assert all(m.weight.dtype == torch.float8_e4m3fn for m in swapped)
    assert all(m.weight_scale.dtype == torch.float32 for m in swapped)
    assert all(m.bias.dtype == torch.float16 for m in swapped)
    assert decoder.proj_out.weight.dtype == torch.float32
    assert decoder.x_embedder.weight.dtype == torch.float32
    # A second call and the fp16 decode cast are no-ops; bf16 is refused.
    assert install_fp8_block_linears(decoder) == 0
    assert decoder.prepare_autocast_linear_weights(torch.float16) == 0
    with pytest.raises(ValueError, match="FP16 autocast"):
        decoder.prepare_autocast_linear_weights(torch.bfloat16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
