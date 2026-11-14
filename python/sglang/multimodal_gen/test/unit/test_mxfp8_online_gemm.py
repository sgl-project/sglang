# SPDX-License-Identifier: Apache-2.0
"""Online MXFP8 GEMM path (``MXFP8Config()`` on a bf16 checkpoint): load-time
block quant, the prequantized (e4m3, swizzled scales) input from the fused
SwiGLU kernel, the per-layer fallback to the per-channel fp8 path, and the Intel XPU
block-scaled ``scaled_mm`` path; plus serialized AutoRound MXFP8 checkpoints on XPU."""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

_XPU = torch.xpu.is_available()
_DEVICE = "cuda" if torch.cuda.is_available() else "xpu"
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() or _XPU), reason="CUDA or XPU required"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _layer(
    in_f: int, out_f: int, bias: bool, params_dtype: torch.dtype = torch.bfloat16
):
    from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config

    return RowParallelLinear(
        in_f,
        out_f,
        bias=bias,
        params_dtype=params_dtype,
        quant_config=MXFP8Config(),
        prefix="mlp.fc2",
    ).to(_DEVICE)


@requires_cuda
def test_mxfp8_linear_matches_bf16_and_accepts_prequantized(request) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuBLASLt MXFP8 block scaling requires Blackwell or newer")
    request.getfixturevalue("single_process_model_parallel")
    from sglang.kernels.ops.activation.swiglu_mxfp8_swizzled import silu_mul_mxfp8

    in_f, out_f, rows = 512, 384, 200
    layer = _layer(in_f, out_f, bias=False)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("cuda", torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.mxfp8 and layer.quant_method.accepts_mxfp8_input(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.float8_e8m0fnu

    x = torch.randn(rows, in_f, generator=g).to("cuda", torch.bfloat16)
    out, _ = layer(x)
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel

    hidden = torch.randn(rows, 2 * in_f, generator=g).to("cuda", torch.bfloat16)
    act = torch.nn.functional.silu(hidden[:, :in_f]) * hidden[:, in_f:]
    out_tensor, _ = layer(act)
    out_tuple, _ = layer(silu_mul_mxfp8(hidden))
    assert torch.equal(out_tensor, out_tuple)


@pytest.mark.skipif(not _XPU, reason="XPU required")
@pytest.mark.parametrize(
    "x_dtype, out_f, strided_x",
    [
        (torch.bfloat16, 384, False),
        (torch.float32, 384, False),
        # oneDNN's block-scaled GEMM has no N % 16 rule (unlike the cuBLASLt path).
        (torch.bfloat16, 100, False),
        # A row-strided slice, as from a fused QKV projection; the quant kernel
        # needs contiguous rows, so the helper copies only in this case.
        (torch.bfloat16, 384, True),
    ],
)
def test_xpu_mxfp8_linear_matches_bf16(request, x_dtype, out_f, strided_x) -> None:
    request.getfixturevalue("single_process_model_parallel")
    in_f, rows = 512, 200
    layer = _layer(in_f, out_f, bias=True)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("xpu", torch.bfloat16)
    bias = torch.randn(out_f, generator=g).to("xpu", torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.bias.copy_(bias)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.xpu_mxfp8 and not layer.quant_method.accepts_mxfp8_input(layer)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.float8_e8m0fnu
    assert layer.weight_scale.shape == (out_f, in_f // 32)

    x = torch.randn(rows, in_f, generator=g).to("xpu", x_dtype)
    if strided_x:
        x = torch.cat([x, x], dim=1)[:, :in_f]
        assert not x.is_contiguous()
    out, _ = layer(x)
    assert out.dtype == x_dtype
    ref = x.float() @ weight.float().t() + bias.float()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


def _assert_aligned_layer_falls_back(params_dtype: torch.dtype) -> None:
    layer = _layer(512, 384, bias=False, params_dtype=params_dtype)
    g = torch.Generator(device="cpu").manual_seed(1)
    weight = (torch.randn(384, 512, generator=g) * 0.02).to(_DEVICE, params_dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.mxfp8 and not layer.quant_method.accepts_mxfp8_input(layer)
    assert not layer.xpu_mxfp8
    x = torch.randn(200, 512, generator=g).to(_DEVICE, params_dtype)
    out, _ = layer(x)
    ref = x.float() @ weight.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


@requires_cuda
def test_pre_blackwell_aligned_layer_falls_back_to_channelwise(request) -> None:
    if torch.cuda.get_device_capability()[0] >= 10:
        pytest.skip("requires a pre-Blackwell GPU")
    request.getfixturevalue("single_process_model_parallel")
    _assert_aligned_layer_falls_back(torch.bfloat16)


def test_fp16_layer_falls_back_to_channelwise(request) -> None:
    """The swizzled quantizer takes bf16 only; an fp16 layer must not fail at load."""
    request.getfixturevalue("single_process_model_parallel")
    _assert_aligned_layer_falls_back(torch.float16)


def test_unaligned_layer_falls_back_to_channelwise(request) -> None:
    """The block-scaled GEMM needs K % 32 == 0; such a layer keeps the
    per-channel fp8 path and still answers a forward. K is 16 rather than a
    smaller odd size because the fallback's scaled GEMM still wants K % 16 == 0
    (ROCm rejects anything else outright)."""
    request.getfixturevalue("single_process_model_parallel")
    layer = _layer(16, 128, bias=True)
    layer.quant_method.process_weights_after_loading(layer)
    assert not layer.xpu_mxfp8
    assert not layer.mxfp8 and not layer.quant_method.accepts_mxfp8_input(layer)
    out, _ = layer(torch.randn(4, 16, device=_DEVICE, dtype=torch.bfloat16))
    assert out.shape == (4, 128)


# ---- AutoRound mx_fp W8A8 checkpoints (auto_round:llm_compressor) ----

# quantization_config of transformer/config.json in intel-ai/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound
_CHECKPOINT_CONFIG = {
    "bits": 8,
    "act_bits": 8,
    "data_type": "mx_fp",
    "act_data_type": "mx_fp",
    "group_size": 32,
    "act_group_size": 32,
    "sym": True,
    "act_sym": True,
    "act_dynamic": True,
    "enable_quanted_input": False,
    "low_gpu_mem_usage": True,
    "autoround_version": "0.17.0",
    "block_name_to_quantize": "blocks",
    "quant_method": "auto-round",
    "packing_format": "auto_round:llm_compressor",
}


def _on_xpu():
    from sglang.multimodal_gen.runtime.layers.quantization import auto_round_mxfp8

    return patch.object(auto_round_mxfp8.current_platform, "is_xpu", return_value=True)


def test_mx_fp_checkpoint_routes_to_the_mxfp8_config():
    """auto-round dispatches on the data type: mx_fp must not reach srt's AutoRound,
    which only supports int (GPTQ/AWQ) packing."""
    from sglang.multimodal_gen.runtime.layers.quantization.auto_round import (
        AutoRoundConfig,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.auto_round_mxfp8 import (
        AutoRoundMXFP8Config,
    )

    with _on_xpu():
        config = AutoRoundConfig.from_config(dict(_CHECKPOINT_CONFIG))
    assert isinstance(config, AutoRoundMXFP8Config)
    assert config.get_name() == "auto-round"
    assert config.block_names == ("blocks",)


@pytest.mark.parametrize(
    "override",
    [
        {"bits": 4, "act_bits": 4},
        {"data_type": "mx_fp_rceil"},
        {"act_dynamic": False},
        {"act_data_type": "int"},
        {"group_size": 128},
        {"act_sym": False},
        {"packing_format": "auto_round:auto_gptq"},
        {"extra_config": {"blocks.0.to_q": {"bits": 16}}},
        {"block_name_to_quantize": ""},
    ],
    ids=[
        "mxfp4",
        "mx_fp_rceil",
        "static_act",
        "int_act",
        "group_128",
        "asym_act",
        "gptq_packing",
        "extra_config",
        "no_blocks",
    ],
)
def test_other_mx_fp_layouts_are_rejected(override):
    """Only dynamic block-32 W8A8 MXFP8 is implemented; any other AutoRound mx_fp
    layout must fail at load with a message naming it, not run with wrong scales."""
    from sglang.multimodal_gen.runtime.layers.quantization.auto_round import (
        AutoRoundConfig,
    )

    with _on_xpu(), pytest.raises(ValueError, match="AutoRound"):
        AutoRoundConfig.from_config({**_CHECKPOINT_CONFIG, **override})


def test_mxfp8_checkpoint_off_xpu_is_rejected():
    from sglang.multimodal_gen.runtime.layers.quantization import auto_round_mxfp8

    with (
        patch.object(auto_round_mxfp8.current_platform, "is_xpu", return_value=False),
        pytest.raises(ValueError, match="require Intel XPU"),
    ):
        auto_round_mxfp8.AutoRoundMXFP8Config.from_config(dict(_CHECKPOINT_CONFIG))


@pytest.mark.parametrize(
    "key, value, expected",
    [
        (
            "block_name_to_quantize",
            "blocks, single_blocks",
            ("blocks", "single_blocks"),
        ),
        ("to_quant_block_names", ["blocks"], ("blocks",)),
        (
            "block_name_to_quantize",
            [["model.blocks"], ["vision.blocks"]],
            ("model.blocks", "vision.blocks"),
        ),
    ],
    ids=["comma_string", "srt_alias", "nested_lists"],
)
def test_block_name_forms_are_accepted(key, value, expected):
    """The block list comes as a string, srt's alias key, or nested lists
    (multimodal AutoRound exports)."""
    from sglang.multimodal_gen.runtime.layers.quantization.auto_round_mxfp8 import (
        AutoRoundMXFP8Config,
    )

    config = {
        k: v for k, v in _CHECKPOINT_CONFIG.items() if k != "block_name_to_quantize"
    }
    with _on_xpu():
        assert (
            AutoRoundMXFP8Config.from_config({**config, key: value}).block_names
            == expected
        )


@pytest.mark.skipif(not _XPU, reason="srt's MXFP8 scheme is XPU-only")
def test_only_linears_in_the_quantized_blocks_get_mxfp8():
    """The checkpoint quantizes the Wan blocks.N linears; condition_embedder and
    proj_out stay BF16 (unquantized tensors in the checkpoint)."""
    from sglang.multimodal_gen.runtime.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.auto_round_mxfp8 import (
        AutoRoundMXFP8Config,
        _DiffusionMXFP8Scheme,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )

    config = AutoRoundMXFP8Config(block_names=("blocks",))
    linear = MagicMock(spec=LinearBase)
    for prefix in ("blocks.0.to_q", "blocks.39.ffn.fc_out"):
        assert isinstance(
            config.get_quant_method(linear, prefix), CompressedTensorsLinearMethod
        )
        assert type(linear.scheme) is _DiffusionMXFP8Scheme
    for prefix in ("condition_embedder.time_proj", "proj_out", "blocks_extra.fc"):
        assert isinstance(
            config.get_quant_method(linear, prefix), UnquantizedLinearMethod
        )
    assert config.get_quant_method(MagicMock(), "blocks.0.norm2") is None


@pytest.mark.skipif(not _XPU, reason="XPU required")
@pytest.mark.parametrize("linear_cls", ["ColumnParallelLinear", "RowParallelLinear"])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
def test_xpu_checkpoint_weights_load_and_match_bf16(request, linear_cls, x_dtype):
    """Checkpoint layout (weight e4m3 [N, K], weight_scale uint8 UE8M0 [N, K/32])
    goes through the diffusion layers' own weight loaders, which srt's parameter
    classes cannot serve (they need tp_rank), then reproduces the BF16 GEMM."""
    request.getfixturevalue("single_process_model_parallel")
    from sglang.multimodal_gen.runtime.layers import linear as linear_mod
    from sglang.multimodal_gen.runtime.layers.quantization.auto_round_mxfp8 import (
        AutoRoundMXFP8Config,
    )
    from sglang.srt.layers.quantization.fp8_utils import xpu_mxfp8_quantize

    in_f, out_f, rows = 512, 384, 200
    layer = getattr(linear_mod, linear_cls)(
        in_f,
        out_f,
        bias=True,
        params_dtype=torch.bfloat16,
        quant_config=AutoRoundMXFP8Config(block_names=("blocks",)),
        prefix="blocks.0.ffn.fc_out",
    ).to("xpu")
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.uint8
    assert layer.weight_scale.shape == (out_f, in_f // 32)

    g = torch.Generator(device="cpu").manual_seed(0)
    weight = (torch.randn(out_f, in_f, generator=g) * 0.02).to("xpu", torch.bfloat16)
    bias = (torch.randn(out_f, generator=g) * 0.02).to("xpu", torch.bfloat16)
    q_weight, scale = xpu_mxfp8_quantize(weight)
    for param, tensor in (
        (layer.weight, q_weight),
        (layer.weight_scale, scale.view(torch.uint8)),
        (layer.bias, bias),
    ):
        param.weight_loader(param, tensor)
    layer.quant_method.process_weights_after_loading(layer)
    # srt's scheme keeps the checkpoint bytes; the GEMM views them as e8m0.
    assert layer.weight_scale.dtype == torch.uint8

    x = torch.randn(rows, in_f, generator=g).to("xpu", x_dtype)
    out = layer(x)[0]
    # srt's MXFP8 scheme returns the input dtype.
    assert out.dtype == x_dtype and out.shape == (rows, out_f)
    ref = x.float() @ weight.float().t() + bias.float()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.05, rel


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
