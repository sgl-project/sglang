# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.dits.ming_image import MingImageDitConfig
from sglang.multimodal_gen.configs.pipeline_configs.ming_image import (
    MingImageLayerPipelineConfig,
    MingImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.ming_image import (
    MingImageLayerSamplingParams,
    MingImageSamplingParams,
)
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.ming_image import (
    MingImageEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.models.dits.ming_image import (
    MingImageTransformer2DModel,
    MingRMSNorm,
    MingSiluAndMul,
)
from sglang.multimodal_gen.runtime.models.encoders.ming_image import ming_position_ids
from sglang.multimodal_gen.runtime.pipelines.ming_image_pipeline import (
    MingImagePipeline,
    prepare_mu,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ming_image import (
    MingImageReferenceStage,
    ming_reference_size,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_context


@pytest.mark.parametrize(
    "checkpoint,config_cls,sampling_cls",
    [
        ("Design", MingImagePipelineConfig, MingImageSamplingParams),
        ("Design-Layer", MingImageLayerPipelineConfig, MingImageLayerSamplingParams),
    ],
)
@pytest.mark.parametrize(
    "path_template",
    [
        "inclusionAI/Ming-Image-0.1-{}",
        "/models/Ming-Image-0.1-{}",
        "/cache/models--inclusionAI--Ming-Image-0.1-{}/snapshots/revision",
    ],
)
def test_registry_resolves_both_checkpoints(
    checkpoint, config_cls, sampling_cls, path_template, monkeypatch
):
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: pytest.fail("Ming checkpoints do not have a model_index.json"),
    )
    info = get_model_info(path_template.format(checkpoint), backend="sglang")
    assert info.pipeline_cls is MingImagePipeline
    assert info.pipeline_config_cls is config_cls
    assert info.sampling_param_cls is sampling_cls


@pytest.mark.parametrize("mode,multi", [("zero_masked", False), ("learned", True)])
def test_checkpoint_contract(mode, multi):
    config = MingImageDitConfig()
    config.update_model_arch(
        {
            "alignment_padding_mode": mode,
            "multi_frame_output": multi,
            "n_layers": 30,
            "n_heads": 30,
        }
    )
    assert config.num_layers == 30
    assert config.num_attention_heads == 30
    assert config.alignment_padding_mode == mode


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"alignment_padding_mode": "learned"},
        {"alignment_padding_mode": "zero_masked", "multi_frame_output": True},
        {"alignment_padding_mode": "learned", "multi_frame_output": False},
    ],
)
def test_checkpoint_contract_rejects_ambiguous_padding(metadata):
    with pytest.raises(ValueError):
        MingImageDitConfig().update_model_arch(metadata)


def test_encoder_loader_is_pipeline_local():
    assert MingImageEncoderLoader.component_names == []
    assert (
        type(ComponentLoader.for_component_type("text_encoder", "transformers"))
        is TextEncoderLoader
    )


@pytest.mark.parametrize(
    "prompt,count",
    [("", 4), ("Decompose this image into 2 layers.", 2), ("Number of layers: 3", 3)],
)
def test_layer_count_request_and_output_contract(prompt, count):
    params = MingImageLayerSamplingParams(prompt=prompt)
    assert params.build_request_extra()["ming_num_layers"] == count
    assert params.num_samples_per_request == count
    assert "num_layers" in params.image_request_extra_fields()
    assert params.default_image_output_format() == "png"


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_layer_count_rejects_empty_output(count):
    with pytest.raises(ValueError, match="positive integer"):
        MingImageLayerSamplingParams(num_layers=count).build_request_extra()


def test_cpu_noise_draw_preserves_official_frame_order():
    config = MingImagePipelineConfig()
    config.check_pipeline_config()
    assert not config.vae_tiling
    assert not config.vae_sp
    assert config.supports_sequential_multi_output_inference()
    batch = SimpleNamespace(extra={"ming_frames": 3}, height=32, width=48)
    shape = config.prepare_latent_shape(batch, 1, 1)
    actual = torch.randn(shape, generator=torch.Generator().manual_seed(42))
    actual = config.maybe_pack_latents(actual, 1, batch)
    expected = (
        torch.randn((3, 16, 4, 6), generator=torch.Generator().manual_seed(42))
        .transpose(0, 1)
        .unsqueeze(0)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert config.get_latent_dtype(torch.bfloat16) == torch.float32
    assert config.shard_latents_for_sp(batch, actual) == (actual, False)
    assert config.get_classifier_free_guidance_scale(batch, 2.0) == 3.0


def test_centered_video_rope_with_reference_and_query_tokens():
    ids = [10, 11] + [99] * 4 + [12, 13] + [99] * 3 + [14]
    positions = ming_position_ids(ids, [(1, 4, 4), (1, 2, 6)], 99)[:, 0]
    assert positions.shape == (3, len(ids))
    torch.testing.assert_close(positions[:, :2], torch.tensor([[0, 1]]).expand(3, -1))
    torch.testing.assert_close(
        positions[:, 2:6], torch.tensor([[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]])
    )
    torch.testing.assert_close(
        positions[:, 8:11], torch.tensor([[5, 5, 5], [5, 5, 5], [4, 5, 6]])
    )
    assert positions[:, -1].tolist() == [6, 6, 6]


@pytest.mark.parametrize(
    "height,width,resolution,expected",
    [
        (512, 512, 512, (512, 512)),
        (720, 1280, 1024, (720, 1280)),
        (1280, 720, 1024, (1280, 720)),
        (100, 400, 512, (256, 1024)),
    ],
)
def test_official_reference_buckets(height, width, resolution, expected):
    assert ming_reference_size(height, width, resolution) == expected


@pytest.mark.parametrize(
    "override,expected", [({}, torch.bfloat16), ({"vae": "fp32"}, torch.float32)]
)
def test_reference_vae_declares_precision_before_first_encode(override, expected):
    args = SimpleNamespace(
        component_precisions=override, pipeline_config=MingImagePipelineConfig()
    )
    (use,) = MingImageReferenceStage(torch.nn.Identity()).component_uses(args)
    assert use.component_name == "vae"
    assert use.target_dtype == expected


def test_ming_norm_and_swiglu_preserve_reference_rounding():
    torch.manual_seed(9)
    x = torch.randn(3, 128, dtype=torch.bfloat16)
    norm = MingRMSNorm(128).to(dtype=x.dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    expected = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    ).to(x.dtype) * norm.weight
    torch.testing.assert_close(norm(x), expected, rtol=0, atol=0)
    gate, up = x.chunk(2, -1)
    torch.testing.assert_close(MingSiluAndMul()(x), F.silu(gate) * up, rtol=0, atol=0)


def test_padding_replacement_keeps_valid_rows_and_registers():
    x = torch.randn(2, 5, 8)
    token = torch.randn(1, 8)
    mask = (torch.arange(5) < 3).unsqueeze(0)
    output = MingImageTransformer2DModel._replace_padding_with_token_mask(
        x, mask, token
    )
    torch.testing.assert_close(output[:, :3], x[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(output[:, 3:], token.expand(2, 2, 8), rtol=0, atol=0)
    assert output.data_ptr() != x.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_moe_preserves_bf16_activation_before_fp32_combine():
    torch.manual_seed(7)
    x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
    w13 = torch.randn(4, 256, 128, dtype=x.dtype, device=x.device) * 0.02
    w2 = torch.randn(4, 128, 128, dtype=x.dtype, device=x.device) * 0.02
    weights, ids = torch.randn(16, 4, device=x.device).softmax(-1).topk(2)
    config = MoeRunnerConfig(
        activation="silu_rounded",
        is_gated=True,
        no_combine=True,
        inplace=False,
        top_k=2,
    )
    with get_context().override_server_args(model_path="dummy"):
        actual = fused_experts(
            x, w13, w2, StandardTopKOutput(weights, ids.int(), None), config
        )
    expected = torch.empty_like(actual)
    for expert in range(4):
        rows, slots = torch.where(ids == expert)
        gate, up = F.linear(x[rows], w13[expert]).chunk(2, -1)
        expected[rows, slots] = F.linear(F.silu(gate) * up, w2[expert])
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-4)
    combined = (actual.float() * weights.unsqueeze(-1)).sum(1).to(x.dtype)
    reference = (expected.float() * weights.unsqueeze(-1)).sum(1).to(x.dtype)
    torch.testing.assert_close(combined, reference, rtol=0, atol=2e-4)


@pytest.mark.parametrize("size,mu", [(256, 0.5), (1024, 1.35), (2048, 1.35)])
def test_flow_shift_uses_spatial_tokens_not_layer_count(size, mu):
    assert prepare_mu(SimpleNamespace(height=size, width=size), None) == ("mu", mu)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
