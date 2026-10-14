# SPDX-License-Identifier: Apache-2.0
"""TinyVAE latent contracts, temporal state isolation and checkpoint loading."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.vaes.tiny_autoencoder import (
    TinyImageAutoencoder,
    TinyVideoAutoencoder,
)
from sglang.multimodal_gen.runtime.models.vaes.tiny_vae import (
    TinyVAEDecodeWrapper,
    load_tiny_autoencoder,
    validate_tiny_vae_config,
    with_tiny_vae,
)


@pytest.mark.parametrize("channels", [16, 24, 32, 48, 128])
def test_video_batch_and_request_isolation(channels):
    torch.manual_seed(0)
    model = TinyVideoAutoencoder(channels, load_encoder=False).eval()
    z = torch.randn(2, channels, 2, 1, 1)
    with torch.inference_mode():
        together = model.decode(z)
        separately = torch.cat([model.decode(sample) for sample in z.split(1)])
        torch.testing.assert_close(together, separately, rtol=1e-5, atol=1e-6)
        assert torch.equal(together, model.decode(z))
    expected_frames = (
        5 if channels == 24 else 2 * model.temporal_ratio - (model.temporal_ratio - 1)
    )
    assert together.shape == (
        2,
        3,
        expected_frames,
        8 * model.patch_size,
        8 * model.patch_size,
    )


@pytest.mark.parametrize("channels", [4, 16, 64, 128])
def test_image_video_layout(channels):
    model = TinyImageAutoencoder(channels, load_encoder=False).eval()
    z = torch.randn(1, channels, 2, 1, 2)
    with torch.inference_mode():
        frames = model.decode(z)
        separately = torch.stack([model.decode(z[:, :, t]) for t in range(2)], dim=2)
    torch.testing.assert_close(frames, separately)
    assert frames.shape[1] == (4 if channels == 64 else 3)


class Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.register_buffer("latents_mean", torch.tensor([2.0]))
        self.register_buffer("latents_std", torch.tensor([3.0]))

    def encode(self, pixels):
        return pixels + self.weight


class Capture(nn.Module):
    latent_channels = 1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def decode(self, latents):
        self.input = latents.clone()
        return latents.expand(-1, 3, -1, -1, -1)


def config(**arch):
    return SimpleNamespace(
        vae_config=SimpleNamespace(arch_config=SimpleNamespace(**arch)),
        get_decode_scale_and_shift=lambda device, dtype, vae: (0.5, 2.0),
    )


@pytest.mark.parametrize(
    "model_name", ["taesd", "taehv", "taeh3", "taew2_1", "taew2_2", "taeltx2"]
)
def test_wrapper_raw_latents_and_original_encoder(model_name):
    original, tiny = Original(), Capture()
    cfg = config(latents_mean=[2.0], latents_std=[3.0], video_decoder_variant="ltx_2")
    wrapped = TinyVAEDecodeWrapper(original, tiny, model_name, cfg)
    raw = torch.full((1, 1, 1, 1, 1), 5.0)
    parts = []
    pixels = wrapped.decode(raw, on_frames=parts.append)
    expected = 1.0 if model_name in ("taeh3", "taew2_1", "taew2_2", "taeltx2") else 1.5
    assert torch.equal(tiny.input, torch.full_like(raw, expected))
    assert torch.equal(pixels, parts[0])
    assert torch.equal(wrapped.encode(raw), original.encode(raw))
    assert not wrapped.use_parallel_decode
    wrapped.to(torch.float64)
    assert wrapped.tiny.weight.dtype == wrapped.original.weight.dtype == torch.float64


@pytest.mark.parametrize("variant", ["ltx_2_3", "ltx_2_5"])
def test_ltx_incompatible_decoder_is_rejected(variant):
    cfg = config(video_decoder_variant=variant)
    cfg.tiny_vae_model, cfg.tiny_vae_path = "taeltx2", "tiny.pth"
    with pytest.raises(ValueError, match="LTX-2.3/2.5"):
        validate_tiny_vae_config(cfg)


@pytest.mark.parametrize("model_name", ["taeh3", "taew2_1", "taew2_2"])
def test_normalization_buffers_follow_module_dtype(model_name):
    cfg = config(latents_mean=[2.0], latents_std=[3.0])
    wrapped = TinyVAEDecodeWrapper(Original(), Capture(), model_name, cfg)
    buffers = dict(wrapped.named_buffers())
    assert "tiny_latents_mean" in buffers and "tiny_latents_std" in buffers
    wrapped.to(dtype=torch.float64)
    assert (
        wrapped.tiny_latents_mean.dtype
        == wrapped.tiny_latents_std.dtype
        == torch.float64
    )
    # Decoding uses cached metadata, without rebuilding it from configuration.
    cfg.vae_config.arch_config.latents_mean = None
    cfg.vae_config.arch_config.latents_std = None
    raw = torch.full((1, 1, 1, 1, 1), 5.0, dtype=torch.float64)
    torch.testing.assert_close(wrapped._normalized_latents(raw), torch.ones_like(raw))


def test_flux2_repatchifies_before_normalization():
    tiny = Capture()
    wrapped = TinyVAEDecodeWrapper(Original(), tiny, "taef2", config())
    raw = torch.arange(32 * 4 * 6.0).reshape(1, 32, 4, 6)
    expected = torch.nn.functional.pixel_unshuffle(raw, 2)
    assert torch.equal(wrapped._normalized_latents(raw), (expected - 2) * 0.5)


def test_h3_pixel_contract():
    wrapped = TinyVAEDecodeWrapper(
        Original(), Capture(), "taeh3", config(latents_mean=[2], latents_std=[3])
    )
    raw = torch.full((1, 1, 1, 1, 1), 3.5)
    actual = wrapped.decode_base(raw)
    mean = raw.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = raw.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    torch.testing.assert_close(actual * std + mean, torch.full_like(actual, 0.5))


@pytest.mark.parametrize("prefix", ["", "decoder.", "taesd_decoder."])
def test_decoder_checkpoint_prefixes(tmp_path, prefix):
    from safetensors.torch import save_file

    model = TinyImageAutoencoder(4, load_encoder=False).eval()
    state = {prefix + key: value for key, value in model.decoder.state_dict().items()}
    state["encoder.ignored"] = torch.ones(1)
    state["vae_scale"] = torch.tensor(0.18215)
    path = tmp_path / "tiny.safetensors"
    save_file(state, path)
    restored = load_tiny_autoencoder(str(path), "taesd", torch.float32)
    assert restored.encoder is None
    for key, value in model.decoder.state_dict().items():
        assert torch.equal(value, restored.decoder.state_dict()[key])
    state.pop(prefix + "1.weight")
    save_file(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_tiny_autoencoder(str(path), "taesd", torch.float32)


@pytest.mark.parametrize(
    "path,name", [(None, "taeh3"), ("tiny.pth", None), ("tiny.pth", "unknown")]
)
def test_invalid_configuration(path, name):
    with pytest.raises(ValueError):
        validate_tiny_vae_config(
            SimpleNamespace(tiny_vae_path=path, tiny_vae_model=name)
        )


def test_audio_and_default_vae_are_unchanged():
    original = Original()
    args = SimpleNamespace(pipeline_config=SimpleNamespace(tiny_vae_path=None))
    assert with_tiny_vae(original, args, "vae") is original
    args.pipeline_config.tiny_vae_path = "unused.safetensors"
    assert with_tiny_vae(original, args, "audio_vae") is original


def test_cli_and_python_config_fields():
    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
    from sglang.multimodal_gen.runtime.utils.argparse import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    PipelineConfig.add_cli_args(parser)
    parsed = parser.parse_args(
        ["--tiny-vae-model", "taeh3", "--tiny-vae-path", "tiny.safetensors"]
    )
    assert parsed.tiny_vae_model == "taeh3"
    assert parsed.tiny_vae_path == "tiny.safetensors"
    cfg = PipelineConfig(tiny_vae_model="taeh3", tiny_vae_path="tiny.safetensors")
    validate_tiny_vae_config(cfg)


@pytest.mark.parametrize("kind", ["flux", "wan21", "wan22", "qwen21", "ltx2"])
def test_pipeline_normalization_roundtrip(kind):
    from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig
    from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
        QwenImage21PipelineConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.wan import (
        Wan2_2_TI2V_5B_Config,
        WanT2V480PConfig,
    )

    cases = {
        "flux": (FluxPipelineConfig, "taef1", 16),
        "wan21": (WanT2V480PConfig, "taew2_1", 16),
        "wan22": (Wan2_2_TI2V_5B_Config, "taew2_2", 48),
        "qwen21": (QwenImage21PipelineConfig, "taeqi2_1", 64),
        "ltx2": (LTX2PipelineConfig, "taeltx2", 128),
    }
    cls, name, channels = cases[kind]
    cfg = cls()
    # Mirror VAELoader: pipeline defaults are placeholders until config.json
    # updates the VAE architecture and derived normalization tensors.
    if kind == "flux":
        cfg.vae_config.update_model_arch(
            {"scaling_factor": 0.3611, "shift_factor": 0.1159}
        )
    elif kind == "wan22":
        cfg.vae_config.update_model_arch(
            {
                "z_dim": 48,
                "latents_mean": tuple(torch.linspace(-0.5, 0.5, 48).tolist()),
                "latents_std": tuple(torch.linspace(0.5, 1.5, 48).tolist()),
            }
        )
    original = Original()
    if kind == "ltx2":
        original.latents_mean = torch.randn(channels)
        original.latents_std = torch.rand(channels) + 0.5
    normalized = torch.randn(1, channels, 2, 2, 2)
    scale, shift = cfg.get_decode_scale_and_shift(
        normalized.device, normalized.dtype, original
    )
    raw = normalized / scale
    if shift is not None:
        raw = raw + shift
    wrapped = TinyVAEDecodeWrapper(original, Capture(), name, cfg)
    torch.testing.assert_close(wrapped._normalized_latents(raw), normalized)


def test_generation_stage_uses_tiny_decoder(monkeypatch):
    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )

    cfg = PipelineConfig(vae_tiling=False, vae_precision="fp32")
    cfg.vae_config.update_model_arch({"scaling_factor": 0.5, "shift_factor": 2.0})
    original, tiny = Original(), Capture()
    wrapped = TinyVAEDecodeWrapper(original, tiny, "taesd", cfg)
    args = SimpleNamespace(
        pipeline_config=cfg,
        disable_autocast=True,
        enable_torch_compile=False,
    )
    z = torch.full((1, 1, 2, 1, 1), 0.25)
    stage = DecodingStage(wrapped)
    parts = []
    result = stage.decode(z, args, vae_dtype=torch.float32, on_frames=parts.append)
    assert torch.equal(tiny.input, z)
    assert torch.equal(result, z.expand(-1, 3, -1, -1, -1))
    assert torch.equal(parts[0], result)


@pytest.mark.parametrize(
    "name", ["lighttaew2_1", "lighttaew2_2", "lighttaehy1_5", "taeltx_2"]
)
def test_comfy_video_names(name):
    cfg = config(
        video_decoder_variant="ltx_2", latents_mean=[2.0], latents_std=[3.0]
    )
    cfg.tiny_vae_model, cfg.tiny_vae_path = name, "tiny.pth"
    validate_tiny_vae_config(cfg)
    wrapped = TinyVAEDecodeWrapper(Original(), Capture(), name, cfg)
    assert wrapped.model_name != name


def test_torch_checkpoint_and_unexpected_weights(tmp_path):
    model = TinyVideoAutoencoder(24, load_encoder=False)
    path = tmp_path / "tiny.pth"
    torch.save(model.state_dict(), path)
    restored = load_tiny_autoencoder(str(path), "taeh3", torch.float32)
    assert restored.latent_channels == 24
    state = model.state_dict()
    state["invalid.weight"] = torch.ones(1)
    torch.save(state, path)
    with pytest.raises(ValueError, match="Unexpected TinyVAE"):
        load_tiny_autoencoder(str(path), "taeh3", torch.float32)


@pytest.mark.parametrize(
    "high_precision,component_precisions",
    [(None, {}), ("bf16", {}), ("fp32", {"video_vae": "bf16"})],
)
def test_loader_wraps_only_visual_component(
    monkeypatch, high_precision, component_precisions
):
    from sglang.multimodal_gen.runtime.models.vaes import tiny_vae

    cfg = config(latents_mean=[2.0], latents_std=[3.0])
    cfg.tiny_vae_model, cfg.tiny_vae_path = "taeh3", "tiny.pth"
    cfg.vae_precision, cfg.vae_decode_precision = "fp32", "bf16"
    cfg.vae_decode_precision_high = high_precision
    args = SimpleNamespace(
        pipeline_config=cfg, component_precisions=component_precisions
    )
    calls = []

    def load(path, name, dtype):
        calls.append((path, name, dtype))
        return Capture().to(dtype)

    monkeypatch.setattr(tiny_vae, "load_tiny_autoencoder", load)
    original = Original()
    assert with_tiny_vae(original, args, "audio_vae") is original
    wrapped = with_tiny_vae(original, args, "video_vae")
    assert wrapped.original is original
    assert calls == [("tiny.pth", "taeh3", torch.bfloat16)]
    assert wrapped.tiny.weight.device == original.weight.device


def test_loader_rejects_request_precision_changes_before_loading(monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes import tiny_vae

    cfg = config()
    cfg.tiny_vae_model, cfg.tiny_vae_path = "taeh3", "unused.pth"
    cfg.vae_precision, cfg.vae_decode_precision = "fp32", "bf16"
    cfg.vae_decode_precision_high = "fp32"
    args = SimpleNamespace(pipeline_config=cfg, component_precisions={})

    def unexpected_load(*args, **kwargs):
        pytest.fail("Incompatible precision must be rejected before loading weights")

    monkeypatch.setattr(tiny_vae, "load_tiny_autoencoder", unexpected_load)
    with pytest.raises(ValueError, match="request-scoped precision changes"):
        with_tiny_vae(Original(), args, "video_vae")
