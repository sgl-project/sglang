# SPDX-License-Identifier: Apache-2.0
"""Strict loading, dtype and config contracts for tiny Kandinsky 6 SR components."""

import copy
import json

import pytest
import torch
from kandinsky6_sr_tiny_components import TINY_DIT, TINY_KVAE, TINY_LU_MODEL
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import (
    component_loader as component_loader_module,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.latent_upscaler_loader import (
    LatentUpscalerLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.kandinsky6_sr import (
    Kandinsky6SRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.upsampler.kandinsky6_sr_latent_upscaler import (
    Kandinsky6SRLatentUpscalerBank,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_sr_vae import Kandinsky6SRVAE

pytestmark = pytest.mark.usefixtures("single_process_model_parallel")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """Load on CPU; the forward test moves its DiT to CUDA for fused kernels."""
    monkeypatch.setattr(
        component_loader_module.ComponentLoader,
        "target_device",
        staticmethod(lambda component_starts_on_cpu: torch.device("cpu")),
    )


def _dit_config(**changes):
    flat = dict(TINY_DIT, **changes)
    config = Kandinsky6SRDitConfig()
    config.update_model_arch(flat)
    return config, flat


def _dit(**changes):
    """Tiny DiT with deterministic random weights (the linears start uninitialized)."""
    config, flat = _dit_config(**changes)
    model = Kandinsky6SRTransformer3DModel(config, flat)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
    return model


def _official_state_dict(model):
    """State dict as the current Diffusers checkpoint stores it."""
    return {
        name.replace(".feed_forward.mlp.fc_in.", ".feed_forward.net.0.proj.")
        .replace(".feed_forward.mlp.fc_out.", ".feed_forward.net.2.")
        .replace(
            "time_embeddings.in_layer.",
            "time_embeddings.timestep_embedder.linear_1.",
        )
        .replace(
            "time_embeddings.out_layer.",
            "time_embeddings.timestep_embedder.linear_2.",
        ): tensor
        for name, tensor in model.state_dict().items()
    }


def _load_through_the_real_loader(checkpoint, **changes):
    """Meta-init the DiT and load ``checkpoint`` like ``maybe_load_fsdp_model`` does."""
    config, flat = _dit_config(**changes)
    with torch.device("meta"):
        model = Kandinsky6SRTransformer3DModel(config, flat)
    load_model_from_full_model_state_dict(
        model,
        model.preprocess_loaded_state_dict(iter(checkpoint.items())),
        torch.device("cpu"),
        torch.float32,
        strict=False,
        param_names_mapping=get_param_names_mapping(model.param_names_mapping),
    )
    return model


def test_complete_checkpoint_loads_and_meta_buffers_are_rebuilt():
    """Meta-init must rebuild non-checkpoint RoPE and time-embedding buffers."""
    reference = _dit()
    checkpoint = _official_state_dict(reference)
    model = _load_through_the_real_loader(checkpoint)
    model.post_load_weights()
    assert not any(t.is_meta for t in [*model.parameters(), *model.buffers()])
    torch.testing.assert_close(model.pooled_bias, reference.pooled_bias)
    assert model.out_layer.out_layer.weight.shape[0] == 4 * TINY_DIT["out_visual_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randn(1, 2, 4, 4, 4, device=device)
    rope_pos = [torch.arange(2, device=device) for _ in range(3)]
    with set_forward_context(current_timestep=0, attn_metadata=None):
        out = model(x, torch.tensor([500.0], device=device), rope_pos)
    assert out.shape == (1, 2, 4, 4, 12) and torch.isfinite(out).all()


@pytest.mark.parametrize(
    "key,unexpected,message",
    [
        ("pooled_bias", False, "was not loaded"),
        (
            "visual_transformer_blocks.0.visual_modulation.out_layer.bias",
            False,
            "was not loaded",
        ),
        ("text_embeddings.in_layer.weight", True, "unexpected keys"),
    ],
)
def test_incomplete_or_unexpected_checkpoint_fails_the_load(key, unexpected, message):
    # even bias parameters must fail instead of the generic loader's zero-fill fallback
    checkpoint = _official_state_dict(_dit())
    if unexpected:
        checkpoint[key] = torch.zeros(2, 2)
    else:
        del checkpoint[key]
    with pytest.raises(ValueError, match=message):
        _load_through_the_real_loader(checkpoint)


def test_wide_input_checkpoint_needs_visual_cond_to_run_under_noise():
    """A hybrid_anchor checkpoint has a ``2 * in + 1`` input layer; the default ``noise``
    override would feed ``in`` channels.  That mismatch must be reported, and the
    documented ``visual_cond`` override must make it run."""
    wide = dict(
        instruct_type="hybrid_anchor", attribute_overrides={"instruct_type": "noise"}
    )
    with pytest.raises(ValueError, match="visual_cond"):
        _dit(**wide)

    model = _dit(
        instruct_type="hybrid_anchor",
        attribute_overrides={"instruct_type": "noise", "visual_cond": True},
    )
    assert model.instruct_type == "noise" and model.visual_cond is True
    assert model.visual_embed_dim == 9
    assert model.config.instruct_type == "hybrid_anchor"  # the trained value is kept


def test_rope_angle_tables_stay_fp32_when_the_module_is_cast():
    """The residency manager casts a module to its target dtype (bf16) on every move to
    the GPU; the RoPE angle tables must not be rounded to bf16 with the weights."""
    model = _dit().to(torch.bfloat16)
    assert model.pooled_bias.dtype == torch.bfloat16
    for name, buffer in model.visual_rope_embeddings.named_buffers():
        assert buffer.dtype == torch.float32, name


class FakeServerArgs:
    def __init__(self):
        self.pipeline_config = Kandinsky6SRPipelineConfig()
        self.model_paths = {}
        self.component_paths = {}
        self.component_precisions = {}
        self.component_weights_paths = {}
        self.component_quantizations = {}
        self.num_gpus = 1
        self.disable_autocast = False
        self.revision = None
        self.trust_remote_code = False

    def should_start_component_on_cpu(self, component_name):
        return False

    def should_direct_gpu_weight_load_component(self, component_name):
        return False

    def should_use_fsdp_for_component(self, component_name):
        return False

    def requested_component_attention_backend(self, component_name):
        return None

    def resolve_component_attention_backend(self, *names):
        return None, None

    def resolve_component_backend_by_role(self, *names):
        return {}


def _write_component(directory, config, tensors):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    save_file(tensors, str(directory / "diffusion_pytorch_model.safetensors"))


def _tiny_vae_bundle(tmp_path):
    raw_config = dict(
        _class_name="Kandinsky6SRVAE",
        vae_type="video-kvae",
        encoder_config=dict(TINY_KVAE, in_channels=3),
        decoder_config=dict(TINY_KVAE, out_ch=3),
        scaling_factor=0.5,
        spatial_factor=16,
        temporal_factor=4,
    )
    config = Kandinsky6SRVAEConfig()
    config.update_model_arch(raw_config)
    wrapper = Kandinsky6SRVAE(config)
    directory = tmp_path / "vae"
    _write_component(
        directory,
        raw_config,
        {k: v.contiguous() for k, v in wrapper.state_dict().items()},
    )
    return directory


def test_vae_loader_loads_the_wrapper_strictly(tmp_path):
    directory = _tiny_vae_bundle(tmp_path)
    server_args = FakeServerArgs()
    vae = VAELoader().load_customized(str(directory), server_args, "vae")
    assert isinstance(vae, Kandinsky6SRVAE) and vae.scaling_factor == 0.5
    assert next(vae.parameters()).dtype == torch.bfloat16  # vae_precision
    assert (vae.spatial_factor, vae.temporal_factor) == (16, 4)

    # one missing tensor must fail: non-strict VAE loading would only log a warning
    weights_path = directory / "diffusion_pytorch_model.safetensors"
    weights = load_file(str(weights_path))
    torch.testing.assert_close(
        vae.state_dict(),
        {name: value.bfloat16() for name, value in weights.items()},
        rtol=0,
        atol=0,
    )
    weights.pop(next(iter(weights)))
    save_file(weights, str(weights_path))
    with pytest.raises(RuntimeError, match="Missing key"):
        VAELoader().load_customized(str(directory), FakeServerArgs(), "vae")


def _tiny_lu_bundle(tmp_path, models=None, scales=(2, 4), include_scales=True):
    models = models or [
        {"target_scale": scale, "model": copy.deepcopy(TINY_LU_MODEL)}
        for scale in ("2x", "4x")
    ]
    bank = Kandinsky6SRLatentUpscalerBank(models, 0.5, scales=scales)
    directory = tmp_path / "latent_upscaler"
    config = {
        "_class_name": "Kandinsky6SRLatentUpscalerBank",
        "models": models,
        "scaling_factor": 0.5,
    }
    if include_scales:
        config["scales"] = list(scales)
    _write_component(
        directory,
        config,
        {k: v.contiguous() for k, v in bank.state_dict().items()},
    )
    return directory


@pytest.mark.parametrize("legacy,library", [(False, "diffusers"), (True, "kandinsky6")])
def test_latent_upscaler_loading_and_default_scale_order(tmp_path, legacy, library):
    models = [
        {"target_scale": scale, "model": copy.deepcopy(TINY_LU_MODEL)}
        for scale in (("4x", "2x") if legacy else ("2x", "4x"))
    ]
    directory = _tiny_lu_bundle(tmp_path, models=models, include_scales=not legacy)
    loader = component_loader_module.ComponentLoader.for_component_type(
        "latent_upscaler", library
    )
    assert isinstance(loader, LatentUpscalerLoader)
    bank, _ = loader.load(str(directory), FakeServerArgs(), "latent_upscaler", library)
    assert bank.scales == (2, 4)
    assert bank._models[0] is bank.for_scale(2)
    assert bank._models[1] is bank.for_scale(4)
    assert not bank.training
    assert all(
        p.dtype == torch.bfloat16 and not p.requires_grad for p in bank.parameters()
    )
    z = torch.randn(1, 4, 3, 4, 6, dtype=torch.bfloat16)
    assert bank.upscale(z, scale=2).shape == (1, 4, 3, 8, 12)
    assert bank.upscale(z, scale=4).shape == (1, 4, 3, 16, 24)
    with pytest.raises(ValueError, match="no x8 entry"):
        bank.upscale(z, scale=8)


def test_latent_upscaler_rejects_the_wrong_library():
    with pytest.raises(AssertionError, match="latent_upscaler must be loaded from"):
        component_loader_module.ComponentLoader.for_component_type(
            "latent_upscaler", "transformers"
        )


@pytest.mark.parametrize("error_kind", ["Missing key", "bogus_field"])
def test_latent_upscaler_load_errors_are_raised_not_swallowed(tmp_path, error_kind):
    directory = _tiny_lu_bundle(tmp_path)
    if error_kind == "Missing key":
        path = directory / "diffusion_pytorch_model.safetensors"
        weights = load_file(str(path))
        weights.pop(next(iter(weights)))
        save_file(weights, str(path))
    else:
        path = directory / "config.json"
        config = json.loads(path.read_text())
        config["models"][0]["model"]["bogus_field"] = 1
        path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="native fallback is disabled") as error:
        LatentUpscalerLoader().load(
            str(directory), FakeServerArgs(), "latent_upscaler", "diffusers"
        )
    assert error_kind in str(error.value.__cause__)


def test_motion_attention_is_not_a_supported_entry_key():
    """Unsupported motion attention must fail instead of silently using dense attention."""
    with_motion = {
        **copy.deepcopy(TINY_LU_MODEL),
        "enable_x2_entry": False,
        "motion_attention": {
            "spatial_kernel_size": 3,
            "temporal_offsets": [-1, 1],
            "after_mid_blocks": [1],
            "num_heads": 2,
        },
    }
    with pytest.raises(ValueError, match="motion_attention"):
        Kandinsky6SRLatentUpscalerBank(
            [{"target_scale": "4x", "model": with_motion}], 0.5
        )
