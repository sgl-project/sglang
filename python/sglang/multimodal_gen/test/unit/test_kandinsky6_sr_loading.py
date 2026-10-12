# SPDX-License-Identifier: Apache-2.0
"""Strict loading, dtype and config contracts for tiny Kandinsky 6 SR components."""

import copy
import json
import os

import pytest
import torch
from kandinsky6_sr_tiny_components import TINY_KVAE, TINY_LU_MODEL
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
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
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
from sglang.multimodal_gen.runtime.models.dits.kandinsky6_sr import (
    Kandinsky6SRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.upsampler.kandinsky6_sr_latent_upscaler import (
    Kandinsky6SRLatentUpscalerBank,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_sr_vae import Kandinsky6SRVAE

TINY_DIT = dict(
    in_visual_dim=4,
    in_text_dim=8,
    in_text_dim2=8,
    time_dim=16,
    out_visual_dim=4,
    patch_size=[1, 2, 2],
    model_dim=32,
    ff_dim=64,
    num_text_blocks=0,
    num_visual_blocks=2,
    axes_dims=[8, 4, 4],
    visual_cond=False,
    instruct_type="noise",
    use_text=False,
    n_grid=3,
    attribute_overrides={"instruct_type": "noise"},
)


@pytest.fixture(scope="module", autouse=True)
def single_process_model_parallel():
    """The K6 feed-forward uses TP-aware linears, which need a (size-1) TP group."""
    if not model_parallel_is_initialized():
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29508",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            os.environ.setdefault(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


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


# --------------------------------------------------------------------------- #
# DiT
# --------------------------------------------------------------------------- #
def test_complete_checkpoint_loads_and_meta_buffers_are_rebuilt():
    """Positive control for the failure tests below, and the meta-device path: RoPE
    tables and time-embedding frequencies are not in the checkpoint and must be rebuilt
    by ``post_load_weights`` so that a loaded model actually runs."""
    reference = _dit()
    checkpoint = _official_state_dict(reference)
    model = _load_through_the_real_loader(checkpoint)
    model.post_load_weights()
    assert not any(t.is_meta for t in [*model.parameters(), *model.buffers()])
    assert not model.time_embeddings.freqs.is_meta
    torch.testing.assert_close(model.pooled_bias, reference.pooled_bias)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.randn(1, 2, 4, 4, 4, device=device)
    rope_pos = [torch.arange(2, device=device) for _ in range(3)]
    out = model(x, torch.tensor([500.0], device=device), rope_pos)
    assert out.shape == (1, 2, 4, 4, 12) and torch.isfinite(out).all()


@pytest.mark.parametrize(
    "missing",
    [
        "pooled_bias",
        "visual_transformer_blocks.0.visual_modulation.out_layer.bias",
    ],
)
def test_missing_checkpoint_tensor_fails_the_load(missing):
    """The generic loader zero-fills missing parameters whose names contain ``bias`` (and
    would only warn about others).  For this model every parameter is required, so a
    partial checkpoint must raise instead of yielding a silently different network."""
    checkpoint = _official_state_dict(_dit())
    del checkpoint[missing]
    with pytest.raises(ValueError, match="was not loaded"):
        _load_through_the_real_loader(checkpoint)


def test_unexpected_checkpoint_tensor_fails_the_load():
    checkpoint = _official_state_dict(_dit())
    checkpoint["text_embeddings.in_layer.weight"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="unexpected keys"):
        _load_through_the_real_loader(checkpoint)


def test_official_config_builds_the_head_from_the_total_width_and_loads_official_keys():
    """An official config stores the TOTAL DX head width in ``out_visual_dim`` (n_grid lives
    in the scheduler config) plus nested ``sr_params`` and ``attribute_overrides: null``.
    """
    official = {
        key: value
        for key, value in TINY_DIT.items()
        if key not in ("n_grid", "attribute_overrides")
    }
    official.update(
        out_visual_dim=12,
        attribute_overrides=None,
        sr_params=dict(
            scale_factor={"512": [1.0, 1.0, 1.0]},
            visual_size=[512],
            scheduler_scale=5.0,
            lq_noise_scale=0.7,
            lq_noise_type="ddpm",
            lq_channel_noise_scale=0.0,
            cap_noise_timestep=False,
            fps=24,
        ),
    )
    config = Kandinsky6SRDitConfig()
    config.update_model_arch(official)
    with torch.device("meta"):
        model = Kandinsky6SRTransformer3DModel(config, official)
    # head width = prod(patch_size) * the TOTAL out_visual_dim
    assert model.state_dict()["out_layer.out_layer.weight"].shape[0] == 4 * 12
    assert model.base_out_visual_dim == 4 and model.n_grid == 1

    load_model_from_full_model_state_dict(
        model,
        model.preprocess_loaded_state_dict(
            iter(
                _official_state_dict(_dit()).items()
            )  # same shapes as the legacy DX head
        ),
        torch.device("cpu"),
        torch.float32,
        strict=False,
        param_names_mapping=get_param_names_mapping(model.param_names_mapping),
    )
    model.post_load_weights()
    assert not any(t.is_meta for t in [*model.parameters(), *model.buffers()])


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


# --------------------------------------------------------------------------- #
# Fake server args for the component loaders
# --------------------------------------------------------------------------- #
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


def test_latent_upscaler_loader_loads_bank_in_bf16_eval(tmp_path):
    directory = _tiny_lu_bundle(tmp_path)
    bank, _ = LatentUpscalerLoader().load(
        str(directory), FakeServerArgs(), "latent_upscaler", "diffusers"
    )
    assert bank.scales == (2, 4)
    assert not bank.training
    assert all(
        p.dtype == torch.bfloat16 and not p.requires_grad for p in bank.parameters()
    )
    z = torch.randn(1, 4, 3, 4, 6, dtype=torch.bfloat16)
    assert bank.upscale(z, scale=2).shape == (1, 4, 3, 8, 12)
    assert bank.upscale(z, scale=4).shape == (1, 4, 3, 16, 24)
    with pytest.raises(ValueError, match="no x8 entry"):
        bank.upscale(z, scale=8)


def test_latent_upscaler_legacy_config_uses_current_default_scale_order(tmp_path):
    models = [
        {"target_scale": scale, "model": copy.deepcopy(TINY_LU_MODEL)}
        for scale in ("4x", "2x")
    ]
    directory = _tiny_lu_bundle(
        tmp_path,
        models=models,
        include_scales=False,
    )
    bank, _ = LatentUpscalerLoader().load(
        str(directory), FakeServerArgs(), "latent_upscaler", "diffusers"
    )
    assert bank.scales == (2, 4)
    assert bank._models[0] is bank.for_scale(2)
    assert bank._models[1] is bank.for_scale(4)


def test_latent_upscaler_may_be_declared_with_the_kandinsky6_library():
    """The official ``save_pretrained`` writes ["kandinsky6", ...], the Hub repo ["diffusers", ...]."""
    for library in ("diffusers", "kandinsky6"):
        loader = component_loader_module.ComponentLoader.for_component_type(
            "latent_upscaler", library
        )
        assert isinstance(loader, LatentUpscalerLoader)
    with pytest.raises(AssertionError, match="latent_upscaler must be loaded from"):
        component_loader_module.ComponentLoader.for_component_type(
            "latent_upscaler", "transformers"
        )


def test_latent_upscaler_load_errors_are_raised_not_swallowed(tmp_path):
    """``ComponentLoader.load`` falls back to a diffusers ``AutoModel`` when the
    customized loader raises, which hides real errors.  This component must re-raise,
    with the offending config key or tensor named."""
    directory = _tiny_lu_bundle(tmp_path)
    weights_path = directory / "diffusion_pytorch_model.safetensors"
    weights = load_file(str(weights_path))
    weights.pop(next(iter(weights)))
    save_file(weights, str(weights_path))
    with pytest.raises(RuntimeError, match="native fallback is disabled") as error:
        LatentUpscalerLoader().load(
            str(directory), FakeServerArgs(), "latent_upscaler", "diffusers"
        )
    assert "Missing key" in str(error.value.__cause__)

    bad = copy.deepcopy(TINY_LU_MODEL)
    bad["bogus_field"] = 1
    directory = _tiny_lu_bundle(
        tmp_path / "bad",
        models=[{"target_scale": "2x", "model": TINY_LU_MODEL}],
        scales=(2,),
    )
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text())
    config["models"][0]["model"] = bad
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="native fallback is disabled") as error:
        LatentUpscalerLoader().load(
            str(directory), FakeServerArgs(), "latent_upscaler", "diffusers"
        )
    assert "bogus_field" in str(error.value.__cause__)


def test_motion_attention_is_not_a_supported_entry_key():
    """The consolidated architecture (mirroring FastVideo's ``kandinsky6_sr.py`` /
    commit 950a5edb) only implements the released checkpoint, which never trains
    natten/shifted-window motion attention: a config that asks for it must fail loudly
    at construction instead of silently building an unsupported backend."""
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
