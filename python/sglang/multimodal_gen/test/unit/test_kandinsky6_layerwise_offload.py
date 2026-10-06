# SPDX-License-Identifier: Apache-2.0
"""Exercise the declared offload groups, not just component selection."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.kandinsky6 import (
    Kandinsky6Transformer3DModel,
)
from sglang.multimodal_gen.runtime.models.upsampler.kandinsky6_sr_latent_upscaler import (
    Kandinsky6SRLatentUpscalerBank,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_audio import (
    Kandinsky6AudioVAE,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_sr_vae import Kandinsky6SRVAE
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils import precision
from sglang.multimodal_gen.test.unit.kandinsky6_sr_tiny_components import TINY_LU_MODEL
from sglang.srt.utils.network import get_free_port_below_ephemeral

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def single_gpu(monkeypatch, default_global_server_args):
    if not model_parallel_is_initialized():
        port = get_free_port_below_ephemeral()
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            monkeypatch.setenv(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    monkeypatch.setattr(
        precision._mixed_precision_state,
        "state",
        precision.MixedPrecisionState(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        ),
    )
    args = get_global_server_args()
    args.attention_backend = "fa"
    args.pipeline_config = Kandinsky6TI2VAPipelineConfig()
    with global_force_attn_backend_context_manager(AttentionBackendEnum.FA):
        yield


def _dit(multimodal):
    config = Kandinsky6VideoAudioConfig()
    config.update_model_arch(
        dict(
            model_dim=128,
            time_dim=32,
            ff_dim=256,
            axes_dims=(32, 48, 48),
            in_visual_dim=4,
            out_visual_dim=4,
            in_text_dim=16,
            in_text_dim2=8,
            in_audio_dim=4,
            out_audio_dim=4,
            num_text_blocks=2,
            num_visual_blocks=2,
            patch_size=(1, 1, 1),
            visual_cond=False,
            visual_token_type_num_embeddings=0,
            is_multimodal=multimodal,
            model_dim_a=128,
            time_dim_a=32,
            ff_dim_a=256,
            axes_dims_a=(32, 48, 48),
            ca_rope=True,
            cross_gates=True,
            fix_modulation=True,
        )
    )
    model = Kandinsky6Transformer3DModel(config, {})
    # TP-aware linears can use uninitialized storage before checkpoint loading
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.02)
    model.to(device="cuda", dtype=torch.bfloat16).eval()
    inputs = dict(
        hidden_states=torch.randn(1, 2, 3, 3, 4, device="cuda", dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(
            1, 7, 16, device="cuda", dtype=torch.bfloat16
        ),
        pooled_projections=torch.randn(1, 8, device="cuda", dtype=torch.bfloat16),
        timestep=torch.tensor([500.0], device="cuda"),
        visual_rope_pos=[torch.arange(n, device="cuda") for n in (2, 3, 3)],
        text_rope_pos=torch.arange(7, device="cuda"),
    )
    if multimodal:
        inputs["hidden_states_audio"] = torch.randn(
            1, 5, 4, device="cuda", dtype=torch.bfloat16
        )
    groups = (
        ["video_text_transformer_blocks", "audio_text_transformer_blocks"]
        if multimodal
        else ["text_transformer_blocks"]
    )
    return model, lambda: model(**inputs), groups + ["visual_transformer_blocks"]


def _audio_vae():
    config = Kandinsky6AudioVAEConfig()
    config.arch_config.mode = "16k"
    config.arch_config.vocoder_config = dict(
        resblock="1",
        num_mels=80,
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        upsample_initial_channel=8,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 1, 1]],
        activation="snake",
        snake_logscale=False,
    )
    model = Kandinsky6AudioVAE(config).to("cuda").eval()
    mel = torch.randn(1, 80, 8, device="cuda")

    def forward():
        latent = model.vae.encode(mel).mean
        decoded = model.decode(latent)
        return latent, decoded, model.vocode(decoded)

    groups = [f"vae.encoder.down.{i}.block" for i in range(3)]
    groups += [f"vae.decoder.up.{i}.block" for i in range(3)]
    return model, forward, groups + ["vocoder.ups.0", "vocoder.resblocks"]


def _sr_vae():
    config = Kandinsky6SRVAEConfig()
    common = dict(
        ch=8,
        ch_mult=(1, 1, 2, 2, 2),
        num_res_blocks=2,
        z_channels=4,
        temporal_compress_times=4,
        norm_type="rms_norm",
    )
    config.update_model_arch(
        dict(
            vae_type="video-kvae",
            encoder_config=dict(common, in_channels=3),
            decoder_config=dict(common, out_ch=3),
        )
    )
    model = Kandinsky6SRVAE(config).to("cuda").eval()
    pixels = torch.randn(1, 3, 33, 32, 32, device="cuda")

    def forward():
        latent, segments = model.encode(pixels)
        assert segments == [17, 16]
        return latent, model.decode(latent).sample

    groups = [f"encoder.down.{i}.block" for i in range(5)]
    groups += [f"decoder.up.{i}.block" for i in range(5)]
    return model, forward, groups


def _upscaler():
    model = (
        Kandinsky6SRLatentUpscalerBank(
            [{"target_scale": scale, "model": dict(TINY_LU_MODEL)} for scale in (2, 4)]
        )
        .to("cuda")
        .eval()
    )
    latent = torch.randn(1, 4, 3, 3, 3, device="cuda")

    def forward():
        # exercise both branches of every loaded entry, including width changes
        return tuple(
            entry(latent, scale) for entry in model._models for scale in (2, 4)
        )

    groups = [
        f"_models.{index}.{group}"
        for index in range(2)
        for group in (
            "pre_blocks",
            "mid_blocks",
            "post_blocks",
            "x2_branch.adapter",
            "x2_branch.mid_blocks",
            "x2_branch.blocks",
        )
    ]
    return model, forward, groups


@pytest.mark.parametrize(
    "component", ["video-dit", "joint-dit", "audio-vae", "sr-vae", "upscaler"]
)
@pytest.mark.parametrize("resident_layers", [0, 1])
@torch.no_grad()
def test_layerwise_groups_preserve_repeated_forwards(component, resident_layers):
    torch.manual_seed(13)
    if component.endswith("dit"):
        model, forward, groups = _dit(component == "joint-dit")
    elif component == "audio-vae":
        model, forward, groups = _audio_vae()
    elif component == "sr-vae":
        model, forward, groups = _sr_vae()
    else:
        model, forward, groups = _upscaler()
    assert isinstance(model, LayerwiseOffloadableModuleMixin)
    assert set(groups) <= set(model.layer_names)
    assert model.layerwise_offload_dit_group_enabled == component.endswith("dit")
    state = {name: value.cpu().clone() for name, value in model.state_dict().items()}
    with (
        torch.inference_mode(),
        set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        expected = forward()
        model.configure_layerwise_offload(
            SimpleNamespace(
                performance_mode="speed",
                pin_cpu_memory=True,
                layerwise_tuning_for=lambda *args, **kwargs: (
                    1,
                    resident_layers,
                    "leading",
                    "forward",
                ),
            )
        )
        assert {
            manager.layers_attr_str for manager in model.layerwise_offload_managers
        } == set(groups)
        for _ in range(2):
            model.prepare_for_next_req()
            torch.testing.assert_close(forward(), expected, rtol=0, atol=0)
            for manager in model.layerwise_offload_managers:
                assert manager._last_forwarded_layer == manager.num_layers - 1
                manager.release_all()
        model.disable_offload()
        torch.testing.assert_close(forward(), expected, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), state[name], rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
