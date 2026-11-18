# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.qwenimage import (
    QwenImageDitConfig,
    QwenImageEditPlus_2511_DitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPlus_2511_PipelineConfig,
    QwenImageEditPlusPipelineConfig,
    QwenImageLayeredPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)

_DEVICE = torch.device("cuda")
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the tiny DiT runs fused CUDA kernels"
)
# 32x32 pixels pack into 2x2 latent tokens; the edit reference into 3x2
HEIGHT = WIDTH = 32
REF_SIZE = (32, 48)
CHANNELS = 8
TEXT_DIM = 12


def _tiny_dit(dit_config):
    for key, value in dict(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        num_layers=2,
        attention_head_dim=128,
        num_attention_heads=2,
        joint_attention_dim=TEXT_DIM,
    ).items():
        setattr(dit_config.arch_config, key, value)
    model = QwenImageTransformer2DModel(dit_config, {})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    return model.to(device=_DEVICE, dtype=torch.bfloat16)


def _request(cond_len, uncond_len, edit, dtype):
    generator = torch.Generator().manual_seed(1)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(_DEVICE, dtype)

    noise_tokens = (HEIGHT // 16) * (WIDTH // 16)
    batch = SimpleNamespace(
        height=HEIGHT,
        width=WIDTH,
        latents=randn(1, noise_tokens, CHANNELS),
        prompt_embeds=[randn(1, cond_len, TEXT_DIM)],
        negative_prompt_embeds=[randn(1, uncond_len, TEXT_DIM)],
        vae_image_sizes=[REF_SIZE],
        do_classifier_free_guidance=True,
        is_cfg_negative=False,
        true_cfg_scale=None,
        guidance_scale=4.0,
        cfg_normalization=0,
        guidance_rescale=0,
        sampling_params=SimpleNamespace(quality=None),
        extra={},
    )
    latent_input = batch.latents
    if edit:
        ref_tokens = (REF_SIZE[1] // 16) * (REF_SIZE[0] // 16)
        reference = randn(1, ref_tokens, CHANNELS)
        latent_input = torch.cat([batch.latents, reference], dim=1)
    return batch, latent_input


def _stage(compile_only_offload=None):
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = SimpleNamespace(
        dit_layerwise_offload=compile_only_offload is None,
        is_dit_layerwise_offload_selected=False,
    )
    stage._cache_dit_enabled = False
    stage._extra_func_kwarg_names_cache = {}
    stage._offloaded_dit_modules_for_compile = (
        [] if compile_only_offload is None else [compile_only_offload]
    )
    return stage


@torch.no_grad()
def _denoise_step(
    pipeline_config,
    model,
    cond_len,
    uncond_len,
    *,
    edit,
    streamed,
    stage=None,
    resident_layers=0,
):
    dtype = next(model.parameters()).dtype
    batch, latent_input = _request(cond_len, uncond_len, edit, dtype)
    branch_kwargs = []
    for embeds, prepare in (
        (batch.prompt_embeds, pipeline_config.prepare_pos_cond_kwargs),
        (batch.negative_prompt_embeds, pipeline_config.prepare_neg_cond_kwargs),
    ):
        kwargs = prepare(batch, _DEVICE, model.rotary_emb, dtype)
        branch_kwargs.append(kwargs | dict(encoder_hidden_states=embeds))
    policy = CFGPolicy().build(batch, {}, *branch_kwargs)
    stage = stage or _stage()
    stage.server_args.enable_breakable_cuda_graph = False
    widths = []
    forward = model.forward

    def counting_forward(**kwargs):
        widths.append(kwargs["hidden_states"].shape[0])
        return forward(**kwargs)

    server_args = SimpleNamespace(
        enable_cfg_parallel=False,
        enable_breakable_cuda_graph=False,
        pipeline_config=pipeline_config,
    )
    with (
        patch.object(model, "forward", counting_forward),
        patch.object(
            model,
            "layerwise_offload_managers",
            [
                SimpleNamespace(
                    enabled=streamed,
                    resident_layers=resident_layers,
                    num_layers=len(model.transformer_blocks),
                )
            ],
        ),
    ):
        out = stage._predict_noise_with_cfg(
            current_model=model,
            latent_model_input=latent_input,
            timestep=torch.tensor([500.0], device=_DEVICE, dtype=dtype),
            batch=batch,
            timestep_index=0,
            attn_metadata=None,
            target_dtype=dtype,
            current_guidance_scale=4.0,
            cfg_policy=policy,
            cfg_gate_state=None,
            server_args=server_args,
            guidance=None,
            latents=batch.latents,
        )
    return out, widths


CASES = [
    (QwenImagePipelineConfig, QwenImageDitConfig, False),
    (QwenImageEditPlusPipelineConfig, QwenImageDitConfig, True),
    (QwenImageEditPlus_2511_PipelineConfig, QwenImageEditPlus_2511_DitConfig, True),
]


@requires_cuda
@pytest.mark.parametrize("pipeline_cls,dit_cls,edit", CASES)
@pytest.mark.parametrize("cond_len,uncond_len", [(7, 3), (3, 7), (5, 5)])
def test_streamed_dit_runs_both_cfg_branches_in_one_forward(
    single_process_model_parallel, pipeline_cls, dit_cls, edit, cond_len, uncond_len
):
    model = _tiny_dit(dit_cls())
    serial, serial_widths = _denoise_step(
        pipeline_cls(), model, cond_len, uncond_len, edit=edit, streamed=False
    )
    batched, batched_widths = _denoise_step(
        pipeline_cls(), model, cond_len, uncond_len, edit=edit, streamed=True
    )
    assert serial_widths == [1, 1]
    assert batched_widths == [2]
    assert batched.shape == serial.shape
    # bf16 tolerance; a wrong mask or timestep row moves the output far more
    rel_l2 = (batched.float() - serial.float()).norm() / serial.float().norm()
    assert rel_l2 < 0.05, rel_l2


def test_layered_stays_serial():
    kwargs = dict(
        encoder_hidden_states=[torch.zeros(1, 3, TEXT_DIM)],
        encoder_hidden_states_mask=None,
        txt_seq_lens=[3],
        img_shapes=[[(1, 2, 2), (1, 2, 2)]],
        freqs_cis=(torch.zeros(8, 4), torch.zeros(3, 4)),
        additional_t_cond=torch.zeros(1, dtype=torch.long),
    )
    config = QwenImageLayeredPipelineConfig()
    assert config.batch_cfg_branch_kwargs(kwargs, dict(kwargs)) is None


@requires_cuda
def test_dit_offloaded_only_for_compile_warmup_stays_serial(
    single_process_model_parallel,
):
    model = _tiny_dit(QwenImageDitConfig())
    stage = _stage(compile_only_offload=model)
    _, widths = _denoise_step(
        QwenImagePipelineConfig(), model, 7, 3, edit=False, streamed=True, stage=stage
    )
    assert widths == [1, 1]

    stage.server_args.dit_layerwise_offload = True
    _, widths = _denoise_step(
        QwenImagePipelineConfig(), model, 7, 3, edit=False, streamed=True, stage=stage
    )
    assert widths == [2]


@requires_cuda
def test_dit_with_every_layer_resident_stays_serial(single_process_model_parallel):
    model = _tiny_dit(QwenImageDitConfig())
    num_layers = len(model.transformer_blocks)
    _, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        7,
        3,
        edit=False,
        streamed=True,
        resident_layers=num_layers,
    )
    assert widths == [1, 1]

    _, widths = _denoise_step(
        QwenImagePipelineConfig(),
        model,
        7,
        3,
        edit=False,
        streamed=True,
        resident_layers=num_layers - 1,
    )
    assert widths == [2]
