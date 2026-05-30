# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)


class _Progress:
    def __init__(self):
        self.count = 0

    def update(self):
        self.count += 1


def test_causal_dmd_chunk_loop_uses_model_input_builder():
    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    predict_calls = []
    add_noise_calls = []

    def fake_predict(self, *args, **kwargs):
        del self, args
        latent_model_input = kwargs["latent_model_input"]
        predict_calls.append(
            (
                latent_model_input.shape,
                latent_model_input.dtype,
                int(kwargs["current_timestep"]),
            )
        )
        x0_btchw = latent_model_input[:, :2].permute(0, 2, 1, 3, 4).float()
        return x0_btchw + int(kwargs["current_timestep"]), kwargs["current_timestep"]

    def fake_add_noise(self, *args, **kwargs):
        del self, args
        add_noise_calls.append(int(kwargs["next_timestep"].item()))
        return kwargs["x0_btchw"] + 10

    stage._predict_x0_btchw = MethodType(fake_predict, stage)
    stage._add_noise_for_next_timestep = MethodType(fake_add_noise, stage)

    chunk_latents = torch.zeros(1, 2, 2, 1, 1)
    condition = torch.ones(1, 1, 2, 1, 1)
    prepare_call_count = 0

    def prepare_model_input(current_latents):
        nonlocal prepare_call_count
        prepare_call_count += 1
        return torch.cat([current_latents, condition], dim=1)

    progress = _Progress()
    result, attn_metadata = stage._denoise_causal_dmd_chunk(
        SimpleNamespace(generator=None),
        SimpleNamespace(),
        chunk_latents=chunk_latents,
        scheduler=SimpleNamespace(),
        timesteps=torch.tensor([7, 3]),
        prompt_embeds=None,
        kv_cache=[],
        crossattn_cache=[],
        current_start_tokens=0,
        start_frame=0,
        image_kwargs={},
        pos_cond_kwargs={},
        target_dtype=torch.float16,
        autocast_enabled=False,
        device=torch.device("cpu"),
        attn_raw_latent_shape=(2, 1, 1),
        prepare_model_input=prepare_model_input,
        progress_bar=progress,
    )

    assert prepare_call_count == 2
    assert predict_calls == [
        (torch.Size([1, 3, 2, 1, 1]), torch.float16, 0),
        (torch.Size([1, 3, 2, 1, 1]), torch.float16, 1),
    ]
    assert add_noise_calls == [3]
    assert progress.count == 2
    assert attn_metadata == 1
    assert torch.equal(result, torch.full_like(result, 11))
