# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.pipelines_core.comfyui_session import (
    bind_comfyui_session,
    release_comfyui_session,
)


class _Req:
    def __init__(self):
        self.extra = {}
        self.prompt_embeds = []
        self.prompt_seq_lens = None
        self.pooled_embeds = []


def test_latent_prep_verify_input_restores_cached_embeds() -> None:
    """Later ComfyUI steps omit embeds; verify_input must restore before checks."""
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.comfyui_latent_preparation import (
        ComfyUILatentPreparationStage,
    )

    sid = "run-verify"
    embeds = [torch.ones(2, 4)]
    first = _Req()
    first.extra["comfyui_session_id"] = sid
    first.prompt_embeds = embeds
    first.prompt_seq_lens = [[2]]
    bind_comfyui_session(first)

    batch = SimpleNamespace(
        extra={"comfyui_session_id": sid},
        prompt_embeds=[],
        prompt=" ",
        num_outputs_per_prompt=1,
        generator=torch.Generator("cpu"),
        num_frames=1,
        height=64,
        width=64,
        latents=None,
        prompt_seq_lens=None,
        pooled_embeds=None,
        negative_prompt_embeds=None,
        negative_prompt_seq_lens=None,
        neg_pooled_embeds=None,
        image_latent=None,
        vae_image_sizes=None,
        prompt_attention_mask=None,
        negative_attention_mask=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
    )
    stage = ComfyUILatentPreparationStage(scheduler=None, transformer=None)
    result = stage.verify_input(batch, server_args=None)
    assert result.is_valid()
    assert torch.equal(batch.prompt_embeds[0], embeds[0])
    release_comfyui_session(sid)


def test_session_restores_conditioning_on_later_steps() -> None:
    sid = "run-1"
    first = _Req()
    first.extra["comfyui_session_id"] = sid
    first.prompt_embeds = [torch.ones(4, 8)]
    first.prompt_seq_lens = [[4]]
    bind_comfyui_session(first)

    second = _Req()
    second.extra["comfyui_session_id"] = sid
    bind_comfyui_session(second)

    assert torch.equal(second.prompt_embeds[0], first.prompt_embeds[0])
    assert second.prompt_seq_lens == [[4]]
    release_comfyui_session(sid)

    third = _Req()
    third.extra["comfyui_session_id"] = sid
    bind_comfyui_session(third)
    assert third.prompt_embeds == []
