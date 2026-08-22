# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
    get_run_state,
    release_comfyui_session,
    release_run_state,
    set_run_state,
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


def test_release_run_state_keeps_conditioning():
    sid = "run-keep-cond"
    first = _Req()
    first.extra = {"comfyui_session_id": sid, "model_payload": {"audio_scale": 0.5}}
    first.prompt_embeds = [torch.ones(2, 4)]
    bind_comfyui_session(first)
    set_run_state(first, {"branch": "alive"})
    release_run_state(sid)
    assert get_run_state(first) is None
    second = _Req()
    second.extra = {"comfyui_session_id": sid}
    bind_comfyui_session(second)
    assert torch.equal(second.prompt_embeds[0], first.prompt_embeds[0])
    assert second.extra["model_payload"]["audio_scale"] == 0.5
    release_comfyui_session(sid)


def test_session_restores_extra_keys_on_later_steps() -> None:
    sid = "run-extra"
    payload = {"audio_scale": 0.5}
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    first = _Req()
    first.extra = {
        "comfyui_session_id": sid,
        "model_payload": payload,
        "sample_sigmas": sigmas,
    }
    first.prompt_embeds = [torch.ones(3, 4)]
    bind_comfyui_session(first)

    second = _Req()
    second.extra = {"comfyui_session_id": sid}
    bind_comfyui_session(second)
    assert second.extra["model_payload"] is payload
    assert torch.equal(second.extra["sample_sigmas"], sigmas)
    assert torch.equal(second.prompt_embeds[0], first.prompt_embeds[0])
    release_comfyui_session(sid)


def test_new_run_id_evicts_previous_run_for_same_executor() -> None:
    first = _Req()
    first.extra = {"comfyui_session_id": "exec1:1", "h3_layout": {"signature": (1, 2)}}
    first.prompt_embeds = [torch.ones(2, 4)]
    bind_comfyui_session(first)
    set_run_state(first, {"branch": "old"})

    second = _Req()
    second.extra = {"comfyui_session_id": "exec1:2"}
    bind_comfyui_session(second)
    assert "h3_layout" not in second.extra
    assert second.prompt_embeds == []
    old = _Req()
    old.extra = {"comfyui_session_id": "exec1:1"}
    assert get_run_state(old) is None
    release_comfyui_session("exec1:2")


def test_fingerprint_mismatch_does_not_restore_extras() -> None:
    sid = "exec2:1"
    first = _Req()
    first.extra = {
        "comfyui_session_id": sid,
        "comfyui_cache_fp": {"spatial": (5, 15, 27)},
        "h3_layout": {"signature": (8, 5, 15, 27, 4)},
    }
    first.prompt_embeds = [torch.ones(2, 4)]
    bind_comfyui_session(first)

    second = _Req()
    second.extra = {
        "comfyui_session_id": sid,
        "comfyui_cache_fp": {"spatial": (5, 30, 54)},
    }
    bind_comfyui_session(second)
    assert "h3_layout" not in second.extra
    assert torch.equal(second.prompt_embeds[0], first.prompt_embeds[0])
    release_comfyui_session(sid)


def test_cond_keys_keep_positive_and_negative_apart() -> None:
    sid = "exec3:1"
    pos = _Req()
    pos.extra = {"comfyui_session_id": sid, "comfyui_cond_key": "pos"}
    pos.prompt_embeds = [torch.ones(2, 4)]
    bind_comfyui_session(pos)
    neg = _Req()
    neg.extra = {"comfyui_session_id": sid, "comfyui_cond_key": "neg"}
    neg.prompt_embeds = [torch.zeros(2, 4)]
    bind_comfyui_session(neg)

    later_pos = _Req()
    later_pos.extra = {"comfyui_session_id": sid, "comfyui_cond_key": "pos"}
    bind_comfyui_session(later_pos)
    later_neg = _Req()
    later_neg.extra = {"comfyui_session_id": sid, "comfyui_cond_key": "neg"}
    bind_comfyui_session(later_neg)
    assert torch.equal(later_pos.prompt_embeds[0], pos.prompt_embeds[0])
    assert torch.equal(later_neg.prompt_embeds[0], neg.prompt_embeds[0])
    release_comfyui_session(sid)
