# SPDX-License-Identifier: Apache-2.0

import uuid

import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    ComfyUIModelAdapter,
    PackedForward,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.minimax_h3 import (
    MiniMaxH3Adapter,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.zimage import (
    ZImageAdapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
    get_run_state,
    initialize_comfyui_pipeline,
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


class _Executor(SGLDiffusionExecutor):
    """Real executor bookkeeping, without a generator or model."""

    def __init__(self, adapter):
        torch.nn.Module.__init__(self)
        self.adapter = adapter
        self.session_id = uuid.uuid4().hex
        self._run_id = 0
        self._sent_conds = set()
        self.begin_sampler_run()
        self.sid = self.comfyui_session_id()

    def send(self, packed) -> _Req:
        """Executor half of _execute_packed, then the worker-side bind."""
        self._mark_and_maybe_drop(packed)
        req = _Req()
        self.adapter.fill_req(req, packed)
        req.extra = {
            **(req.extra or {}),
            "comfyui_session_id": self.sid,
            "comfyui_cond_key": packed.extra_req["comfyui_cond_key"],
        }
        return bind_comfyui_session(req)


def test_cond_key_flux_same_pooled_different_t5_not_conflated() -> None:
    # ComfyUI's Flux pooled `y` comes from CLIP-L's first 77-token chunk only,
    # so prompts that differ later share `y` but not the T5 context.
    ex = _Executor(FluxAdapter())
    x, t = torch.zeros(1, 16, 8, 8), torch.tensor([0.5])
    y = torch.randn(1, 768)
    ctx_a, ctx_b = torch.randn(1, 8, 4096), torch.randn(1, 8, 4096)
    first = ex.send(ex.adapter.pack(x, t, ctx_a, y=y))
    second = ex.send(ex.adapter.pack(x, t, ctx_b, y=y.clone()))
    assert torch.equal(first.prompt_embeds[1], ctx_a)
    assert torch.equal(second.prompt_embeds[1], ctx_b)
    release_comfyui_session(ex.sid)


def test_cond_key_same_first_last_scalar_not_conflated() -> None:
    ex = _Executor(ZImageAdapter())
    x, t = torch.zeros(1, 16, 8, 8), torch.tensor([0.5])
    ctx_a, ctx_b = torch.randn(1, 6, 32), torch.randn(1, 6, 32)
    ctx_b.view(-1)[0] = ctx_a.view(-1)[0]
    ctx_b.view(-1)[-1] = ctx_a.view(-1)[-1]
    ex.send(ex.adapter.pack(x, t, ctx_a))
    second = ex.send(ex.adapter.pack(x, t, ctx_b))
    assert torch.equal(second.prompt_embeds[0], ctx_b.squeeze(0))
    release_comfyui_session(ex.sid)


def test_cond_key_covers_image_latent() -> None:
    def packed(image_latent):
        return PackedForward(
            latents=torch.zeros(1, 4, 8),
            timesteps=torch.tensor([500.0]),
            prompt_embeds=[torch.ones(3, 8)],
            prompt_seq_lens=[[3]],
            height=64,
            width=64,
            extra_req={"image_latent": image_latent},
        )

    ex = _Executor(ComfyUIModelAdapter())
    ref_a, ref_b = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    ex.send(packed(ref_a))
    second = ex.send(packed(ref_b))
    assert torch.equal(second.image_latent, ref_b)
    release_comfyui_session(ex.sid)


def test_cond_key_repeat_still_uses_cache() -> None:
    ex = _Executor(FluxAdapter())
    x, t = torch.zeros(1, 16, 8, 8), torch.tensor([0.5])
    y, ctx = torch.randn(1, 768), torch.randn(1, 8, 4096)
    ex.send(ex.adapter.pack(x, t, ctx, y=y))
    # ComfyUI hands over fresh tensors each step; equal content must still hit.
    repeat = ex.adapter.pack(x, t, ctx.clone(), y=y.clone())
    restored = ex.send(repeat)
    assert repeat.prompt_embeds == []
    assert torch.equal(restored.prompt_embeds[1], ctx)
    release_comfyui_session(ex.sid)


def test_cond_key_flux_without_pooled_keeps_t5_apart() -> None:
    ex = _Executor(FluxAdapter())
    x, t = torch.zeros(1, 16, 8, 8), torch.tensor([0.5])
    ctx_a, ctx_b = torch.randn(1, 8, 4096), torch.randn(1, 8, 4096)
    first = ex.send(ex.adapter.pack(x, t, ctx_a, y=None))
    second = ex.send(ex.adapter.pack(x, t, ctx_b, y=None))
    assert torch.equal(first.prompt_embeds[1], ctx_a)
    assert torch.equal(second.prompt_embeds[1], ctx_b)
    repeat = ex.adapter.pack(x, t, ctx_a.clone(), y=None)
    restored = ex.send(repeat)
    assert repeat.prompt_embeds == []  # cached now that y is a tensor
    assert torch.equal(restored.prompt_embeds[1], ctx_a)
    release_comfyui_session(ex.sid)


def _h3_packed(text, payload):
    return PackedForward(
        latents=torch.zeros(1, 4, 2, 2),
        timesteps=torch.tensor([500.0]),
        prompt_embeds=[text],
        prompt_seq_lens=[[int(text.shape[0])]],
        height=2,
        width=2,
        extra_req={
            "h3_payload": payload,
            "h3_context": text,
            "comfyui_cache_fp": {"spatial": (1, 2, 2)},
        },
    )


def test_h3_cache_hit_restores_own_extras() -> None:
    ex = _Executor(MiniMaxH3Adapter())
    pos, neg = torch.ones(3, 4), torch.zeros(3, 4)
    pos_payload, neg_payload = {"text_token_tags": [1, 2]}, {"text_token_tags": [3]}
    ex.send(_h3_packed(pos, pos_payload))
    ex.send(_h3_packed(neg, neg_payload))
    later_pos = _h3_packed(pos.clone(), dict(pos_payload))
    restored = ex.send(later_pos)
    assert "h3_context" not in later_pos.extra_req  # cache hit, extras dropped
    assert torch.equal(restored.extra["h3_context"], pos)
    assert restored.extra["h3_payload"] == pos_payload
    release_comfyui_session(ex.sid)


def test_h3_same_text_different_payload_not_conflated() -> None:
    ex = _Executor(MiniMaxH3Adapter())
    text = torch.ones(3, 4)
    ex.send(_h3_packed(text, {"refs": [torch.zeros(2, 2)]}))
    second = ex.send(_h3_packed(text.clone(), {"refs": [torch.ones(2, 2)]}))
    assert torch.equal(second.extra["h3_payload"]["refs"][0], torch.ones(2, 2))
    release_comfyui_session(ex.sid)


def test_cond_key_hashes_large_tensors_inside_lists() -> None:
    ex = _Executor(MiniMaxH3Adapter())
    text = torch.ones(3, 4)
    latent_a = torch.zeros(4096)
    latent_b = latent_a.clone()
    latent_b[2048] = 1.0  # outside what repr() would print
    key_a = ex._cond_key(_h3_packed(text, {"cond_video_latents": [latent_a]}))
    key_b = ex._cond_key(_h3_packed(text, {"cond_video_latents": [latent_b]}))
    assert key_a != key_b


def test_extras_cached_per_cond_key() -> None:
    sid = "exec4:1"
    pos_ctx, neg_ctx = torch.ones(2, 4), torch.zeros(2, 4)
    for key, ctx in (("pos", pos_ctx), ("neg", neg_ctx)):
        req = _Req()
        req.extra = {
            "comfyui_session_id": sid,
            "comfyui_cond_key": key,
            "h3_context": ctx,
        }
        req.prompt_embeds = [ctx]
        bind_comfyui_session(req)

    later_pos = _Req()
    later_pos.extra = {"comfyui_session_id": sid, "comfyui_cond_key": "pos"}
    bind_comfyui_session(later_pos)
    assert torch.equal(later_pos.extra["h3_context"], pos_ctx)
    release_comfyui_session(sid)


class _FakePipeline:
    def __init__(self):
        self.modules = {}


def _init_vae_geometry(pipeline_config):
    from types import SimpleNamespace

    initialize_comfyui_pipeline(
        _FakePipeline(),
        SimpleNamespace(pipeline_config=pipeline_config, comfyui_mode=True),
    )
    return pipeline_config.vae_config.arch_config


def test_comfyui_mode_derives_flux_vae_scale_factor() -> None:
    from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig

    arch = _init_vae_geometry(FluxPipelineConfig())
    assert arch.vae_scale_factor == 8


def test_comfyui_mode_derives_qwen_vae_scale_factor() -> None:
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
        QwenImagePipelineConfig,
    )

    arch = _init_vae_geometry(QwenImagePipelineConfig())
    assert arch.vae_scale_factor == 8


def test_comfyui_mode_h3_vae_post_init_without_latent_stats() -> None:
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        MiniMaxH3PipelineConfig,
    )

    arch = _init_vae_geometry(MiniMaxH3PipelineConfig())
    assert arch.latents_mean is None
    assert arch.latents_std is None
