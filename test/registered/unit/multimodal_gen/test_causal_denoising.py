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


def test_causal_dmd_block_updates_context_after_denoising():
    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    seen = {}

    def fake_denoise(self, *args, **kwargs):
        del self, args
        seen["model_input"] = kwargs["prepare_model_input"](
            kwargs["chunk_latents"]
        ).clone()
        seen["denoise_pos"] = (
            kwargs["current_start_tokens"],
            kwargs["start_frame"],
        )
        return kwargs["chunk_latents"] + 2, "metadata"

    def fake_update(self, *args, **kwargs):
        del self, args
        seen["context_input"] = kwargs["context_input"].clone()
        seen["update_pos"] = (
            kwargs["current_start_tokens"],
            kwargs["start_frame"],
            kwargs["attn_metadata"],
        )

    stage._denoise_causal_dmd_chunk = MethodType(fake_denoise, stage)
    stage._update_causal_context_cache = MethodType(fake_update, stage)

    chunk_latents = torch.ones(1, 2, 1, 1, 1)
    condition = torch.full((1, 1, 1, 1, 1), 4.0)
    result = stage._denoise_and_update_causal_block(
        SimpleNamespace(),
        SimpleNamespace(),
        chunk_latents=chunk_latents,
        scheduler=SimpleNamespace(),
        timesteps=torch.tensor([1]),
        prompt_embeds=None,
        kv_cache=[],
        crossattn_cache=[],
        current_start_tokens=8,
        start_frame=2,
        image_kwargs={},
        pos_cond_kwargs={},
        target_dtype=torch.float16,
        autocast_enabled=False,
        device=torch.device("cpu"),
        attn_raw_latent_shape=(1, 1, 1),
        prepare_model_input=lambda x: torch.cat([x, condition], dim=1),
        prepare_context_input=lambda x: torch.cat([x, condition], dim=1),
    )

    assert torch.equal(result, chunk_latents + 2)
    assert seen["denoise_pos"] == (8, 2)
    assert seen["update_pos"] == (8, 2, "metadata")
    assert torch.equal(
        seen["model_input"], torch.cat([chunk_latents, condition], dim=1)
    )
    assert torch.equal(
        seen["context_input"], torch.cat([chunk_latents + 2, condition], dim=1)
    )


def test_causal_context_warmup_uses_context_cache_update_path():
    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    stage.frame_seq_length = 4
    seen = {}

    def fake_update(self, *args, **kwargs):
        del self, args
        seen.update(kwargs)

    stage._update_causal_context_cache = MethodType(fake_update, stage)
    context_input = torch.randn(1, 2, 3, 4, 4)
    stage._warm_up_causal_context_cache(
        SimpleNamespace(),
        SimpleNamespace(),
        context_input=context_input,
        prompt_embeds="prompt",
        kv_cache=["kv"],
        crossattn_cache=["cross"],
        current_start_frame=3,
        image_kwargs={},
        pos_cond_kwargs={},
        target_dtype=torch.float16,
        autocast_enabled=False,
    )

    assert seen["context_input"] is context_input
    assert seen["current_start_tokens"] == 12
    assert seen["start_frame"] == 3
    assert seen["attn_metadata"] is None


def test_causal_cache_helpers_reset_and_forward_model_specific_kwargs():
    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    stage.num_transformer_blocks = 2
    kv_cache = [
        {
            "global_end_index": torch.tensor([7]),
            "local_end_index": torch.tensor([3]),
            "global_end_index_int": 7,
            "local_end_index_int": 3,
        },
        {"global_end_index": torch.tensor([5]), "local_end_index": torch.tensor([2])},
    ]
    crossattn_cache = [{"is_init": True}, {"is_init": True}]
    stage._reset_causal_caches(
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        device=torch.device("cpu"),
    )

    assert [block["is_init"] for block in crossattn_cache] == [False, False]
    assert [int(block["global_end_index"].item()) for block in kv_cache] == [0, 0]
    assert [int(block["local_end_index"].item()) for block in kv_cache] == [0, 0]
    assert kv_cache[0]["global_end_index_int"] == 0
    assert kv_cache[0]["local_end_index_int"] == 0

    calls = []

    def fake_initialize_kv_cache(self, batch_size, dtype, device, **kwargs):
        calls.append(("kv", batch_size, dtype, device, kwargs))
        self.kv_cache1 = ["kv-cache"]

    def fake_initialize_crossattn_cache(self, batch_size, max_text_len, dtype, device):
        calls.append(("cross", batch_size, max_text_len, dtype, device))
        self.crossattn_cache = ["cross-cache"]

    stage._initialize_kv_cache = MethodType(fake_initialize_kv_cache, stage)
    stage._initialize_crossattn_cache = MethodType(
        fake_initialize_crossattn_cache, stage
    )

    kv_cache, crossattn_cache = stage._initialize_causal_caches(
        batch_size=2,
        max_text_len=128,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        kv_cache_kwargs={"sequence_shard_enabled": True},
    )

    assert kv_cache == ["kv-cache"]
    assert crossattn_cache == ["cross-cache"]
    assert calls == [
        (
            "kv",
            2,
            torch.bfloat16,
            torch.device("cpu"),
            {"sequence_shard_enabled": True},
        ),
        ("cross", 2, 128, torch.bfloat16, torch.device("cpu")),
    ]
