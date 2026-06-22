# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
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


def test_causal_dmd_forward_context_uses_prepare_hooks(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages import causal_denoising

    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    stage.attn_backend = SimpleNamespace(get_enum=lambda: None)
    monkeypatch.setattr(
        causal_denoising, "get_local_torch_device", lambda: torch.device("cpu")
    )
    latents = torch.zeros(2, 3, 4, 5, 6)
    seen = {}

    def fake_prepare_frame_seq_length(self, h, w):
        self.num_token_per_frame = h * w
        seen["frame_shape"] = (h, w)
        return self.num_token_per_frame

    stage._target_dtype = MethodType(lambda self: torch.float16, stage)
    stage._autocast_enabled = MethodType(lambda self, dtype, server_args: False, stage)
    stage._get_causal_dmd_scheduler = MethodType(
        lambda self, batch, server_args: "scheduler", stage
    )
    stage._get_causal_dmd_latents = MethodType(lambda self, batch: latents, stage)
    stage._prepare_frame_seq_length = MethodType(fake_prepare_frame_seq_length, stage)
    stage._prepare_causal_dmd_timesteps = MethodType(
        lambda self, batch, server_args, scheduler, device: torch.tensor(
            [9], device=device
        ),
        stage,
    )
    stage._prepare_causal_dmd_image_kwargs = MethodType(
        lambda self, batch, server_args, target_dtype: {"image": target_dtype},
        stage,
    )
    stage._prepare_causal_dmd_pos_cond_kwargs = MethodType(
        lambda self, batch, server_args, target_dtype: {"pos": target_dtype},
        stage,
    )
    stage._prepare_causal_dmd_prompt_embeds = MethodType(
        lambda self, batch, server_args, target_dtype: ["prompt"],
        stage,
    )

    ctx = stage._prepare_causal_dmd_forward_context(
        SimpleNamespace(),
        SimpleNamespace(),
    )

    assert ctx.target_dtype == torch.float16
    assert ctx.autocast_enabled is False
    assert ctx.device == torch.device("cpu")
    assert ctx.scheduler == "scheduler"
    assert ctx.timesteps.tolist() == [9]
    assert ctx.latents is latents
    assert ctx.prompt_embeds == ["prompt"]
    assert ctx.image_kwargs == {"image": torch.float16}
    assert ctx.pos_cond_kwargs == {"pos": torch.float16}
    assert (ctx.batch_size, ctx.channels, ctx.num_frames, ctx.height, ctx.width) == (
        2,
        3,
        4,
        5,
        6,
    )
    assert seen["frame_shape"] == (5, 6)


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
    stage.num_token_per_frame = 4
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


def test_causal_dmd_timestep_expansion_preserves_fractional_dtype():
    timestep = torch.tensor(996.3375, dtype=torch.float32)

    expanded = CausalDMDDenoisingStage._expand_timestep(
        timestep,
        batch_size=3,
        device=torch.device("cpu"),
    )

    assert expanded.dtype == torch.float32
    assert torch.equal(expanded, torch.tensor([996.3375, 996.3375, 996.3375]))


def test_causal_cache_helpers_reset_and_forward_model_specific_kwargs():
    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    stage.num_transformer_blocks = 2
    kv_cache = [
        CausalSelfAttentionKVCache(
            k=torch.empty(1, 1, 1, 1),
            v=torch.empty(1, 1, 1, 1),
            global_end_index=torch.tensor([7]),
            local_end_index=torch.tensor([3]),
            global_end_index_int=7,
            local_end_index_int=3,
        ),
        CausalSelfAttentionKVCache(
            k=torch.empty(1, 1, 1, 1),
            v=torch.empty(1, 1, 1, 1),
            global_end_index=torch.tensor([5]),
            local_end_index=torch.tensor([2]),
        ),
    ]
    crossattn_cache = [
        CrossAttentionKVCache(
            k=torch.empty(1, 1, 1, 1),
            v=torch.empty(1, 1, 1, 1),
            is_init=True,
        ),
        CrossAttentionKVCache(
            k=torch.empty(1, 1, 1, 1),
            v=torch.empty(1, 1, 1, 1),
            is_init=True,
        ),
    ]
    stage._reset_causal_caches(
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
    )

    assert [block.is_init for block in crossattn_cache] == [False, False]
    assert [int(block.global_end_index.item()) for block in kv_cache] == [0, 0]
    assert [int(block.local_end_index.item()) for block in kv_cache] == [0, 0]
    assert kv_cache[0].global_end_index_int == 0
    assert kv_cache[0].local_end_index_int == 0

    calls = []

    def fake_initialize_kv_cache(self, batch_size, dtype, device, **kwargs):
        calls.append(("kv", batch_size, dtype, device, kwargs))
        self.causal_kv_cache = ["kv-cache"]

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


def test_causal_kv_cache_block_resets_indices_in_place():
    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    k = torch.ones(1, 4, 2, 3)
    v = torch.ones(1, 4, 2, 3)
    global_end_index = torch.tensor([7])
    local_end_index = torch.tensor([4])
    cache = CausalSelfAttentionKVCache(
        k=k,
        v=v,
        global_end_index=global_end_index,
        local_end_index=local_end_index,
        global_end_index_int=7,
        local_end_index_int=4,
    )

    assert cache.k is k
    assert cache.global_end_index_int == 7
    detached_k = k.detach()
    cache.k = detached_k
    assert cache.k is detached_k

    stage._reset_kv_cache([cache])

    assert cache.global_end_index is global_end_index
    assert cache.local_end_index is local_end_index
    assert int(cache.global_end_index.item()) == 0
    assert int(cache.local_end_index.item()) == 0
    assert cache.global_end_index_int == 0
    assert cache.local_end_index_int == 0


def test_causal_kv_cache_allocation_sets_shapes_and_optional_int_indices():
    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    stage.num_transformer_blocks = 2

    cache = stage._allocate_causal_kv_cache(
        batch_size=3,
        kv_cache_size=5,
        num_attention_heads=7,
        attention_head_dim=11,
        dtype=torch.float16,
        device=torch.device("cpu"),
        use_int_indices=True,
        sink_tokens=13,
        attention_window_size=3,
    )

    assert len(cache) == 2
    assert cache[0].k.shape == (3, 5, 7, 11)
    assert cache[0].v.shape == (3, 5, 7, 11)
    assert cache[0].global_end_index.shape == (1,)
    assert cache[0].local_end_index.shape == (1,)
    assert cache[0].global_end_index_int == 0
    assert cache[0].local_end_index_int == 0
    assert cache[0].cache_size == 5
    assert cache[0].sink_tokens == 13
    assert cache[0].attention_window_size == 3
    assert cache[0].allow_growth is False


def test_causal_kv_cache_update_handles_append_roll_and_recompute():
    cache = CausalSelfAttentionKVCache(
        k=torch.zeros(1, 4, 1, 1),
        v=torch.zeros(1, 4, 1, 1),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        global_end_index_int=0,
        local_end_index_int=0,
    )

    first_view = cache.update_and_get_attention_kv(
        key=torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]]),
        value=torch.tensor([[[[10.0]], [[20.0]], [[30.0]]]]),
        current_chunk_start=0,
    )

    assert first_view.visible_global_end == 3
    assert first_view.visible_local_end == 3
    assert cache.global_end_index_int == 3
    assert cache.local_end_index_int == 3
    assert first_view.k.flatten().tolist() == [1.0, 2.0, 3.0]

    rolled_view = cache.update_and_get_attention_kv(
        key=torch.tensor([[[[4.0]], [[5.0]], [[6.0]]]]),
        value=torch.tensor([[[[40.0]], [[50.0]], [[60.0]]]]),
        current_chunk_start=3,
    )

    assert rolled_view.visible_global_end == 6
    assert rolled_view.visible_local_end == 4
    assert cache.k.flatten().tolist() == [3.0, 4.0, 5.0, 6.0]

    cache.attention_window_size = 2
    recompute_view = cache.update_and_get_attention_kv(
        key=torch.tensor([[[[50.0]]]]),
        value=torch.tensor([[[[500.0]]]]),
        current_chunk_start=4,
    )

    assert recompute_view.local_start_index == 2
    assert recompute_view.local_end_index == 3
    assert recompute_view.visible_global_end == 6
    assert recompute_view.visible_local_end == 4
    assert recompute_view.k.flatten().tolist() == [50.0, 6.0]


def test_causal_kv_cache_update_for_cache_head_slice_returns_local_view():
    cache = CausalSelfAttentionKVCache(
        k=torch.full((1, 4, 4, 1), 99.0),
        v=torch.full((1, 4, 4, 1), 999.0),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        global_end_index_int=0,
        local_end_index_int=0,
    )

    view = cache.update_and_get_attention_kv(
        key=torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),
        value=torch.tensor([[[[10.0], [20.0]], [[30.0], [40.0]]]]),
        current_chunk_start=0,
        cache_head_start=1,
    )

    assert view.k.shape == (1, 2, 2, 1)
    assert view.v.shape == (1, 2, 2, 1)
    assert view.k.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert view.v.flatten().tolist() == [10.0, 20.0, 30.0, 40.0]
    assert cache.k[0, :2, :, 0].tolist() == [
        [99.0, 1.0, 2.0, 99.0],
        [99.0, 3.0, 4.0, 99.0],
    ]

    cache.update_and_get_attention_kv(
        key=torch.tensor([[[[5.0], [6.0]], [[7.0], [8.0]]]]),
        value=torch.tensor([[[[50.0], [60.0]], [[70.0], [80.0]]]]),
        current_chunk_start=2,
        cache_head_start=1,
    )
    rolled_view = cache.update_and_get_attention_kv(
        key=torch.tensor([[[[9.0], [10.0]], [[11.0], [12.0]]]]),
        value=torch.tensor([[[[90.0], [100.0]], [[110.0], [120.0]]]]),
        current_chunk_start=4,
        cache_head_start=1,
    )

    assert rolled_view.k.flatten().tolist() == [
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ]
    assert cache.k[0, :, :, 0].tolist() == [
        [99.0, 5.0, 6.0, 99.0],
        [99.0, 7.0, 8.0, 99.0],
        [99.0, 9.0, 10.0, 99.0],
        [99.0, 11.0, 12.0, 99.0],
    ]


def test_causal_kv_cache_update_grows_without_rolling_when_enabled():
    cache = CausalSelfAttentionKVCache(
        k=torch.zeros(1, 2, 1, 1),
        v=torch.zeros(1, 2, 1, 1),
        global_end_index=torch.zeros(1, dtype=torch.long),
        local_end_index=torch.zeros(1, dtype=torch.long),
        global_end_index_int=0,
        local_end_index_int=0,
        allow_growth=True,
    )

    cache.update_and_get_attention_kv(
        key=torch.tensor([[[[1.0]], [[2.0]]]]),
        value=torch.tensor([[[[10.0]], [[20.0]]]]),
        current_chunk_start=0,
    )
    view = cache.update_and_get_attention_kv(
        key=torch.tensor([[[[3.0]], [[4.0]]]]),
        value=torch.tensor([[[[30.0]], [[40.0]]]]),
        current_chunk_start=2,
    )

    assert cache.cache_size == 4
    assert cache.attention_window_size == 4
    assert cache.global_end_index_int == 4
    assert cache.local_end_index_int == 4
    assert view.local_start_index == 2
    assert view.local_end_index == 4
    assert view.k.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]


def test_crossattn_cache_block_stores_detached_tensors_and_resets():
    stage = CausalDMDDenoisingStage.__new__(CausalDMDDenoisingStage)
    stage.num_transformer_blocks = 1
    k = torch.ones(1, 8, 2, 3)
    v = torch.ones(1, 8, 2, 3)
    cache = CrossAttentionKVCache(k=k, v=v, is_init=True)

    assert cache.k is k
    assert cache.is_init is True
    detached_v = v.detach()
    cache.v = detached_v
    assert cache.v is detached_v
    new_k = torch.full_like(k, 2.0, requires_grad=True)
    new_v = torch.full_like(v, 3.0, requires_grad=True)
    cache.store(new_k, new_v)
    assert cache.k is not new_k
    assert cache.v is not new_v
    assert cache.k.requires_grad is False
    assert cache.v.requires_grad is False

    stage._reset_crossattn_cache([cache])

    assert cache.k.flatten().tolist() == [2.0] * cache.k.numel()
    assert cache.v.flatten().tolist() == [3.0] * cache.v.numel()
    assert cache.is_init is False
