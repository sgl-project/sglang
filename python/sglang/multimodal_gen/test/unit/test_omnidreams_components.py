# SPDX-License-Identifier: Apache-2.0
"""CPU component tests for the OmniDreams port (no checkpoint, no GPU).

Covers the pure-torch building blocks that the GPU phases depend on:
- 3D NeoX RoPE (``omnidreams_rope``): layout, rotation correctness, ``shift_t``.
- ``BlockKVCache``: fill -> roll -> steady-state, sink retention, overwrite.
- ``OmniDreamsFlowMatchScheduler``: 2-step sigmas, self-forcing ``sample``, ``add_noise``.
- Cosmos-Reason1 ``full_concat_embeddings``: drop embedding layer, per-layer norm, 100352.
- A tiny-config ``OmniDreamsDiT`` end-to-end forward (single-chunk + KV-cache path).
- The ``OmniDreamsDenoisingStage`` autoregressive rollout orchestration.

The structural/fixture checks live in ``test_omnidreams_scaffold.py``.
"""

import types
from collections import Counter

import numpy as np
import PIL.Image
import torch

from sglang.multimodal_gen.configs.models.dits.omnidreams import (
    OmniDreamsDiTArchConfig,
    OmniDreamsDiTConfig,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
from sglang.multimodal_gen.runtime.models.dits.omnidreams_kvcache import BlockKVCache
from sglang.multimodal_gen.runtime.models.dits.omnidreams_rope import (
    ROPE_IS_NEOX_STYLE,
    RotaryPositionEmbedding3D,
    apply_rope_freqs,
    rope_dims,
)
from sglang.multimodal_gen.runtime.models.encoders.omnidreams_text import (
    FULL_CONCAT_DIM,
    full_concat_embeddings,
    mean_normalize,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_omnidreams_flow_match import (  # noqa: E501
    OmniDreamsFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (  # noqa: E501
    OmniDreamsBeforeDenoisingStage,
)


# --------------------------------------------------------------------- RoPE -- #
def test_rope_dims_44_42_42_neox():
    assert rope_dims(128) == (44, 42, 42)
    assert sum(rope_dims(128)) == 128
    assert ROPE_IS_NEOX_STYLE is True


def test_apply_rope_matches_neox_reference_and_preserves_norm():
    torch.manual_seed(0)
    emb = RotaryPositionEmbedding3D(
        head_dim=128,
        len_h=4,
        len_w=5,
        len_t=2,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    freqs = emb.shift_t(0)
    L = freqs.shape[0]
    assert L == 2 * 4 * 5
    assert freqs.shape[-1] == 128
    # NeoX builds freqs as [first | first] -> the two halves are identical.
    assert torch.allclose(freqs[..., :64], freqs[..., 64:])

    x = torch.randn(1, L, 16, 128)
    out = apply_rope_freqs(x, freqs)
    half = 64
    f = freqs[..., :half].reshape(L, half).view(1, L, 1, half)
    cos, sin = f.cos(), f.sin()
    a, b = x[..., :half], x[..., half:]
    ref = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)
    assert torch.allclose(out, ref, atol=1e-6)
    # rotations are norm-preserving
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_shift_t_advances_only_time_frequencies():
    emb = RotaryPositionEmbedding3D(head_dim=128, len_h=4, len_w=5, len_t=2)
    dim_t_half = rope_dims(128)[0] // 2  # 22
    f0, f1 = emb.shift_t(0), emb.shift_t(1)
    # time band (first dim_t_half angles) changes with ar_idx ...
    assert not torch.allclose(f0[..., :dim_t_half], f1[..., :dim_t_half])
    # ... spatial bands (h then w) do not.
    assert torch.allclose(f0[..., dim_t_half:64], f1[..., dim_t_half:64])


# ----------------------------------------------------------------- KV cache -- #
def _chunk(val, B=1, n=2, d=3, size=2):
    t = torch.full((B, size, n, d), float(val))
    return t, t.clone()


def test_kv_cache_fill_roll_steady_state():
    c = BlockKVCache(
        k_shape=(1, 4, 2, 3),
        v_shape=(1, 4, 2, 3),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
    )
    # chunk 0: [10,10], visible 2
    c.before_update(0)
    k, v = _chunk(10)
    c.update(k, v)
    ck = c.cached_k().clone()
    c.after_update(0)
    assert ck.shape[1] == 2 and bool((ck == 10).all())
    # chunk 1: [10,10,11,11], visible 4 (full)
    c.before_update(1)
    k, v = _chunk(11)
    c.update(k, v)
    ck = c.cached_k().clone()
    c.after_update(1)
    assert ck.shape[1] == 4
    assert bool((ck[:, :2] == 10).all()) and bool((ck[:, 2:] == 11).all())
    # chunk 2: steady-state roll left -> drop 10, keep 11, add 12
    c.before_update(2)
    k, v = _chunk(12)
    c.update(k, v)
    ck = c.cached_k().clone()
    c.after_update(2)
    assert bool((ck[:, :2] == 11).all()) and bool((ck[:, 2:] == 12).all())


def test_kv_cache_overwrite_same_chunk_idx():
    c = BlockKVCache(
        k_shape=(1, 4, 2, 3),
        v_shape=(1, 4, 2, 3),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        sink_size=0,
    )
    for idx, val in enumerate([10, 11]):
        c.before_update(idx)
        k, v = _chunk(val)
        c.update(k, v)
        c.after_update(idx)
    # rewriting the same chunk_idx refreshes the rightmost slots in place
    c.before_update(1)
    k, v = _chunk(99)
    c.update(k, v)
    ck = c.cached_k().clone()
    c.after_update(1)
    assert bool((ck[:, 2:] == 99).all()) and bool((ck[:, :2] == 10).all())


def test_kv_cache_sink_is_never_evicted():
    c = BlockKVCache(
        k_shape=(1, 6, 2, 3),
        v_shape=(1, 6, 2, 3),
        seq_dim=1,
        chunk_size=2,
        window_size=4,
        sink_size=2,
    )
    ck = None
    for idx, val in enumerate([1, 2, 3, 4, 5]):
        c.before_update(idx)
        k, v = _chunk(val)
        c.update(k, v)
        ck = c.cached_k().clone()
        c.after_update(idx)
    # sink keeps the first chunk; rolling window keeps the last two chunks
    assert bool((ck[:, :2] == 1).all())
    assert bool((ck[:, 2:4] == 4).all()) and bool((ck[:, 4:6] == 5).all())


def test_kv_cache_from_tensor_roundtrip():
    kk, vv = torch.randn(1, 3, 2, 3), torch.randn(1, 3, 2, 3)
    fc = BlockKVCache.from_tensor(kk, vv, seq_dim=1)
    assert torch.allclose(fc.cached_k(), kk) and torch.allclose(fc.cached_v(), vv)


# ---------------------------------------------------------------- scheduler -- #
def test_scheduler_two_step_sigmas_and_timesteps():
    s = OmniDreamsFlowMatchScheduler()
    sig = s.denoising_sigmas.tolist()
    ts = s.denoising_step_list.tolist()
    assert len(sig) == 2
    assert abs(sig[0] - 1.0) < 1e-6
    assert abs(sig[1] - 0.8036) < 1e-3
    assert abs(ts[0] - 1000.0) < 1e-3
    assert abs(ts[1] - 803.57) < 1.0
    # context-noise raw timestep 128 snaps to sigma ~= 0.128
    assert abs(s.sigma_for_timestep(128) - 0.128) < 0.02


def test_scheduler_sample_ideal_flow_recovers_target():
    s = OmniDreamsFlowMatchScheduler()
    init = torch.randn(1, 4)
    target = torch.randn(1, 4)
    step2sigma = {
        round(t, 4): sg
        for t, sg in zip(s.denoising_step_list.tolist(), s.denoising_sigmas.tolist())
    }

    def ideal_flow(noisy, t):
        return (noisy - target) / step2sigma[round(float(t), 4)]

    out = s.sample(init, predict_flow=ideal_flow, rng=torch.Generator().manual_seed(7))
    assert torch.allclose(out, target, atol=1e-4)


def test_scheduler_add_noise_scales_by_sigma():
    s = OmniDreamsFlowMatchScheduler()
    clean = torch.zeros(1, 4096)
    noisy = s.add_noise(
        clean, torch.tensor(128.0), rng=torch.Generator().manual_seed(3)
    )
    # clean == 0 -> noisy == sigma * noise -> std approximates sigma
    assert abs(noisy.std().item() - s.sigma_for_timestep(128)) < 0.05


# ------------------------------------------------------------------- text ---- #
def test_full_concat_dim_and_per_layer_normalization():
    torch.manual_seed(0)
    hs = [torch.randn(2, 7, 3584) for _ in range(29)]  # 28 layers + embedding layer
    emb = full_concat_embeddings(hs)
    assert emb.shape == (2, 7, FULL_CONCAT_DIM) == (2, 7, 100352)
    block0 = emb[..., :3584]
    assert block0.mean(dim=-1).abs().max().item() < 1e-4
    assert (block0.std(dim=-1) - 1.0).abs().max().item() < 1e-2
    # dropping the embedding layer == normalizing layers 1..28 and concatenating
    manual = torch.cat([mean_normalize(hs[i]) for i in range(1, 29)], dim=-1)
    assert torch.allclose(emb, manual, atol=1e-6)


# ------------------------------------------------------- tiny DiT forward ---- #
def _tiny_dit() -> OmniDreamsDiT:
    """A small CPU-constructible OmniDreamsDiT for end-to-end forward testing.

    head_dim = 24/2 = 12 keeps the RoPE 6-way split valid (dim_t/h/w = 4/4/4).
    """
    arch = OmniDreamsDiTArchConfig(
        in_channels=4,
        out_channels=4,
        model_channels=24,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        adaln_lora_dim=8,
        crossattn_proj_in_channels=32,
        crossattn_emb_channels=16,
        additional_concat_ch=4,
    )
    model = OmniDreamsDiT(config=OmniDreamsDiTConfig(arch_config=arch), hf_config={})
    model.post_load_weights()  # fuse padding-mask (24->20) + last-layer shuffle
    return model.eval()


def _tiny_inputs(model, grid=(2, 2, 2), B=1, lctx=5):
    arch = model.arch
    gt, gh, gw = grid
    L = gt * gh * gw
    pdim = arch.patch_temporal * arch.patch_spatial**2  # kt*kh*kw
    hidden = torch.randn(B, L, arch.in_channels * pdim)
    cond_mask = torch.zeros(B, L, pdim)
    hdmap = torch.randn(B, L, arch.additional_concat_ch * pdim)
    ctx = torch.randn(B, lctx, arch.crossattn_proj_in_channels)
    head_dim = arch.model_channels // arch.num_heads
    rope = RotaryPositionEmbedding3D(head_dim=head_dim, len_h=gh, len_w=gw, len_t=gt)
    return hidden, cond_mask, hdmap, ctx, rope, (gt, gh, gw, L)


@torch.no_grad()
def test_tiny_dit_single_chunk_forward_and_unpatchify():
    torch.manual_seed(0)
    model = _tiny_dit()
    hidden, cond_mask, hdmap, ctx, rope, (gt, gh, gw, L) = _tiny_inputs(model)
    out = model(
        hidden_states=hidden,
        encoder_hidden_states=ctx,
        timestep=torch.tensor([500.0]),
        condition_video_input_mask=cond_mask,
        rope_freqs=rope.shift_t(0),
        hdmap_condition=hdmap,
    )
    pdim = model.arch.patch_temporal * model.arch.patch_spatial**2
    assert out.shape == (1, L, model.arch.out_channels * pdim)
    assert torch.isfinite(out).all()
    video = model.unpatchify(out, gt, gh, gw)
    assert video.shape == (
        1,
        model.arch.out_channels,
        gt * model.arch.patch_temporal,
        gh * model.arch.patch_spatial,
        gw * model.arch.patch_spatial,
    )


@torch.no_grad()
def test_tiny_dit_autoregressive_kv_cache_path():
    torch.manual_seed(0)
    model = _tiny_dit()
    hidden, cond_mask, hdmap, ctx, rope, (gt, gh, gw, L) = _tiny_inputs(model)
    # window holds two chunks so chunk 1's Q (L) attends K/V of length 2L.
    caches = model.init_kv_caches(
        batch_size=1,
        chunk_tokens=L,
        window_tokens=2 * L,
        sink_tokens=0,
        dtype=torch.float32,  # match the float32 CPU test model (bf16 in production)
    )

    def run_chunk(idx):
        for c in caches:
            c.before_update(idx)
        out = model(
            hidden_states=hidden,
            encoder_hidden_states=ctx,
            timestep=torch.tensor([500.0]),
            condition_video_input_mask=cond_mask,
            rope_freqs=rope.shift_t(idx),
            hdmap_condition=hdmap,
            kv_caches=caches,
        )
        for c in caches:
            c.after_update(idx)
        return out

    out0 = run_chunk(0)
    out1 = run_chunk(1)
    pdim = model.arch.patch_temporal * model.arch.patch_spatial**2
    expected = (1, L, model.arch.out_channels * pdim)
    assert out0.shape == expected and out1.shape == expected
    assert torch.isfinite(out0).all() and torch.isfinite(out1).all()
    # chunk 1 attends a larger cached window than chunk 0 -> outputs differ.
    assert not torch.allclose(out0, out1)


# ------------------------------------------------ AR denoising rollout ------ #
def _ar_stage_and_args(arch, dit, scheduler, monkeypatch):
    """Build an OmniDreamsDenoisingStage bypassing the heavy base __init__.

    Forces CPU device (production runs uniformly on cuda) and fakes the minimal
    server_args.pipeline_config the AR forward reads.
    """
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams as od_stage
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (  # noqa: E501
        OmniDreamsDenoisingStage,
    )

    monkeypatch.setattr(od_stage, "get_local_torch_device", lambda: torch.device("cpu"))
    stage = OmniDreamsDenoisingStage.__new__(OmniDreamsDenoisingStage)
    stage.transformer = dit
    stage.scheduler = scheduler
    stage.vae = None
    server_args = types.SimpleNamespace(
        pipeline_config=types.SimpleNamespace(
            dit_precision="fp32",
            dit_config=types.SimpleNamespace(arch_config=arch),
        )
    )
    return stage, server_args


def _ar_batch(
    arch, image_token, num_chunks, text, gen, hp=2, wp=2, len_t=2, window_size_t=2
):
    tokens_per_frame = hp * wp
    return types.SimpleNamespace(
        scheduler=None,
        prompt_embeds=[text],
        generator=gen,
        latents=None,
        extra={
            "omnidreams": {
                "hp": hp,
                "wp": wp,
                "len_t": len_t,
                "tokens_per_frame": tokens_per_frame,
                "chunk_tokens": len_t * tokens_per_frame,
                "latent_h": hp * arch.patch_spatial,
                "latent_w": wp * arch.patch_spatial,
                "num_chunks": num_chunks,
                "window_size_t": window_size_t,
                "sink_size_t": 0,
                "context_noise": 128.0,
                "image_token": image_token,
                "hdmap_tokens": None,
            }
        },
    )


@torch.no_grad()
def test_ar_denoising_unconditioned_rollout(monkeypatch):
    torch.manual_seed(0)
    arch = OmniDreamsDiTArchConfig(
        in_channels=4,
        out_channels=4,
        model_channels=24,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        adaln_lora_dim=8,
        crossattn_proj_in_channels=32,
        crossattn_emb_channels=16,
        additional_concat_ch=4,
    )
    dit = OmniDreamsDiT(config=OmniDreamsDiTConfig(arch_config=arch), hf_config={})
    dit.post_load_weights()
    dit.eval()
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, dit, sched, monkeypatch)

    text = torch.randn(1, 5, arch.crossattn_proj_in_channels)
    gen = torch.Generator().manual_seed(1)
    batch = _ar_batch(arch, image_token=None, num_chunks=3, text=text, gen=gen)
    out = stage.forward(batch, server_args)
    assert tuple(out.latents.shape) == (1, 4, 3 * 2, 2 * 2, 2 * 2)
    assert torch.isfinite(out.latents).all()


@torch.no_grad()
def test_ar_denoising_i2v_pins_frame0(monkeypatch):
    torch.manual_seed(0)
    arch = OmniDreamsDiTArchConfig(
        in_channels=4,
        out_channels=4,
        model_channels=24,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        adaln_lora_dim=8,
        crossattn_proj_in_channels=32,
        crossattn_emb_channels=16,
        additional_concat_ch=4,
    )
    dit = OmniDreamsDiT(config=OmniDreamsDiTConfig(arch_config=arch), hf_config={})
    dit.post_load_weights()
    dit.eval()
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, dit, sched, monkeypatch)

    in_d = arch.in_channels * arch.patch_temporal * arch.patch_spatial**2  # 16
    tokens_per_frame = 4
    image_token = torch.randn(1, tokens_per_frame, in_d)
    text = torch.randn(1, 5, arch.crossattn_proj_in_channels)
    gen = torch.Generator().manual_seed(1)
    batch = _ar_batch(arch, image_token=image_token, num_chunks=2, text=text, gen=gen)
    out = stage.forward(batch, server_args)
    assert tuple(out.latents.shape) == (1, 4, 2 * 2, 2 * 2, 2 * 2)
    assert torch.isfinite(out.latents).all()
    # chunk-0 frame-0 must equal the (unpatchified) pinned reference latent.
    ref_f0 = dit.unpatchify(
        torch.cat([image_token, torch.zeros(1, tokens_per_frame, in_d)], dim=1), 2, 2, 2
    )[:, :, 0]
    assert torch.allclose(out.latents[:, :, 0], ref_f0, atol=1e-4)


@torch.no_grad()
def test_ar_denoising_window_roll_many_chunks(monkeypatch):
    torch.manual_seed(0)
    arch = OmniDreamsDiTArchConfig(
        in_channels=4,
        out_channels=4,
        model_channels=24,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        adaln_lora_dim=8,
        crossattn_proj_in_channels=32,
        crossattn_emb_channels=16,
        additional_concat_ch=4,
    )
    dit = OmniDreamsDiT(config=OmniDreamsDiTConfig(arch_config=arch), hf_config={})
    dit.post_load_weights()
    dit.eval()
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, dit, sched, monkeypatch)

    text = torch.randn(1, 5, arch.crossattn_proj_in_channels)
    gen = torch.Generator().manual_seed(2)
    # window of 4 latent frames (2 chunks) exercises the steady-state left-roll.
    batch = _ar_batch(
        arch, image_token=None, num_chunks=4, text=text, gen=gen, window_size_t=4
    )
    out = stage.forward(batch, server_args)
    assert tuple(out.latents.shape) == (1, 4, 4 * 2, 2 * 2, 2 * 2)
    assert torch.isfinite(out.latents).all()


# ----------------------------------------------- A.2 reference preprocess ---- #
_PRE = OmniDreamsBeforeDenoisingStage._preprocess_pixels
_CPU = torch.device("cpu")


def test_reference_preprocess_pil_resizes_and_normalizes():
    # off-target-size PIL [0,255] -> resized [1,3,1,H,W] in [-1,1].
    pil = PIL.Image.fromarray((np.random.rand(40, 60, 3) * 255).astype("uint8"))
    out = _PRE(pil, height=32, width=48, device=_CPU, dtype=torch.float32)
    assert tuple(out.shape) == (1, 3, 1, 32, 48)
    assert out.min() >= -1.0 - 1e-4 and out.max() <= 1.0 + 1e-4
    assert out.min() < 0.0  # mapped into the signed VAE input range


def test_reference_preprocess_from_path(tmp_path):
    pil = PIL.Image.fromarray((np.random.rand(20, 30, 3) * 255).astype("uint8"))
    p = tmp_path / "ref.png"
    pil.save(p)
    out = _PRE(str(p), height=16, width=24, device=_CPU, dtype=torch.float32)
    assert tuple(out.shape) == (1, 3, 1, 16, 24)


def test_reference_preprocess_signed_tensor_passthrough():
    # already-[-1,1] tensor: no re-normalize, just gains the temporal axis.
    t = torch.rand(1, 3, 32, 48) * 2 - 1
    out = _PRE(t, height=32, width=48, device=_CPU, dtype=torch.float32)
    assert tuple(out.shape) == (1, 3, 1, 32, 48)
    assert torch.allclose(out[:, :, 0], t)


def test_reference_preprocess_unsigned_tensor_is_normalized():
    # [0,1] 3D tensor -> [1,3,1,H,W] normalized into [-1,1].
    u = torch.rand(3, 16, 16)
    out = _PRE(u, height=16, width=16, device=_CPU, dtype=torch.float32)
    assert tuple(out.shape) == (1, 3, 1, 16, 16)
    assert out.min() < 0.0
    assert torch.allclose(out[0, :, 0], u * 2 - 1)


# ----------------------------------------------- A.1 per-chunk HDMap -------- #
class _RecordingDiT:
    """Wraps a DiT, recording the ``hdmap_condition`` of each forward call.

    Delegates every other attribute (``init_kv_caches``/``patchify``/...) to the
    wrapped module so the AR stage runs unchanged.
    """

    def __init__(self, dit):
        self._dit = dit
        self.hdmap_calls: list[torch.Tensor] = []

    def __call__(self, *args, **kwargs):
        self.hdmap_calls.append(kwargs.get("hdmap_condition"))
        return self._dit(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._dit, name)


def _ar_setup(monkeypatch, num_chunks, window_size_t=4):
    arch = OmniDreamsDiTArchConfig(
        in_channels=4,
        out_channels=4,
        model_channels=24,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        adaln_lora_dim=8,
        crossattn_proj_in_channels=32,
        crossattn_emb_channels=16,
        additional_concat_ch=4,
    )
    dit = OmniDreamsDiT(config=OmniDreamsDiTConfig(arch_config=arch), hf_config={})
    dit.post_load_weights()
    dit.eval()
    rec = _RecordingDiT(dit)
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, rec, sched, monkeypatch)
    text = torch.randn(1, 5, arch.crossattn_proj_in_channels)
    gen = torch.Generator().manual_seed(3)
    batch = _ar_batch(
        arch,
        image_token=None,
        num_chunks=num_chunks,
        text=text,
        gen=gen,
        window_size_t=window_size_t,
    )
    return arch, stage, server_args, rec, batch


@torch.no_grad()
def test_ar_hdmap_per_chunk_indexing(monkeypatch):
    torch.manual_seed(0)
    num_chunks = 3
    arch, stage, server_args, rec, batch = _ar_setup(monkeypatch, num_chunks)

    pdim = arch.patch_temporal * arch.patch_spatial**2
    hdmap_d = arch.additional_concat_ch * pdim
    chunk_tokens = batch.extra["omnidreams"]["chunk_tokens"]
    # Each chunk's HD-map is a constant tensor tagged with its chunk index.
    hdmap_tokens = [
        torch.full((1, chunk_tokens, hdmap_d), float(ci)) for ci in range(num_chunks)
    ]
    batch.extra["omnidreams"]["hdmap_tokens"] = hdmap_tokens

    stage.forward(batch, server_args)

    # Each chunk is tagged with its index; verify routing without hard-coding the
    # per-chunk call count (predict_flow x steps + 1 cache-write).
    assert rec.hdmap_calls and all(h is not None for h in rec.hdmap_calls)
    tags = [float(h.flatten()[0]) for h in rec.hdmap_calls]
    counts = Counter(tags)
    # distinct tags == {0..n-1}: each chunk received its OWN tensor (not shared);
    assert sorted(counts) == [float(ci) for ci in range(num_chunks)]
    # uniform #calls per chunk + non-decreasing order -> each chunk's calls are
    # contiguous and never cross-contaminated with another chunk's tensor.
    assert len(set(counts.values())) == 1
    assert tags == sorted(tags)


@torch.no_grad()
def test_ar_hdmap_none_falls_back_to_zeros(monkeypatch):
    torch.manual_seed(0)
    arch, stage, server_args, rec, batch = _ar_setup(monkeypatch, num_chunks=2)
    assert batch.extra["omnidreams"]["hdmap_tokens"] is None
    stage.forward(batch, server_args)
    # Disabled HDMap -> every forward gets an all-zero condition.
    assert rec.hdmap_calls
    assert all(bool((h == 0).all()) for h in rec.hdmap_calls)


def test_encode_hdmap_broadcast_clamp_and_none(monkeypatch):
    """`_encode_hdmap` assembles the per-chunk list (broadcast / clamp / None).

    Stubs preprocess+encode+patchify so the test isolates the per-chunk control
    flow (the VAE numerics are a GPU concern). Each src tag flows through as a
    1-element tensor so chunk identity is traceable.
    """
    stage = OmniDreamsBeforeDenoisingStage.__new__(OmniDreamsBeforeDenoisingStage)
    stage.transformer = types.SimpleNamespace(patchify=lambda latent: latent)
    monkeypatch.setattr(
        stage,
        "_preprocess_pixels",
        lambda src, h, w, d, dt: torch.tensor([float(src)]),
    )
    monkeypatch.setattr(stage, "_vae_encode_normalized", lambda x: x)
    dev = torch.device("cpu")

    # Single (non-list) input broadcasts to every chunk.
    b1 = types.SimpleNamespace(hdmap_path=7, hdmap_pixels=None)
    toks = stage._encode_hdmap(b1, dev, torch.float32, torch.float32, 3, 16, 16)
    assert [float(t) for t in toks] == [7.0, 7.0, 7.0]

    # Per-chunk list shorter than num_chunks clamps to the last entry.
    b2 = types.SimpleNamespace(hdmap_path=[1, 2], hdmap_pixels=None)
    toks2 = stage._encode_hdmap(b2, dev, torch.float32, torch.float32, 4, 16, 16)
    assert [float(t) for t in toks2] == [1.0, 2.0, 2.0, 2.0]

    # No HD-map input -> None (AR stage falls back to zeros).
    b3 = types.SimpleNamespace(hdmap_path=None, hdmap_pixels=None)
    assert stage._encode_hdmap(b3, dev, torch.float32, torch.float32, 2, 16, 16) is None
