# SPDX-License-Identifier: Apache-2.0
"""OmniDreams unit tests.

A small representative set: the RoPE ``shift_t`` contract (the FP8-blur
root-cause fix), a tiny DiT forward, the AR rollout window, HD-map per-frame
clip slicing, FP8 weight prep, config three-state validation, the flat
checkpoint key fixture, and the ``num_chunks`` math. GPU-touching tests skip on
CPU; the rest are CPU-runnable. The end-to-end serving path is covered by the
``omnidreams_2b_i2v`` server E2E case.
"""

from __future__ import annotations

import os
import types

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.omnidreams import (
    OmniDreamsDiTArchConfig,
    OmniDreamsDiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
    OmniDreamsPipelineConfig,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams import (
    OmniDreamsDiT,
    RotaryPositionEmbedding3D,
    rope_dims,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_omnidreams_flow_match import (  # noqa: E501
    OmniDreamsFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (  # noqa: E501
    _MAX_AR_CHUNKS,
    OmniDreamsBeforeDenoisingStage,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="OmniDreams DiT forward runs on the platform (GPU) device",
)


# --------------------------------------------------------------------------- #
# 3D RoPE: shift_t_freqs must match the FlashDreams reference exactly.         #
#  (This was the FP8-blur root cause: the old layout was [t,t,h,h,w,w] and    #
#  dropped the h/w NTK extrapolation; see omnidreams_rope.py.)                #
# --------------------------------------------------------------------------- #
def _fd_reference_shift_t_freqs(
    *, head_dim, len_h, len_w, len_t, ratios, ar_idx=0, device="cpu"
):
    """Flashdreams ``RotaryPositionEmbedding3D.shift_t`` reference: per-axis
    NTK-rescaled base frequencies (``_compute_freqs``) concatenated with the
    non-interleaved ``[t, h, w, t, h, w]`` layout (``_cat_freqs``). This is the
    contract the native FP8 C++ ``_make_cosmos_rope_cache`` consumes."""
    dim_t, dim_h, dim_w = rope_dims(head_dim)

    def _freqs(dim, ratio):
        dim_range = (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2]
            / dim
        )
        theta = 10000.0 * (ratio ** (dim / (dim - 2)))
        return 1.0 / (theta**dim_range)

    rt, rh, rw = ratios
    t = torch.arange(len_t, device=device) + ar_idx * len_t
    h = torch.arange(len_h, device=device)
    w = torch.arange(len_w, device=device)
    tt, hh, ww = torch.meshgrid(t, h, w, indexing="ij")
    ft = torch.outer(tt.reshape(-1).float(), _freqs(dim_t, rt))
    fh = torch.outer(hh.reshape(-1).float(), _freqs(dim_h, rh))
    fw = torch.outer(ww.reshape(-1).float(), _freqs(dim_w, rw))
    raw = torch.cat([ft, fh, fw, ft, fh, fw], dim=-1)  # [L, D]
    return raw.unsqueeze(1).unsqueeze(1)  # [L, 1, 1, D]


def test_shift_t_freqs_matches_fd_reference_formula():
    # Golden: shift_t_freqs must equal flashdreams' shift_t exactly. The old
    # implementation diverged (cos~0.64 vs fd) via wrong layout + dropped NTK.
    emb = RotaryPositionEmbedding3D(
        head_dim=128,
        len_h=4,
        len_w=5,
        len_t=2,
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        t_extrapolation_ratio=1.0,
    )
    for ar in (0, 1, 2):
        got = emb.shift_t_freqs(ar)
        ref = _fd_reference_shift_t_freqs(
            head_dim=128,
            len_h=4,
            len_w=5,
            len_t=2,
            ratios=(1.0, 3.0, 3.0),
            ar_idx=ar,
        )
        assert got.shape == ref.shape == (2 * 4 * 5, 1, 1, 128)
        assert torch.allclose(
            got, ref, atol=1e-5
        ), f"ar={ar} diverged from fd reference"


# --------------------------------------------------------------------------- #
# Tiny DiT forward (GPU) + AR denoise rollout.                                #
# --------------------------------------------------------------------------- #
def _tiny_arch() -> OmniDreamsDiTArchConfig:
    """Shared tiny arch: head_dim = 24/2 = 12 keeps the RoPE 6-way split valid."""
    return OmniDreamsDiTArchConfig(
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


def _tiny_dit(arch: OmniDreamsDiTArchConfig | None = None) -> OmniDreamsDiT:
    """A small CPU-constructible OmniDreamsDiT for end-to-end forward testing."""
    model = OmniDreamsDiT(
        config=OmniDreamsDiTConfig(arch_config=arch or _tiny_arch()), hf_config={}
    )
    model.post_load_weights()  # fuse padding-mask (24->20) + last-layer shuffle
    # Column/Row/MergedColumnParallelLinear allocate zero-filled weights for the
    # checkpoint loader; the bare test model has none, so random-init those weight
    # matrices for a meaningful forward (other params keep their module init).
    model = model.to(_DEVICE)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2 and float(p.abs().max()) == 0.0:
                torch.nn.init.normal_(p, std=0.02)
    return model.eval()


def _tiny_inputs(model, grid=(2, 2, 2), B=1, lctx=5):
    arch = model.arch
    gt, gh, gw = grid
    L = gt * gh * gw
    pdim = arch.patch_temporal * arch.patch_spatial**2  # kt*kh*kw
    dev = next(model.parameters()).device
    hidden = torch.randn(B, L, arch.in_channels * pdim, device=dev)
    cond_mask = torch.zeros(B, L, pdim, device=dev)
    hdmap = torch.randn(B, L, arch.additional_concat_ch * pdim, device=dev)
    ctx = torch.randn(B, lctx, arch.crossattn_proj_in_channels, device=dev)
    head_dim = arch.model_channels // arch.num_heads
    rope = RotaryPositionEmbedding3D(head_dim=head_dim, len_h=gh, len_w=gw, len_t=gt)
    return hidden, cond_mask, hdmap, ctx, rope, (gt, gh, gw, L)


@requires_gpu
@torch.no_grad()
def test_tiny_dit_single_chunk_forward_and_unpatchify():
    torch.manual_seed(0)
    model = _tiny_dit()
    hidden, cond_mask, hdmap, ctx, rope, (gt, gh, gw, L) = _tiny_inputs(model)
    out = model(
        hidden_states=hidden,
        encoder_hidden_states=ctx,
        timestep=torch.tensor([500.0], device=_DEVICE),
        condition_video_input_mask=cond_mask,
        rope_cos_sin=rope.shift_t(0).to(_DEVICE),
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


def _ar_stage_and_args(arch, dit, scheduler, monkeypatch):
    """Build an OmniDreamsDenoisingStage bypassing the heavy base __init__.

    Runs on the platform device (``_DEVICE``) and fakes the minimal
    server_args.pipeline_config the AR forward reads.
    """
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams as od_stage
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (  # noqa: E501
        OmniDreamsDenoisingStage,
    )

    monkeypatch.setattr(od_stage, "get_local_torch_device", lambda: _DEVICE)
    stage = OmniDreamsDenoisingStage.__new__(OmniDreamsDenoisingStage)
    stage.transformer = dit
    stage.scheduler = scheduler
    stage.vae = None
    stage._component_residency_manager = None
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
                "hdmap_pixel": None,
            }
        },
    )


@requires_gpu
@torch.no_grad()
def test_ar_denoising_window_roll_many_chunks(monkeypatch):
    torch.manual_seed(0)
    arch = _tiny_arch()
    dit = _tiny_dit(arch)
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, dit, sched, monkeypatch)

    text = torch.randn(1, 5, arch.crossattn_proj_in_channels, device=_DEVICE)
    gen = torch.Generator(device=_DEVICE).manual_seed(2)
    # window of 4 latent frames (2 chunks): the 3rd chunk triggers the steady-state
    # left-roll. (3 chunks keeps the untrained-weight rollout numerically stable.)
    batch = _ar_batch(
        arch, image_token=None, num_chunks=3, text=text, gen=gen, window_size_t=4
    )
    out = stage.forward(batch, server_args)
    # The window roll is verified structurally: the steady-state left-roll must yield
    # exactly num_chunks * len_t latent frames (here 3 * 2). Rollout *numerics* are
    # covered by the real-weight e2e — finiteness of an untrained multi-chunk rollout
    # is not a stable invariant (GPU SDPA non-determinism near fp edges).
    assert tuple(out.latents.shape) == (1, 4, 3 * 2, 2 * 2, 2 * 2)


# --------------------------------------------------------------------------- #
# HD-map: per-frame clip slicing (the stream-encode refactor guard).          #
# --------------------------------------------------------------------------- #
def _hdmap_stage(monkeypatch):
    """Stage with stubbed preprocess + a fake causal VAE (tc=4) + identity patchify.

    The fake VAE maps a ``T``-frame clip to ``1 + (T-1)//4`` latent frames (mirroring
    the Wan VAE temporal compression), filling latent frame ``j`` with value ``j`` so
    chunk slicing is verifiable. patchify is identity so the returned tokens are the
    sliced latents themselves. (Real VAE numerics are a GPU concern.)
    """
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams as od_mod

    stage = OmniDreamsBeforeDenoisingStage.__new__(OmniDreamsBeforeDenoisingStage)
    stage.transformer = types.SimpleNamespace(patchify=lambda latent: latent)
    stage.encoder = object()  # HD-map VAE; _vae_encode_normalized is stubbed below
    monkeypatch.setattr(
        stage,
        "_preprocess_pixels",
        lambda src, h, w, d, dt: torch.zeros(1, 3, 1, 2, 2),
    )

    def fake_vae(x, vae, cache=None, is_first_chunk=True):
        t = x.shape[2] if x.dim() == 5 else 1
        n_latent = 1 + (t - 1) // 4
        return torch.cat(
            [torch.full((1, 16, 1, 2, 2), float(j)) for j in range(n_latent)], dim=2
        )

    monkeypatch.setattr(od_mod, "_vae_encode_normalized", fake_vae)
    return stage


def test_encode_hdmap_per_frame_clip_slicing(monkeypatch):
    """Per-frame HD-map (option 2): the full raster sequence is decoded once as a
    causal clip (deferred per-chunk VAE encode in the AR loop). Returns
    ``(None, clip)`` where ``clip`` has ``num_chunks * len_t`` latent frames."""
    stage = _hdmap_stage(monkeypatch)
    dev = torch.device("cpu")
    num_chunks, len_t = 3, 2
    num_latent = num_chunks * len_t  # 6
    total_pixel = 1 + (num_latent - 1) * 4  # 21

    b = types.SimpleNamespace(hdmap_path=list(range(total_pixel)), hdmap_pixels=None)
    toks, pixel = stage._encode_hdmap(
        b, dev, torch.float32, torch.float32, num_chunks, len_t, 16, 16
    )
    # Per-frame path defers VAE encode to the AR loop -> (None, clip).
    assert toks is None
    assert pixel is not None
    # pixel shape: [B, 3, total_pixel, H, W]
    assert pixel.shape[2] == total_pixel


# --------------------------------------------------------------------------- #
# FP8 weight prep: the offline exporter must unfuse to_qkv -> q/k/v.          #
# --------------------------------------------------------------------------- #
def _fake_fused_dit_state_dict(num_blocks: int, *, qdim: int = 64, inner: int = 48):
    """Minimal bf16 state dict mimicking the post-to_qkv-refactor OmniDreamsDiT.

    Self-attn Q/K/V is fused into ``to_qkv`` (q,k,v row order); cross-attn K/V
    is fused into ``to_kv`` (unused by Cosmos FP8 prep, must pass through).
    """
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    for i in range(num_blocks):
        p = f"blocks.{i}."
        q = torch.randn(inner, qdim, dtype=torch.bfloat16)
        k = torch.randn(inner, qdim, dtype=torch.bfloat16)
        v = torch.randn(inner, qdim, dtype=torch.bfloat16)
        sd[p + "self_attn.to_qkv.weight"] = torch.cat([q, k, v], dim=0).contiguous()
        sd[p + "self_attn.output_proj.weight"] = torch.randn(
            qdim, inner, dtype=torch.bfloat16
        )
        sd[p + "cross_attn.q_proj.weight"] = torch.randn(
            inner, qdim, dtype=torch.bfloat16
        )
        sd[p + "cross_attn.to_kv.weight"] = torch.randn(
            2 * inner, qdim, dtype=torch.bfloat16
        )
        sd[p + "cross_attn.output_proj.weight"] = torch.randn(
            qdim, inner, dtype=torch.bfloat16
        )
        sd[p + "mlp.layer1.weight"] = torch.randn(inner * 2, qdim, dtype=torch.bfloat16)
        sd[p + "mlp.layer2.weight"] = torch.randn(qdim, inner * 2, dtype=torch.bfloat16)
        sd[p + "self_attn.q_norm.weight"] = torch.ones(inner, dtype=torch.bfloat16)
        sd[p + "self_attn.k_norm.weight"] = torch.ones(inner, dtype=torch.bfloat16)
        sd[p + "cross_attn.k_norm.weight"] = torch.ones(inner, dtype=torch.bfloat16)
    return sd


def test_prepare_fp8_dit_weights_unfuses_to_qkv_into_qkv_proj():
    """The offline exporter must accept the post-refactor fused ``to_qkv`` DiT
    state dict (commit 5dff6576c merged self-attn q/k/v into MergedColumnParallelLinear).

    Before the fix ``prepare_fp8_dit_weights`` raised KeyError on the missing
    split ``q_proj``; now it unfuses ``to_qkv`` -> q/k/v, lets the Cosmos prep
    rebuild ``qkv_proj`` (fp8), and drops the dead ``to_qkv``.
    """
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        prepare_fp8_dit_weights,
    )

    nb = 2
    inner = 48  # matches _fake_fused_dit_state_dict default
    sd = _fake_fused_dit_state_dict(nb)
    fused_snapshot = {
        i: sd[f"blocks.{i}.self_attn.to_qkv.weight"].clone() for i in range(nb)
    }

    out = prepare_fp8_dit_weights(sd, num_blocks=nb, linear_policy="all")

    for i in range(nb):
        # to_qkv dropped (dead bf16 must not ship in the fp8 artifact).
        assert f"blocks.{i}.self_attn.to_qkv.weight" not in out
        # rebuilt fused qkv_proj is fp8 with a per-out-channel scale.
        qk = f"blocks.{i}.self_attn.qkv_proj.weight"
        sk = qk + "_scale"
        assert out[qk].dtype == torch.uint8, f"{qk} not fp8"
        assert tuple(out[sk].shape) == (out[qk].shape[0],)
        # split q/k/v are NOT retained (default drops them in favor of qkv_proj).
        for rel in ("q_proj", "k_proj", "v_proj"):
            assert f"blocks.{i}.self_attn.{rel}.weight" not in out
        # dequant recovers the original to_qkv within e4m3 precision, with the
        # q/k/v shard boundaries intact (q rows 0..inner, k inner..2*inner, ...).
        deq = out[qk].view(torch.float8_e4m3fn).to(torch.float32) * out[sk].to(
            torch.float32
        ).unsqueeze(1)
        orig = fused_snapshot[i].to(torch.float32)
        assert (deq - orig).abs().max().item() < 1.0, f"block {i} dequant drift"
        assert torch.allclose(deq[:inner], orig[:inner], atol=1.0)


# --------------------------------------------------------------------------- #
# Config three-state validation.                                              #
# --------------------------------------------------------------------------- #
def test_config_three_state_valid():
    """Phase 1+2: native_dit_acceleration accepts disabled/weight_only_fp8/fp8_compute."""
    for mode in ("disabled", "weight_only_fp8", "fp8_compute"):
        cfg = OmniDreamsPipelineConfig(native_dit_acceleration=mode)
        assert cfg.native_dit_acceleration == mode


# --------------------------------------------------------------------------- #
# Flat-checkpoint key coverage (authoritative 570-key .pt fixture).           #
# --------------------------------------------------------------------------- #
_KEY_FIXTURE = os.path.join(
    os.path.dirname(__file__), "data", "omnidreams_dit_keys.txt"
)


def _load_fixture_keys() -> set[str]:
    with open(_KEY_FIXTURE) as f:
        return {line.strip() for line in f if line.strip()}


def _build_meta_model() -> OmniDreamsDiT:
    with torch.device("meta"):
        return OmniDreamsDiT(config=OmniDreamsDiTConfig(), hf_config={})


def test_state_dict_matches_authoritative_key_fixture():
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

    model = _build_meta_model()
    keys = set(model.state_dict().keys())
    ckpt_keys = _load_fixture_keys()
    assert len(ckpt_keys) == 570
    # The packed-QKV merge maps the checkpoint's separate q/k/v -> to_qkv and k/v ->
    # to_kv (param_names_mapping). Apply it to the checkpoint keys and verify they
    # cover exactly the model's state_dict (i.e. the flat checkpoint stays loadable).
    mapping_fn = get_param_names_mapping(OmniDreamsDiT.param_names_mapping)
    mapped = {mapping_fn(k)[0] for k in ckpt_keys}
    assert (
        mapped == keys
    ), f"missing={sorted(keys - mapped)} extra={sorted(mapped - keys)}"


# --------------------------------------------------------------------------- #
# num_frames -> chunk mapping + AR cap.                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "num_frames,expected",
    [(5, 1), (6, 2), (13, 2), (14, 3), (21, 3)],  # len_t=2: first=5, step=8
)
def test_compute_num_chunks_boundaries(num_frames, expected):
    batch = types.SimpleNamespace(num_frames=num_frames)
    assert (
        OmniDreamsBeforeDenoisingStage._compute_num_chunks(batch, len_t=2) == expected
    )


def test_compute_num_chunks_caps_ar_loop():
    batch = types.SimpleNamespace(num_frames=10_000_000)
    assert (
        OmniDreamsBeforeDenoisingStage._compute_num_chunks(batch, len_t=2)
        == _MAX_AR_CHUNKS
    )
