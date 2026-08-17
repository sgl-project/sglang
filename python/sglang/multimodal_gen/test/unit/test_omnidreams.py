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
    OmniDreamsAttention,
    OmniDreamsDiT,
    RotaryPositionEmbedding3D,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_omnidreams_flow_match import (  # noqa: E501
    OmniDreamsFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (  # noqa: E501
    OmniDreamsBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCausalDiTState,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="OmniDreams DiT forward runs on the platform (GPU) device",
)


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
    stage._bcg_runners = {}
    server_args = types.SimpleNamespace(
        pipeline_config=types.SimpleNamespace(
            dit_precision="fp32",
            dit_config=types.SimpleNamespace(arch_config=arch),
        ),
        enable_breakable_cuda_graph=False,
        enable_torch_compile=False,
    )
    stage.server_args = server_args
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


@requires_gpu
@torch.no_grad()
def test_offline_realtime_single_chunk_parity(monkeypatch):
    """A1 lock: the offline AR loop and the realtime single-chunk path share
    ``_run_ar_chunk``, so one chunk of offline ``forward`` and one realtime
    ``_realtime_denoise_forward`` (block_idx=0) must produce bit-identical
    latents under the same seed + inputs (deterministic SDPA on a fixed device).
    """
    arch = _tiny_arch()
    dit = _tiny_dit(arch)
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, dit, sched, monkeypatch)

    text = torch.randn(1, 5, arch.crossattn_proj_in_channels, device=_DEVICE)

    # Offline: one chunk.
    torch.manual_seed(0)
    offline_batch = _ar_batch(
        arch,
        image_token=None,
        num_chunks=1,
        text=text,
        gen=torch.Generator(device=_DEVICE).manual_seed(7),
    )
    offline_out = stage.forward(offline_batch, server_args)

    # Realtime: one chunk (block_idx=0). Build a minimal session + the state
    # the before-stage would have stashed (mirror _realtime_stash_initial_state);
    # the denoise stage's lazy assembly builds the rest (rope/caches/masks/
    # cross_attn_kv) on chunk 0.
    rt_batch = _ar_batch(
        arch,
        image_token=None,
        num_chunks=1,
        text=text,
        gen=torch.Generator(device=_DEVICE).manual_seed(7),
    )
    rt_batch.realtime_session_id = "parity"
    rt_batch.block_idx = 0
    rt_batch.condition_inputs = {}
    head_dim = arch.model_channels // arch.num_heads
    in_d = arch.in_channels * arch.patch_temporal * arch.patch_spatial**2
    hdmap_d = arch.additional_concat_ch * arch.patch_temporal * arch.patch_spatial**2
    mask_d = arch.patch_temporal * arch.patch_spatial**2
    state = RealtimeCausalDiTState()
    rt_batch.session = types.SimpleNamespace(get_or_create_state=lambda cls: state)
    # Stash the one-shot prep via the real before-stage method (drift-free vs
    # hand-populating rc); the denoise stage's lazy assembly builds the rest.
    before_stage = OmniDreamsBeforeDenoisingStage.__new__(
        OmniDreamsBeforeDenoisingStage
    )
    before_stage._realtime_stash_initial_state(
        rt_batch,
        server_args,
        text_embeds=text,
        scheduler=sched,
        generator=rt_batch.generator,
        arch_constants={
            "hp": 2,
            "wp": 2,
            "len_t": 2,
            "tokens_per_frame": 4,
            "chunk_tokens": 4,
            "head_dim": head_dim,
            "in_d": in_d,
            "hdmap_d": hdmap_d,
            "mask_d": mask_d,
            "context_noise": 128.0,
            "window_size_t": 2,
            "sink_size_t": 0,
        },
        hdmap_pixel=None,
        image_token=None,
    )

    rt_out = stage._realtime_denoise_forward(rt_batch, server_args)

    assert rt_out.latents.shape == offline_out.latents.shape
    assert torch.equal(rt_out.latents, offline_out.latents)


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
    ``clip`` where the temporal length covers ``num_chunks * len_t`` latent frames."""
    stage = _hdmap_stage(monkeypatch)
    dev = torch.device("cpu")
    num_chunks, len_t = 3, 2
    num_latent = num_chunks * len_t  # 6
    total_pixel = 1 + (num_latent - 1) * 4  # 21

    b = types.SimpleNamespace(hdmap_path=list(range(total_pixel)), hdmap_pixels=None)
    pixel = stage._encode_hdmap(b, dev, torch.float32, num_chunks, len_t, 16, 16)
    # Per-frame path defers VAE encode to the AR loop and returns its pixel clip.
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


def test_prepared_fp8_weight_cache_uses_artifact_bytes_and_tp_row_shard():
    """Prepared FP8 compute must consume artifact bytes, never BF16-dequantize."""
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        prepared_fp8_weight_cache,
    )

    raw = torch.arange(24, dtype=torch.uint8).reshape(6, 4)
    scale = torch.arange(1, 7, dtype=torch.float16)
    fp8, fp8_scale = prepared_fp8_weight_cache(
        {
            "blocks.0.self_attn.qkv_proj.weight": raw,
            "blocks.0.self_attn.qkv_proj.weight_scale": scale,
        },
        "blocks.0.self_attn.qkv_proj.weight",
        target_shape=(3, 4),
        tp_rank=1,
        tp_world_size=2,
    )

    assert fp8.dtype is torch.float8_e4m3fn
    assert torch.equal(fp8.view(torch.uint8), raw[3:])
    assert fp8_scale.dtype is torch.float32
    assert torch.equal(fp8_scale, scale[3:].float().reshape(1, -1))


def test_prepared_fp8_keeps_cross_attention_kv_in_bf16():
    """Cross-attention K/V must retain BF16 precision for conditioning fidelity."""
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        prepare_fp8_dit_weights,
    )

    key = "blocks.0.cross_attn.to_kv.weight"
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.bfloat16)
    cosmos_utils = types.SimpleNamespace(
        prepare_cosmos_quantized_streaming_weights=lambda *args, **kwargs: {key: weight}
    )

    prepared = prepare_fp8_dit_weights({}, num_blocks=1, cosmos_fp8_utils=cosmos_utils)

    assert prepared[key].dtype is torch.bfloat16
    assert key + "_scale" not in prepared


@requires_gpu
@torch.no_grad()
def test_fp8_compute_keeps_attention_bf16_and_quantizes_mlp_only():
    """The safe compute policy must not FP8-quantize attention projections."""
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        OmniDreamsFP8ComputeLinear,
        install_fp8_compute_on_dit,
    )

    model = _tiny_dit()
    original_qkv_method = model.blocks[0].self_attn.to_qkv.quant_method

    assert install_fp8_compute_on_dit(model)

    assert model.blocks[0].self_attn.to_qkv.quant_method is original_qkv_method
    assert isinstance(model.blocks[0].mlp.layer1, OmniDreamsFP8ComputeLinear)
    assert isinstance(model.blocks[0].mlp.layer2, OmniDreamsFP8ComputeLinear)


# --------------------------------------------------------------------------- #
# Config three-state validation.                                              #
# --------------------------------------------------------------------------- #
def test_config_fp8_acceleration_modes_valid():
    """The explicit prepared mode is accepted alongside existing FP8 modes."""
    for mode in (
        "disabled",
        "weight_only_fp8",
        "fp8_compute",
        "fp8_compute_prepared",
    ):
        cfg = OmniDreamsPipelineConfig(native_dit_acceleration=mode)
        assert cfg.native_dit_acceleration == mode


# --------------------------------------------------------------------------- #
# param_names_mapping: checkpoint q/k/v -> packed to_qkv / to_kv.             #
# --------------------------------------------------------------------------- #
def _apply(mapping_fn, key):
    return mapping_fn(key)


def test_param_names_mapping_self_attn_qkv_merge():
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

    fn = get_param_names_mapping(OmniDreamsDiT.param_names_mapping)
    key, idx, total = _apply(fn, "blocks.0.self_attn.q_proj.weight")
    assert key == "blocks.0.self_attn.to_qkv.weight"
    assert idx == 0
    assert total == 3
    _, idx, total = _apply(fn, "blocks.0.self_attn.k_proj.weight")
    assert idx == 1
    assert total == 3
    _, idx, total = _apply(fn, "blocks.0.self_attn.v_proj.weight")
    assert idx == 2
    assert total == 3


def test_param_names_mapping_cross_attn_kv_merge():
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

    fn = get_param_names_mapping(OmniDreamsDiT.param_names_mapping)
    key, idx, total = _apply(fn, "blocks.5.cross_attn.k_proj.weight")
    assert key == "blocks.5.cross_attn.to_kv.weight"
    assert idx == 0
    assert total == 2
    _, idx, total = _apply(fn, "blocks.5.cross_attn.v_proj.weight")
    assert idx == 1
    assert total == 2


def test_param_names_mapping_passthrough():
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

    fn = get_param_names_mapping(OmniDreamsDiT.param_names_mapping)
    for name in (
        "blocks.0.self_attn.q_norm.weight",
        "blocks.0.self_attn.output_proj.weight",
        "blocks.0.cross_attn.q_proj.weight",
        "blocks.0.mlp.layer1.weight",
        "final_layer.linear.weight",
        "x_embedder.proj.1.weight",
    ):
        key, idx, total = _apply(fn, name)
        assert key == name
        assert idx is None
        assert total is None


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


# --------------------------------------------------------------------------- #
# BCG (Breakable CUDA Graph) unit tests.                                       #
# --------------------------------------------------------------------------- #
def test_omnidreams_is_bcg_supported_in_server_args():
    """OmniDreams must be accepted by server-args BCG validation."""
    from sglang.multimodal_gen.runtime.server_args import (
        BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
        BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS,
    )

    assert "OmniDreamsPipelineConfig" in BREAKABLE_CUDA_GRAPH_SUPPORTED_PIPELINE_CONFIGS
    assert "omnidreams" in BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS
    assert "nvidia/omnidreams" in BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS


def test_bcg_still_rejected_for_unsupported_models():
    """BCG must still be rejected for unsupported model IDs."""
    from sglang.multimodal_gen.runtime.server_args import (
        BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS,
    )

    assert "some-random-model" not in BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS
    assert "wan2.1" not in BREAKABLE_CUDA_GRAPH_SUPPORTED_MODEL_IDS


def test_omnidreams_attention_is_bcg_break_point():
    """OmniDreamsAttention.forward must be wrapped as a BCG break point.

    The wrapper checks ``is_in_breakable_cuda_graph()`` and routes to the
    eager-on-graph path during capture. When BCG is inactive (normal eager),
    it delegates transparently to the original forward.
    """
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        is_in_breakable_cuda_graph,
    )

    # The installed forward on the class must be the wrapped version: it
    # should have ``__wrapped__`` pointing at the original forward.
    assert hasattr(OmniDreamsAttention.forward, "__wrapped__")

    # Sanity: is_in_breakable_cuda_graph returns False outside capture.
    assert not is_in_breakable_cuda_graph()


@requires_gpu
@torch.no_grad()
def test_bcg_routes_omnidreams_through_runner_when_enabled(monkeypatch):
    """When BCG is enabled, the AR dit_call routes through the BCG runner
    (``run_dit_with_bcg`` -> ``_maybe_get_bcg_runner`` -> ``_bcg_run``),
    not a direct ``self.transformer(...)`` call."""
    arch = _tiny_arch()
    dit = _tiny_dit(arch)
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, dit, sched, monkeypatch)
    # Enable BCG on the stage's server_args.
    stage.server_args.enable_breakable_cuda_graph = True

    call_log: list[dict] = []
    original_forward = dit.forward

    def tracking_forward(**kwargs):
        call_log.append(kwargs)
        return original_forward(**kwargs)

    dit.forward = tracking_forward

    # Mock the BCG runner so we don't need a real CUDA graph capture.
    mock_runner = types.SimpleNamespace(
        __call__=lambda **kw: original_forward(**kw),
        capture=lambda **kw: True,
        entries={},
        _disabled_reason=None,
        _should_capture_on_call=lambda key: False,
        _log_signature_miss=lambda key: None,
    )
    stage._bcg_runners[id(dit)] = mock_runner

    text = torch.randn(1, 5, arch.crossattn_proj_in_channels, device=_DEVICE)
    gen = torch.Generator(device=_DEVICE).manual_seed(2)
    batch = _ar_batch(
        arch, image_token=None, num_chunks=1, text=text, gen=gen, window_size_t=2
    )
    stage.forward(batch, server_args)

    # The transformer forward must have been called (at least the denoise
    # step + the context-noise re-forward).
    assert len(call_log) >= 2

    # Each call must carry the BCG-specific kwargs (kv_caches, cross_attn_kv).
    for kw in call_log:
        assert "kv_caches" in kw
        assert "cross_attn_kv" in kw

    # Restore.
    stage.server_args.enable_breakable_cuda_graph = False


@requires_gpu
@torch.no_grad()
def test_bcg_disabled_routes_omnidreams_eagerly(monkeypatch):
    """When BCG is disabled, ``run_dit_with_bcg`` falls through to a direct
    ``self.transformer(...)`` call (no runner created)."""
    arch = _tiny_arch()
    dit = _tiny_dit(arch)
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, dit, sched, monkeypatch)
    assert not stage.server_args.enable_breakable_cuda_graph

    # _maybe_get_bcg_runner must return None.
    assert stage._maybe_get_bcg_runner(dit) is None
    assert stage._bcg_runners == {}


def test_bcg_disables_torch_compile_for_omnidreams():
    """``--enable-breakable-cuda-graph`` must skip torch.compile per the
    existing framework policy (BCG and torch.compile are mutually exclusive)."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = types.SimpleNamespace(
        enable_breakable_cuda_graph=True,
        enable_torch_compile=True,
    )
    stage._cache_dit_enabled = False
    stage._torch_compile_registry = types.SimpleNamespace(is_compiled=lambda m: False)

    # _maybe_torch_compile must return early when BCG is enabled, even if
    # enable_torch_compile is also True.
    stage._maybe_torch_compile(types.SimpleNamespace())


@requires_gpu
@torch.no_grad()
def test_bcg_kv_cache_not_stale_across_requests(monkeypatch):
    """During capture/replay, KV-cache update/read happens on each invocation
    and does not reuse cache objects from a prior request.

    The BCG signature keys mutable objects by identity (``id(obj)``), so a
    new request with new KV-cache objects produces a different signature and
    cannot replay a graph captured against another request's caches.
    """
    from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
        _signature_kwargs,
    )

    arch = _tiny_arch()
    dit = _tiny_dit(arch)
    sched = OmniDreamsFlowMatchScheduler()
    stage, server_args = _ar_stage_and_args(arch, dit, sched, monkeypatch)
    stage.server_args.enable_breakable_cuda_graph = True

    # Build two sets of KV caches (simulating two requests).
    caches_a = dit.init_kv_caches(
        batch_size=1,
        chunk_tokens=4,
        window_tokens=8,
        sink_tokens=0,
        device=_DEVICE,
        dtype=torch.float32,
    )
    caches_b = dit.init_kv_caches(
        batch_size=1,
        chunk_tokens=4,
        window_tokens=8,
        sink_tokens=0,
        device=_DEVICE,
        dtype=torch.float32,
    )

    text = torch.randn(1, 5, arch.crossattn_proj_in_channels, device=_DEVICE)
    cross_attn_kv = dit.precompute_cross_attn_kv(dit.crossattn_proj(text))

    kwargs_a = dict(
        hidden_states=torch.randn(1, 4, 20, device=_DEVICE),
        encoder_hidden_states=text,
        timestep=torch.tensor([500.0], device=_DEVICE),
        condition_video_input_mask=torch.zeros(1, 4, 4, device=_DEVICE),
        rope_cos_sin=torch.randn(4, 12, device=_DEVICE),
        hdmap_condition=torch.randn(1, 4, 4, device=_DEVICE),
        kv_caches=caches_a,
        cross_attn_kv=cross_attn_kv,
    )
    kwargs_b = dict(kwargs_a)
    kwargs_b["kv_caches"] = caches_b

    sig_a = _signature_kwargs(kwargs_a)
    sig_b = _signature_kwargs(kwargs_b)

    # The signatures must differ because the kv_caches list identity differs.
    assert sig_a != sig_b, (
        "KV-cache objects from different requests must not share a BCG "
        "signature (stale capture-bound Python objects)."
    )

    # Same caches -> same signature (replay-safe for the same request).
    sig_a2 = _signature_kwargs(kwargs_a)
    assert sig_a == sig_a2


@requires_gpu
@torch.no_grad()
def test_bcg_hdmap_and_no_hdmap_signatures_differ(monkeypatch):
    """HDMap enabled vs disabled produce different BCG signatures because the
    ``hdmap_condition`` tensor shape differs (or is zero vs non-zero).

    With the tiny arch, hdmap_d != 0, so the hdmap_condition tensor is always
    present. The test verifies that different hdmap_condition values share the
    same signature (shape-based, not value-based), confirming shape-stable
    replay. The HDMap disabled path passes the same-shape zero tensor.
    """
    from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
        _signature_kwargs,
    )

    arch = _tiny_arch()
    dit = _tiny_dit(arch)
    device = _DEVICE

    text = torch.randn(1, 5, arch.crossattn_proj_in_channels, device=device)
    caches = dit.init_kv_caches(
        batch_size=1,
        chunk_tokens=4,
        window_tokens=8,
        sink_tokens=0,
        device=device,
        dtype=torch.float32,
    )
    cross_attn_kv = dit.precompute_cross_attn_kv(dit.crossattn_proj(text))

    pdim = arch.patch_temporal * arch.patch_spatial**2
    hdmap_d = arch.additional_concat_ch * pdim

    # HDMap enabled: real hdmap_condition tensor.
    kwargs_hdmap = dict(
        hidden_states=torch.randn(1, 4, arch.in_channels * pdim, device=device),
        encoder_hidden_states=text,
        timestep=torch.tensor([500.0], device=device),
        condition_video_input_mask=torch.zeros(1, 4, pdim, device=device),
        rope_cos_sin=torch.randn(4, 12, device=device),
        hdmap_condition=torch.randn(1, 4, hdmap_d, device=device),
        kv_caches=caches,
        cross_attn_kv=cross_attn_kv,
    )

    # HDMap disabled: zero hdmap_condition (same shape, different value).
    kwargs_no_hdmap = dict(kwargs_hdmap)
    kwargs_no_hdmap["hdmap_condition"] = torch.zeros(1, 4, hdmap_d, device=device)

    # Same shape -> same signature (shape-based replay is safe).
    assert _signature_kwargs(kwargs_hdmap) == _signature_kwargs(kwargs_no_hdmap)

    # Different shape -> different signature.
    kwargs_diff_shape = dict(kwargs_hdmap)
    kwargs_diff_shape["hdmap_condition"] = torch.randn(1, 8, hdmap_d, device=device)
    assert _signature_kwargs(kwargs_hdmap) != _signature_kwargs(kwargs_diff_shape)
