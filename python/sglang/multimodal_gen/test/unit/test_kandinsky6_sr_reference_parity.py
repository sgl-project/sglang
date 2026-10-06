# SPDX-License-Identifier: Apache-2.0
"""CPU parity of the Kandinsky 6 SR port against the k6_video reference (``kandinsky_sr``).

The reference package is not part of this repository.  Point
``KANDINSKY_SR_REFERENCE_SRC`` at the directory that contains ``kandinsky_sr`` (for
example ``k6_video/src``); without it the whole module is skipped.  The reference needs
``flash_attn`` and ``loguru``, which are stubbed here (per-sequence SDPA for the dense
varlen attention), and its base resolutions are shrunk so that tiles stay tiny.  All
components use tiny random weights (every parameter re-randomized: the reference
zero-initializes most modulation / output layers, which would hide bugs) and float32
math on CPU.  The port loads the reference ``state_dict()`` (unprefixed, i.e. exactly the official
Diffusers layout) with ``strict=True`` through its own name mapping.
"""

import copy
import importlib.machinery
import os
import re
import sys
import types
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.models.dits.kandinsky6_sr import (
    Kandinsky6SRTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.schedulers.kandinsky6_piflow import (
    PiflowScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.upsampler.kandinsky6_sr_latent_upscaler import (
    Kandinsky6SRLatentUpscalerBank,
)
from sglang.multimodal_gen.runtime.models.vaes.kandinsky6_sr_vae import Kandinsky6SRVAE
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr import (
    tiling,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    build_dit_spec,
    build_sampling_spec,
    effective_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    denoise_with_scheduler,
    euler_start_timestep,
    make_dit_fn,
    module_dtype,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    plan_tiles,
    stitch_tiles,
    super_resolve,
)

REFERENCE_ENV = "KANDINSKY_SR_REFERENCE_SRC"
_reference_dir = os.environ.get(REFERENCE_ENV)
if not _reference_dir or not os.path.isdir(
    os.path.join(_reference_dir, "kandinsky_sr")
):
    pytest.skip(
        f"set {REFERENCE_ENV} to the directory containing the kandinsky_sr package",
        allow_module_level=True,
    )

# --------------------------------------------------------------------------- #
# Reference harness (test-only): stubs, tiny resolutions, tiny random components
# --------------------------------------------------------------------------- #
TINY_RESOLUTIONS = {512: [(64, 64), (64, 96), (96, 64)]}
TINY_KVAE_ENC = dict(
    ch=8,
    ch_mult=(1, 1, 2, 2, 2),
    num_res_blocks=2,
    in_channels=3,
    z_channels=4,
    temporal_compress_times=4,
    norm_type="rms_norm",
)
TINY_KVAE_DEC = dict(
    ch=8,
    out_ch=3,
    ch_mult=(1, 1, 2, 2, 2),
    num_res_blocks=2,
    z_channels=4,
    temporal_compress_times=4,
    norm_type="rms_norm",
)
TINY_SCALING_FACTOR = 0.5
TINY_DIT_CFG = dict(
    in_visual_dim=4,
    in_text_dim=8,
    in_text_dim2=8,
    time_dim=16,
    out_visual_dim=4,
    patch_size=(1, 2, 2),
    model_dim=32,
    ff_dim=64,
    num_text_blocks=0,
    num_visual_blocks=2,
    axes_dims=(8, 4, 4),
    visual_cond=False,
    instruct_type="noise",
    attention_params={"512": {"type": "flash"}},
    use_text=False,
)
TINY_PIFLOW = dict(
    nfe=2,
    num_policy_substeps=8,
    final_step_size_scale=0.5,
    shift=5.0,
    n_grid=3,
    eps=1e-6,
)
TINY_SR_PARAMS = dict(
    scale_factor={512: [1.0, 1.0, 1.0]},
    visual_size=[512],
    scheduler_scale=5.0,
    lq_noise_scale=0.7,
    lq_noise_type="ddpm",
    lq_channel_noise_scale=0.0,
    cap_noise_timestep=False,
    fps=24,
)


def _varlen_sdpa(q, k, v, cu_q, cu_k):
    outs = []
    for i in range(len(cu_q) - 1):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_k[i]), int(cu_k[i + 1])
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi)
        outs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _flash_attn_stub() -> types.ModuleType:
    module = types.ModuleType("flash_attn")

    def varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        return_attn_probs=False,
        **_,
    ):
        out = _varlen_sdpa(q, k, v, cu_seqlens_q, cu_seqlens_k)
        return (out, torch.zeros(1), None) if return_attn_probs else out

    def qkvpacked(qkv, cu_seqlens, max_seqlen, return_attn_probs=False, **_):
        q, k, v = qkv.unbind(dim=1)
        out = _varlen_sdpa(q, k, v, cu_seqlens, cu_seqlens)
        return (out, torch.zeros(1), None) if return_attn_probs else out

    module.flash_attn_varlen_func = varlen
    module.flash_attn_varlen_qkvpacked_func = qkvpacked
    # transformers probes flash_attn with importlib.util.find_spec
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    return module


def _loguru_stub() -> types.ModuleType:
    module = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    module.logger = _Logger()
    module.__spec__ = importlib.machinery.ModuleSpec("loguru", loader=None)
    return module


def _install_reference() -> None:
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    if _reference_dir not in sys.path:
        sys.path.insert(0, _reference_dir)
    if "flash_attn" not in sys.modules:
        sys.modules["flash_attn"] = _flash_attn_stub()
    if "loguru" not in sys.modules:
        sys.modules["loguru"] = _loguru_stub()
    from kandinsky_sr import constants

    constants.RESOLUTIONS.clear()
    constants.RESOLUTIONS.update(TINY_RESOLUTIONS)


_install_reference()

from kandinsky_sr.core.components.model.dit import get_dit  # noqa: E402
from kandinsky_sr.core.components.model.dx_dit import DXDiTWrapper  # noqa: E402
from kandinsky_sr.core.components.video_kvae.cached_model import (  # noqa: E402
    CachedCausalVAE,
)
from omegaconf import OmegaConf  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_process_model_parallel():
    """The K6 feed-forward uses TP-aware linears, which need a (size-1) TP group."""
    if not model_parallel_is_initialized():
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29507",
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            os.environ.setdefault(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def randomize(module: torch.nn.Module, seed: int, std: float = 0.05) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * std)


def build_reference_dit(*, piflow: dict | None, seed: int = 2, cfg: dict | None = None):
    cfg = dict(cfg or TINY_DIT_CFG)
    if piflow is not None:
        dit = DXDiTWrapper(
            cfg, out_visual_dim=cfg["out_visual_dim"], n_grid=piflow["n_grid"]
        )
        dit.piflow_params = dict(piflow)
    else:
        dit = get_dit(cfg)
    dit.eval()
    randomize(dit, seed)
    return dit


def flat_transformer_config(*, cfg: dict, piflow: dict | None, overrides: dict) -> dict:
    """The flat legacy dict form of a transformer config (n_grid / piflow_* / sr_* fields)."""
    flat = dict(cfg)
    flat["patch_size"] = list(flat["patch_size"])
    flat["axes_dims"] = list(flat["axes_dims"])
    flat["n_grid"] = piflow["n_grid"] if piflow is not None else 1
    if piflow is not None:
        flat.update(
            piflow_nfe=piflow["nfe"],
            piflow_num_policy_substeps=piflow["num_policy_substeps"],
            piflow_final_step_size_scale=piflow["final_step_size_scale"],
            piflow_shift=piflow["shift"],
            piflow_eps=piflow["eps"],
        )
    flat["attribute_overrides"] = dict(overrides)
    flat.update(
        sr_visual_size=[512],
        sr_scale_factor={"512": [1.0, 1.0, 1.0]},
        sr_scheduler_scale=TINY_SR_PARAMS["scheduler_scale"],
        sr_lq_noise_scale=TINY_SR_PARAMS["lq_noise_scale"],
        sr_lq_noise_type=TINY_SR_PARAMS["lq_noise_type"],
        sr_lq_channel_noise_scale=TINY_SR_PARAMS["lq_channel_noise_scale"],
        sr_cap_noise_timestep=TINY_SR_PARAMS["cap_noise_timestep"],
        sr_fps=TINY_SR_PARAMS["fps"],
    )
    return flat


def build_port_dit(
    reference_dit, *, cfg: dict | None = None, piflow: dict | None, overrides: dict
) -> Kandinsky6SRTransformer3DModel:
    """Port DiT loaded with the reference weights (``model.`` prefix), strictly."""
    flat = flat_transformer_config(
        cfg=dict(cfg or TINY_DIT_CFG), piflow=piflow, overrides=overrides
    )
    dit_config = Kandinsky6SRDitConfig()
    dit_config.update_model_arch(flat)
    model = Kandinsky6SRTransformer3DModel(dit_config, flat)
    official = dict(reference_dit.state_dict())  # unprefixed = the official layout
    mapped, _ = hf_to_custom_state_dict(
        official,
        get_param_names_mapping(model.param_names_mapping),
        valid_target_names=set(model.state_dict()),
    )
    model.load_state_dict(mapped, strict=True)
    return model.eval()


# --------------------------------------------------------------------------- #
# DiT forward parity
# --------------------------------------------------------------------------- #
def _reference_forward(dit, x, time, *, scale_factor, motion_score=None):
    """Run the packed reference DiT on a batched ``[B, T, H, W, C]`` input."""
    batch, frames, height, width, channels = x.shape
    packed = x.reshape(batch * frames, height, width, channels)
    visual_cu = frames * torch.arange(batch + 1, dtype=torch.int32)
    text_cu = torch.zeros(batch + 1, dtype=torch.int32)
    rope_pos = [
        torch.cat([torch.arange(frames) for _ in range(batch)]),
        torch.arange(height // dit.patch_size[1]),
        torch.arange(width // dit.patch_size[2]),
    ]
    out = dit(
        packed,
        torch.zeros(0, 1),
        torch.zeros(batch, 1),
        time,
        visual_cu,
        text_cu,
        rope_pos,
        torch.zeros(0, dtype=torch.long),
        scale_factor=scale_factor,
        motion_score=motion_score,
    )
    return out.reshape(batch, frames, *out.shape[1:]) if out.dim() == 4 else out


def _port_forward(model, x, time, *, scale_factor, motion_score=None):
    batch, frames, height, width, _ = x.shape
    rope_pos = [
        torch.arange(frames),
        torch.arange(height // model.patch_size[1]),
        torch.arange(width // model.patch_size[2]),
    ]
    return model(
        x,
        time,
        rope_pos,
        scale_factor=scale_factor,
        motion_score=motion_score,
    )


def _random_latent(
    width_channels: int, *, batch=2, frames=3, height=4, width=6, seed=0
):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        batch, frames, height, width, width_channels, generator=generator
    )


@pytest.mark.parametrize("piflow", [TINY_PIFLOW, None], ids=["dx_head", "plain_head"])
def test_dit_forward_matches_reference(piflow):
    """Guards the block math, RoPE, modulation, time bias and the DX head layout.

    The port is batched ``[B, T, H, W, C]`` while the reference packs samples with
    ``cu_seqlens``; different per-sample times make a wrong batching/broadcast visible.
    """
    reference = build_reference_dit(piflow=piflow)
    model = build_port_dit(
        reference, piflow=piflow, overrides={"instruct_type": "noise"}
    )
    x = _random_latent(4)
    time = torch.tensor([300.0, 850.0])
    scale = (1.0, 2.0, 2.0)
    with torch.no_grad():
        expected = _reference_forward(reference, x, time, scale_factor=scale)
        actual = _port_forward(model, x, time, scale_factor=scale)
    n_grid = 1 if piflow is None else piflow["n_grid"]
    if piflow is not None:
        # reference DX wrapper already splits the grids: [B*T, n_grid, H, W, C]
        expected = expected.movedim(1, -2).reshape(*actual.shape)
    assert actual.shape[-1] == 4 * n_grid
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_dit_forward_matches_reference_with_motion_score_and_wide_input():
    """Guards the optional motion embedding and the ``2 * in + 1`` conditioned input."""
    cfg = dict(
        TINY_DIT_CFG,
        use_motion_score=True,
        instruct_type="hybrid_anchor",
        use_lq_noise_cond=True,
    )
    reference = build_reference_dit(piflow=TINY_PIFLOW, cfg=cfg)
    model = build_port_dit(
        reference,
        cfg=cfg,
        piflow=TINY_PIFLOW,
        overrides={"instruct_type": "noise", "visual_cond": True},
    )
    reference.instruct_type = "noise"
    reference.visual_cond = True
    x = _random_latent(2 * 4 + 1)
    time = torch.tensor([120.0, 999.0])
    motion = torch.full((1,), 900.0)
    with torch.no_grad():
        expected = _reference_forward(
            reference, x, time, scale_factor=(1.0, 1.0, 1.0), motion_score=motion
        )
        actual = _port_forward(
            model, x, time, scale_factor=(1.0, 1.0, 1.0), motion_score=motion
        )
        without_motion = _port_forward(model, x, time, scale_factor=(1.0, 1.0, 1.0))
    expected = expected.movedim(1, -2).reshape(*actual.shape)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(actual, without_motion)


def test_state_dict_keys_equal_mapped_reference_keys():
    """Guards the module tree / key mapping: extra or missing keys would break real checkpoints."""
    reference = build_reference_dit(piflow=TINY_PIFLOW)
    model = build_port_dit(reference, piflow=TINY_PIFLOW, overrides={})
    mapped, _ = hf_to_custom_state_dict(
        dict(reference.state_dict()),
        get_param_names_mapping(model.param_names_mapping),
    )
    assert set(mapped) == set(model.state_dict())


# --------------------------------------------------------------------------- #
# Sampler loops (pi-Flow / Euler) against the reference generate functions
# --------------------------------------------------------------------------- #
def _reference_loop_inputs(x):
    batch, frames, height, width, _ = x.shape
    visual_cu = frames * torch.arange(batch + 1, dtype=torch.int32)
    text = {"text_embeds": torch.zeros(0, 1), "pooled_embed": torch.zeros(batch, 1)}
    text_cu = torch.zeros(batch + 1, dtype=torch.int32)
    rope_pos = [
        torch.cat([torch.arange(frames) for _ in range(batch)]),
        torch.arange(height // 2),
        torch.arange(width // 2),
    ]
    return visual_cu, text, text_cu, rope_pos, torch.zeros(0, dtype=torch.long)


def _port_dit_fn(model, x, scale_factor, use_motion_score=False):
    return make_dit_fn(
        model,
        latent_frames_hw=tuple(x.shape[1:4]),
        patch_size=model.patch_size,
        scale_factor=scale_factor,
        use_motion_score=use_motion_score,
    )


def test_piflow_loop_matches_reference_piflow_generate():
    """Guards the segment schedule, DXPolicy grid layout and rollout on a batch of tiles.

    The reference packs the tiles along time with one per-frame sigma; the port batches
    ``[B, T, H, W, C]``.  Same weights and same start latent, fp32 on CPU.

    The port no longer re-derives the pi-Flow segment schedule itself (``latents.piflow_schedule``
    / ``sampling.denoise_piflow`` are gone): it drives a real ``PiflowScheduler`` object through
    ``denoise_with_scheduler`` instead, which the analysis in commit d8d0e79f's port (and the
    ``PiflowScheduler._policy_step`` implementation itself) shows reproduces the exact same
    per-segment ``(raw_src, seg, raw_dst, sigma_src)`` schedule as the removed pure function did.
    """
    from kandinsky_sr.core.algo.piflow_sampler import piflow_generate

    reference = build_reference_dit(piflow=TINY_PIFLOW)
    model = build_port_dit(
        reference, piflow=TINY_PIFLOW, overrides={"instruct_type": "noise"}
    )
    x = _random_latent(4, seed=5)
    scale = (1.0, 2.0, 2.0)
    visual_cu, text, text_cu, rope_pos, text_rope = _reference_loop_inputs(x)
    packed = x.reshape(-1, *x.shape[2:]).clone()
    scheduler = PiflowScheduler(**TINY_PIFLOW)
    scheduler.set_timesteps(TINY_PIFLOW["nfe"], device="cpu")
    with torch.no_grad():
        expected = piflow_generate(
            packed,
            reference,
            text,
            visual_cu,
            text_cu,
            rope_pos,
            text_rope,
            scale,
            out_dim=4,
            start_timestep=1.0,
            device="cpu",
            **TINY_PIFLOW,
        )
        actual = denoise_with_scheduler(
            x.clone(),
            _port_dit_fn(model, x, scale),
            scheduler,
            channels=4,
            is_piflow=True,
        )
    torch.testing.assert_close(
        actual, expected.reshape(*x.shape[:4], 4), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize(
    "capped", [False, True], ids=["full_range", "capped_noise_start"]
)
def test_euler_loop_matches_reference_generate(capped):
    """Guards the warped timesteps, per-step deltas and (for a wide input) the in-place
    update of the first channels only, incl. the ``cap_noise_timestep`` start time.

    The reference's ``generate(..., 5, ...)`` counts timestep *grid points* (5 points -> 4
    Euler steps); the port's own convention counts DiT *calls* directly, so the equivalent
    port call is ``num_inference_steps=4``. Both grids are the same 5-point warped-linspace
    array (``FlowMatchEulerDiscreteScheduler.set_timesteps`` applies the configured ``shift``
    warp to the 4 unwarped sigmas this test passes in, then appends the terminal 0 itself --
    algebraically the same 5th point the reference's own ``warp(0) == 0`` produces), so the 4
    step-to-step deltas ``denoise_with_scheduler`` consumes via ``scheduler.step`` are bit-for-
    bit the 4 deltas the old, removed ``denoise_euler`` consumed from that same array.
    """
    from kandinsky_sr.core.algo.utils import generate

    cfg = dict(TINY_DIT_CFG, use_motion_score=True, instruct_type="hybrid_anchor")
    reference = build_reference_dit(piflow=None, cfg=cfg)
    model = build_port_dit(
        reference,
        cfg=cfg,
        piflow=None,
        overrides={"instruct_type": "noise", "visual_cond": True},
    )
    reference.instruct_type, reference.visual_cond = "noise", True
    x = _random_latent(2 * 4 + 1, seed=9)
    scale = (1.0, 1.5, 1.5)
    start = euler_start_timestep(
        cap_noise_timestep=capped, lq_noise_scale=0.7, instruct_type="noise"
    )
    assert start == (0.7 if capped else 1.0)
    visual_cu, text, text_cu, rope_pos, text_rope = _reference_loop_inputs(x)
    packed = x.reshape(-1, *x.shape[2:]).clone()
    num_inference_steps = 4  # the reference's 5 grid points == 4 actual Euler steps
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    sigmas = torch.linspace(start, 0.0, num_inference_steps + 1)[:-1].tolist()
    scheduler.set_timesteps(sigmas=sigmas, device="cpu")
    with torch.no_grad():
        expected = generate(
            packed,
            reference,
            "cpu",
            5,
            text,
            text,
            visual_cu,
            text_cu,
            text_cu,
            rope_pos,
            text_rope,
            text_rope,
            scale,
            1.0,
            5.0,
            start_timestep=start,
        )
        actual = denoise_with_scheduler(
            x.clone(),
            _port_dit_fn(model, x, scale, use_motion_score=True),
            scheduler,
            channels=4,
            is_piflow=False,
        )
    torch.testing.assert_close(
        actual, expected.reshape(*x.shape[:4], 4), rtol=1e-5, atol=1e-5
    )


# --------------------------------------------------------------------------- #
# Tiling / stitching against the reference helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "frame_hw,scale", [((64, 96), 2), ((64, 64), 4), ((96, 64), 2), ((80, 144), 2)]
)
def test_tile_plan_and_stitch_match_reference(frame_hw, scale):
    """Guards the tile grid (aspect -> base resolution -> even grid) and that the
    frame-chunked stitch equals one reference stitch over the whole clip (T=19, not a
    multiple of the chunk size)."""
    from kandinsky_sr.core.algo.tiling_utils import stitch_tiles_hanning
    from kandinsky_sr.pipeline.sr_pipeline import _tile_geometry

    height, width = frame_hw
    plan = plan_tiles(
        frame_hw=frame_hw,
        visual_size=512,
        tiling_scale=scale,
        tile_min_overlap=0.2,
        resolutions=TINY_RESOLUTIONS,
    )
    base, tile_hw, grid = _tile_geometry(
        height, width, 512, scale, 0.25, 16, "even", 0.2
    )
    assert (plan.base_hw, plan.tile_hw) == (base, tile_hw)
    assert tuple(plan.pixel_grid) == tuple(grid)
    generator = torch.Generator().manual_seed(3)
    tiles = [
        torch.randint(0, 256, (3, 19, *base), dtype=torch.uint8, generator=generator)
        for _ in range(grid.total_tiles)
    ]
    expected = (
        stitch_tiles_hanning(
            [t.float() for t in tiles], grid, height, width, scale=scale
        )
        .clamp(0, 255)
        .to(torch.uint8)
    )
    assert torch.equal(stitch_tiles(tiles, plan), expected)


# --------------------------------------------------------------------------- #
# End to end: pure orchestration vs kandinsky_sr.pipeline.stages.run_tiled_sr*
# --------------------------------------------------------------------------- #
TINY_LU_MODEL = {
    "architecture": "multi_scale",
    "in_channels": 4,
    "hidden_channels": 8,
    "num_pre_blocks": 1,
    "num_mid_blocks": 2,
    "num_post_blocks": 2,
    "expand_ratio": 2,
    "dims": 3,
    "bare_stem": True,
    "input_skip": False,
    "modulated_norm": True,
    "modulated_output_proj": True,
    "temporal_padding": "replicate",
    "upsample_mode": "pxs_v2",
    "upsample_padding_mode": "zeros",
    "enable_x2_entry": True,
    "x2_adapter_blocks": 1,
    "x2_tail_mode": "private_full",
    "x2_finisher": "pxs_residual",
}


def build_reference_lu_bank(target_scales=("2x", "4x"), seed=0):
    """Reference LU bank with random weights and its bundle-style ``models`` config."""
    import pydantic
    from kandinsky_sr.core.algo.latent_upscaler import LatentUpscalerBank
    from kandinsky_sr.core.components.latent_upscaler.config import ModelConfig
    from kandinsky_sr.core.components.latent_upscaler.model.factory import (
        build_upsampler,
    )

    models = [
        {"target_scale": scale, "model": copy.deepcopy(TINY_LU_MODEL)}
        for scale in target_scales
    ]
    adapter = pydantic.TypeAdapter(ModelConfig)
    bank = LatentUpscalerBank()
    for index, spec in enumerate(models):
        upsampler = build_upsampler(adapter.validate_python(spec["model"])).eval()
        randomize(upsampler, seed + 10 + index)
        upsampler.target_scale = spec["target_scale"]
        upsampler.scaling_factor = TINY_SCALING_FACTOR
        bank[spec["target_scale"]] = upsampler
    return bank, models


def build_stacks(*, piflow, use_lu):
    """(reference SRComponents, port dict) sharing identical random weights."""
    from kandinsky_sr.pipeline.sr_pipeline import SRComponents, SRParams

    torch.manual_seed(0)
    ref_vae = CachedCausalVAE(
        encoder_conf=OmegaConf.create(dict(TINY_KVAE_ENC)),
        decoder_conf=OmegaConf.create(dict(TINY_KVAE_DEC)),
    ).eval()
    randomize(ref_vae, 1, std=0.1)
    ref_vae.config = SimpleNamespace(scaling_factor=TINY_SCALING_FACTOR)
    ref_dit = build_reference_dit(piflow=piflow, seed=2)
    ref_bank, models = build_reference_lu_bank() if use_lu else (None, None)
    reference = SRComponents(
        dit=ref_dit,
        vae=ref_vae,
        latent_upscaler=ref_bank,
        sr_params=SRParams(**TINY_SR_PARAMS),
        spatial_factor=16,
    )

    vae_config = Kandinsky6SRVAEConfig()
    vae_config.update_model_arch(
        dict(
            vae_type="video-kvae",
            encoder_config=dict(TINY_KVAE_ENC),
            decoder_config=dict(TINY_KVAE_DEC),
            scaling_factor=TINY_SCALING_FACTOR,
            spatial_factor=16,
            temporal_factor=4,
        )
    )
    vae = Kandinsky6SRVAE(vae_config)
    # unprefixed reference keys = the official layout, mapped by the wrapper's load hook
    vae.load_state_dict(dict(ref_vae.state_dict()), strict=True)
    port = {
        "vae": vae.eval(),
        "dit": build_port_dit(
            ref_dit, piflow=piflow, overrides={"instruct_type": "noise"}
        ),
        "bank": None,
    }
    if use_lu:
        bank = Kandinsky6SRLatentUpscalerBank(models, TINY_SCALING_FACTOR)
        converted = {}
        scale_indices = {"2x": 0, "4x": 1}
        for key, value in ref_bank.state_dict().items():
            scale, suffix = key.split(".", 1)
            suffix = suffix.replace("private_", "")
            suffix = re.sub(r"(output_proj\.)0\.", r"\1norm.", suffix)
            suffix = re.sub(r"(output_proj\.)2\.", r"\1conv.", suffix)
            converted[f"_models.{scale_indices[scale]}.{suffix}"] = value
        bank.load_state_dict(
            converted,
            strict=True,
        )
        port["bank"] = bank.eval()
    return reference, port


def run_reference(video, reference, *, scale, seed, tiles_batch_size, num_steps):
    from kandinsky_sr.core.algo.latent_upscaler import latent_upscaler_for_scale
    from kandinsky_sr.pipeline.sr_pipeline import (
        RunConfig,
        encode_lq_video_to_lr_latent,
    )
    from kandinsky_sr.pipeline.stages import run_tiled_sr, run_tiled_sr_from_pixels
    from kandinsky_sr.pipeline.upscale_utils import (
        pre_upscale_video,
        resolve_scale_request,
    )

    tiling_scale, pre = resolve_scale_request(float(scale))
    if pre != 1.0:
        video = pre_upscale_video(video, pre, 16)
    run_config = RunConfig(
        device="cpu",
        num_steps=num_steps,
        seed=seed,
        overlap=0.25,
        tiles_batch_size=tiles_batch_size,
        resolution_scale=tiling_scale,
    )
    if latent_upscaler_for_scale(reference.latent_upscaler, tiling_scale) is not None:
        lr_latent = encode_lq_video_to_lr_latent(video, reference.vae, "cpu")
        return run_tiled_sr(lr_latent, reference, run_config)
    return run_tiled_sr_from_pixels(video, reference, run_config)


def run_port(video, port, *, scale, seed, tiles_batch_size, num_steps):
    """The stage logic (scale request, pre-upscale, specs) around the pure function.

    ``num_steps`` here is the reference's convention (timestep grid points); for a
    flow-matching (non pi-Flow) bundle the port's own ``num_inference_steps`` is
    ``num_steps - 1`` (DiT calls), so it is translated once, here, at the one remaining
    boundary between the two conventions. A pi-Flow bundle ignores it either way (its
    scheduler's own ``nfe`` always wins), so no translation is needed for that case.
    """
    tiling_scale, pre = tiling.resolve_scale_request(float(scale))
    if pre != 1.0:
        video = tiling.pre_upscale_video(video, pre, 16)
    arch = port["dit"].config
    port_num_steps = num_steps if arch.is_piflow else num_steps - 1
    spec = build_sampling_spec(
        arch=arch,
        tiling_scale=tiling_scale,
        seed=seed,
        num_steps=port_num_steps,
        tiles_batch_size=tiles_batch_size,
        tile_min_overlap=0.2,
    )
    bank = port["bank"]
    use_lu = bank is not None and bank.for_scale(tiling_scale) is not None
    return super_resolve(
        video,
        vae=port["vae"],
        scaling_factor=port["vae"].scaling_factor,
        dit=port["dit"],
        dit_spec=build_dit_spec(port["dit"]),
        spec=spec,
        scheduler=effective_scheduler(spec, None),
        device="cpu",
        upscale_fn=partial(bank.upscale, scale=tiling_scale) if use_lu else None,
        lu_dtype=module_dtype(bank) if use_lu else None,
        resolutions=TINY_RESOLUTIONS,
    )


def _random_video(frames, height, width, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 256, (frames, 3, height, width), dtype=torch.uint8, generator=generator
    )


E2E_CASES = [
    # scale, (H, W), tiles_batch_size
    (2, (64, 96), 1),
    (2, (64, 96), 3),
    (4, (64, 64), 2),
    (2.25, (64, 96), 1),
]


def _assert_uint8_close(actual, expected, *, max_diff: int = 1):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    diff = (actual.int() - expected.int()).abs()
    exact = (diff == 0).float().mean().item()
    assert diff.max().item() <= max_diff and exact > 0.999, (diff.max().item(), exact)


@pytest.mark.parametrize("use_lu", [False, True], ids=["pixel_path", "lu_path"])
@pytest.mark.parametrize("scale,hw,tiles_batch_size", E2E_CASES)
def test_tiled_piflow_super_resolution_matches_reference(
    use_lu, scale, hw, tiles_batch_size
):
    """Guards the whole orchestration: scale request + pre-upscale, tile geometry, LU or
    tile encode, chunk seeding (``seed + first tile index``), pi-Flow loop, decode with
    truncation to uint8 and Hann stitching, against ``run_tiled_sr`` /
    ``run_tiled_sr_from_pixels`` with identical random weights.

    ``max_diff=2`` (not the default 1): the reference's ``piflow_generate`` is the
    training-only inline schedule, which clamps the *final* segment's ``raw_dst`` to
    0.0 (dead code on every real checkpoint - see GAP 2 / ``latents.piflow_schedule``).
    The port now deliberately matches the real ``PiflowScheduler._policy_step``
    convention (``raw_dst=eps``) instead, which real checkpoints always run through.
    That is a ~1e-6 raw-timestep difference in the last segment only, so it shows up
    here as at most a 1-LSB-wider uint8 rounding tie on top of the pre-existing
    rounding-boundary noise (still >99.99% exact pixels), not as a structural bug.
    """
    reference, port = build_stacks(piflow=TINY_PIFLOW, use_lu=use_lu)
    video = _random_video(9, *hw)
    kwargs = dict(scale=scale, seed=42, tiles_batch_size=tiles_batch_size, num_steps=5)
    expected = run_reference(video, reference, **kwargs)
    actual = run_port(video, port, **kwargs)
    _assert_uint8_close(actual, expected, max_diff=2)


@pytest.mark.parametrize("use_lu", [False, True], ids=["pixel_path", "lu_path"])
def test_tiled_euler_super_resolution_matches_reference(use_lu):
    """Same as above for a plain (non pi-Flow) DiT: the Euler loop with ``num_steps``."""
    reference, port = build_stacks(piflow=None, use_lu=use_lu)
    video = _random_video(9, 64, 96, seed=1)
    kwargs = dict(scale=2, seed=7, tiles_batch_size=2, num_steps=4)
    expected = run_reference(video, reference, **kwargs)
    actual = run_port(video, port, **kwargs)
    _assert_uint8_close(actual, expected)


def test_seed_and_chunking_change_the_result_like_the_reference():
    """The initial noise of chunk k is seeded with ``seed + chunk start``: another seed or
    another chunk size must change the output (a port that ignored them would not)."""
    _, port = build_stacks(piflow=TINY_PIFLOW, use_lu=False)
    video = _random_video(9, 64, 96, seed=2)
    base = run_port(video, port, scale=2, seed=42, tiles_batch_size=1, num_steps=5)
    assert not torch.equal(
        base, run_port(video, port, scale=2, seed=43, tiles_batch_size=1, num_steps=5)
    )
    assert not torch.equal(
        base, run_port(video, port, scale=2, seed=42, tiles_batch_size=4, num_steps=5)
    )
