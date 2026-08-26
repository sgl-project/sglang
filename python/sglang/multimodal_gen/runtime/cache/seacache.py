# SPDX-License-Identifier: Apache-2.0
"""
SeaCache: spectral-evolution-aware step caching for diffusion transformers.

SeaCache reuses TeaCache's accumulate-and-refresh schedule -- accumulate a
relative-L1 distance between the timestep-modulated inputs of consecutive steps,
re-run the transformer blocks once the sum reaches a threshold, and otherwise add
back the cached block-stack residual -- but measures that distance after a
timestep-dependent Wiener filter instead of on the raw feature.

Diffusion builds low-frequency structure early and refines high-frequency detail
late, so the optimal linear denoiser's frequency response widens as sampling
proceeds. Filtering the decision feature with that response makes the distance
track content change rather than stochastic noise, which yields a schedule that
concentrates refreshes on the early steps that matter. Unlike TeaCache there are
no per-checkpoint fitted coefficients.

Reference: SeaCache, Chung et al., CVPR 2026 (https://arxiv.org/abs/2602.18993),
official implementation at https://github.com/jiwoogit/SeaCache.
"""

from __future__ import annotations

import msgspec
import torch

from sglang.multimodal_gen.configs.sample.seacache import SeaCacheParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Keeps the flow-matching endpoints off the degenerate a=0 / b=0 filters. FLUX's
# first sigma is exactly 1.0, so this clamp is reached on every run.
_SIGMA_CLAMP = 1e-6

# Regularizes the power-law signal spectrum at DC. Without it Sx(0) is infinite;
# with it Sx(0) = 1/eps, which makes the DC gain 1/a -- finite, and large enough
# that mean normalization still leaves a low-pass shape.
_SPECTRUM_EPS = 1e-16

_REL_L1_EPS = 1e-16

# Exponent beta of the assumed power-law signal spectrum S(f) ~ |f|^-beta. The paper
# fixes it per modality rather than tuning it: 2 for images, 3 for video.
_POWER_EXP_IMAGE = 2.0


def ab_from_sigma(sigma: float) -> tuple[float, float]:
    """Flow-matching mixture coefficients: x_t = a * x_0 + b * noise."""
    clamped = max(_SIGMA_CLAMP, min(1.0 - _SIGMA_CLAMP, float(sigma)))
    return 1.0 - clamped, clamped


def sea_filter_response(
    *,
    shape: torch.Size | tuple[int, ...],
    dims: tuple[int, ...],
    a: float,
    b: float,
    power_exp: float,
    norm_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Separable Wiener gain broadcast over `shape`, one 1-D factor per filtered axis."""
    response = None
    for axis in dims:
        freq = torch.fft.fftfreq(shape[axis], device=device, dtype=torch.float32).abs()
        signal_power = 1.0 / (freq**power_exp + _SPECTRUM_EPS)
        gain = (a * signal_power) / (a * a * signal_power + b * b + _SPECTRUM_EPS)
        view = [1] * len(shape)
        view[axis] = gain.shape[0]
        gain = gain.reshape(view)
        response = gain if response is None else response * gain

    if norm_mode == "mean":
        return response / response.mean()
    if norm_mode == "peak":
        return response / response.amax()
    return response


def apply_sea_filter(
    x: torch.Tensor,
    *,
    a: float,
    b: float,
    power_exp: float = 2.0,
    norm_mode: str = "mean",
    dims: tuple[int, ...] = (-2, -3),
) -> torch.Tensor:
    """Filter `x` along `dims` with the SEA response, in fp32, returning `x.dtype`."""
    x32 = x.contiguous().to(torch.float32)
    response = sea_filter_response(
        shape=x32.shape,
        dims=dims,
        a=a,
        b=b,
        power_exp=power_exp,
        norm_mode=norm_mode,
        device=x32.device,
    )
    spectrum = torch.fft.fftn(x32, dim=dims)
    return torch.fft.ifftn(spectrum * response, dim=dims).real.to(x.dtype)


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Relative L1 distance, normalized by the previous step's magnitude.

    Reduced in fp32 rather than the bf16 input dtype the reference uses: a
    bf16-rounded ratio accumulated over tens of steps drifts by a few percent
    against a threshold of order 0.3.
    """
    numerator = (current - previous).abs().float().mean()
    denominator = previous.abs().float().mean() + _REL_L1_EPS
    return float(numerator / denominator)


class _BranchState(msgspec.Struct):
    previous_modulated_input: torch.Tensor | None = None
    previous_residual: torch.Tensor | None = None
    accumulated_rel_l1_distance: float = 0.0
    real_steps: int = 0
    skipped_steps: int = 0


class _StepContext(msgspec.Struct):
    params: SeaCacheParams
    step: int
    num_steps: int
    sigmas: torch.Tensor
    is_cfg_negative: bool
    debug: bool


class SeaCache:
    """Per-DiT SeaCache state and skip decision.

    Held by `CachableDiT` and driven by the model's forward, which supplies the
    block-0 modulated input and the latent grid shape.
    """

    # Models whose forward is entered once per CFG branch, so each branch needs
    # its own accumulator. FLUX.1-dev is single-branch by default (embedded
    # guidance, no negative prompt) but becomes two-branch with --negative-prompt.
    _CFG_SUPPORTED_PREFIXES = frozenset({"flux"})

    def __init__(self, *, prefix: str) -> None:
        self.supports_cfg_cache = prefix.lower() in self._CFG_SUPPORTED_PREFIXES
        self._branches: dict[bool, _BranchState] = {}
        self.reset()

    def reset(self) -> None:
        self._branches = {False: _BranchState(), True: _BranchState()}

    def should_run_blocks(
        self, *, modulated_inp: torch.Tensor, grid_hw: tuple[int, int]
    ) -> bool:
        context = self._resolve()
        if context is None:
            return True

        if context.step == 0 and not context.is_cfg_negative:
            self.reset()

        state = self._branch_state(context)
        should_run = self._decide(
            context=context, state=state, modulated_inp=modulated_inp, grid_hw=grid_hw
        )

        if should_run:
            state.real_steps += 1
        else:
            state.skipped_steps += 1
        if context.step == context.num_steps - 1:
            self._log_summary(context=context, state=state)
        return should_run

    def record_residual(
        self, *, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        context = self._resolve()
        if context is None:
            return
        self._branch_state(context).previous_residual = (
            hidden_states - original_hidden_states
        )

    def retrieve(self, *, hidden_states: torch.Tensor) -> torch.Tensor:
        context = self._resolve()
        return hidden_states + self._branch_state(context).previous_residual

    def _branch_state(self, context: _StepContext) -> _BranchState:
        if self.supports_cfg_cache:
            return self._branches[context.is_cfg_negative]
        return self._branches[False]

    def _decide(
        self,
        *,
        context: _StepContext,
        state: _BranchState,
        modulated_inp: torch.Tensor,
        grid_hw: tuple[int, int],
    ) -> bool:
        # The reference stores the unfiltered feature on force-computed steps and
        # the filtered one elsewhere, so the first gated distance compares across
        # representations. Kept as-is: this is the schedule the published FLUX
        # numbers come from.
        last_step = context.num_steps - 1
        if (
            context.step == 0
            or context.step == last_step
            or state.previous_modulated_input is None
        ):
            state.accumulated_rel_l1_distance = 0.0
            state.previous_modulated_input = modulated_inp
            return True

        a, b = ab_from_sigma(float(context.sigmas[context.step]))
        height, width = grid_hw
        filtered = apply_sea_filter(
            modulated_inp.reshape(modulated_inp.shape[0], height, width, -1),
            a=a,
            b=b,
            power_exp=_POWER_EXP_IMAGE,
            norm_mode=context.params.norm_mode,
        ).reshape(modulated_inp.shape)

        distance = state.accumulated_rel_l1_distance + rel_l1(
            filtered, state.previous_modulated_input
        )
        # Ranks must agree: the blocks contain collectives, so one rank skipping
        # while another computes deadlocks. Syncing the accumulator rather than
        # the boolean keeps the two branches' state in lockstep too.
        state.accumulated_rel_l1_distance = _sync_distance(
            distance, device=modulated_inp.device
        )
        state.previous_modulated_input = filtered

        should_run = state.accumulated_rel_l1_distance >= context.params.thresh
        if should_run:
            state.accumulated_rel_l1_distance = 0.0
        return should_run

    def _resolve(self) -> _StepContext | None:
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        forward_context = get_forward_context()
        batch = forward_context.forward_batch
        if (
            batch is None
            or not batch.enable_seacache
            or batch.seacache_params is None
            # Warmup and breakable-CUDA-graph capture both issue several forwards
            # under one timestep index and carry a truncated step count.
            or batch.is_warmup
        ):
            return None

        if batch.progressive_mode != "fullres":
            logger.warning_once(
                "SeaCache is disabled for progressive resolution: the latent shape "
                "changes between stages and step indices can be revisited."
            )
            return None

        if get_global_server_args().enable_breakable_cuda_graph:
            logger.warning_once(
                "SeaCache is disabled under --enable-breakable-cuda-graph: a replayed "
                "graph freezes the skip decision taken at capture time."
            )
            return None

        if batch.did_sp_shard_latents:
            raise RuntimeError(
                "SeaCache is not compatible with sequence parallelism: its filter "
                "needs the full 2-D latent grid, but each rank holds a contiguous "
                "slice of rows. Drop --ulysses-degree / --ring-degree, or use "
                "--tp-size to spend the extra GPUs on tensor parallelism instead."
            )

        return _StepContext(
            params=batch.seacache_params,
            step=forward_context.current_timestep,
            num_steps=batch.num_inference_steps,
            sigmas=batch.scheduler.sigmas,
            is_cfg_negative=batch.is_cfg_negative,
            debug=batch.debug,
        )

    def _log_summary(self, *, context: _StepContext, state: _BranchState) -> None:
        if not context.debug:
            return
        total = state.real_steps + state.skipped_steps
        if total == 0:
            return
        branch = "negative" if context.is_cfg_negative else "positive"
        logger.info(
            "[SeaCache] %s branch: %d/%d steps refreshed (refresh ratio %.2f), "
            "thresh=%.3f",
            branch,
            state.real_steps,
            total,
            state.real_steps / total,
            context.params.thresh,
        )


def _sync_distance(distance: float, *, device: torch.device) -> float:
    from sglang.multimodal_gen.runtime.distributed import (
        get_tp_group,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        return distance
    group = get_tp_group()
    if group.world_size == 1:
        return distance
    buffer = torch.tensor([distance], dtype=torch.float64, device=device)
    group.broadcast(buffer, src=0)
    return float(buffer[0])
