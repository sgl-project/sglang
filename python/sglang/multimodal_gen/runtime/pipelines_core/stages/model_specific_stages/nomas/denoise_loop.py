# SPDX-License-Identifier: Apache-2.0
"""Nomas segment denoising: per-segment Euler loop and KV cache write."""

import math
import threading

import torch
from msgspec import Struct

from sglang.multimodal_gen.runtime.models.dits.nomas import NomasTransformer3DModel


def _segment_is_finite(latent, entries):
    # Keep validation on-device until all reductions finish: one host decision,
    # rather than one synchronization for each layer's K and V.
    flags = [torch.isfinite(latent).all()]
    flags.extend(
        torch.isfinite(entry[key]).all() for entry in entries for key in ("k", "v")
    )
    return bool(torch.stack(flags).all())


class NomasSegmentCondition(Struct, frozen=True):
    text_states: torch.Tensor
    text_mask: torch.Tensor
    camera_embedding: torch.Tensor | None = None
    camera_has_mask: torch.Tensor | None = None
    text_offsets: tuple[int, ...] | None = None
    video_windows: tuple[tuple[int, int, int], ...] | None = None
    action_start: int = -1


class NomasLatentSession:
    """One bounded, sequential session; model weights may be shared between sessions.

    One call represents one complete prompt segment, with all current text
    appended exactly once. Inter-segment cache visibility is explicit. No
    chunk subdivision, CFG, image conditioning, or distributed mode.
    """

    def __init__(
        self,
        model: NomasTransformer3DModel,
        *,
        cache_capacity: int,
        seed: int,
        steps: int = 8,
        shift: float = 5.0,
        cache_noise_sigma: float = 0.206897,
        generator: torch.Generator | None = None,
        sigmas: torch.Tensor | None = None,
        max_cache_segments: int | None = None,
        max_cache_capacity: int | None = None,
        cache_growth_factor: float = 1.5,
    ):
        if (
            type(steps) is not int
            or steps <= 0
            or not math.isfinite(shift)
            or shift <= 0
        ):
            raise ValueError("Euler steps and shift must be positive and finite")
        if not 0 <= cache_noise_sigma <= 1:
            raise ValueError("cache noise sigma must be in [0,1]")
        self.model = model.eval().requires_grad_(False)
        self.cache = model.allocate_cache(cache_capacity)
        self.seed = seed
        self.generator = (
            generator
            if generator is not None
            else torch.Generator(device="cpu").manual_seed(seed)
        )
        if self.generator.device.type != "cpu":
            raise ValueError("Nomas requires a CPU noise generator")
        self.sigmas = (
            None if sigmas is None else sigmas.detach().to("cpu", torch.float32).clone()
        )
        if self.sigmas is not None and (
            self.sigmas.shape != (steps + 1,)
            or not torch.isfinite(self.sigmas).all()
            or self.sigmas[-1] != 0
            or not (self.sigmas[:-1] > self.sigmas[1:]).all()
        ):
            raise ValueError(
                "sigmas must be a finite decreasing steps+1 schedule ending at zero"
            )
        self.steps, self.shift = steps, shift
        self.cache_noise_sigma = cache_noise_sigma
        if max_cache_segments is not None and max_cache_segments < 1:
            raise ValueError("max_cache_segments must be positive or None")
        # Upstream max_cache_segments: FIFO by whole segment, None keeps all.
        self.max_cache_segments = max_cache_segments
        # The cache grows geometrically up to this many tokens before evicting.
        if max_cache_capacity is not None and max_cache_capacity < cache_capacity:
            raise ValueError("max_cache_capacity cannot be below the initial capacity")
        self.max_cache_capacity = max_cache_capacity or cache_capacity
        if cache_growth_factor <= 1:
            raise ValueError("cache_growth_factor must exceed 1")
        self.cache_growth_factor = cache_growth_factor
        self._lock = threading.Lock()
        self._finished = False

    def _make_room(self, tokens: int) -> bool:
        """Enforce max_cache_segments, grow toward max_cache_capacity, then evict FIFO.

        Returns whether any segment was evicted, i.e. whether history changed.
        """
        cache = self.cache
        try:
            evicted = 0
            if self.max_cache_segments is not None:
                while len(cache.segment_lengths) >= self.max_cache_segments:
                    evicted += cache.evict_oldest_segment()
            needed = cache.valid_length + tokens
            if needed > cache.capacity and cache.capacity < self.max_cache_capacity:
                target = max(
                    needed, math.ceil(cache.capacity * self.cache_growth_factor)
                )
                cache.grow(min(target, self.max_cache_capacity))
            evicted += cache.evict_for(tokens)
            return evicted > 0
        except Exception:
            # This function may already have evicted history or replaced some
            # layer buffers. Its caller has not received `evicted` yet.
            cache.failed = True
            raise

    def reset(self, seed=None):
        with self._lock:
            self.cache.reset()
            self._finished = False
            if seed is not None:
                self.seed = seed
            self.generator.manual_seed(self.seed)

    @torch.inference_mode()
    def generate_segment(
        self,
        shape: tuple[int, int, int],
        condition: NomasSegmentCondition,
        *,
        initial_latents=None,
        return_trace=False,
        cache_segment=True,
        evict=False,
    ):
        """Return [1,C,T,H,W] latent; optionally record every Euler step for parity.

        ``initial_latents`` bypasses initial-noise RNG consumption, useful for
        independent numerical comparison. Cache noise still uses session RNG.
        Set cache_segment=False for the final offline segment; this ends the
        session until reset() without a cache forward or cache-noise RNG draw.
        With evict=True, older segments are evicted FIFO after denoising and
        before the cache forward, as upstream does at cache-write time: the
        denoise steps see up to max_cache_segments of history, the committed
        K/V see the post-eviction history.
        On failure RNG is restored. A failed cache commit, or a failure after
        eviction already rewrote the history, marks the cache failed and
        requires reset().
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent calls on one Nomas session are unsupported")
        rng_state = self.generator.get_state()
        evicted = False
        try:
            if self._finished:
                raise RuntimeError(
                    "uncached final segment ended the session; reset first"
                )
            if len(shape) != 3 or any(type(x) is not int or x <= 0 for x in shape):
                raise ValueError("segment shape must be positive integer (T,H,W)")
            if condition.text_mask.ndim != 2:
                raise ValueError("text mask must be [1,S]")
            if cache_segment:
                tokens = math.prod(shape) + condition.text_mask.shape[1]
                if evict:
                    if tokens > self.max_cache_capacity:
                        raise ValueError(
                            "a single Nomas segment exceeds the cache capacity"
                        )
                else:
                    self.cache.reserve(tokens)
            p = next(self.model.parameters())
            full_shape = (1, self.model.config.in_channels, *shape)
            if initial_latents is None:
                latent = torch.randn(
                    full_shape, generator=self.generator, dtype=p.dtype
                ).to(p.device)
            else:
                if (
                    initial_latents.shape != full_shape
                    or initial_latents.device != p.device
                    or initial_latents.dtype != p.dtype
                ):
                    raise ValueError(
                        "initial latents shape, dtype and device must match model"
                    )
                latent = initial_latents.clone()
            kwargs = dict(
                encoder_hidden_states=condition.text_states,
                text_mask=condition.text_mask,
                cache=self.cache,
                camera_embedding=condition.camera_embedding,
                camera_has_mask=condition.camera_has_mask,
                text_offsets=condition.text_offsets,
                video_windows=condition.video_windows,
                hide_cached_action_tails=True,
            )
            sigmas = self.sigmas
            if sigmas is None:
                sigmas = torch.linspace(1, 0, self.steps + 1, dtype=torch.float32)
                sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
            trace = []
            # Camera gating promotes residuals to FP32, as in the upstream model.
            with torch.autocast(
                p.device.type,
                dtype=p.dtype,
                enabled=p.dtype in (torch.bfloat16, torch.float16),
            ):
                context = self.model.prepare_segment_context(
                    latent,
                    condition.text_mask,
                    self.cache,
                    text_offsets=condition.text_offsets,
                    video_windows=condition.video_windows,
                    hide_cached_action_tails=True,
                )
                kwargs["_context"] = context
                for index in range(self.steps):
                    timestep = (sigmas[index : index + 1] * 1000).to(p.device)
                    pred, _ = self.model.predict(latent, timestep, **kwargs)
                    latent = (
                        latent.float()
                        + pred.float() * (sigmas[index + 1] - sigmas[index])
                    ).to(p.dtype)
                    if return_trace:
                        trace.append(
                            {"prediction": pred.clone(), "latent": latent.clone()}
                        )
                entries = []
                if cache_segment:
                    if evict:
                        evicted = self._make_room(tokens)
                        if evicted:
                            kwargs["_context"] = self.model.prepare_segment_context(
                                latent,
                                condition.text_mask,
                                self.cache,
                                text_offsets=condition.text_offsets,
                                video_windows=condition.video_windows,
                                hide_cached_action_tails=True,
                                rope=context.rope,
                            )
                    sigma = self.cache_noise_sigma
                    cache_latent = latent
                    if sigma:
                        noise = torch.randn(
                            latent.shape, generator=self.generator, dtype=latent.dtype
                        ).to(p.device)
                        cache_latent = (
                            (1 - sigma) * latent.float() + sigma * noise.float()
                        ).to(latent.dtype)
                    _, entries = self.model.predict(
                        cache_latent,
                        torch.tensor([sigma * 1000], device=p.device),
                        **kwargs,
                        collect_cache=True,
                    )
            if not _segment_is_finite(latent, entries):
                raise RuntimeError("non-finite latent/cache; segment was not committed")
            if cache_segment:
                action_start = condition.action_start
                if action_start < 0 and condition.text_offsets is not None:
                    action_start = condition.text_offsets[0]
                self.cache.commit(
                    entries,
                    condition.text_mask,
                    math.prod(shape),
                    shape[0],
                    action_start,
                )
            else:
                self._finished = True
            return (latent, trace) if return_trace else latent
        except Exception:
            self.generator.set_state(rng_state)
            if evicted:
                # A retry would denoise against a different history.
                self.cache.failed = True
            raise
        finally:
            self._lock.release()
