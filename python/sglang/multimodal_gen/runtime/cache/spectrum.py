# SPDX-License-Identifier: Apache-2.0
"""
Spectrum: Adaptive Spectral Feature Forecasting for diffusion sampling acceleration.

Training-free step skipping with Chebyshev polynomial ridge regression over
denoiser block outputs. See https://arxiv.org/abs/2603.01623
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.spectrum import SpectrumParams


logger = logging.getLogger(__name__)


def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Reshape tensor to (1, -1) for ridge regression, preserving original shape."""
    return x.reshape(1, -1), x.shape


def _unflatten(x_flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Restore flattened tensor to original shape."""
    return x_flat.reshape(shape)


def _is_missing_linalg_backend_error(err: RuntimeError) -> bool:
    """Detect the "no LAPACK / no MAGMA" RuntimeError PyTorch raises when the
    build lacks a linear algebra backend for the tensor's device.

    Some PyTorch builds (notably several ROCm wheels) ship without LAPACK for
    CPU tensors *and* without MAGMA for CUDA/HIP tensors, making
    ``torch.linalg.cholesky``/``torch.cholesky_solve`` unusable on any device.
    """
    msg = str(err)
    return "LAPACK" in msg or "MAGMA" in msg


def _cholesky_lower(a: torch.Tensor) -> torch.Tensor:
    """Cholesky factorization ``A = L @ L.T`` without ``torch.linalg``.

    Used as a fallback when the backend has no LAPACK/MAGMA support. ``a`` is
    expected to be tiny (Chebyshev design matrix size), so the Python-level
    loop is cheap.
    """
    n = a.shape[0]
    L = torch.zeros_like(a)
    for i in range(n):
        for j in range(i + 1):
            s = L[i, :j] @ L[j, :j]
            if i == j:
                L[i, j] = torch.sqrt(a[i, i] - s)
            else:
                L[i, j] = (a[i, j] - s) / L[j, j]
    return L


def _cholesky_solve_lower(l: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve ``(L @ L.T) @ x = b`` given lower-triangular Cholesky factor ``l``."""
    n = l.shape[0]
    y = torch.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - l[i, :i] @ y[:i]) / l[i, i]
    x = torch.zeros_like(b)
    for i in reversed(range(n)):
        x[i] = (y[i] - l[i + 1 :, i] @ x[i + 1 :]) / l[i, i]
    return x


def _ridge_cholesky_solve(xtx: torch.Tensor, xth: torch.Tensor) -> torch.Tensor:
    """Solve the ridge normal equations ``xtx @ coef = xth`` via Cholesky.

    Prefers ``torch.linalg.cholesky``/``torch.cholesky_solve``, retrying once
    with jitter if ``xtx`` is only marginally non-positive-definite. Falls
    back to a manual, LAPACK/MAGMA-free Cholesky solve when the backend has
    no linear algebra support at all for this device (see
    `_is_missing_linalg_backend_error`).
    """
    p = xtx.shape[0]
    eye = torch.eye(p, device=xtx.device, dtype=xtx.dtype)
    try:
        chol = torch.linalg.cholesky(xtx)
        return torch.cholesky_solve(xth, chol)
    except RuntimeError as e:
        if _is_missing_linalg_backend_error(e):
            chol = _cholesky_lower(xtx)
            if not torch.isfinite(chol).all():
                jitter = 1e-6 * xtx.diag().mean()
                chol = _cholesky_lower(xtx + jitter * eye)
            return _cholesky_solve_lower(chol, xth)
        jitter = 1e-6 * xtx.diag().mean()
        chol = torch.linalg.cholesky(xtx + jitter * eye)
        return torch.cholesky_solve(xth, chol)


class ChebyshevForecaster(nn.Module):
    """Chebyshev-basis ridge regression forecaster over diffusion step index.

    Keeps a sliding window of (step, flattened_feature) pairs from recent *real*
    DiT forwards. On predict(), fits ridge regression coefficients for Chebyshev
    bases T_0..T_M on normalized time, then evaluates at the requested step.
    """

    def __init__(
        self,
        M: int = 4,
        K: int = 100,
        lam: float = 0.1,
        num_steps: int = 50,
        device: Optional[torch.device] = None,
        feature_shape: Optional[torch.Size] = None,
    ) -> None:
        super().__init__()
        self.M = M
        self.K = K
        self.lam = lam
        self.num_steps = num_steps
        # Preallocated (K,) / (K, N) ring buffer, written in place by update().
        self.register_buffer("t_buf", torch.empty(0))
        self._H_buf: Optional[torch.Tensor] = None
        self._write_idx = 0
        self._count = 0
        self._shape: Optional[torch.Size] = feature_shape
        self._feature_dtype: Optional[torch.dtype] = None
        self._coef: Optional[torch.Tensor] = None
        self.device_ref = device
        self._tau_scale = 2.0 / float(self.num_steps) if self.num_steps != 0 else 0.0

    @property
    def P(self) -> int:
        return self.M + 1

    def _taus(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize timesteps to [-1, 1] range for Chebyshev basis."""
        if self.num_steps == 0:
            return torch.zeros_like(t)
        return t * self._tau_scale - 1.0

    def _build_design(self, taus: torch.Tensor) -> torch.Tensor:
        """Build Chebyshev basis design matrix [T_0(tau), T_1(tau), ..., T_M(tau)]."""
        taus = taus.reshape(-1, 1)
        k = taus.shape[0]
        t0 = torch.ones((k, 1), device=taus.device, dtype=taus.dtype)
        if self.M == 0:
            return t0
        t1 = taus
        cols = [t0, t1]
        for _ in range(2, self.M + 1):
            cols.append(2 * taus * cols[-1] - cols[-2])
        return torch.cat(cols[: self.M + 1], dim=1)

    def update(self, t: float | torch.Tensor, h: torch.Tensor) -> None:
        device = self.device_ref or h.device
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=device)
        h_flat, shape = _flatten(h)
        if self._feature_dtype is None:
            self._feature_dtype = h_flat.dtype
        h_flat = h_flat.to(device).to(torch.float32)
        if self._shape is None:
            self._shape = shape
        else:
            assert shape == self._shape, "Spectrum feature shape must remain constant"

        if self._H_buf is None:
            # Preallocate the ring buffer once the flattened feature size is known
            self.t_buf = torch.zeros(self.K, dtype=torch.float32, device=device)
            self._H_buf = torch.zeros(
                self.K, h_flat.shape[1], dtype=torch.float32, device=device
            )

        self.t_buf[self._write_idx] = t_tensor
        self._H_buf[self._write_idx] = h_flat[0]
        self._write_idx = (self._write_idx + 1) % self.K
        self._count = min(self._count + 1, self.K)

        # Invalidate cached ridge coefficients; will be refit on next predict()
        self._coef = None

    def _recent(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the last `min(n, count)` valid (t, h) entries in chronological
        order (oldest first, most recent last).

        Only used by the discrete Taylor blend, which needs true temporal
        order for its finite differences (unlike the ridge fit, see
        `_fit_if_needed`). `n` is small (<= taylor_order + 1), so even the
        rare wraparound `cat` below is cheap.
        """
        count = min(self._count, n)
        start = (self._write_idx - count) % self.K
        if start + count <= self.K:
            # Contiguous window -- cheap view, no copy.
            return self.t_buf[start : start + count], self._H_buf[start : start + count]
        # Wraps past the end of the physical buffer -- must reorder.
        tail = self.K - start
        t = torch.cat([self.t_buf[start:], self.t_buf[: count - tail]])
        h = torch.cat([self._H_buf[start:], self._H_buf[: count - tail]])
        return t, h

    def ready(self) -> bool:
        return self._count >= 1

    def _fit_if_needed(self) -> None:
        """Fit ridge regression coefficients on cached (t, h) pairs if not cached."""
        if self._coef is not None:
            return
        assert self.ready()
        assert self._H_buf is not None
        assert self._feature_dtype is not None
        feature_dtype = self._feature_dtype
        t, h = self.t_buf[: self._count], self._H_buf[: self._count]
        taus = self._taus(t)
        # Ridge solve in fp32; autocast would keep matmuls in bf16 and break
        # torch.cholesky_solve dtype requirements.
        with torch.autocast(device_type=self._H_buf.device.type, enabled=False):
            x = self._build_design(taus).to(torch.float32)
            p = x.shape[1]
            lam_i = self.lam * torch.eye(p, device=x.device, dtype=x.dtype)
            xt = x.transpose(0, 1)
            xtx = xt @ x + lam_i
            xth = xt @ h
            self._coef = _ridge_cholesky_solve(xtx, xth).to(feature_dtype)

    @torch.no_grad()
    def predict(self, t_star: float | torch.Tensor) -> torch.Tensor:
        assert self._shape is not None
        device = self.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=torch.float32, device=device)
        self._fit_if_needed()
        assert self._coef is not None
        tau_star = self._taus(t_star)
        x_star = self._build_design(tau_star[None]).to(self._coef.dtype)
        h_flat = x_star @ self._coef
        return _unflatten(h_flat, self._shape)


class SpectrumForecaster(nn.Module):
    """Chebyshev + discrete Taylor blend forecaster.

    The paper uses pure Chebyshev; the reference repo blends with a local
    discrete Taylor predictor. ``w=1`` is Chebyshev-only; lower ``w`` adds Taylor.
    """

    def __init__(
        self,
        cheb: ChebyshevForecaster,
        *,
        taylor_order: int = 1,
        w: float = 1.0,
    ) -> None:
        super().__init__()
        self.cheb = cheb
        self.taylor_order = taylor_order
        self.w = w

    @torch.no_grad()
    def _local_taylor_discrete(self, t_star: torch.Tensor) -> torch.Tensor:
        """Predict hidden state at t_star using discrete Taylor expansion from recent real steps."""
        assert self.cheb._H_buf is not None
        assert self.cheb._shape is not None
        assert self.cheb._feature_dtype is not None
        feature_dtype = self.cheb._feature_dtype
        t, h = self.cheb._recent(self.taylor_order + 1)
        h_i = h[-1]
        if t.numel() < 2:
            return _unflatten(h_i.reshape(1, -1), self.cheb._shape).to(feature_dtype)
        h_im1 = h[-2]
        t_i = t[-1]
        t_im1 = t[-2]
        dh1 = h_i - h_im1
        dt_last = (t_i - t_im1).clamp_min(1e-8)
        k = ((t_star - t_i) / dt_last).to(h_i.dtype)
        out = h_i + k * dh1
        if self.taylor_order >= 2 and t.numel() >= 3:
            h_im2 = h[-3]
            d2 = h_i - 2 * h_im1 + h_im2
            out = out + 0.5 * k * (k - 1.0) * d2
        if self.taylor_order >= 3 and t.numel() >= 4:
            h_im3 = h[-4]
            d3 = h_i - 3 * h_im1 + 3 * h_im2 - h_im3
            out = out + (k * (k - 1.0) * (k - 2.0) / 6.0) * d3
        return _unflatten(out.reshape(1, -1), self.cheb._shape).to(feature_dtype)

    @torch.no_grad()
    def predict(self, t_star: float | torch.Tensor) -> torch.Tensor:
        """Blend Chebyshev regression and local Taylor predictions."""
        device = self.cheb.t_buf.device
        t_star = torch.as_tensor(t_star, dtype=torch.float32, device=device)
        if self.w >= 1.0:
            return self.cheb.predict(t_star)
        elif self.w <= 0.0:
            return self._local_taylor_discrete(t_star)
        return torch.lerp(
            self._local_taylor_discrete(t_star), self.cheb.predict(t_star), self.w
        )

    def update(self, t, h) -> None:
        self.cheb.update(t, h)

    def ready(self) -> bool:
        return self.cheb.ready()


@dataclass
class SpectrumContext:
    current_step: int
    num_inference_steps: int
    total_forward_steps: int
    do_cfg: bool
    is_cfg_negative: bool
    spectrum_params: SpectrumParams
    debug: bool


class SpectrumMixin:
    """Mixin providing Spectrum step-skipping and feature forecasting.

    Wired into ``CachableDiT`` (see ``runtime/models/dits/base.py``). Concrete
    models call three hooks from ``forward()`` around their transformer blocks:

    - ``begin_spectrum_step()`` — advance skip schedule; returns whether to run
      blocks or forecast instead.
    - ``spectrum_record_features()`` — after a real forward, store block outputs.
    - ``spectrum_predict_features()`` — on skipped steps, return forecasted outputs.

    Models with separate CFG branches (Wan, Hunyuan, SD3) list their model prefix
    in ``_CFG_SUPPORTED_PREFIXES`` so cond/uncond maintain independent counters and
    forecasters. All other ``CachableDiT`` subclasses share one counter.
    """

    # DiT model prefixes that run separate cond/uncond forwards (see TeaCache).
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "sd3"}

    def _init_spectrum_state(self) -> None:
        """Initialize Spectrum state variables. Dual-branch models (Wan, Hunyuan) track separate
        cond/uncond forecasters and counters; others share one.
        """
        # Positive branch (cond) or single-branch state
        self.spectrum_cnt = 0
        self.spectrum_num_consecutive_cached_steps = 0
        self.spectrum_curr_ws: Optional[float] = None
        self.spectrum_forecaster: Optional[SpectrumForecaster] = None
        self.spectrum_real_steps = 0
        self.spectrum_skipped_steps = 0
        self.spectrum_shadow_rel_l2_sum = 0.0
        self.spectrum_shadow_rel_l2_count = 0

        # Negative branch (uncond) state for dual-branch CFG models
        self.spectrum_cnt_negative = 0
        self.spectrum_num_consecutive_cached_steps_negative = 0
        self.spectrum_curr_ws_negative: Optional[float] = None
        self.spectrum_forecaster_negative: Optional[SpectrumForecaster] = None
        self.spectrum_real_steps_negative = 0
        self.spectrum_skipped_steps_negative = 0
        self.spectrum_shadow_rel_l2_sum_negative = 0.0
        self.spectrum_shadow_rel_l2_count_negative = 0

        # Runtime branch tracking
        self.spectrum_is_cfg_negative = False
        self._spectrum_ctx: Optional[SpectrumContext] = None
        self._spectrum_supports_cfg_cache = (
            self.prefix.lower() in self._CFG_SUPPORTED_PREFIXES
        )

    def reset_spectrum_state(self, spectrum_params: SpectrumParams) -> None:
        self.spectrum_cnt = 0
        self.spectrum_num_consecutive_cached_steps = 0
        self.spectrum_curr_ws = spectrum_params.window_size
        self.spectrum_forecaster = None
        self.spectrum_is_cfg_negative = False
        self.spectrum_real_steps = 0
        self.spectrum_skipped_steps = 0
        self.spectrum_shadow_rel_l2_sum = 0.0
        self.spectrum_shadow_rel_l2_count = 0
        if self._spectrum_supports_cfg_cache:
            self.spectrum_cnt_negative = 0
            self.spectrum_num_consecutive_cached_steps_negative = 0
            self.spectrum_curr_ws_negative = spectrum_params.window_size
            self.spectrum_forecaster_negative = None
            self.spectrum_real_steps_negative = 0
            self.spectrum_skipped_steps_negative = 0
            self.spectrum_shadow_rel_l2_sum_negative = 0.0
            self.spectrum_shadow_rel_l2_count_negative = 0

    def _get_spectrum_branch_state(self) -> tuple[int, int, float]:
        """Get schedule state for current branch (cond or uncond)."""
        if self.spectrum_is_cfg_negative and self._spectrum_supports_cfg_cache:
            return (
                self.spectrum_cnt_negative,
                self.spectrum_num_consecutive_cached_steps_negative,
                self.spectrum_curr_ws_negative or 0.0,
            )
        return (
            self.spectrum_cnt,
            self.spectrum_num_consecutive_cached_steps,
            self.spectrum_curr_ws or 0.0,
        )

    def _set_spectrum_branch_state(
        self, cnt: int, consecutive: int, curr_ws: float
    ) -> None:
        """Set schedule state for current branch (cond or uncond)."""
        if self.spectrum_is_cfg_negative and self._spectrum_supports_cfg_cache:
            self.spectrum_cnt_negative = cnt
            self.spectrum_num_consecutive_cached_steps_negative = consecutive
            self.spectrum_curr_ws_negative = curr_ws
        else:
            self.spectrum_cnt = cnt
            self.spectrum_num_consecutive_cached_steps = consecutive
            self.spectrum_curr_ws = curr_ws

    def _get_spectrum_forecaster(self) -> Optional[SpectrumForecaster]:
        """Get forecaster for current branch (cond or uncond)."""
        if self.spectrum_is_cfg_negative and self._spectrum_supports_cfg_cache:
            return self.spectrum_forecaster_negative
        return self.spectrum_forecaster

    def _set_spectrum_forecaster(self, forecaster: SpectrumForecaster) -> None:
        """Set forecaster for current branch (cond or uncond)."""
        if self.spectrum_is_cfg_negative and self._spectrum_supports_cfg_cache:
            self.spectrum_forecaster_negative = forecaster
        else:
            self.spectrum_forecaster = forecaster

    def _get_spectrum_context(self) -> Optional[SpectrumContext]:
        """Retrieve current Spectrum context from forward batch. Returns None if disabled."""
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        try:
            forward_context = get_forward_context()
        except AssertionError:
            return None
        forward_batch = forward_context.forward_batch
        if (
            forward_batch is None
            or not forward_batch.enable_spectrum
            or forward_batch.spectrum_params is None
        ):
            return None

        spectrum_params = forward_batch.spectrum_params
        do_cfg = forward_batch.do_classifier_free_guidance
        is_cfg_negative = forward_batch.is_cfg_negative
        num_inference_steps = forward_batch.num_inference_steps
        total_forward_steps = spectrum_params.get_total_forward_steps(
            num_inference_steps,
            do_cfg,
            self._spectrum_supports_cfg_cache,
        )

        return SpectrumContext(
            current_step=forward_context.current_timestep,
            num_inference_steps=num_inference_steps,
            total_forward_steps=total_forward_steps,
            do_cfg=do_cfg,
            is_cfg_negative=is_cfg_negative,
            spectrum_params=spectrum_params,
            debug=bool(getattr(forward_batch, "debug", False)),
        )

    def _record_spectrum_step_stat(self, actual_forward: bool) -> None:
        if self.spectrum_is_cfg_negative and self._spectrum_supports_cfg_cache:
            if actual_forward:
                self.spectrum_real_steps_negative += 1
            else:
                self.spectrum_skipped_steps_negative += 1
            return

        if actual_forward:
            self.spectrum_real_steps += 1
        else:
            self.spectrum_skipped_steps += 1

    def _record_shadow_error_stat(self, rel_l2: float) -> None:
        if self.spectrum_is_cfg_negative and self._spectrum_supports_cfg_cache:
            self.spectrum_shadow_rel_l2_sum_negative += rel_l2
            self.spectrum_shadow_rel_l2_count_negative += 1
            return

        self.spectrum_shadow_rel_l2_sum += rel_l2
        self.spectrum_shadow_rel_l2_count += 1

    def _emit_spectrum_summary(self, ctx: SpectrumContext) -> None:
        if not ctx.debug:
            return

        if self.spectrum_is_cfg_negative and self._spectrum_supports_cfg_cache:
            real = self.spectrum_real_steps_negative
            skipped = self.spectrum_skipped_steps_negative
            err_sum = self.spectrum_shadow_rel_l2_sum_negative
            err_count = self.spectrum_shadow_rel_l2_count_negative
            branch = "negative"
        else:
            real = self.spectrum_real_steps
            skipped = self.spectrum_skipped_steps
            err_sum = self.spectrum_shadow_rel_l2_sum
            err_count = self.spectrum_shadow_rel_l2_count
            branch = "positive"

        total = real + skipped
        skip_ratio = (skipped / total) if total > 0 else 0.0
        avg_rel_l2 = (err_sum / err_count) if err_count > 0 else float("nan")
        logger.info(
            "[Spectrum/%s] total=%d real=%d skipped=%d skip_ratio=%.3f shadow_rel_l2_avg=%s n=%d window_start=%.3f flex=%.3f",
            branch,
            total,
            real,
            skipped,
            skip_ratio,
            f"{avg_rel_l2:.4f}" if err_count > 0 else "NA",
            err_count,
            float(ctx.spectrum_params.window_size),
            float(ctx.spectrum_params.flex_window),
        )

    def begin_spectrum_step(self) -> bool:
        """Advance Spectrum schedule. Returns True when transformer blocks should run.

        Schedule (after ``warmup_steps`` real forwards):
        - ``window = floor(curr_ws)`` — run a real forward when
          ``(consecutive_cached + 1) % window == 0``, otherwise skip.
        - Each real forward increases ``curr_ws`` by ``flex_window``, widening
          gaps over time (paper alpha).
        """
        ctx = self._get_spectrum_context()
        self._spectrum_ctx = ctx
        if ctx is None:
            # Spectrum disabled — always run blocks (normal DiT path).
            return True

        # Reset at the very first denoising step of each generation.
        # Only the positive (or sole) branch triggers the reset so that:
        # - single-branch models (FLUX, Hunyuan embedded guidance) reset once.
        # - dual-branch models (Wan with true CFG) reset both counters from the
        #   positive-branch call and leave the negative-branch call unaffected,
        #   keeping both branches synchronised (both start at cnt=0 → cnt=1).
        # Doing the reset here (not inside _get_spectrum_context) guarantees it
        # fires exactly once per step, preventing the double-reset that would
        # desync the two branches.
        if ctx.current_step == 0 and not ctx.is_cfg_negative:
            if ctx.debug:
                logger.info(
                    "[Spectrum] Debug mode enables shadow-error validation; runtime perf is not representative of non-debug runs."
                )
            self.reset_spectrum_state(ctx.spectrum_params)

        self.spectrum_is_cfg_negative = ctx.is_cfg_negative
        params = ctx.spectrum_params
        cnt, consecutive, curr_ws = self._get_spectrum_branch_state()

        # Warmup: first ``warmup_steps`` calls on this branch always run the DiT.
        actual_forward = True
        if cnt >= params.warmup_steps:
            # After warmup, skip most steps and only run a real forward every
            # ``window`` cached steps. ``consecutive`` counts skips since last real.
            window = max(1, math.floor(curr_ws))
            actual_forward = (consecutive + 1) % window == 0
            if actual_forward:
                # Widen the gap for the next stretch (paper alpha / flex_window).
                curr_ws += params.flex_window
                curr_ws = round(curr_ws, 3)

        # One denoising forward completed on this branch.
        cnt += 1
        if actual_forward:
            consecutive = 0
        else:
            consecutive += 1
        self._record_spectrum_step_stat(actual_forward)

        # End-of-run wrap: after ``total_steps`` forwards on this branch, reset
        # counters so state does not leak if the same module is reused. (A fresh
        # run also resets via ``reset_spectrum_state`` at denoising timestep 0.)
        total_steps = ctx.total_forward_steps
        if cnt >= total_steps:
            self._emit_spectrum_summary(ctx)
            cnt = 0
            consecutive = 0
            curr_ws = params.window_size

        self._set_spectrum_branch_state(cnt, consecutive, curr_ws)
        return actual_forward

    def spectrum_record_features(self, features: torch.Tensor) -> None:
        """Append block outputs from a real forward to the branch forecaster."""
        ctx = self._spectrum_ctx or self._get_spectrum_context()
        if ctx is None:
            return

        params = ctx.spectrum_params
        forecaster = self._get_spectrum_forecaster()
        step_idx = float(ctx.current_step)

        # Initialize forecaster on first real forward
        if forecaster is None:
            cheb = ChebyshevForecaster(
                M=params.m,
                K=params.history_size,
                lam=params.lam,
                num_steps=params.tau_num_steps,
                device=features.device,
                feature_shape=features.shape,
            )
            forecaster = SpectrumForecaster(
                cheb, taylor_order=params.taylor_order, w=params.w
            )
            self._set_spectrum_forecaster(forecaster)

        # In debug mode, compute shadow prediction error for validation
        if ctx.debug and forecaster.ready() and step_idx >= float(params.warmup_steps):
            predicted = forecaster.predict(step_idx).to(
                dtype=features.dtype, device=features.device
            )
            pred_f = predicted.float()
            feat_f = features.detach().float()
            denom = torch.norm(feat_f).item()
            if denom > 0:
                rel_l2 = (torch.norm(pred_f - feat_f).item()) / denom
                self._record_shadow_error_stat(rel_l2)

        # Update forecaster with actual features from this real step
        forecaster.update(step_idx, features.detach())
        self._spectrum_ctx = None

    def spectrum_predict_features(self, template: torch.Tensor) -> torch.Tensor:
        """Return forecasted block outputs for a skipped step (same shape as template)."""
        forecaster = self._get_spectrum_forecaster()
        if forecaster is None or not forecaster.ready():
            return template
        ctx = self._spectrum_ctx or self._get_spectrum_context()
        if ctx is None:
            return template
        step_idx = float(ctx.current_step)
        predicted = forecaster.predict(step_idx)
        self._spectrum_ctx = None
        return predicted.to(dtype=template.dtype, device=template.device)
