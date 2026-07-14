# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import CacheParams


@dataclass
class SpectrumParams(CacheParams):
    """
    Parameters for [Spectrum](https://arxiv.org/abs/2603.01623) spectral feature
    forecasting.

    Attributes:
        cache_type: (`str`, defaults to `spectrum`):
            A string labeling these parameters as belonging to spectrum.
        window_size (`float`, defaults to `2.0`):
            Initial skip window size (paper: N). After warmup, a real forward is
            run every `floor(window_size)` consecutive cached steps; when a real
            forward runs, the window grows by `flex_window`. Recommended: `2.0`.
        flex_window (`float`, defaults to `0.75`):
            Adaptive window slope (paper: alpha). Controls how quickly skip intervals
            widen as sampling progresses. Recommended: `0.75` for ~3.5x speedup,
            `3.0` for ~5x speedup (fewer network evaluations, lower quality).
        warmup_steps (`int`, defaults to `5`):
            Number of initial forward passes that always run the full DiT before
            Spectrum caching begins. Early steps define global structure and are
            kept exact in the paper setup.
        w (`float`, defaults to `1.0`):
            Blend weight between a local Taylor predictor and the Chebyshev
            spectral predictor (``(1 - w) * h_taylor + w * h_cheb``). This
            mix is not in the paper; ``w=1.0`` is pure Chebyshev and matches
            the [reference repo](https://github.com/hanjq17/Spectrum#-text-to-image-t2i)
            default. They recommend ``0.5``–``1.0`` for extra robustness.
        lam (`float`, defaults to `0.1`):
            Ridge regularization strength (lambda) when fitting Chebyshev
            coefficients over recent cached features.
        m (`int`, defaults to `4`):
            Number of Chebyshev bases (polynomial degree M). The design matrix has
            `M + 1` columns.
        history_size (`int`, defaults to `100`):
            Maximum number of recent (step, feature) pairs kept for coefficient
            fitting (paper reference code uses a sliding window of up to 100).
        tau_num_steps (`int`, defaults to `50`):
            Denominator for mapping denoising step indices to Chebyshev time
            ``tau in [-1, 1]``. The reference repo hardcodes 50 regardless of
            ``num_inference_steps``; keep this at 50 unless you know you need
            another horizon.
        taylor_order (`int`, defaults to `1`):
            Order of the local discrete Taylor fallback used in the blend when
            `w < 1.0`. Supported values: `1`, `2`, or `3`.
        separate_cfg_branches (not a field — determined by the DiT model):
            Wan and Hunyuan maintain independent Spectrum state per CFG branch.
            Other ``CachableDiT`` models use one counter for all forwards; see
            ``get_total_forward_steps()``.
    """

    cache_type: str = "spectrum"
    window_size: float = 2.0
    flex_window: float = 0.75
    warmup_steps: int = 5
    w: float = 1.0
    lam: float = 0.1
    m: int = 4
    history_size: int = 100
    tau_num_steps: int = 50
    taylor_order: int = 1

    def get_total_forward_steps(
        self, num_inference_steps: int, do_cfg: bool, separate_cfg_branches: bool
    ) -> int:
        """How many DiT forward calls one Spectrum counter sees per generation.

        Used by (1) ``ChebyshevForecaster`` to map step indices onto [-1, 1] and
        by (2) ``begin_spectrum_step`` to wrap branch counters at end-of-run.
        Not used for logging.

        - **Separate counters** (Wan, Hunyuan): each CFG branch has its own
          ``spectrum_cnt`` / forecaster. Every denoising step triggers one cond
          forward and one uncond forward, but each counter only advances on its
          branch → ``num_inference_steps`` calls per counter.

        - **Single counter** (FLUX, SD3, …): one counter interleaves cond and
          uncond forwards when CFG is enabled → ``2 * num_inference_steps`` calls.

        When CFG is off, every model performs one forward per denoising step.
        """
        if do_cfg and not separate_cfg_branches:
            return num_inference_steps * 2
        return num_inference_steps
