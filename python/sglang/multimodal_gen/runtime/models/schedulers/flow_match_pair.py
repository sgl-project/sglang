# Copied and adapted from: https://github.com/OpenMOSS/MOVA/tree/main/mova/diffusion/schedulers/flow_match.py and flow_match_pair.py
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch

from sglang.multimodal_gen.runtime.models.schedulers.base import BaseScheduler


class FlowMatchScheduler(BaseScheduler):
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self.order = 1
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.train_timesteps = None
        self.train_sigmas = None
        self.set_timesteps(num_train_timesteps)
        self.set_timesteps(num_inference_steps)
        BaseScheduler.__init__(self)

    def set_shift(self, shift: float) -> None:
        self.shift = shift

    def set_timesteps(
        self,
        num_inference_steps=100,
        denoising_strength=1.0,
        training=False,
        shift=None,
        dynamic_shift_len=None,
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = (
            self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        )
        if self.extra_one_step:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps + 1
            )[:-1]
        else:
            self.sigmas = torch.linspace(
                sigma_start, self.sigma_min, num_inference_steps
            )
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = (
                self.calculate_shift(dynamic_shift_len)
                if dynamic_shift_len is not None
                else self.exponential_shift_mu
            )
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = (
                self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
            )
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        # Initialize train_timesteps on first set.
        if self.train_timesteps is None:
            self.train_timesteps = self.timesteps
            self.train_sigmas = self.sigmas
        if training:
            x = self.timesteps
            y = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2
            )
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None):
        return sample

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        timestep_id = torch.argmin(
            (self.timesteps - timestep.to(self.timesteps.device)).abs()
        )
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu


class FlowMatchPairScheduler(FlowMatchScheduler):
    """Pairing scheduler built on FlowMatchScheduler.

    Provides a convenient pairing interface for timesteps or sigmas.

    Attributes:
        pair_timesteps: Cached timestep pairs of shape [num_timesteps, 2].
        pair_sigmas: Cached sigma pairs of shape [num_timesteps, 2].
    """

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        self._pair_postprocess_fn = None
        self._pair_postprocess_requires_source = False
        self.pair_timesteps: torch.Tensor | None = None
        self.pair_sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        super().__init__(
            num_inference_steps=num_inference_steps,
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            inverse_timesteps=inverse_timesteps,
            extra_one_step=extra_one_step,
            reverse_sigmas=reverse_sigmas,
            exponential_shift=exponential_shift,
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
        )

    def set_pair_postprocess(self, fn):
        """Set a postprocess function to customize pairs after construction.

        Args:
            fn: Callable with signature fn(pairs: torch.Tensor) -> torch.Tensor.
                The returned tensor must have the same shape as input pairs.

        Raises:
            TypeError: If fn is not callable or None.
            RuntimeError: If scheduler is not initialized.
        """
        if fn is not None and not callable(fn):
            raise TypeError("pair_postprocess must be callable or None")
        self._pair_postprocess_fn = fn
        self._pair_postprocess_requires_source = (
            False if fn is None else bool(getattr(fn, "_requires_source", False))
        )
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Scheduler not initialized; call set_timesteps() first")
        self._refresh_pair_cache()

    def set_pair_postprocess_by_name(self, name: str | None, **kwargs):
        """Configure a postprocess function by name.

        Supported names:
            - None/"none"/"off"/"false"/"no": disable
            - "quadratic_perp_bulge_swap": x2=x+d, y2=x-d, where d=4*amp*s*(1-s), s=t/T
            - "v2a_sequential": assume pairs are (t,t); sample half sequence from column 0
              with stride 2, then let column 0 follow this sequence first, followed by column 1
            - "a2v_sequential": same as above, but column 1 first then column 0
            - "dual_sigma_shift": use only timestep count; rebuild two columns independently using
              FlowMatchScheduler sigma transform logic; configurable visual_shift/audio_shift

        Args:
            name: Postprocess name or None to disable.
            **kwargs: Extra parameters for the named postprocess. For example:
                - amp: Float amplitude, default 150.0.

        Raises:
            ValueError: If name is unknown.
        """

        if name is None or str(name).lower() in ("none", "off", "false", "no"):
            self.set_pair_postprocess(None)
            return
        if name == "quadratic_perp_bulge_swap":
            amp = float(kwargs.get("amp", 150.0))

            def _quadratic_perp_bulge_swap(pairs: torch.Tensor):
                if (
                    not isinstance(pairs, torch.Tensor)
                    or pairs.ndim != 2
                    or pairs.shape[1] != 2
                ):
                    raise ValueError("pairs must be a torch.Tensor of shape [N, 2]")
                x = pairs[:, 0]
                T = float(self.num_train_timesteps)
                s = x / T
                d = 4.0 * amp * s * (1.0 - s)
                x2 = x + d
                y2 = x - d
                return torch.stack([x2, y2], dim=1)

            self.set_pair_postprocess(_quadratic_perp_bulge_swap)
            return
        if name == "v2a_sequential":

            def _v2a(pairs: torch.Tensor):
                if (
                    not isinstance(pairs, torch.Tensor)
                    or pairs.ndim != 2
                    or pairs.shape[1] != 2
                ):
                    raise ValueError("pairs must be a torch.Tensor of shape [N, 2]")
                N = pairs.shape[0]
                base = pairs[:, 0]
                seq_half = base[::2]
                m = int(seq_half.shape[0])
                col0 = torch.cat([seq_half, seq_half[-1:].repeat(m)], dim=0)[:N]
                col1 = torch.cat([seq_half[0:1].repeat(m), seq_half], dim=0)[:N]
                return torch.stack(
                    [
                        col0.to(dtype=pairs.dtype, device=pairs.device),
                        col1.to(dtype=pairs.dtype, device=pairs.device),
                    ],
                    dim=1,
                )

            self.set_pair_postprocess(_v2a)
            return
        if name == "a2v_sequential":

            def _a2v(pairs: torch.Tensor):
                if (
                    not isinstance(pairs, torch.Tensor)
                    or pairs.ndim != 2
                    or pairs.shape[1] != 2
                ):
                    raise ValueError("pairs must be a torch.Tensor of shape [N, 2]")
                N = pairs.shape[0]
                base = pairs[:, 0]
                seq_half = base[::2]
                m = int(seq_half.shape[0])
                col0 = torch.cat([seq_half[0:1].repeat(m), seq_half], dim=0)[:N]
                col1 = torch.cat([seq_half, seq_half[-1:].repeat(m)], dim=0)[:N]
                return torch.stack(
                    [
                        col0.to(dtype=pairs.dtype, device=pairs.device),
                        col1.to(dtype=pairs.dtype, device=pairs.device),
                    ],
                    dim=1,
                )

            self.set_pair_postprocess(_a2v)
            return
        if name == "v2a":

            def _v2a_classic(pairs: torch.Tensor):
                if (
                    not isinstance(pairs, torch.Tensor)
                    or pairs.ndim != 2
                    or pairs.shape[1] != 2
                ):
                    raise ValueError("pairs must be a torch.Tensor of shape [N, 2]")
                zeros = torch.zeros_like(pairs[:, 0])
                return torch.stack([zeros, pairs[:, 1]], dim=1)

            self.set_pair_postprocess(_v2a_classic)
            return
        if name == "a2v":

            def _a2v_classic(pairs: torch.Tensor):
                if (
                    not isinstance(pairs, torch.Tensor)
                    or pairs.ndim != 2
                    or pairs.shape[1] != 2
                ):
                    raise ValueError("pairs must be a torch.Tensor of shape [N, 2]")
                zeros = torch.zeros_like(pairs[:, 1])
                return torch.stack([pairs[:, 0], zeros], dim=1)

            self.set_pair_postprocess(_a2v_classic)
            return
        if name == "dual_sigma_shift":
            visual_shift = float(kwargs.get("visual_shift", self.shift))
            audio_shift = float(kwargs.get("audio_shift", self.shift))
            visual_denoising_strength = float(
                kwargs.get("visual_denoising_strength", 1.0)
            )
            audio_denoising_strength = float(
                kwargs.get("audio_denoising_strength", 1.0)
            )
            visual_mu = kwargs.get(
                "visual_exponential_shift_mu", self.exponential_shift_mu
            )
            audio_mu = kwargs.get(
                "audio_exponential_shift_mu", self.exponential_shift_mu
            )

            def _dual_sigma_shift(pairs: torch.Tensor, *, source: str):
                if not isinstance(pairs, torch.Tensor):
                    raise TypeError("pairs must be a torch.Tensor")
                if pairs.ndim != 2 or pairs.shape[1] != 2:
                    raise ValueError("pairs must be a torch.Tensor of shape [N, 2]")
                if pairs.shape[0] == 0:
                    raise ValueError("pairs length must be greater than 0")
                if source not in ("timesteps", "sigmas"):
                    raise ValueError("source must be 'timesteps' or 'sigmas'")

                num_steps = pairs.shape[0]
                device = pairs.device
                dtype = pairs.dtype

                def _build_column(
                    shift_value: float, denoising_strength: float, mu_override
                ):
                    if shift_value <= 0:
                        raise ValueError("shift must be positive")
                    if denoising_strength <= 0:
                        raise ValueError("denoising_strength must be positive")

                    sigma_start = (
                        self.sigma_min
                        + (self.sigma_max - self.sigma_min) * denoising_strength
                    )
                    if self.extra_one_step:
                        base = torch.linspace(
                            sigma_start,
                            self.sigma_min,
                            num_steps + 1,
                            device=device,
                            dtype=dtype,
                        )[:-1]
                    else:
                        base = torch.linspace(
                            sigma_start,
                            self.sigma_min,
                            num_steps,
                            device=device,
                            dtype=dtype,
                        )

                    if self.inverse_timesteps:
                        base = torch.flip(base, dims=[0])

                    if self.exponential_shift:
                        mu_value = mu_override
                        if mu_value is None:
                            raise RuntimeError(
                                "exponential_shift enabled but exponential_shift_mu is missing"
                            )
                        exp_mu = math.exp(float(mu_value))
                        base = exp_mu / (exp_mu + (1 / base - 1))
                    else:
                        base = shift_value * base / (1 + (shift_value - 1) * base)

                    if self.shift_terminal is not None:
                        one_minus_z = 1 - base
                        scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
                        base = 1 - (one_minus_z / scale_factor)

                    if self.reverse_sigmas:
                        base = 1 - base

                    if source == "timesteps":
                        return base * self.num_train_timesteps
                    return base

                col0 = _build_column(visual_shift, visual_denoising_strength, visual_mu)
                col1 = _build_column(audio_shift, audio_denoising_strength, audio_mu)
                return torch.stack([col0, col1], dim=1)

            _dual_sigma_shift._requires_source = True
            self.set_pair_postprocess(_dual_sigma_shift)
            return
        raise ValueError(f"Unknown pair_postprocess name: {name}")

    def _make_pairs_from_vector(self, vec: torch.Tensor) -> torch.Tensor:
        if vec.ndim != 1:
            raise ValueError("vec must be 1D")
        return torch.stack([vec, vec], dim=1)

    def get_pairs(self, source: str = "timesteps") -> torch.Tensor:
        if source == "timesteps":
            if self.pair_timesteps is None:
                self._refresh_pair_cache()
            return self.pair_timesteps
        if source == "sigmas":
            if self.pair_sigmas is None:
                self._refresh_pair_cache()
            return self.pair_sigmas
        raise ValueError("source must be 'timesteps' or 'sigmas'")

    def timestep_to_sigma(self, timestep: torch.Tensor | float) -> torch.Tensor:
        """Return sigma for a scalar timestep via nearest neighbor lookup.

        Args:
            timestep: Scalar timestep value.

        Returns:
            Sigma corresponding to the nearest timestep.
        """
        t_value = float(timestep)
        t_cpu = torch.tensor(t_value)
        idx = torch.argmin((self.train_timesteps - t_cpu).abs())
        return self.train_sigmas[idx]

    def step_from_to(
        self,
        model_output: torch.Tensor,
        timestep_from: torch.Tensor,
        timestep_to: torch.Tensor | None,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one step using an explicit (from, to) timestep pair.

        The update rule is:
            x_{to} = x_{from} + model_output * (sigma(to) - sigma(from))

        Args:
            model_output: Predicted model output.
            timestep_from: Source timestep.
            timestep_to: Target timestep or None for terminal.
            sample: Current sample at timestep_from.

        Returns:
            Updated sample at timestep_to.
        """
        sigma_from = self.timestep_to_sigma(timestep_from)
        if timestep_to is None:
            sigma_to = torch.tensor(
                1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0,
                device=sigma_from.device,
                dtype=sigma_from.dtype,
            )
        else:
            sigma_to = self.timestep_to_sigma(timestep_to)
        prev_sample = sample + model_output * (sigma_to - sigma_from)
        return prev_sample

    def _refresh_pair_cache(self) -> None:
        if self.timesteps is None or self.sigmas is None:
            raise RuntimeError("Scheduler not initialized; call set_timesteps() first")

        def _apply_postprocess(pairs: torch.Tensor, source: str) -> torch.Tensor:
            if self._pair_postprocess_fn is None:
                return pairs
            if self._pair_postprocess_requires_source:
                modified = self._pair_postprocess_fn(pairs, source=source)
            else:
                modified = self._pair_postprocess_fn(pairs)
            if not isinstance(modified, torch.Tensor):
                raise TypeError("pair_postprocess must return a torch.Tensor")
            if modified.shape != pairs.shape:
                raise ValueError("pair_postprocess must return the same shape as input")
            return modified

        base_pairs_timesteps = self._make_pairs_from_vector(self.timesteps)
        base_pairs_sigmas = self._make_pairs_from_vector(self.sigmas)

        self.pair_timesteps = _apply_postprocess(base_pairs_timesteps, "timesteps")
        self.pair_sigmas = _apply_postprocess(base_pairs_sigmas, "sigmas")


EntryClass = FlowMatchPairScheduler
