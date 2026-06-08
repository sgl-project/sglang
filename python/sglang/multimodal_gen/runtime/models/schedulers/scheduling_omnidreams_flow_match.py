# SPDX-License-Identifier: Apache-2.0
"""OmniDreams self-forcing flow-match scheduler (port of FlashDreams ``fm.py``).

Why a dedicated scheduler instead of reusing
``scheduling_self_forcing_flow_match.SelfForcingFlowMatchScheduler``:

1. Schedule construction differs. OmniDreams warps the FULL ``num_train_timesteps``
   table and selects entries by ``denoising_timesteps`` (e.g. [1000, 450] ->
   indices [0, 550] -> sigmas [1.0, 0.8036]). Building a 2-point linspace
   directly instead yields [1.0, 0.8333] -- a silent ~3.7% sigma error.
2. Step semantics differ. OmniDreams uses self-forcing renoise
   (``clean = noisy - sigma*flow``; ``noisy = (1-sigma)*clean + sigma*noise``
   with FRESH noise each step), not the deterministic Euler
   ``prev = sample + v*(sigma_next - sigma)``.

Distilled single-view defaults: num_inference_steps=2,
denoising_timesteps=[1000, 450], shift=5.0, sigma_min=0.0 -> sigmas
[1.0, 0.8036], warped network timesteps [1000.0, 803.57].
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


def warp_sigmas(sigmas: Tensor, shift: float) -> Tensor:
    """DiffSynth schedule warp: ``shift * s / (1 + (shift - 1) * s)``."""
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


class OmniDreamsFlowMatchScheduler:
    """Self-forcing flow-match scheduler for OmniDreams.

    Attributes:
        denoising_sigmas: per-step sigmas, shape ``[num_inference_steps]``.
        denoising_step_list: per-step (warped) timesteps fed to the network.
    """

    def __init__(
        self,
        num_inference_steps: int = 2,
        denoising_timesteps: tuple[int, ...] = (1000, 450),
        shift: float = 5.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.0,
        num_train_timesteps: int = 1000,
        extra_one_step: bool = True,
        warp_denoising_step: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        assert num_inference_steps == len(denoising_timesteps), (
            f"num_inference_steps ({num_inference_steps}) must equal "
            f"len(denoising_timesteps) ({len(denoising_timesteps)})"
        )
        N = num_train_timesteps
        self.num_train_timesteps = N
        self.num_inference_steps = num_inference_steps

        if extra_one_step:
            base = torch.linspace(sigma_max, sigma_min, N + 1, dtype=torch.float32)[:-1]
        else:
            base = torch.linspace(sigma_max, sigma_min, N, dtype=torch.float32)
        full_sigmas = warp_sigmas(base, shift)
        full_timesteps = full_sigmas * N

        idxs = [N - t for t in denoising_timesteps]
        if warp_denoising_step:
            step_list = [full_timesteps[i].item() if i < N else 0.0 for i in idxs]
            sigma_list = [full_sigmas[i if i < N else N - 1].item() for i in idxs]
        else:
            step_list = [float(t) for t in denoising_timesteps]
            snapped = [
                int(torch.argmin((full_timesteps - t).abs()).item()) for t in step_list
            ]
            sigma_list = [full_sigmas[i].item() for i in snapped]

        self.denoising_step_list = torch.tensor(
            step_list, dtype=torch.float32, device=device
        )
        self.denoising_sigmas = torch.tensor(
            sigma_list, dtype=torch.float32, device=device
        )
        self._full_sigmas = full_sigmas.to(device)
        self._full_timesteps = full_timesteps.to(device)

    def to(self, device: torch.device | str) -> "OmniDreamsFlowMatchScheduler":
        self.denoising_step_list = self.denoising_step_list.to(device)
        self.denoising_sigmas = self.denoising_sigmas.to(device)
        self._full_sigmas = self._full_sigmas.to(device)
        self._full_timesteps = self._full_timesteps.to(device)
        return self

    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: Callable[[Tensor, Tensor], Tensor],
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Self-forcing denoising loop; returns the clean ``x0`` estimate.

        ``predict_flow(noisy, timestep)`` returns the network's flow prediction.
        Iteration 0 trusts ``initial_noise`` as the sigma=1 sample; later
        iterations re-noise the previous clean estimate with FRESH noise.
        """
        input_dtype = initial_noise.dtype
        noisy = initial_noise
        clean: Tensor | None = None
        for i in range(self.denoising_step_list.shape[0]):
            sigma = self.denoising_sigmas[i]
            timestep = self.denoising_step_list[i].to(dtype=input_dtype)
            if i > 0:
                assert clean is not None
                noise = torch.empty_like(noisy).normal_(generator=rng)
                noisy = ((1.0 - sigma) * clean + sigma * noise).to(input_dtype)
            flow = predict_flow(noisy, timestep)
            clean = noisy - sigma * flow
        assert clean is not None, "denoising_step_list is empty"
        return clean.to(input_dtype)

    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Forward corruption at an arbitrary timestep (snapped to the table).

        Used for context-noise on cached/clean frames (raw timestep 128 ->
        sigma ~= 0.128).
        """
        assert timestep.shape == (), f"expected scalar timestep, got {timestep.shape}"
        idx = torch.argmin(
            (self._full_timesteps - timestep.to(self._full_timesteps.dtype)).abs()
        ).reshape(1)
        sigma = self._full_sigmas.index_select(0, idx).reshape(())
        noise = torch.empty_like(clean_input).normal_(generator=rng)
        return ((1.0 - sigma) * clean_input + sigma * noise).to(clean_input.dtype)

    def sigma_for_timestep(self, timestep: float) -> float:
        """Return the warped-table sigma nearest to a raw ``timestep`` (for tests/debug)."""
        idx = int(torch.argmin((self._full_timesteps - float(timestep)).abs()).item())
        return float(self._full_sigmas[idx].item())


EntryClass = OmniDreamsFlowMatchScheduler
