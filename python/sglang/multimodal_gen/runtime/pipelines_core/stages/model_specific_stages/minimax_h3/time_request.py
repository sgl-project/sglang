# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def minimax_h3_align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    return current + (5 - current) % 17


def minimax_h3_video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def minimax_h3_audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
    dmd_steps: tuple[int, ...] | None = None,
) -> list[float]:
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")
    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    import torch

    if dmd_steps is not None:
        if int(num_steps) != len(dmd_steps):
            raise ValueError(
                f"this checkpoint is distilled for {len(dmd_steps)} DiT forwards "
                f"({len(dmd_steps) + 1} sigma grid points); got "
                f"num_inference_steps={num_steps}"
            )
        # trained rungs are unshifted timesteps over 1000; append the final step to sigma 0
        base = torch.tensor(
            [step / 1000.0 for step in dmd_steps] + [0.0], dtype=torch.float32
        )
    else:
        # The rectified-flow sigma range is fixed at [1.0, 0.0].
        base = torch.linspace(
            1.0,
            0.0,
            int(num_steps) + 1,
            device="cpu",
            dtype=torch.float32,
        )
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    if shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]


def minimax_h3_image_pdd_grid(config: dict) -> tuple[float, ...]:
    """The ``sigmas`` of an image release's pdd_config.json.

    The image PDD heads are distilled on an explicit grid rather than the
    shifted video grid, so the config lists it: ``num_inference_steps + 1``
    points decreasing strictly from 1.0 to 0.0, one interval per fused head.
    """
    sigmas = config.get("sigmas")
    steps = config.get("num_inference_steps")
    if not isinstance(sigmas, list) or any(
        isinstance(s, bool) or not isinstance(s, (int, float)) for s in sigmas
    ):
        raise ValueError(
            f"image pdd_config.json sigmas must be a list of numbers, got {sigmas!r}"
        )
    sigmas = tuple(float(s) for s in sigmas)
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or len(sigmas) != steps + 1
        or steps < 1
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or any(a <= b for a, b in zip(sigmas, sigmas[1:]))
    ):
        raise ValueError(
            "image pdd_config.json sigmas must decrease strictly from 1.0 to 0.0 "
            f"in num_inference_steps + 1 points; got num_inference_steps={steps!r}, "
            f"sigmas={list(sigmas)!r}"
        )
    return sigmas


def minimax_h3_image_pdd_sigmas(
    pdd_sigmas: tuple[float, ...],
    *,
    task: str,
    num_steps: int,
    is_warmup: bool,
) -> list[float]:
    """The PDD grid as the denoise loop runs it.

    One fused head serves each interval of ``pdd_sigmas``, so a serving
    request runs exactly ``len(pdd_sigmas) - 1`` steps; a warmup runs a
    prefix of the same grid. Each sigma is rounded through its float32
    timestep ``1 - sigma`` and back, and the grid ends at 0, the schedule
    the heads were distilled with.
    """
    import torch

    if task not in ("t2i", "edit"):
        raise ValueError(f"MiniMax H3 image PDD requires t2i or edit, got {task!r}")
    steps = len(pdd_sigmas) - 1
    if is_warmup:
        steps = min(max(1, num_steps), steps)
    elif num_steps != steps:
        raise ValueError(
            f"this MiniMax H3 PDD release has {steps} fused output heads, "
            f"so num_inference_steps must be {steps}, got {num_steps!r}"
        )
    timesteps = torch.tensor(
        [1.0 - sigma for sigma in pdd_sigmas[:-1]], dtype=torch.float32
    )
    sigmas = torch.cat((1.0 - timesteps, torch.zeros(1, dtype=torch.float32)))
    return sigmas.tolist()[: steps + 1]
