# SPDX-License-Identifier: Apache-2.0
import math
from typing import Protocol


class MiniMaxH3LatentStatsConfig(Protocol):
    latent_channels: int
    latents_mean: list[float] | None
    latents_std: list[float] | None


class MiniMaxH3VAEContractError(ValueError):
    def __init__(self, component_name: str, detail: str) -> None:
        super().__init__(f"MiniMax H3 {component_name} {detail}")
        self.component_name = component_name
        self.detail = detail

    def __reduce__(self):
        # BaseException pickles via cls(*args); rebuild from the two ctor args
        # so the error propagates cleanly across process boundaries.
        return (type(self), (self.component_name, self.detail))


def validate_minimax_h3_vae_latent_stats(
    arch_config: MiniMaxH3LatentStatsConfig,
    component_name: str,
    expected_channels: int,
) -> None:
    if arch_config.latent_channels != expected_channels:
        raise MiniMaxH3VAEContractError(
            component_name,
            "latent_channels must be "
            f"{expected_channels}, got {arch_config.latent_channels!r}",
        )

    for field_name, values in (
        ("latents_mean", arch_config.latents_mean),
        ("latents_std", arch_config.latents_std),
    ):
        if values is None:
            raise MiniMaxH3VAEContractError(
                component_name,
                f"config.json missing {field_name}",
            )
        if not isinstance(values, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            raise MiniMaxH3VAEContractError(
                component_name,
                f"config.json {field_name} must be a list of numbers",
            )
        if len(values) != expected_channels:
            raise MiniMaxH3VAEContractError(
                component_name,
                f"config.json {field_name} must contain exactly "
                f"{expected_channels} values, got {len(values)}",
            )
        if field_name == "latents_mean" and not all(
            math.isfinite(value) for value in values
        ):
            raise MiniMaxH3VAEContractError(
                component_name,
                "config.json latents_mean values must be finite",
            )
        if field_name == "latents_std" and not all(
            math.isfinite(value) and value > 0 for value in values
        ):
            raise MiniMaxH3VAEContractError(
                component_name,
                "config.json latents_std values must be finite and greater than zero",
            )


__all__ = [
    "MiniMaxH3VAEContractError",
    "validate_minimax_h3_vae_latent_stats",
]
