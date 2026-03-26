# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class HybridAttentionSchedule:
    """Configuration for hybrid attention scheduling.

    During denoising, the first ``high_precision_first_steps`` and last
    ``high_precision_last_steps`` steps use the high-precision backend,
    while the middle steps use the low-precision backend.
    """

    high_precision_backend: AttentionBackendEnum
    low_precision_backend: AttentionBackendEnum
    high_precision_first_steps: int
    high_precision_last_steps: int

    def get_backend_for_step(
        self, step_index: int, total_steps: int
    ) -> AttentionBackendEnum:
        """Return which backend should be active for a given denoising step."""
        if step_index < self.high_precision_first_steps:
            return self.high_precision_backend
        if step_index >= total_steps - self.high_precision_last_steps:
            return self.high_precision_backend
        return self.low_precision_backend

    @classmethod
    def from_string(cls, schedule_str: str) -> "HybridAttentionSchedule":
        """Parse from CLI string ``'high_backend:low_backend:first:last'``.

        Example: ``'fa:sdpa:3:3'``
        """
        parts = schedule_str.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"hybrid-attention-schedule must have 4 colon-separated parts "
                f"'high_backend:low_backend:first_steps:last_steps', got: {schedule_str!r}"
            )

        high_name, low_name, first_str, last_str = parts

        try:
            high_backend = AttentionBackendEnum[high_name.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown high-precision attention backend '{high_name}'. "
                f"Available: {[e.name.lower() for e in AttentionBackendEnum]}"
            )

        try:
            low_backend = AttentionBackendEnum[low_name.upper()]
        except KeyError:
            raise ValueError(
                f"Unknown low-precision attention backend '{low_name}'. "
                f"Available: {[e.name.lower() for e in AttentionBackendEnum]}"
            )

        try:
            first_steps = int(first_str)
            last_steps = int(last_str)
        except ValueError:
            raise ValueError(
                f"first_steps and last_steps must be integers, got: "
                f"{first_str!r} and {last_str!r}"
            )

        if first_steps < 0 or last_steps < 0:
            raise ValueError(
                f"first_steps and last_steps must be non-negative, got: "
                f"{first_steps} and {last_steps}"
            )

        return cls(
            high_precision_backend=high_backend,
            low_precision_backend=low_backend,
            high_precision_first_steps=first_steps,
            high_precision_last_steps=last_steps,
        )
