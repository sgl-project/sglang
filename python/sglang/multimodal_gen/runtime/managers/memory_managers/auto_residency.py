# SPDX-License-Identifier: Apache-2.0
"""Warmup memory estimates for the default workload.

Peak-memory extrapolation separates persistent weights from workload-scaled
activations.
"""

from __future__ import annotations

from typing import Iterable

import msgspec

GIB_BYTES = 1024**3

# Activation memory rarely scales perfectly linearly with workload units;
# pad the extrapolated activation part before checking the budget.
ACTIVATION_EXTRAPOLATION_MARGIN = 1.2


class WarmupMemoryRecord(msgspec.Struct, frozen=True):
    """Per-rank memory measurement of one server warmup forward."""

    width: int
    height: int
    num_frames: int
    baseline_allocated_bytes: int
    peak_allocated_bytes: int
    succeeded: bool

    def workload_units(self) -> int:
        return max(1, self.width) * max(1, self.height) * max(1, self.num_frames)


def estimate_default_workload_peak_bytes(
    *,
    records: Iterable[WarmupMemoryRecord],
    target_units: int | None,
    constant_weight_bytes: int = 0,
) -> int | None:
    """Extrapolate live warmup memory to the default workload.

    Real measurements use allocated bytes. Cached allocator blocks are
    reclaimable and are covered by the explicit VRAM reserve; treating them as
    live memory would charge the same storage again when adding resident
    weights.

    Preference order:
    1. A measurement at or above the target workload bounds the peak directly.
    2. Two distinct measured workload sizes fit ``peak = constant + slope *
       units``; only the fitted linear part is extrapolated (and padded).
       Under offload the pre-forward baseline is nearly empty, so a
       single-point split cannot separate constant costs (streamed weights,
       attention workspace, tiled VAE decode) from workload-linear
       activations -- measured on Wan2.1-14B, single-point scaling
       overestimated a ~30 GiB peak as ~183 GiB.
    3. One usable size: scale everything above the pre-forward allocated
       baseline (conservative; may block adjustment but never over-allocates).

    Returns None when the estimate cannot be trusted: no successful records,
    the target workload is unknown (an unknown target would silently equate the
    area/frame-capped warmup peak with the real serving peak), or a probe at or
    below the target ran out of memory. That last case is not missing data but
    a measurement: the card could not hold the target as it is already
    configured, so making more weights resident can only make it worse. A probe
    that failed strictly above the target says nothing about the target and is
    dropped instead.
    """
    records = list(records)
    if target_units is None:
        return None
    failed_units = [
        record.workload_units() for record in records if not record.succeeded
    ]
    if any(units <= target_units for units in failed_units):
        return None
    records = [record for record in records if record.succeeded]
    if not records:
        return None

    peak_by_units: dict[int, int] = {}
    for record in records:
        units = record.workload_units()
        peak = record.peak_allocated_bytes
        peak_by_units[units] = max(peak_by_units.get(units, 0), peak)

    covering_peaks = [
        peak for units, peak in peak_by_units.items() if units >= target_units
    ]
    if covering_peaks:
        return max(covering_peaks)

    if len(peak_by_units) >= 2:
        # fit on the two largest sizes: closest to the target, best local slope
        (large_units, large_peak), (small_units, small_peak) = sorted(
            peak_by_units.items(), reverse=True
        )[:2]
        slope = (large_peak - small_peak) / (large_units - small_units)
        if slope >= 0:
            constant = max(
                large_peak - slope * large_units,
                min(constant_weight_bytes, large_peak),
            )
            return int(
                constant + slope * target_units * ACTIVATION_EXTRAPOLATION_MARGIN
            )
        # negative slope is measurement noise; fall through to the
        # conservative single-point estimate

    estimates = []
    for record in records:
        peak = record.peak_allocated_bytes
        baseline = min(
            max(record.baseline_allocated_bytes, constant_weight_bytes), peak
        )
        activation = peak - baseline
        ratio = target_units / record.workload_units()
        estimates.append(
            baseline + int(activation * ratio * ACTIVATION_EXTRAPOLATION_MARGIN)
        )
    return max(estimates)
