"""
    utils for analyzing the perf log dumped by perf_logger.py, for testcase usage, benchmark reporting
"""

import re
from typing import Any, Dict, List, Tuple


def calculate_diff(base: float, new: float) -> Tuple[float, float]:
    """Returns (diff, diff_percent)."""
    diff = new - base
    if base == 0:
        percent = 0.0
    else:
        percent = (diff / base) * 100
    return diff, percent


def calculate_upper_bound(
    baseline: float,
    rel_tol: float,
    min_abs_tol: float
) -> float:
    """Calculates the upper bound for performance regression check."""
    rel_limit = baseline * (1 + rel_tol)
    abs_limit = baseline + min_abs_tol
    return max(rel_limit, abs_limit)


def calculate_lower_bound(
    baseline: float,
    rel_tol: float,
    min_abs_tol: float
) -> float:
    """Calculates the lower bound for performance improvement check."""
    rel_lower = baseline * (1 - rel_tol)
    abs_lower = baseline - min_abs_tol
    return min(rel_lower, abs_lower)


def get_perf_status_emoji(
    baseline: float,
    new: float,
    rel_tol: float = 0.05,
    min_abs_tol: float = 10.0,
) -> str:
    """
    Determines the status emoji based on performance difference.

    Logic:
      Upper bound (Slower): max(baseline * (1 + rel_tol), baseline + min_abs_tol)
      Lower bound (Faster): min(baseline * (1 - rel_tol), baseline - min_abs_tol)
    """
    upper_bound = calculate_upper_bound(baseline, rel_tol, min_abs_tol)
    lower_bound = calculate_lower_bound(baseline, rel_tol, min_abs_tol)

    if new > upper_bound:
        return "🔴"  # Significant Deterioration
    elif new < lower_bound:
        return "🟢"  # Significant Optimization
    else:
        return "⚪️"  # Normal Fluctuation


def consolidate_steps(
    steps_list: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], List[str], Dict[str, int]]:
    """
    Aggregates specific repeating steps (like denoising_step_*) into groups.
    Returns:
        - aggregated_durations: {name: duration_ms}
        - ordered_names: list of names in execution order
        - counts: {name: count_of_steps_aggregated}
    """
    durations = {}
    counts = {}
    ordered_names = []
    seen_names = set()

    # Regex for steps to group
    # Group "denoising_step_0", "denoising_step_1" -> "Denoising Loop"
    denoise_pattern = re.compile(r"^denoising_step_(\d+)$")
    denoising_group_name = "Denoising Loop"

    for step in steps_list:
        name = step.get("name", "unknown")
        dur = step.get("duration_ms", 0.0)

        match = denoise_pattern.match(name)
        if match:
            key = denoising_group_name
            if key not in durations:
                durations[key] = 0.0
                counts[key] = 0
                if key not in seen_names:
                    ordered_names.append(key)
                    seen_names.add(key)
            durations[key] += dur
            counts[key] += 1
        else:
            # Standard stage (preserve order)
            if name not in durations:
                durations[name] = 0.0
                counts[name] = 0
                if name not in seen_names:
                    ordered_names.append(name)
                    seen_names.add(name)
            durations[name] += dur
            counts[name] += 1

    return durations, ordered_names, counts
