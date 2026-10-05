# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/utils.py

from typing import Any

import torch


def dict_to_3d_list(
    mask_strategy: dict[str, Any] | None = None,
    t_max: int | None = None,
    l_max: int | None = None,
    h_max: int | None = None,
) -> list[list[list[torch.Tensor | None]]]:
    """
    Convert a dictionary of mask indices to a 3D list of tensors.
    Args:
        mask_strategy: keys are "t_l_h", values are torch.Tensor masks.
        t_max, l_max, h_max: if provided (all three), force the output shape to (t_max, l_max, h_max).
                            If all three are None, infer shape from the data.
    """
    # Case 1: no data, but fixed shape requested
    if mask_strategy is None:
        assert t_max is not None and l_max is not None and h_max is not None, (
            "If mask_strategy is None, you must provide t_max, l_max, and h_max"
        )
        return [
            [[None for _ in range(h_max)] for _ in range(l_max)] for _ in range(t_max)
        ]

    # Parse all keys into integer tuples
    indices = [tuple(map(int, key.split("_"))) for key in mask_strategy]

    # Decide on dimensions
    if t_max is None and l_max is None and h_max is None:
        # fully dynamic: infer from data
        max_timesteps_idx = max(t for t, _, _ in indices) + 1
        max_layer_idx = max(l for _, l, _ in indices) + 1  # noqa: E741
        max_head_idx = max(h for _, _, h in indices) + 1
    else:
        # require all three to be provided
        assert t_max is not None and l_max is not None and h_max is not None, (
            "Either supply none of (t_max, l_max, h_max) to infer dimensions, "
            "or supply all three to fix the shape."
        )
        max_timesteps_idx = t_max
        max_layer_idx = l_max
        max_head_idx = h_max

    # Preallocate
    result = [
        [[None for _ in range(max_head_idx)] for _ in range(max_layer_idx)]
        for _ in range(max_timesteps_idx)
    ]

    # Fill in, skipping any out-of-bounds entries
    for key, value in mask_strategy.items():
        t, l, h = map(int, key.split("_"))  # noqa: E741
        if (
            0 <= t < max_timesteps_idx
            and 0 <= l < max_layer_idx
            and 0 <= h < max_head_idx
        ):
            result[t][l][h] = value
        # else: silently ignore any key that doesn't fit

    return result
