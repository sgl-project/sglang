# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Host-side helpers for the prepared CuTe providers."""

import torch


def validate_init_barrier_group(group, init_barrier_group=None):
    """Select an initialization-only barrier without starting a CUDA collective.

    The caller must finish its local GPU initialization before this barrier.
    An explicit CPU group must preserve the device group's ordered membership;
    a different group could let peers read uninitialized mailboxes or deadlock.
    Omitting it preserves the original device-group barrier API.
    """
    if init_barrier_group is None:
        return group
    dist = torch.distributed
    if not dist.is_initialized():
        raise RuntimeError("Kimi initialization barrier needs initialized groups")
    if dist.get_backend(init_barrier_group) != "gloo":
        raise ValueError("Kimi initialization barrier group must use Gloo")
    members = tuple(dist.get_process_group_ranks(group))
    if (
        not members
        or tuple(dist.get_process_group_ranks(init_barrier_group)) != members
        or dist.get_rank(init_barrier_group) < 0
    ):
        raise ValueError(
            "Kimi initialization barrier group must match ordered device-group ranks"
        )
    return init_barrier_group
