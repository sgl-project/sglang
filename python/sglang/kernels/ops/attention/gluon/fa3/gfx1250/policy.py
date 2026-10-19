# SPDX-License-Identifier: MIT
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Host execution policy for gfx1250 FAv3; importing it does not load Gluon."""

from __future__ import annotations

import functools
import math
from enum import Enum, IntFlag
from typing import NamedTuple

import torch


class LaunchConfig(NamedTuple):
    schedule: str
    block_m: int
    block_n: int
    num_warps: int
    tdm_output: bool
    llvm_fn_attrs: str


class NumericalMode(str, Enum):
    ONLINE = "online"
    PERSISTENT = "persistent"
    GUARDED = "guarded"


class GuardFlag(IntFlag):
    NONE = 0
    NORM_UNSAFE = 1
    FUSED_EXP_UNSAFE = 2


class ExecutionPolicy(NamedTuple):
    launch_config: LaunchConfig
    numerical_mode: NumericalMode
    softmax_scale: float


@functools.lru_cache(maxsize=None)
def _get_num_cus(device_index: int) -> int:
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties["multiprocessor_count"])


def get_num_cus(device: torch.device | int | None = None) -> int:
    """Return the CU count for a specific device, cached per device index."""
    from triton.runtime import driver

    if device is None:
        device_index = driver.active.get_device_interface().current_device()
    elif isinstance(device, torch.device):
        device_index = device.index
        if device_index is None:
            device_index = driver.active.get_device_interface().current_device()
    else:
        device_index = int(device)
    return _get_num_cus(device_index)


def _get_persistent_query_groups(
    batch: int, num_heads: int, seqlen_q: int, num_cus: int
) -> int:
    """Balance persistent query work without underfilling the device."""
    if num_cus <= 0:
        raise ValueError("num_cus must be positive")
    query_blocks = math.ceil(seqlen_q / 128)
    query_groups = math.ceil(query_blocks / 29)
    batch_heads = batch * num_heads

    # Align the grid to a whole multiple of the CU count, while retaining
    # at least two workgroups per CU. Small grids keep the existing grouping.
    group_multiple = num_cus // math.gcd(batch_heads, num_cus)
    aligned_groups = query_groups // group_multiple * group_multiple
    if aligned_groups * batch_heads < 2 * num_cus:
        return query_groups

    # Compare the estimated query tiles on the most heavily loaded CU.
    # Fewer groups reuse K/V longer, but must not increase the longest workload.
    original_tiles = math.ceil(query_groups * batch_heads / num_cus) * math.ceil(
        query_blocks / query_groups
    )
    aligned_tiles = (aligned_groups * batch_heads // num_cus) * math.ceil(
        query_blocks / aligned_groups
    )
    return aligned_groups if aligned_tiles < original_tiles else query_groups


def select_launch_config(
    batch: int,
    num_heads: int,
    seqlen_q: int,
    *,
    seqlen_k: int | None = None,
    num_cus: int | None = None,
    head_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> LaunchConfig:
    """Select a gfx1250 launch while retaining the BF16/D128 specializations."""
    if head_size not in (64, 128, 256):
        raise ValueError(
            f"Gluon FAv3 requires head_size in (64, 128, 256), got {head_size}"
        )
    if dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("Gluon FAv3 requires BF16 or FP16")
    if min(batch, num_heads, seqlen_q) <= 0:
        raise ValueError("batch, num_heads, and seqlen_q must be positive")
    if num_cus is None:
        num_cus = get_num_cus()

    persistent_groups = _get_persistent_query_groups(
        batch, num_heads, seqlen_q, num_cus
    )
    persistent_workgroups = batch * num_heads * persistent_groups
    if (
        head_size == 128
        and dtype == torch.bfloat16
        and seqlen_k is not None
        and 257 <= seqlen_k <= 512
        and persistent_workgroups >= num_cus
    ):
        return LaunchConfig(
            "persistent",
            128,
            256,
            4,
            True,
            "amdgpu-sched-strategy=iterative-ilp",
        )

    wide_block_m = 256
    workgroups = batch * num_heads * math.ceil(seqlen_q / wide_block_m)
    # D256 spills heavily with the wide eight-wave tile. The four-wave
    # pipeline keeps its larger output accumulator in registers.
    if head_size != 256 and workgroups >= num_cus:
        # TDM output is profitable at and above the same full-device boundary
        # that admits the wide ping-pong tile.
        scheduler = (
            "amdgpu-sched-strategy=max-ilp"
            if seqlen_k is not None and (seqlen_k <= 512 or seqlen_k >= 16_384)
            else ""
        )
        return LaunchConfig("pingpong", 256, 64, 8, head_size == 128, scheduler)
    # max-ilp intermittently corrupts FP16 inputs with nonzero storage offsets.
    # Use default pipeline scheduling until the affected cases are understood.
    return LaunchConfig(
        "pipeline",
        128,
        64,
        4,
        False,
        "" if dtype == torch.float16 else "amdgpu-sched-strategy=max-ilp",
    )


def resolve_execution_policy(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    launch_config: LaunchConfig | None = None,
    num_cus: int | None = None,
) -> ExecutionPolicy:
    """Resolve the effective launch and numerical policy from tensor metadata.

    The caller validates the Q/K/V tensor contract. Supplying num_cus allows
    inspection with CPU or meta tensors without importing Gluon or querying a
    device. GUARDED describes a device-side choice among fixed-shift, bounded
    fused-exponent, and online softmax; it never caches an input-data decision.
    """
    batch, seqlen_q, num_heads, head_size = query.shape
    seqlen_k = key.shape[1]
    scale = head_size**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("softmax_scale must be finite and positive")

    config = launch_config
    if config is None:
        config = select_launch_config(
            batch,
            num_heads,
            seqlen_q,
            seqlen_k=seqlen_k,
            num_cus=get_num_cus(query.device) if num_cus is None else num_cus,
            head_size=head_size,
            dtype=query.dtype,
        )
    if config.schedule == "persistent":
        if query.dtype != torch.bfloat16 or head_size != 128:
            raise ValueError("persistent requires BF16 with head_size=128")
        expected_geometry = (128, 256, 4)
        actual_geometry = (config.block_m, config.block_n, config.num_warps)
        if actual_geometry != expected_geometry:
            raise ValueError(
                "persistent requires (block_m, block_n, num_warps)="
                f"{expected_geometry}, got {actual_geometry}"
            )
        if not config.tdm_output:
            raise ValueError("persistent requires TDM output and stable softmax")
        if query.is_contiguous() and key.is_contiguous() and value.is_contiguous():
            if not 257 <= seqlen_k <= 512:
                raise ValueError("persistent FAv3 requires 257 <= seqlen_k <= 512")
            if 128 * num_heads * head_size * query.element_size() >= 2**31:
                raise ValueError(
                    "persistent FAv3 local query-tile span exceeds the buffer range"
                )
            # The persistent entry point always uses this scheduler, including
            # when a caller supplied an explicit configuration.
            config = config._replace(
                llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp"
            )
            return ExecutionPolicy(config, NumericalMode.PERSISTENT, scale)
        config = LaunchConfig(
            "pingpong", 256, 64, 8, True, "amdgpu-sched-strategy=max-ilp"
        )
    if config.schedule not in ("pipeline", "pingpong"):
        raise ValueError(f"unsupported FAv3 schedule: {config.schedule}")
    expected_geometry = (128, 64, 4) if config.schedule == "pipeline" else (256, 64, 8)
    actual_geometry = (config.block_m, config.block_n, config.num_warps)
    if actual_geometry != expected_geometry:
        raise ValueError(
            f"{config.schedule} requires (block_m, block_n, num_warps)="
            f"{expected_geometry}, got {actual_geometry}"
        )

    mode = NumericalMode.ONLINE
    if (
        launch_config is None
        and config.schedule == "pingpong"
        and query.dtype == torch.bfloat16
        and head_size == 128
        and min(seqlen_q, seqlen_k) >= 65_536
        and seqlen_k < 2**31
        and scale == head_size**-0.5
    ):
        mode = NumericalMode.GUARDED
    return ExecutionPolicy(config, mode, scale)
