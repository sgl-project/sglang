# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn.functional as F


class GPTQCPUGroupPadding:
    """Map a logical TP K shard into complete checkpoint quantization groups."""

    def __init__(
        self, *, input_size: int, shard_size: int, group_size: int, tp_rank: int
    ):
        if shard_size % 8:
            raise ValueError("CPU GPTQ requires logical K shards divisible by 8.")
        self.shard_size = shard_size
        self.shard_start = tp_rank * shard_size
        self.channelwise = group_size == -1
        if self.channelwise:
            self.group_start = 0
            self.left = 0
            self.physical_size = (shard_size + 127) // 128 * 128
            self.num_groups = 1
            self.group_size = self.physical_size
        else:
            if group_size < 32 or group_size % 32:
                raise ValueError("CPU GPTQ group_size must be a multiple of 32.")
            self.group_start = self.shard_start // group_size
            self.left = self.shard_start % group_size
            self.num_groups = (self.left + shard_size + group_size - 1) // group_size
            self.physical_size = self.num_groups * group_size
            self.group_size = group_size
        self.right = self.physical_size - self.left - shard_size
        self.global_groups = (
            1 if self.channelwise else (input_size + group_size - 1) // group_size
        )

    def group_indices(self, *, device=None):
        return (
            torch.arange(self.physical_size, dtype=torch.int32, device=device)
            // self.group_size
        )

    def load_weight(
        self,
        param,
        loaded_weight: torch.Tensor,
        *args,
        name: str,
        weight_loader,
        row_parallel: bool,
        presharded: bool,
        **kwargs,
    ):
        if name in ("scales", "qzeros"):
            if not presharded:
                if loaded_weight.size(0) != self.global_groups:
                    raise ValueError("GPTQ checkpoint has an unexpected group count.")
                loaded_weight = loaded_weight.narrow(
                    0, self.group_start, self.num_groups
                )
        elif name == "qweight":
            if presharded and loaded_weight.size(0) == self.physical_size // 8:
                pass
            else:
                if not presharded:
                    loaded_weight = loaded_weight.narrow(
                        0, self.shard_start // 8, self.shard_size // 8
                    )
                if loaded_weight.size(0) != self.shard_size // 8:
                    raise ValueError(
                        "Presharded GPTQ qweight has an unexpected logical K size."
                    )
                loaded_weight = F.pad(
                    loaded_weight, (0, 0, self.left // 8, self.right // 8)
                )
        else:
            expected = self.group_indices(device=loaded_weight.device)
            if presharded and loaded_weight.numel() == self.physical_size:
                logical = loaded_weight[self.left : self.left + self.shard_size]
            else:
                logical = (
                    loaded_weight
                    if presharded
                    else loaded_weight.narrow(0, self.shard_start, self.shard_size)
                )
            local = expected[self.left : self.left + self.shard_size]
            if not (
                torch.equal(logical, local)
                or torch.equal(logical, local + self.group_start)
            ):
                raise ValueError(
                    "CPU GPTQ padding requires static, contiguous "
                    "quantization groups in g_idx."
                )
            loaded_weight = expected

        if row_parallel:
            if param.shape != loaded_weight.shape:
                raise ValueError(
                    f"GPTQ shard shape mismatch: {param.shape} != {loaded_weight.shape}"
                )
            param.data.copy_(loaded_weight)
        else:
            weight_loader(param, loaded_weight, *args, **kwargs)

    def fill_weight_padding(self, *, qweight: torch.Tensor, qzeros: torch.Tensor):
        if not (self.left or self.right):
            return
        # AMX compensation includes padded lanes; their dequantized weights must be zero.
        shifts = torch.arange(8, device=qzeros.device) * 4
        zeros = ((qzeros.unsqueeze(-1).to(torch.int64) >> shifts) & 15).flatten(1) + 1
        for start, size, group in (
            (0, self.left, 0),
            (self.left + self.shard_size, self.right, self.num_groups - 1),
        ):
            if size:
                zero = zeros[group]
                if torch.any(zero == 16):
                    raise ValueError(
                        "CPU GPTQ padding requires zero points representable "
                        "in INT4 (1 through 15)."
                    )
                packed = (zero * 0x11111111).to(torch.int32)
                qweight[start // 8 : (start + size) // 8].copy_(packed)

    def pad_activation(self, x: torch.Tensor):
        if x.size(-1) != self.shard_size:
            raise ValueError(
                f"Expected logical GPTQ K={self.shard_size}, got {x.size(-1)}."
            )
        return F.pad(x, (self.left, self.right)) if self.left or self.right else x
