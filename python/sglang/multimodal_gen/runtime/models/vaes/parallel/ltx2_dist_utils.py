import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.vaes.parallel.halo_exchange_utils import (
    calc_bottom_halo_size,
    calc_patch_height_index,
    calc_top_halo_size,
    halo_exchange,
)
from sglang.multimodal_gen.runtime.models.vaes.parallel.parallel_utils import (
    get_vae_group,
    get_vae_parallel_rank,
    get_vae_parallel_world_size,
)


class DistLTX2VideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        )
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        self.groups = groups
        self.spatial_padding_mode = spatial_padding_mode

        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        self._spatial_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=groups,
            padding=0,
            padding_mode="zeros",
        )

        self.rank = get_vae_parallel_rank()
        self.world_size = get_vae_parallel_world_size()
        self.group = get_vae_group()

    def _pad_spatial(self, x: torch.Tensor, padding: list[int]) -> torch.Tensor:
        if self.spatial_padding_mode == "zeros":
            return F.pad(x, padding)
        return F.pad(x, padding, mode=self.spatial_padding_mode)

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]

        if causal:
            pad_left = hidden_states[:, :, :1, :, :].repeat(
                (1, 1, time_kernel_size - 1, 1, 1)
            )
            hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)
        else:
            half = (time_kernel_size - 1) // 2
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, half, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, half, 1, 1))
            hidden_states = torch.concatenate(
                [pad_left, hidden_states, pad_right], dim=2
            )

        padding = list(self._spatial_padding)
        if self.world_size <= 1 or self.group is None:
            hidden_states = self._pad_spatial(hidden_states, padding)
            return self.conv(hidden_states)

        height = hidden_states.shape[-2]
        device = hidden_states.device
        patch_height_list = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(self.world_size)
        ]
        dist.all_gather(
            patch_height_list,
            torch.tensor([height], dtype=torch.int64, device=device),
            group=self.group,
        )
        patch_height_index = calc_patch_height_index(patch_height_list)
        self.patch_height_index = patch_height_index.cpu().tolist()

        height_padding = self._spatial_padding[2]
        self.curr_top_halo_size = calc_top_halo_size(
            self.rank,
            self.world_size,
            self.patch_height_index,
            self.kernel_size[1],
            height_padding,
            self.stride[1],
        )

        self.curr_bottom_halo_size = calc_bottom_halo_size(
            self.rank,
            self.world_size,
            self.patch_height_index,
            self.kernel_size[1],
            height_padding,
            self.stride[1],
        )

        self.prev_bottom_halo_size = 0
        if self.rank != 0:
            self.prev_bottom_halo_size = calc_bottom_halo_size(
                self.rank - 1,
                self.world_size,
                self.patch_height_index,
                self.kernel_size[1],
                height_padding,
                self.stride[1],
            )

        self.next_top_halo_size = 0
        if self.rank != self.world_size - 1:
            self.next_top_halo_size = calc_top_halo_size(
                self.rank + 1,
                self.world_size,
                self.patch_height_index,
                self.kernel_size[1],
                height_padding,
                self.stride[1],
            )

        hidden_states = halo_exchange(
            hidden_states,
            rank=self.rank,
            group=self.group,
            prev_bottom_halo_size=self.prev_bottom_halo_size,
            next_top_halo_size=self.next_top_halo_size,
            curr_top_halo_size=self.curr_top_halo_size,
            curr_bottom_halo_size=self.curr_bottom_halo_size,
        )

        if self.rank == 0:
            padding[3] = 0
        elif self.rank == self.world_size - 1:
            padding[2] = 0
        else:
            padding[2] = 0
            padding[3] = 0

        hidden_states = self._pad_spatial(hidden_states, padding)
        return self.conv(hidden_states)
