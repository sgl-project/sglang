# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/rotary_embedding.py

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
import functools
from collections import OrderedDict
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.layers.triton_ops import apply_rotary_embedding
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size] or [num_tokens, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    # cos = cos.unsqueeze(-2).to(x.dtype)
    # sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = (x1.float() * cos - x2.float() * sin).type_as(x)
        o2 = (x2.float() * cos + x1.float() * sin).type_as(x)
        return torch.cat((o1, o2), dim=-1)
    else:
        return apply_rotary_embedding(x, cos, sin, interleaved)


@CustomOp.register("rotary_embedding")
class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int | float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: int | float) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_cuda(self, *args, **kwargs) -> Any:
        return self.forward_native(*args, **kwargs)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


class OneDRotaryEmbedding(torch.nn.Module):
    """1D rotary positional embedding with caching."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        theta_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
        dtype: torch.dtype = torch.float32,
        use_real: bool = False,
        repeat_interleave_real: bool = False,
    ):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.theta = theta
        self.theta_rescale_factor = theta_rescale_factor
        self.interpolation_factor = interpolation_factor
        # dtype of freqs
        self.dtype = dtype
        self.use_real = use_real
        self.repeat_interleave_real = repeat_interleave_real

    def build_freqs(self, device):
        freqs = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=self.dtype, device=device)[
                    : (self.dim // 2)
                ]
                / self.dim
            ).to(device=device)
        )
        return freqs

    def build_freqs_outer(self, pos: torch.Tensor, device):
        theta = self.theta
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        if self.theta_rescale_factor != 1.0:
            theta *= self.theta_rescale_factor ** (self.dim / (self.dim - 2))

        freqs = self.build_freqs(device)

        freqs = torch.outer(pos * self.interpolation_factor, freqs)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()

        if self.use_real and self.repeat_interleave_real:
            freqs_cos = freqs_cos.repeat_interleave(2, dim=1)
            freqs_sin = freqs_sin.repeat_interleave(2, dim=1)

        return freqs_cos.float(), freqs_sin.float()

    @functools.lru_cache(maxsize=16)
    def forward_from_grid(
        self, seq_len: int, start_pos: int, device_str: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device_str)
        pos = torch.arange(
            start_pos, start_pos + seq_len, dtype=self.dtype, device=device
        )

        freqs_cos, freqs_sin = self.build_freqs_outer(pos, device)
        return freqs_cos, freqs_sin

    def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates 1D rotary embeddings for the given positions.

        This method converts the input tensor to a hashable representation
        and calls a cached helper method to perform the computation.
        """
        pos_tuple = tuple(pos.tolist())
        device_str = str(pos.device)
        return self._forward_cached(pos_tuple, device_str)

    @functools.lru_cache(maxsize=16)
    def _forward_cached(
        self, pos_tuple: tuple, device_str: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core implementation that computes 1D rotary embeddings.
        This method is wrapped by an LRU cache.
        """
        device = torch.device(device_str)
        pos = torch.as_tensor(pos_tuple, dtype=self.dtype, device=device)
        freqs_cos, freqs_sin = self.build_freqs_outer(pos, device)
        return freqs_cos, freqs_sin


class NDRotaryEmbedding(torch.nn.Module):
    """N-dimensional rotary positional embedding."""

    def __init__(
        self,
        rope_dim_list: list[int],
        rope_theta: float,
        theta_rescale_factor: float | list[float] = 1.0,
        interpolation_factor: float | list[float] = 1.0,
        use_real: bool = False,
        repeat_interleave_real: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.rope_dim_list = rope_dim_list
        self.ndim = len(rope_dim_list)
        self.rope_theta = rope_theta
        # dtype of freqs
        # does not control the output dtype
        self.dtype = dtype

        if isinstance(theta_rescale_factor, (int, float)):
            self.theta_rescale_factor = [theta_rescale_factor] * self.ndim
        elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
            self.theta_rescale_factor = [theta_rescale_factor[0]] * self.ndim
        else:
            self.theta_rescale_factor = theta_rescale_factor
        assert (
            len(self.theta_rescale_factor) == self.ndim
        ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

        if isinstance(interpolation_factor, (int, float)):
            self.interpolation_factor = [interpolation_factor] * self.ndim
        elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
            self.interpolation_factor = [interpolation_factor[0]] * self.ndim
        else:
            self.interpolation_factor = interpolation_factor
        assert (
            len(self.interpolation_factor) == self.ndim
        ), "len(interpolation_factor) should equal to len(rope_dim_list)"

        self.rope_generators: list[OneDRotaryEmbedding] = torch.nn.ModuleList()
        _config_to_gen_idx: dict[tuple, int] = {}
        self.dim_idx_to_gen_idx: list[int] = []

        for i in range(self.ndim):
            dim = self.rope_dim_list[i]
            rescale = self.theta_rescale_factor[i]
            interp = self.interpolation_factor[i]

            config_key = (dim, rescale, interp, use_real, repeat_interleave_real)
            if config_key not in _config_to_gen_idx:
                generator = OneDRotaryEmbedding(
                    dim=dim,
                    theta=self.rope_theta,
                    theta_rescale_factor=rescale,
                    interpolation_factor=interp,
                    dtype=self.dtype,
                    use_real=use_real,
                    repeat_interleave_real=repeat_interleave_real,
                )
                _config_to_gen_idx[config_key] = len(self.rope_generators)
                self.rope_generators.append(generator)

            gen_idx = _config_to_gen_idx[config_key]
            self.dim_idx_to_gen_idx.append(gen_idx)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates n-d rotary embeddings for given absolute positions.

        Args:
            positions (torch.Tensor): A tensor of shape `[num_tokens, ndim]`
                containing the integer coordinates for each token.

        Returns:
            A tuple of (cos, sin) tensors.
        """
        # Caching wrapper: convert tensor to a hashable tuple of tuples.
        pos_tuple = tuple(map(tuple, positions.tolist()))
        device_str = str(positions.device)
        return self._forward_cached(pos_tuple, device_str)

    @functools.lru_cache(maxsize=16)
    def _forward_cached(
        self, pos_tuple: tuple[tuple[int, ...], ...], device_str: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core implementation that computes embeddings from a position tensor.
        This method is wrapped by an LRU cache.
        """
        device = torch.device(device_str)
        positions = torch.tensor(pos_tuple, dtype=torch.long, device=device)
        return self.forward_uncached(pos=positions)

    def forward_uncached(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core implementation that computes embeddings from a position tensor.
        This method is wrapped by an LRU cache.
        """
        device = pos.device

        # Pre-allocate the final tensors for efficiency.
        num_tokens = pos.shape[0]
        first_generator = self.rope_generators[0]
        if first_generator.use_real and first_generator.repeat_interleave_real:
            head_dim = sum(self.rope_dim_list)
        else:
            head_dim = sum(self.rope_dim_list) // 2

        cos = torch.empty((num_tokens, head_dim), device=device, dtype=self.dtype)
        sin = torch.empty((num_tokens, head_dim), device=device, dtype=self.dtype)

        col_offset = 0
        for i in range(self.ndim):
            # Extract position coordinates for the current dimension for all tokens.
            pos_i = pos[:, i].to(self.dtype)

            # Get the appropriate 1D generator.
            gen_idx = self.dim_idx_to_gen_idx[i]
            generator = self.rope_generators[gen_idx]

            # Calculate 1D embeddings.
            cos_1d, sin_1d = generator(pos_i)

            slice_width = cos_1d.shape[1]
            cos[:, col_offset : col_offset + slice_width] = cos_1d
            sin[:, col_offset : col_offset + slice_width] = sin_1d
            col_offset += slice_width

        return cos.float(), sin.float()

    def forward_from_grid(
        self,
        grid_size: tuple[int, ...],
        shard_dim: int = 0,
        start_frame: int = 0,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Caching wrapper: use grid parameters directly as the key.
        # grid_tuple = _to_tuple(grid_size, dim=self.ndim)
        device_str = str(device) if device is not None else "cpu"
        return self._forward_cached_from_grid(
            grid_size, shard_dim, start_frame, device_str
        )

    @functools.lru_cache(maxsize=16)
    def _forward_cached_from_grid(
        self,
        grid_size: tuple[int, ...],
        shard_dim: int,
        start_frame: int,
        device_str: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes embeddings for a structured grid, using a highly efficient
        implementation that avoids materializing the full position tensor.
        This method is wrapped by an LRU cache.
        """
        device = torch.device(device_str)
        sp_group = get_sp_group()
        sp_rank = sp_group.rank_in_group
        sp_world_size = sp_group.world_size

        sizes = _to_tuple(grid_size, dim=self.ndim)
        starts = (0,) * self.ndim

        # Apply sequence parallel sharding to the sizes and compute shard offset
        shard_sizes = list(sizes)
        shard_offsets = [0] * self.ndim
        if sp_world_size > 1:
            assert sizes[shard_dim] % sp_world_size == 0, (
                f"Dimension {shard_dim} with size {sizes[shard_dim]} is not divisible "
                f"by sequence parallel world size {sp_world_size}"
            )
            shard_size = sizes[shard_dim] // sp_world_size
            shard_offsets[shard_dim] = sp_rank * shard_size
            shard_sizes[shard_dim] = shard_size

        # Pre-allocate outputs on the requested device to avoid CPU ops and extra cats
        num_tokens = 1
        for s in shard_sizes:
            num_tokens *= int(s)
        head_dim_half = sum(self.rope_dim_list) // 2
        cos = torch.empty((num_tokens, head_dim_half), device=device, dtype=self.dtype)
        sin = torch.empty((num_tokens, head_dim_half), device=device, dtype=self.dtype)

        # Compute per-axis 1D embeddings once and expand via repeats to [N, d_i/2]
        col_offset = 0
        for i in range(self.ndim):
            dim_i = self.rope_dim_list[i]
            dim_i_half = dim_i // 2
            size_i = int(shard_sizes[i])

            # Starting position for this axis, with optional frame offset for time axis (i==0)
            base_offset = starts[i]
            if i == 0 and start_frame > 0:
                base_offset += start_frame
            if sp_world_size > 1 and i == shard_dim:
                base_offset += shard_offsets[i]

            gen_idx = self.dim_idx_to_gen_idx[i]
            generator = self.rope_generators[gen_idx]
            cos_1d, sin_1d = generator.forward_from_grid(
                size_i, base_offset, device_str
            )

            # Expand to [num_tokens, dim_i/2] matching flatten order (last dims vary fastest)
            repeats_per_entry = 1
            for j in range(i + 1, self.ndim):
                repeats_per_entry *= int(shard_sizes[j])
            tile_count = 1
            for j in range(0, i):
                tile_count *= int(shard_sizes[j])

            cos_expanded = cos_1d.repeat_interleave(repeats_per_entry, dim=0)
            sin_expanded = sin_1d.repeat_interleave(repeats_per_entry, dim=0)
            if tile_count > 1:
                cos_expanded = cos_expanded.repeat(tile_count, 1)
                sin_expanded = sin_expanded.repeat(tile_count, 1)

            cos[:, col_offset : col_offset + dim_i_half] = cos_expanded
            sin[:, col_offset : col_offset + dim_i_half] = sin_expanded
            col_offset += dim_i_half

        return cos.float(), sin.float()


def _to_tuple(x: int | tuple[int, ...], dim: int = 2) -> tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(
    start: int | tuple[int, ...],
    *args: int | tuple[int, ...],
    dim: int = 2,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = tuple(stop[i] - start[i] for i in range(dim))
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=dtype, device=device)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.FloatTensor | int,
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.
        interpolation_factor (float, optional): Factor to scale positions. Defaults to 1.0.

    Returns:
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos, dtype=dtype, device=device)
    elif (
        isinstance(pos, torch.Tensor)
        and device is not None
        and pos.device != torch.device(device)
    ):
        # Ensure positions are on the requested device to avoid implicit CPU ops.
        pos = pos.to(device)

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].to(dtype) / dim).to(
            device=device
        )
    )  # [D/2]
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    freqs_cos = freqs.cos()  # [S, D/2]
    freqs_sin = freqs.sin()  # [S, D/2]
    return freqs_cos, freqs_sin


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    theta_rescale_factor: float | list[float] = 1.0,
    interpolation_factor: float | list[float] = 1.0,
    shard_dim: int = 0,
    sp_rank: int = 0,
    sp_world_size: int = 1,
    dtype: torch.dtype = torch.float32,
    start_frame: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.
    Supports sequence parallelism by allowing sharding of a specific dimension.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.
        interpolation_factor (float): Factor to scale positions. Defaults to 1.0.
        shard_dim (int): Which dimension to shard for sequence parallelism. Defaults to 0.
        sp_rank (int): Rank in the sequence parallel group. Defaults to 0.
        sp_world_size (int): World size of the sequence parallel group. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cos, sin) tensors of shape [HW, D/2]
    """
    # Determine per-axis sizes for the (possibly sharded) grid without materializing it
    ndim = len(rope_dim_list)
    if len(args) == 0:
        # start is grid_size
        sizes = _to_tuple(start, dim=ndim)
        starts = (0,) * ndim
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        starts = _to_tuple(start, dim=ndim)
        stops = _to_tuple(args[0], dim=ndim)
        sizes = tuple(stops[i] - starts[i] for i in range(ndim))
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        starts = _to_tuple(start, dim=ndim)
        _ = _to_tuple(args[0], dim=ndim)  # stop, unused here
        sizes = _to_tuple(args[1], dim=ndim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    assert (
        shard_dim < ndim
    ), f"shard_dim {shard_dim} must be less than number of dimensions {ndim}"

    # Apply sequence parallel sharding to the sizes and compute shard offset
    shard_sizes = list(sizes)
    shard_offsets = [0] * ndim
    if sp_world_size > 1:
        assert sizes[shard_dim] % sp_world_size == 0, (
            f"Dimension {shard_dim} with size {sizes[shard_dim]} is not divisible "
            f"by sequence parallel world size {sp_world_size}"
        )
        shard_size = sizes[shard_dim] // sp_world_size
        shard_offsets[shard_dim] = sp_rank * shard_size
        shard_sizes[shard_dim] = shard_size

    # Handle theta scaling/interpolation factor per-axis
    if isinstance(theta_rescale_factor, int | float):
        theta_rescale_factor = [theta_rescale_factor] * ndim
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * ndim
    assert (
        len(theta_rescale_factor) == ndim
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int | float):
        interpolation_factor = [interpolation_factor] * ndim
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * ndim
    assert (
        len(interpolation_factor) == ndim
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # Pre-allocate outputs on the requested device to avoid CPU ops and extra cats
    num_tokens = 1
    for s in shard_sizes:
        num_tokens *= int(s)
    head_dim_half = sum(rope_dim_list) // 2
    cos = torch.empty((num_tokens, head_dim_half), device=device, dtype=dtype)
    sin = torch.empty((num_tokens, head_dim_half), device=device, dtype=dtype)
    # Compute per-axis 1D embeddings once and expand via repeats to [N, d_i/2]
    col_offset = 0
    for i in range(ndim):
        dim_i = int(rope_dim_list[i])
        dim_i_half = dim_i // 2
        size_i = int(shard_sizes[i])

        # Starting position for this axis, with optional frame offset for time axis (i==0)
        base_offset = starts[i]
        if i == 0 and start_frame > 0:
            base_offset += start_frame
        if sp_world_size > 1 and i == shard_dim:
            base_offset += shard_offsets[i]

        pos_i = torch.arange(size_i, device=device, dtype=dtype) + base_offset

        cos_1d, sin_1d = get_1d_rotary_pos_embed(
            dim_i,
            pos_i,
            theta=theta,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
            dtype=dtype,
            device=device,
        )  # [size_i, dim_i/2]

        # Expand to [num_tokens, dim_i/2] matching flatten order (last dims vary fastest)
        repeats_per_entry = 1
        for j in range(i + 1, ndim):
            repeats_per_entry *= int(shard_sizes[j])
        tile_count = 1
        for j in range(0, i):
            tile_count *= int(shard_sizes[j])

        cos_expanded = cos_1d.repeat_interleave(repeats_per_entry, dim=0)
        sin_expanded = sin_1d.repeat_interleave(repeats_per_entry, dim=0)
        if tile_count > 1:
            cos_expanded = cos_expanded.repeat(tile_count, 1)
            sin_expanded = sin_expanded.repeat(tile_count, 1)

        cos[:, col_offset : col_offset + dim_i_half] = cos_expanded
        sin[:, col_offset : col_offset + dim_i_half] = sin_expanded
        col_offset += dim_i_half

    return cos, sin


def get_rotary_pos_embed(
    rope_sizes,
    hidden_size,
    heads_num,
    rope_dim_list,
    rope_theta,
    theta_rescale_factor=1.0,
    interpolation_factor=1.0,
    shard_dim: int = 0,
    dtype: torch.dtype = torch.float32,
    start_frame: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rotary positional embeddings for the given sizes.

    Args:
        rope_sizes: Tuple of dimensions (t, h, w)
        hidden_size: Hidden dimension size
        heads_num: Number of attention heads
        rope_dim_list: List of dimensions for each axis, or None
        rope_theta: Base for frequency calculations
        theta_rescale_factor: Rescale factor for theta. Defaults to 1.0
        interpolation_factor: Factor to scale positions. Defaults to 1.0
        shard_dim: Which dimension to shard for sequence parallelism. Defaults to 0.

    Returns:
        Tuple of (cos, sin) tensors for rotary embeddings
    """

    target_ndim = 3
    head_dim = hidden_size // heads_num

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]

    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"

    # Get SP info - now handled within NDRotaryEmbedding
    # sp_group = get_sp_group()
    # sp_rank = sp_group.rank_in_group
    # sp_world_size = sp_group.world_size

    # Simple LRU cache keyed by parameters
    global _ND_ROPE_CACHE
    key = (
        tuple(rope_dim_list),
        float(rope_theta),
        (
            tuple(theta_rescale_factor)
            if isinstance(theta_rescale_factor, list)
            else float(theta_rescale_factor)
        ),
        (
            tuple(interpolation_factor)
            if isinstance(interpolation_factor, list)
            else float(interpolation_factor)
        ),
        dtype,
    )

    cache_hit = key in _ND_ROPE_CACHE
    if cache_hit:
        rope_emb = _ND_ROPE_CACHE.pop(key)
        _ND_ROPE_CACHE[key] = rope_emb  # move to end (most-recent)
    else:
        rope_emb = NDRotaryEmbedding(
            rope_dim_list=rope_dim_list,
            rope_theta=rope_theta,
            theta_rescale_factor=theta_rescale_factor,
            interpolation_factor=interpolation_factor,
            dtype=dtype,
        )
        _ND_ROPE_CACHE[key] = rope_emb
        if len(_ND_ROPE_CACHE) > 16:
            # pop least-recently-used
            _ND_ROPE_CACHE.pop(next(iter(_ND_ROPE_CACHE)))

    freqs_cos, freqs_sin = rope_emb.forward_from_grid(
        grid_size=_to_tuple(rope_sizes, dim=3),
        shard_dim=shard_dim,
        start_frame=start_frame,
        device=device,
    )
    return freqs_cos, freqs_sin


_ROPE_DICT: dict[tuple, RotaryEmbedding] = {}
_ND_ROPE_CACHE: "OrderedDict[tuple, NDRotaryEmbedding]" = OrderedDict()
_ROPE_3D_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int | float,
    is_neox_style: bool = True,
    rope_scaling: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        raise ValueError(f"Unknown RoPE scaling {rope_scaling}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
