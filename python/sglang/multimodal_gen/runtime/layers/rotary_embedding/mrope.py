"""MRotaryEmbedding, YaRNScalingMRotaryEmbedding, NDRotaryEmbedding, OneDRotaryEmbedding."""

import functools

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group


def _to_tuple(x: int | tuple[int, ...], dim: int = 2) -> tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


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
        """
        Handles sp internally
        """
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
