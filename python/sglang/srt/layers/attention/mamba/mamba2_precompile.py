"""Pre-compile the Mamba2 prefill Triton kernels during engine init.

Triton builds a kernel binary on first launch for each new specialization: a
new set of constexpr flags, an optional tensor argument switching between None
and a tensor (which also changes the autotuner's cache key, so the whole config
sweep runs again), or an integer argument moving between 1, a multiple of 16,
and anything else. In the packed prefill layout the token count reaches the
kernels through the sequence lengths, the chunk counts and the strides of the
conv1d output, so batches of new shapes keep hitting new specializations.

Models that capture prefill CUDA graphs pay for those compiles at capture
time. Models that cannot (NemotronH) pay for them on live traffic, where one
compile stalls the rank for seconds and, under TP, every rank waiting on its
collectives. ``precompile_mamba2_prefill_kernels`` replays the mixer's prefill
steps (causal conv1d, then the SSD chunk scan) on dummy data over a sweep of
batch layouts that covers those classes, so the compiles happen before the
engine starts serving.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
from torch import nn

from sglang.srt.configs.mamba_utils import Mamba2CacheParams
from sglang.srt.layers.attention.mamba import mamba as mamba_layer
from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata


class Mamba2PrefillDims(msgspec.Struct, frozen=True, kw_only=True):
    """Per-TP-rank sizes and dtypes of one Mamba2 mixer's prefill inputs."""

    num_heads: int
    head_dim: int
    n_groups: int
    state_size: int
    intermediate_size: int
    conv_dim: int
    conv_kernel: int
    chunk_size: int
    dtype: torch.dtype
    conv_state_dtype: torch.dtype
    ssm_state_dtype: torch.dtype


class Mamba2PrefillParams(msgspec.Struct, frozen=True, kw_only=True):
    """The mixer weights the prefill kernels read (A, D, dt_bias, conv1d)."""

    A: torch.Tensor
    D: torch.Tensor
    dt_bias: torch.Tensor
    conv_weight: torch.Tensor
    conv_bias: Optional[torch.Tensor]
    activation: str


@torch.inference_mode()
def precompile_mamba2_prefill_kernels(
    model: nn.Module,
    *,
    chunk_size: int,
    cache_params: Mamba2CacheParams,
    use_triton_causal_conv: bool,
) -> int:
    """Compile the prefill kernels of every distinct Mamba2 mixer in ``model``.

    Mixers with the same sizes and dtypes share their kernels, so each distinct
    shape is swept once. Returns the number of prefill runs (0 when the model
    has no Mamba2 mixer on a CUDA device).
    """
    seen = set()
    cases = 0
    for module in model.modules():
        if not isinstance(module, mamba_layer.MambaMixer2):
            continue
        if module.A.device.type != "cuda":
            continue
        dims, params = _mixer_prefill_inputs(
            mixer=module, chunk_size=chunk_size, cache_params=cache_params
        )
        key = (
            dims,
            params.A.dtype,
            params.D.dtype,
            params.dt_bias.dtype,
            params.conv_bias is None,
            params.activation,
        )
        if key in seen:
            continue
        seen.add(key)
        cases += sweep_mamba2_prefill(
            dims=dims, params=params, use_triton_causal_conv=use_triton_causal_conv
        )
    return cases


def sweep_mamba2_prefill(
    *,
    dims: Mamba2PrefillDims,
    params: Mamba2PrefillParams,
    use_triton_causal_conv: bool,
) -> int:
    """Run every swept layout with and without initial states and tracking."""
    cases = 0
    for seq_lens in prefill_layouts(dims.chunk_size):
        for with_initial_states in (False, True):
            for with_track in (False, True):
                cases += run_mamba2_prefill(
                    dims=dims,
                    params=params,
                    seq_lens=seq_lens,
                    with_initial_states=with_initial_states,
                    with_track=with_track,
                    use_triton_causal_conv=use_triton_causal_conv,
                )
    return cases


def run_mamba2_prefill(
    *,
    dims: Mamba2PrefillDims,
    params: Mamba2PrefillParams,
    seq_lens: tuple[int, ...],
    with_initial_states: bool,
    with_track: bool,
    use_triton_causal_conv: bool,
) -> bool:
    """Run the mixer's prefill conv1d and SSD scan on dummy data.

    Builds the inputs with the same layout ``MambaMixer2.forward`` gives them
    (views into one projection, conv1d output transposed back), because the
    strides are kernel arguments too. Returns False when the requested
    variant does not exist for ``seq_lens`` (no request to track).
    """
    device = params.A.device
    c = dims.chunk_size
    num_tokens = sum(seq_lens)

    track_seq_idx = track_end_locs = None
    if with_track:
        tracked, end_locs = _tracked_requests(seq_lens=seq_lens, chunk_size=c)
        if not tracked:
            return False
        track_seq_idx = torch.tensor(tracked, dtype=torch.int64, device=device)
        track_end_locs = torch.tensor(end_locs, dtype=torch.int64, device=device)

    query_start_loc, seq_idx = _packed_batch_indices(seq_lens=seq_lens, device=device)
    # Slot 0 stays free, as the pool reserves it for padding.
    cache_indices = torch.arange(1, len(seq_lens) + 1, dtype=torch.int32, device=device)
    has_initial_states = torch.full(
        (len(seq_lens),), with_initial_states, dtype=torch.bool, device=device
    )
    x, B, C, dt = _run_prefill_conv(
        dims=dims,
        params=params,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_states=has_initial_states,
        use_triton_causal_conv=use_triton_causal_conv,
    )

    initial_states = chunk_indices = chunk_offsets = None
    if with_initial_states:
        ssm_states = torch.zeros(
            len(seq_lens) + 1,
            dims.num_heads,
            dims.head_dim,
            dims.state_size,
            dtype=dims.ssm_state_dtype,
            device=device,
        )
        initial_states = torch.where(
            has_initial_states[:, None, None, None], ssm_states[cache_indices], 0
        )
        chunk_indices, chunk_offsets = (
            Mamba2Metadata._query_start_loc_to_chunk_indices_offsets(
                query_start_loc=query_start_loc,
                chunk_size=c,
                total_seqlens=num_tokens,
            )
        )

    out = torch.empty(
        num_tokens, dims.num_heads * dims.head_dim, dtype=dims.dtype, device=device
    )
    # Same arguments as the prefill branch of MambaMixer2.forward.
    mamba_layer.mamba_chunk_scan_combined(
        x.view(1, num_tokens, dims.num_heads, dims.head_dim),
        dt.unsqueeze(0),
        params.A,
        B.view(1, num_tokens, dims.n_groups, -1),
        C.view(1, num_tokens, dims.n_groups, -1),
        chunk_size=c,
        D=params.D,
        z=None,
        dt_bias=params.dt_bias,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        cu_seqlens=query_start_loc,
        initial_states=initial_states,
        return_varlen_states=True,
        return_final_states=False,
        return_track_states=True,
        track_seq_idx=track_seq_idx,
        track_end_locs=track_end_locs,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        out=out.view(1, num_tokens, -1, dims.head_dim),
        state_dtype=dims.ssm_state_dtype,
    )
    return True


def prefill_layouts(chunk_size: int) -> list[tuple[int, ...]]:
    """Per-request token counts of the swept prefill batches.

    Crosses request counts with batch token counts so that the request count,
    the token count and the chunk count each take all of Triton's integer
    classes (1, a multiple of 16, neither) in combination, with tokens split
    as evenly as possible. Two more layouts start long requests off the chunk
    grid, so the state-tracking path sees one and two tracked requests.

    With initial states, the kernels also take the number of logical chunks:
    the chunk count plus one for every request boundary that falls inside a
    chunk. The last group of layouts moves that number through its classes for
    each token count by putting 1-token requests in front.
    """
    c = chunk_size
    request_counts = (1, 2, 3, 16, 17)
    token_counts = (1, 7, 16, c + 44, 2 * c, 16 * c - 1, 16 * c)
    layouts = []
    for num_tokens in token_counts:
        for num_seqs in request_counts:
            if num_seqs > num_tokens:
                continue
            base, extra = divmod(num_tokens, num_seqs)
            layouts.append(tuple(base + (i < extra) for i in range(num_seqs)))
    layouts.append((c + 44, 2 * c + 7))
    layouts.append((c + 44, 2 * c + 7, 3 * c))
    # 17 tokens: the only way to get 16 logical chunks in one chunk with a
    # token count that is neither 1 nor a multiple of 16.
    for num_tokens in token_counts + (17,):
        num_chunks = -(-num_tokens // c)
        for num_logical in (1, 16, 32, num_chunks + 1, num_chunks + 2):
            # Each 1-token request in front ends off the chunk grid and adds
            # one logical chunk.
            num_short = num_logical - num_chunks
            if 0 <= num_short < num_tokens:
                layouts.append((1,) * num_short + (num_tokens - num_short,))
    return list(dict.fromkeys(layouts))


def _run_prefill_conv(
    *,
    dims: Mamba2PrefillDims,
    params: Mamba2PrefillParams,
    seq_lens: tuple[int, ...],
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
    use_triton_causal_conv: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the causal conv1d on a dummy projection; return x, B, C and dt."""
    device = params.A.device
    num_tokens = sum(seq_lens)
    projected = torch.zeros(
        num_tokens,
        dims.intermediate_size + dims.conv_dim + dims.num_heads,
        dtype=dims.dtype,
        device=device,
    )
    _, hidden_states_B_C, dt = torch.split(
        projected, [dims.intermediate_size, dims.conv_dim, dims.num_heads], dim=-1
    )
    conv_states = torch.zeros(
        len(seq_lens) + 1,
        dims.conv_dim,
        dims.conv_kernel - 1,
        dtype=dims.conv_state_dtype,
        device=device,
    )
    conv_fn = (
        mamba_layer.causal_conv1d_fn_triton
        if use_triton_causal_conv
        else mamba_layer.causal_conv1d_fn
    )
    # Same arguments as the prefill branch of MambaMixer2.forward.
    hidden_states_B_C = conv_fn(
        hidden_states_B_C.transpose(0, 1),
        params.conv_weight,
        params.conv_bias,
        activation=params.activation,
        conv_states=conv_states,
        has_initial_state=has_initial_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        seq_lens_cpu=list(seq_lens),
    ).transpose(0, 1)[:num_tokens]
    groups_state_size = dims.n_groups * dims.state_size
    x, B, C = torch.split(
        hidden_states_B_C,
        [dims.intermediate_size, groups_state_size, groups_state_size],
        dim=-1,
    )
    return x, B, C, dt


def _packed_batch_indices(
    *, seq_lens: tuple[int, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query_start_loc (int32) and the per-token request index."""
    query_start_loc = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(seq_lens).cumsum(0)
    query_start_loc = query_start_loc.to(device)
    seq_idx = torch.repeat_interleave(
        torch.arange(len(seq_lens), dtype=torch.int32, device=device),
        query_start_loc.diff(),
        output_size=sum(seq_lens),
    ).unsqueeze(0)
    return query_start_loc, seq_idx


def _tracked_requests(
    *, seq_lens: tuple[int, ...], chunk_size: int
) -> tuple[list[int], list[int]]:
    """Requests whose tracked state must be recomputed off the chunk grid.

    Mirrors the Mamba2 branch of the hybrid backend's track metadata: a request
    of at least one chunk that starts off the grid of the flattened batch.
    """
    seq_idx, end_locs = [], []
    start = 0
    for i, seq_len in enumerate(seq_lens):
        aligned_len = (seq_len // chunk_size) * chunk_size
        if aligned_len > 0 and start % chunk_size != 0:
            seq_idx.append(i)
            end_locs.append(start + aligned_len)
        start += seq_len
    return seq_idx, end_locs


def _mixer_prefill_inputs(
    *,
    mixer: mamba_layer.MambaMixer2,
    chunk_size: int,
    cache_params: Mamba2CacheParams,
) -> tuple[Mamba2PrefillDims, Mamba2PrefillParams]:
    """Read one mixer's per-TP-rank sizes, dtypes and weights."""
    tp_size = mixer.tp_size
    conv_weight = mixer.conv1d.weight
    dims = Mamba2PrefillDims(
        num_heads=mixer.num_heads // tp_size,
        head_dim=mixer.head_dim,
        n_groups=mixer.n_groups // tp_size,
        state_size=mixer.ssm_state_size,
        intermediate_size=mixer.intermediate_size // tp_size,
        conv_dim=mixer.conv_dim // tp_size,
        conv_kernel=conv_weight.size(2),
        chunk_size=chunk_size,
        dtype=conv_weight.dtype,
        conv_state_dtype=cache_params.dtype.conv,
        ssm_state_dtype=cache_params.dtype.temporal,
    )
    params = Mamba2PrefillParams(
        A=mixer.A,
        D=mixer.D,
        dt_bias=mixer.dt_bias,
        conv_weight=conv_weight.view(conv_weight.size(0), conv_weight.size(2)),
        conv_bias=mixer.conv1d.bias,
        activation=mixer.activation,
    )
    return dims, params
