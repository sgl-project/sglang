"""FlashInfer 0.7 CuTe DSL KDA prefill with radix-cache checkpoints."""

from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def build_flashinfer_kda_checkpoint_plan(
    forward_batch: ForwardBatch,
    metadata: ForwardMetadata,
    device: torch.device,
    chunk_size: int,
) -> None:
    """Plan scheduler-generated tracked boundaries, asserting their validity.

    Host track metadata follows build_prefill_track_plan's unaligned-row order.
    """
    extend_lens = forward_batch.extend_seq_lens_cpu
    track_lens = forward_batch.mamba_track_seqlens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    track_mask = forward_batch.mamba_prefill_track_mask_cpu

    checkpoint_counts = [length // chunk_size for length in extend_lens]
    checkpoint_starts = list(accumulate(checkpoint_counts, initial=0))
    # NOTE(kpham-sgl): FlashInfer GDN writes all periodic checkpoints for SGLang
    # to select from; FlashInfer KDA writes only mapped destinations (-1 skips).
    checkpoint_destinations = [-1] * checkpoint_starts[-1]
    num_tracked_checkpoints = 0
    for row, tracked in enumerate(track_mask):
        if not tracked:
            continue
        relative_track_len = track_lens[row] - prefix_lens[row]
        if relative_track_len % chunk_size == 0:
            continue
        completed_chunks = relative_track_len // chunk_size
        assert 0 < completed_chunks <= checkpoint_counts[row], (
            f"Invalid KDA checkpoint metadata: row={row}, "
            f"relative_track_len={relative_track_len}, extend_len={extend_lens[row]}, "
            f"chunk_size={chunk_size}"
        )
        checkpoint_destinations[checkpoint_starts[row] + completed_chunks - 1] = (
            num_tracked_checkpoints
        )
        num_tracked_checkpoints += 1

    assert num_tracked_checkpoints == metadata.track_ssm_h_batch_src.numel(), (
        "KDA checkpoint rows disagree with the prefill tracking metadata"
    )
    metadata.state_checkpoint_cu_starts = torch.tensor(
        checkpoint_starts, dtype=torch.int64, device=device
    )
    metadata.num_state_checkpoints = num_tracked_checkpoints
    metadata.state_checkpoint_every_n_tokens = chunk_size
    metadata.state_checkpoint_indices = torch.tensor(
        checkpoint_destinations, dtype=torch.int32, device=device
    )


class FlashInferKDAPrefillKernel(LinearAttnKernelBase):
    """Execute KDA prefill; FlashInfer validates the tensor contract."""

    uses_state_checkpoints = True
    supports_track_state_snapshot = True
    supports_safe_gate = True

    def __init__(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise ValueError("FlashInfer KDA prefill requires SM100 or SM103")
        from flashinfer.kda import recurrent_kda

        self._recurrent_kda = recurrent_kda

    def decode(self, *args, **kwargs):
        raise NotImplementedError("FlashInferKDAPrefillKernel is prefill-only")

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        return_intermediate_states: bool = False,
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        state_checkpoint_indices: Optional[torch.Tensor] = None,
        track_ssm_h_batch_src: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        needs_checkpoint = (
            return_intermediate_states and kwargs.get("track_state") is not None
        )
        checkpoints = (
            ssm_states.new_empty((num_state_checkpoints, *ssm_states.shape[1:]))
            if needs_checkpoint
            else None
        )
        result = self._recurrent_kda(
            **_prefill_inputs(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=ssm_states,
                lower_bound=lower_bound,
            ),
            cu_seqlens=query_start_loc,
            ssm_state_indices=cache_indices.to(torch.int32),
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=(
                state_checkpoint_cu_starts if needs_checkpoint else None
            ),
            checkpoint_state_indices=(
                state_checkpoint_indices if needs_checkpoint else None
            ),
            checkpoint_every_n_tokens=(
                state_checkpoint_every_n_tokens if needs_checkpoint else 0
            ),
            backend="cute-dsl",
        )
        if needs_checkpoint:
            kwargs["track_state"][track_ssm_h_batch_src] = checkpoints.float()
        output = result[0]
        return (output, None) if return_intermediate_states else output

    def extend_in_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        graph_state: FlashInferKDAPrefillGraphState,
        cu_seqlens: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        inputs = _prefill_inputs(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            lower_bound=lower_bound,
        )
        checkpoints = None
        checkpoint_kwargs = {}
        if graph_state.checkpoint_every_n_tokens:
            checkpoints = ssm_states.new_empty(
                (graph_state.max_seqs, *ssm_states.shape[1:])
            )
            checkpoint_kwargs = dict(
                state_checkpoints=checkpoints,
                checkpoint_cu_starts=graph_state.checkpoint_cu_starts,
                checkpoint_every_n_tokens=graph_state.checkpoint_every_n_tokens,
                checkpoint_state_indices=graph_state.checkpoint_state_indices,
            )
        wrapper = graph_state.wrapper(layer_id=layer_id, cu_seqlens=cu_seqlens)
        result = wrapper.run(
            **inputs,
            output=torch.empty_like(inputs["q"]),
            ssm_state_indices=graph_state.state_indices,
            **checkpoint_kwargs,
        )
        graph_state.scratch.adopt(wrapper)
        return result[0], checkpoints


def _prefill_inputs(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    lower_bound: float,
) -> dict:
    return dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g[:, : q.shape[1]].contiguous(),
        beta=beta[:, : q.shape[1]].contiguous(),
        A_log=A_log.reshape(-1).float().contiguous(),
        # FlashInfer's grouped-CTA recurrent kernel (extends with a cached prefix)
        # indexes dt_bias as a flat [H*K] tensor and asserts a unit leading stride.
        dt_bias=dt_bias.reshape(-1).float().contiguous(),
        initial_state=ssm_states,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        beta_is_logit=True,
    )


# kda_chunked_bt16's BT: how far FlashInfer's last tile reads past the batch.
FLASHINFER_KDA_PREFILL_TILE_TOKENS = 16


class FlashInferKDAPrefillGraphScratch(msgspec.Struct):
    buffer: Optional[torch.Tensor] = None

    # Workaround: shares FlashInfer's private planner scratch (calls run serially on
    # one stream). Drop once flashinfer-ai/flashinfer#5069 lands.
    def share(self, wrapper) -> None:
        if self.buffer is not None:
            _planner_workspace(wrapper).__dict__["_cute_dsl_workspace"] = self.buffer

    def adopt(self, wrapper) -> None:
        workspace = _planner_workspace(wrapper)._cute_dsl_workspace
        if workspace is not None and (
            self.buffer is None or workspace.numel() > self.buffer.numel()
        ):
            self.buffer = workspace


def _planner_workspace(wrapper):
    return wrapper._impl.workspace


class FlashInferKDAPrefillGraphState(msgspec.Struct, frozen=True):
    max_seqs: int
    checkpoint_every_n_tokens: int
    state_indices: torch.Tensor
    checkpoint_cu_starts: Optional[torch.Tensor]
    checkpoint_state_indices: Optional[torch.Tensor]
    host_checkpoint_cu_starts: Optional[np.ndarray]
    host_checkpoint_state_indices: Optional[np.ndarray]
    scratch: FlashInferKDAPrefillGraphScratch
    wrappers: dict = msgspec.field(default_factory=dict)

    @classmethod
    def allocate(
        cls,
        *,
        num_tokens: int,
        max_seqs: int,
        checkpoint_every_n_tokens: int,
        device: torch.device,
        scratch: Optional[FlashInferKDAPrefillGraphScratch] = None,
    ) -> FlashInferKDAPrefillGraphState:
        cu_starts = indices = host_cu_starts = host_indices = None
        if checkpoint_every_n_tokens:
            max_checkpoints = num_tokens // checkpoint_every_n_tokens + 1
            cu_starts = torch.zeros(max_seqs + 1, dtype=torch.int64, device=device)
            indices = torch.full(
                (max_checkpoints,), -1, dtype=torch.int32, device=device
            )
            host_cu_starts = np.zeros(max_seqs + 1, dtype=np.int64)
            host_indices = np.full(max_checkpoints, -1, dtype=np.int32)
        return cls(
            max_seqs=max_seqs,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            state_indices=torch.zeros(max_seqs, dtype=torch.int32, device=device),
            checkpoint_cu_starts=cu_starts,
            checkpoint_state_indices=indices,
            host_checkpoint_cu_starts=host_cu_starts,
            host_checkpoint_state_indices=host_indices,
            scratch=FlashInferKDAPrefillGraphScratch() if scratch is None else scratch,
        )

    def fill(
        self,
        *,
        extend_seq_lens: list[int],
        cache_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        checkpoint_rows: list[tuple[int, int]],
    ) -> None:
        self.state_indices.copy_(cache_indices.clamp(min=0))
        if self.checkpoint_every_n_tokens:
            self._fill_checkpoint_tables(
                extend_seq_lens=extend_seq_lens, checkpoint_rows=checkpoint_rows
            )
        for wrapper in self.wrappers.values():
            wrapper.plan(cu_seqlens)

    def _fill_checkpoint_tables(
        self,
        *,
        extend_seq_lens: list[int],
        checkpoint_rows: list[tuple[int, int]],
    ) -> None:
        n = self.checkpoint_every_n_tokens
        bs = len(extend_seq_lens)
        starts = self.host_checkpoint_cu_starts
        starts[0] = 0
        np.cumsum(
            np.asarray(extend_seq_lens, dtype=np.int64) // n, out=starts[1 : bs + 1]
        )
        starts[bs + 1 :] = starts[bs]
        indices = self.host_checkpoint_state_indices
        indices.fill(-1)
        for row, completed_chunks in checkpoint_rows:
            indices[starts[row] + completed_chunks - 1] = row
        self.checkpoint_cu_starts.copy_(
            torch.from_numpy(starts).pin_memory(), non_blocking=True
        )
        self.checkpoint_state_indices.copy_(
            torch.from_numpy(indices).pin_memory(), non_blocking=True
        )

    def wrapper(self, *, layer_id: int, cu_seqlens: torch.Tensor):
        wrapper = self.wrappers.get(layer_id)
        if wrapper is None:
            from flashinfer.kda import RecurrentKDAPrefillWrapper

            wrapper = RecurrentKDAPrefillWrapper(cu_seqlens.device)
            self.scratch.share(wrapper)
            wrapper.plan(cu_seqlens)
            self.wrappers[layer_id] = wrapper
        return wrapper
