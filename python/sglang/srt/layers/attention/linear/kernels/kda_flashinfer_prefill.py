"""FlashInfer 0.7 CuTe DSL KDA prefill with radix-cache checkpoints."""

from __future__ import annotations

import math
from itertools import accumulate
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

if TYPE_CHECKING:
    from flashinfer.kda import RecurrentKDAPrefillWrapper

    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def build_flashinfer_kda_checkpoint_plan(
    forward_batch: ForwardBatch,
    metadata: ForwardMetadata,
    device: torch.device,
    chunk_size: int,
) -> None:
    if chunk_size <= 0 or chunk_size % 32:
        return
    if any(
        values is None
        for values in (
            forward_batch.extend_seq_lens_cpu,
            forward_batch.mamba_track_seqlens_cpu,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.mamba_prefill_track_mask_cpu,
        )
    ):
        return

    extend_lens = forward_batch.extend_seq_lens_cpu
    track_lens = forward_batch.mamba_track_seqlens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    track_mask = forward_batch.mamba_prefill_track_mask_cpu

    checkpoint_counts = [length // chunk_size for length in extend_lens]
    checkpoint_starts = list(accumulate(checkpoint_counts, initial=0))
    # -1 skips boundaries SGLang will never restore from the radix cache.
    checkpoint_destinations = [-1] * checkpoint_starts[-1]
    num_tracked_checkpoints = 0
    for row, tracked in enumerate(track_mask):
        if not tracked:
            continue
        relative_track_len = track_lens[row] - prefix_lens[row]
        if relative_track_len % chunk_size == 0:
            continue  # The final state is copied from the live state pool.
        completed_chunks = relative_track_len // chunk_size
        if completed_chunks == 0 or completed_chunks > checkpoint_counts[row]:
            return  # Triton handles a track point without a complete boundary.
        checkpoint_destinations[checkpoint_starts[row] + completed_chunks - 1] = (
            num_tracked_checkpoints
        )
        num_tracked_checkpoints += 1

    # The backend selects these same unaligned rows with build_prefill_track_plan.
    metadata.state_checkpoint_cu_starts = torch.tensor(
        checkpoint_starts, dtype=torch.int64, device=device
    )
    metadata.num_state_checkpoints = num_tracked_checkpoints
    metadata.state_checkpoint_every_n_tokens = chunk_size
    metadata.state_checkpoint_indices = torch.tensor(
        checkpoint_destinations, dtype=torch.int32, device=device
    )


class FlashInferKDAPrefillKernel(LinearAttnKernelBase):
    uses_state_checkpoints = True
    supports_track_state_snapshot = True
    supports_safe_gate = True
    expects_beta_logits = True

    def __init__(self, triton_fallback: TritonKDAKernel):
        self._triton = triton_fallback

    def decode(self, *args, **kwargs):
        raise NotImplementedError("FlashInferKDAPrefillKernel is prefill-only")

    def plan(self, query_start_loc: torch.Tensor) -> RecurrentKDAPrefillWrapper:
        from flashinfer.kda import RecurrentKDAPrefillWrapper

        wrapper = RecurrentKDAPrefillWrapper(query_start_loc.device)
        wrapper.plan(query_start_loc)
        return wrapper

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
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        state_checkpoint_indices: Optional[torch.Tensor] = None,
        track_ssm_h_batch_src: Optional[torch.Tensor] = None,
        prefill_metadata: Optional[ForwardMetadata] = None,
        prefill_forward_batch: Optional[ForwardBatch] = None,
        prefill_chunk_size: int = 0,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
        needs_checkpoint = (
            return_intermediate_states and kwargs.get("track_state") is not None
        )
        # KDAAttnBackend supplies packed Q/K/V from one convolution, matching
        # gate parameters and device metadata. Both static and envelope state
        # pools keep compact head matrices with 128-D-aligned slot pitches.
        # Only backend limitations belong here; caller invariants do not.
        eligible = (
            torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))
            and not torch.cuda.is_current_stream_capturing()
            and not kwargs.get("is_spec_decode", False)
            and beta_is_raw
            and lower_bound is not None
            and math.isfinite(float(lower_bound))
            and float(lower_bound) < 0
            and q.dtype == g.dtype == beta.dtype == torch.bfloat16
            and q.shape[-1] == 128
            and v.shape[2:] == q.shape[2:]
            and ssm_states.dtype in (torch.bfloat16, torch.float32)
            # Full-rank fused gate projections can have a padded token stride.
            and g[:, : q.shape[1]].is_contiguous()
            # gpu_only batches lack host lengths; DP padding can add empty rows.
            and seq_lens_cpu is not None
            and min(seq_lens_cpu) > 0
            and q.shape[1] > len(seq_lens_cpu)
            # TBO derives query offsets separately from its host length slices.
            and sum(seq_lens_cpu) == q.shape[1]
        )
        if eligible and needs_checkpoint and state_checkpoint_cu_starts is None:
            if prefill_metadata is not None and prefill_forward_batch is not None:
                build_flashinfer_kda_checkpoint_plan(
                    prefill_forward_batch,
                    prefill_metadata,
                    q.device,
                    prefill_chunk_size,
                )
                state_checkpoint_cu_starts = prefill_metadata.state_checkpoint_cu_starts
                state_checkpoint_indices = prefill_metadata.state_checkpoint_indices
                num_state_checkpoints = prefill_metadata.num_state_checkpoints
                state_checkpoint_every_n_tokens = (
                    prefill_metadata.state_checkpoint_every_n_tokens
                )
        eligible = eligible and (
            # A successful plan publishes starts and indices together; the
            # backend already supplies tracked batch rows with track_state.
            not needs_checkpoint or state_checkpoint_cu_starts is not None
        )
        if not eligible:
            return self._triton.extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
                **kwargs,
            )

        prefill_wrapper = (
            prefill_metadata.flashinfer_kda_prefill_wrapper
            if prefill_metadata is not None
            else None
        )
        if prefill_wrapper is None:
            prefill_wrapper = self.plan(query_start_loc)
            if prefill_metadata is not None:
                prefill_metadata.flashinfer_kda_prefill_wrapper = prefill_wrapper
        checkpoints = (
            ssm_states.new_empty((num_state_checkpoints, *ssm_states.shape[1:]))
            if needs_checkpoint
            else None
        )
        result = prefill_wrapper.run(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g,
            # Kimi-K3 slices beta from a fused projection, so its token stride
            # can exceed the number of heads even in a normal prefill.
            beta=beta[:, : q.shape[1]].contiguous(),
            A_log=A_log.reshape(-1).float().contiguous(),
            dt_bias=dt_bias.reshape(q.shape[2], 128).float().contiguous(),
            initial_state=ssm_states,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=lower_bound,
            ssm_state_indices=cache_indices.to(torch.int32),
            beta_is_logit=True,
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
        )
        if needs_checkpoint:
            kwargs["track_state"][track_ssm_h_batch_src] = checkpoints.float()
        output = result[0]
        return (output, None) if return_intermediate_states else output
