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
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _cpu_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        return value.to(device="cpu").tolist()
    return list(value)


def build_flashinfer_kda_checkpoint_plan(
    forward_batch: ForwardBatch,
    metadata: ForwardMetadata,
    device: torch.device,
    chunk_size: int,
) -> None:
    if metadata.track_ssm_h_src is None or metadata.track_ssm_h_src.numel() == 0:
        return
    if chunk_size <= 0 or chunk_size % 32:
        return

    extend_lens = _cpu_list(
        forward_batch.extend_seq_lens_cpu
        if forward_batch.extend_seq_lens_cpu is not None
        else forward_batch.extend_seq_lens
    )
    track_lens = _cpu_list(
        forward_batch.mamba_track_seqlens_cpu
        if forward_batch.mamba_track_seqlens_cpu is not None
        else forward_batch.mamba_track_seqlens
    )
    prefix_lens = _cpu_list(
        forward_batch.extend_prefix_lens_cpu
        if forward_batch.extend_prefix_lens_cpu is not None
        else forward_batch.extend_prefix_lens
    )
    track_mask = _cpu_list(
        forward_batch.mamba_prefill_track_mask_cpu
        if forward_batch.mamba_prefill_track_mask_cpu is not None
        else forward_batch.mamba_track_mask
    )

    checkpoint_counts = [length // chunk_size for length in extend_lens]
    checkpoint_starts = list(accumulate(checkpoint_counts, initial=0))
    track_sources = []
    for row, tracked in enumerate(track_mask):
        if not tracked:
            continue
        relative_track_len = track_lens[row] - prefix_lens[row]
        if relative_track_len % chunk_size == 0:
            continue  # The final state is copied from the live state pool.
        completed_chunks = relative_track_len // chunk_size
        if completed_chunks == 0 or completed_chunks > checkpoint_counts[row]:
            return  # Triton handles a track point without a complete boundary.
        track_sources.append(checkpoint_starts[row] + completed_chunks - 1)

    if len(track_sources) != metadata.track_ssm_h_batch_src.numel():
        return
    metadata.state_checkpoint_cu_starts = torch.tensor(
        checkpoint_starts, dtype=torch.int64, device=device
    )
    metadata.num_state_checkpoints = checkpoint_starts[-1]
    metadata.state_checkpoint_every_n_tokens = chunk_size
    metadata.state_checkpoint_track_src = torch.tensor(
        track_sources, dtype=torch.int64, device=device
    )
    metadata.state_checkpoint_indices = torch.arange(
        checkpoint_starts[-1], dtype=torch.int32, device=device
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
        state_checkpoint_track_src: Optional[torch.Tensor] = None,
        state_checkpoint_indices: Optional[torch.Tensor] = None,
        track_ssm_h_batch_src: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
        num_sequences = query_start_loc.numel() - 1
        needs_checkpoint = (
            return_intermediate_states and kwargs.get("track_state") is not None
        )
        eligible = (
            torch.cuda.get_device_capability(q.device) in ((10, 0), (10, 3))
            and not torch.cuda.is_current_stream_capturing()
            and not kwargs.get("is_spec_decode", False)
            and beta_is_raw
            and lower_bound is not None
            and math.isfinite(float(lower_bound))
            and float(lower_bound) < 0
            and A_log is not None
            and dt_bias is not None
            and q.dtype == k.dtype == v.dtype == g.dtype == beta.dtype == torch.bfloat16
            and ssm_states.dtype in (torch.bfloat16, torch.float32)
            and q.ndim == k.ndim == v.ndim == g.ndim == 4
            and beta.ndim == 3
            and q.shape[0] == k.shape[0] == v.shape[0] == g.shape[0] == 1
            and q.shape[1] == k.shape[1] == v.shape[1]
            and q.shape[2:] == k.shape[2:] == v.shape[2:] == g.shape[2:]
            and q.shape[-1] == 128
            and ssm_states.ndim == 4
            and ssm_states.shape[1:] == (q.shape[2], 128, 128)
            and ssm_states.stride()[1:] == (128 * 128, 128, 1)
            and ssm_states.stride(0) % 16 == 0
            and ssm_states.storage_offset() * ssm_states.element_size() % 32 == 0
            and A_log.numel() == q.shape[2]
            and dt_bias.numel() == q.shape[2] * 128
            and beta.shape[0] == 1
            and beta.shape[1] >= q.shape[1]
            and beta.shape[2] == q.shape[2]
            and beta.data_ptr() % 16 == 0
            and g.shape[1] >= q.shape[1]
            and g[:, : q.shape[1]].is_contiguous()
            and num_sequences > 0
            and q.shape[1] > num_sequences
            and cache_indices.ndim == 1
            and cache_indices.numel() == num_sequences
            and seq_lens_cpu is not None
            and len(seq_lens_cpu) == num_sequences
            and min(seq_lens_cpu) > 0
            and sum(seq_lens_cpu) == q.shape[1]
            and (
                not needs_checkpoint
                or (
                    state_checkpoint_cu_starts is not None
                    and state_checkpoint_indices is not None
                    and state_checkpoint_track_src is not None
                    and track_ssm_h_batch_src is not None
                )
            )
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

        from flashinfer.kda import recurrent_kda

        checkpoints = (
            ssm_states.new_empty((num_state_checkpoints, *ssm_states.shape[1:]))
            if needs_checkpoint
            else None
        )
        result = recurrent_kda(
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
            cu_seqlens=query_start_loc.to(torch.int64),
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
            backend="cute-dsl",
        )
        if needs_checkpoint:
            kwargs["track_state"][track_ssm_h_batch_src] = checkpoints[
                state_checkpoint_track_src
            ].float()
        output = result[0]
        return (output, None) if return_intermediate_states else output
