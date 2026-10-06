"""FlashInfer 0.7 CuTe DSL KDA prefill with radix-cache checkpoints."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import accumulate
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

if TYPE_CHECKING:
    from flashinfer.kda import RecurrentKDAPrefillWrapper

    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def build_flashinfer_kda_checkpoint_plan(
    forward_batch: ForwardBatch,
    metadata: ForwardMetadata,
    device: torch.device,
    chunk_size: int,
) -> bool:
    """Plan tracked boundaries; False selects Triton for this batch.

    Setup validates chunk_size. The host track metadata comes from the same
    producer as build_prefill_track_plan, including its unaligned-row ordering.
    """
    if any(
        values is None
        for values in (
            forward_batch.extend_seq_lens_cpu,
            forward_batch.mamba_track_seqlens_cpu,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.mamba_prefill_track_mask_cpu,
        )
    ):
        return False

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
            return False  # Triton handles a track point without a complete boundary.
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
    return True


class FlashInferKDAPrefillKernel(LinearAttnKernelBase):
    """Execute planned, packed BF16 safe-gate KDA prefill on SM100/SM103.

    KDAAttnBackend validates model dimensions, dtypes and the safe gate at
    setup, then plans each supported batch before any layer runs. Q/K/V and
    raw gate/beta projections share the model's activation dtype; state pools
    supply compact 128x128 head matrices, including envelope-strided slots.
    Unsupported batch modes are dispatched to Triton by KDAKernelDispatcher.
    """

    uses_state_checkpoints = True
    supports_track_state_snapshot = True
    supports_safe_gate = True
    expects_beta_logits = True

    def __init__(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            raise ValueError("FlashInfer KDA prefill requires SM100 or SM103")
        from flashinfer.kda import RecurrentKDAPrefillWrapper

        self._wrapper_cls = RecurrentKDAPrefillWrapper

    def validate_model(
        self,
        *,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        layers: Iterable[RadixLinearAttention],
        chunk_size: Optional[int] = None,
    ) -> None:
        """Establish the fixed contract once, before warmup or graph capture."""
        if dtype != torch.bfloat16:
            raise ValueError("FlashInfer KDA prefill requires BF16 activations")
        if state_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("FlashInfer KDA prefill requires BF16 or FP32 SSM state")
        if chunk_size is not None and (chunk_size <= 0 or chunk_size % 32):
            raise ValueError(
                "FlashInfer KDA prefill requires a positive checkpoint interval "
                "divisible by 32"
            )
        for layer in layers:
            if (layer.head_q_dim, layer.head_k_dim, layer.head_v_dim) != (
                128,
                128,
                128,
            ) or not layer.num_q_heads == layer.num_k_heads == layer.num_v_heads:
                raise ValueError(
                    f"FlashInfer KDA prefill requires equal Q/K/V head counts and "
                    f"128-D heads (layer {layer.layer_id}); use Triton prefill"
                )
            if (
                layer.lower_bound is None
                or not math.isfinite(float(layer.lower_bound))
                or float(layer.lower_bound) >= 0
            ):
                raise ValueError(
                    f"FlashInfer KDA prefill requires a finite negative safe-gate "
                    f"lower bound (layer {layer.layer_id}); use Triton prefill"
                )

    def decode(self, *args, **kwargs):
        raise NotImplementedError("FlashInferKDAPrefillKernel is prefill-only")

    def plan(self, query_start_loc: torch.Tensor) -> RecurrentKDAPrefillWrapper:
        wrapper = self._wrapper_cls(query_start_loc.device)
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
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        lower_bound: float,
        prefill_wrapper: RecurrentKDAPrefillWrapper,
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
        result = prefill_wrapper.run(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            # Fused projections may leave padding between token rows.
            g=g[:, : q.shape[1]].contiguous(),
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
