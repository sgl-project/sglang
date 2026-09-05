"""cuDNN Frontend kernels for GDN (Gated Delta Network) prefill.

The cuDNN 1.28 FROST implementation consumes packed THD tensors and compact
per-sequence states. SGLang keeps recurrent states in a slot-indexed pool, so
this adapter gathers the active rows before the call and writes the returned
final states back afterward.
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

import torch
from packaging.version import Version

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.runtime_context import mamba_cache_chunk_size

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


_MIN_CUDNN_FRONTEND_VERSION = Version("1.28.0")
_MIN_CUTLASS_DSL_VERSION = Version("4.7.0")
_SUPPORTED_SMS = {100, 101, 102, 103, 107}


def _distribution_version(name: str) -> Version:
    try:
        return Version(importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"cuDNN GDN prefill requires {name}; install SGLang's CUDA dependencies."
        ) from exc


def _validate_cudnn_gdn_runtime() -> None:
    frontend_version = _distribution_version("nvidia-cudnn-frontend")
    if frontend_version < _MIN_CUDNN_FRONTEND_VERSION:
        raise RuntimeError(
            "cuDNN GDN prefill requires nvidia-cudnn-frontend>=1.28.0; "
            f"found {frontend_version}."
        )

    cutlass_version = _distribution_version("nvidia-cutlass-dsl")
    if cutlass_version < _MIN_CUTLASS_DSL_VERSION:
        raise RuntimeError(
            "cuDNN FROST GDN prefill requires nvidia-cutlass-dsl>=4.7.0; "
            f"found {cutlass_version}."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("cuDNN GDN prefill requires an NVIDIA CUDA device.")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    if sm not in _SUPPORTED_SMS:
        supported = ", ".join(f"SM{x}" for x in sorted(_SUPPORTED_SMS))
        raise RuntimeError(
            f"cuDNN FROST GDN prefill supports {supported}; found SM{sm}."
        )


class CudnnGDNKernel(LinearAttnKernelBase):
    """cuDNN Frontend 1.28 FROST GDN prefill kernel.

    Decode and speculative verification remain on SGLang's existing kernels.
    Select this implementation with ``--linear-attn-prefill-backend cudnn``.
    """

    uses_state_checkpoints = True

    def __init__(self) -> None:
        _validate_cudnn_gdn_runtime()
        from cudnn.linear_attention.ops import gated_delta_net

        self._gated_delta_net = gated_delta_net

    def prepare_state_checkpoint_plan(
        self,
        forward_batch: ForwardBatch,
        forward_metadata: ForwardMetadata,
        device: str,
    ) -> None:
        del forward_batch, device
        if (
            forward_metadata.track_ssm_h_src is not None
            and forward_metadata.track_ssm_h_src.numel() > 0
        ):
            # cuDNN packs the incoming state followed by states at each full
            # chunk boundary. That is the same layout used by SGLang's native
            # GDN tracking indices, so only the checkpoint cadence is needed.
            forward_metadata.state_checkpoint_every_n_tokens = mamba_cache_chunk_size()

    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "cuDNN GDN is a prefill-only backend; use Triton for decode."
        )

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
        state_checkpoint_every_n_tokens: int = 0,
        batch_invariant: bool = False,
        **kwargs,
    ) -> tuple:
        del kwargs
        if ssm_states.dtype != torch.float32:
            raise ValueError(
                "cuDNN GDN prefill requires float32 recurrent states; "
                f"got {ssm_states.dtype}. Use --mamba-ssm-dtype float32."
            )
        if q.shape[-1] != 128 or v.shape[-1] != 128:
            raise ValueError(
                "cuDNN FROST GDN prefill requires key/value head dimensions "
                f"of 128; got K={q.shape[-1]}, V={v.shape[-1]}."
            )

        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]
        state_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,
        ).to(torch.int64)
        initial_state = ssm_states[state_indices].contiguous()

        checkpoint_every = int(state_checkpoint_every_n_tokens)
        result = self._gated_delta_net(
            q=q[0].contiguous(),
            k=k[0].contiguous(),
            v=v[0].contiguous(),
            g=g[0].to(torch.float32).contiguous(),
            beta=beta[0].to(torch.float32).contiguous(),
            cu_seqlens=query_start_loc.to(torch.int32),
            scale=q.shape[-1] ** -0.5,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            batch_invariant=batch_invariant,
            checkpoint_every_n_tokens=checkpoint_every,
            plan_name="gdn_frost",
        )
        if checkpoint_every > 0:
            output, final_state, state_checkpoints = result
        else:
            output, final_state = result
            state_checkpoints = None

        ssm_states.index_copy_(0, state_indices, final_state)
        core_attn_out = output.view(1, total_seq_len, num_v_heads, head_v_dim)
        h = state_checkpoints.unsqueeze(0) if state_checkpoints is not None else None
        return core_attn_out, None, h

    def target_verify(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "cuDNN GDN does not support SGLang target verification; use Triton."
        )
