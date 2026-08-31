"""FlashInfer-based kernels for GDN (Gated Delta Network) linear attention.

Both SM90 and SM100 use the same pool layout: [pool, HV, V, K] (K-last).

SM90 (Hopper): full support — decode, prefill, MTP.  State dtype: fp32.
SM100 (Blackwell): full support — decode, prefill, MTP.

Requires flashinfer >= 0.6.14.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.runtime_context import (
    mamba_cache_chunk_size,
)
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_FLASHINFER_GDN_ALIGNMENT = 32


def _empty_aligned_like(
    tensor: torch.Tensor, alignment: int = _FLASHINFER_GDN_ALIGNMENT
) -> torch.Tensor:
    element_size = tensor.dtype.itemsize
    alignment_elements = max(1, alignment // element_size)
    storage = torch.empty(
        tensor.numel() + alignment_elements - 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    start_bytes = (-storage.data_ptr()) % alignment
    if start_bytes % element_size:
        raise RuntimeError(
            f"Cannot align {tensor.dtype} storage at {storage.data_ptr()} "
            f"to {alignment} bytes"
        )
    start = start_bytes // element_size
    return storage[start : start + tensor.numel()].view(tensor.shape)


# ---------------------------------------------------------------------------
# Lazy import for FlashInfer GDN kernels
# ---------------------------------------------------------------------------
_flashinfer_gdn_available: Optional[bool] = None
_flashinfer_chunk_gated_delta_rule = None
_flashinfer_gated_delta_rule_mtp = None
_flashinfer_gated_delta_rule_decode = None
_flashinfer_gated_delta_rule_mtp_bf16 = None


def maybe_build_flashinfer_checkpoint_plan(
    forward_batch: ForwardBatch,
    forward_metadata: ForwardMetadata,
    device: str,
) -> None:
    """Populate packed FlashInfer checkpoint metadata when tracking requires it."""
    if (
        forward_metadata.track_ssm_h_src is None
        or forward_metadata.track_ssm_h_src.numel() == 0
    ):
        return

    checkpoint_every_n_tokens = mamba_cache_chunk_size()
    extend_seq_lens = forward_batch.extend_seq_lens.to(device="cpu", dtype=torch.int64)
    track_mask = forward_batch.mamba_track_mask.to(device="cpu", dtype=torch.bool)
    relative_track_lens = forward_batch.mamba_track_seqlens.to(
        device="cpu", dtype=torch.int64
    ) - forward_batch.extend_prefix_lens.to(device="cpu", dtype=torch.int64)

    checkpoint_counts = extend_seq_lens // checkpoint_every_n_tokens
    checkpoint_cu_starts = torch.zeros(checkpoint_counts.numel() + 1, dtype=torch.int64)
    checkpoint_cu_starts[1:] = torch.cumsum(checkpoint_counts, dim=0)

    use_checkpoint = track_mask & (relative_track_lens % checkpoint_every_n_tokens != 0)
    track_checkpoint_src = checkpoint_cu_starts[:-1][use_checkpoint] + (
        relative_track_lens[use_checkpoint] // checkpoint_every_n_tokens - 1
    )
    if track_checkpoint_src.numel() and track_checkpoint_src.min() < 0:
        raise ValueError("Tracked GDN state precedes the first FlashInfer checkpoint.")
    assert track_checkpoint_src.numel() == forward_metadata.track_ssm_h_dst.numel()

    forward_metadata.track_ssm_h_src = track_checkpoint_src.to(
        device, non_blocking=True
    )
    forward_metadata.state_checkpoint_cu_starts = checkpoint_cu_starts.to(
        device, non_blocking=True
    )
    forward_metadata.num_state_checkpoints = int(checkpoint_cu_starts[-1])
    forward_metadata.state_checkpoint_every_n_tokens = checkpoint_every_n_tokens


def _get_flashinfer_gdn_kernels():
    """Lazy import for FlashInfer GDN prefill, decode and verify (MTP) kernels.

    Returns (available, prefill_fn, mtp_fn, decode_fn, mtp_bf16_fn).
    """
    global _flashinfer_gdn_available, _flashinfer_chunk_gated_delta_rule, _flashinfer_gated_delta_rule_mtp, _flashinfer_gated_delta_rule_decode, _flashinfer_gated_delta_rule_mtp_bf16
    if _flashinfer_gdn_available is None:
        try:
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            from flashinfer.gdn_decode import (
                gated_delta_rule_decode_pretranspose,
                gated_delta_rule_mtp,
            )
            from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
                gated_delta_rule_mtp as gated_delta_rule_mtp_bf16,
            )
            from flashinfer.gdn_prefill import chunk_gated_delta_rule

            _flashinfer_chunk_gated_delta_rule = chunk_gated_delta_rule
            _flashinfer_gated_delta_rule_mtp = gated_delta_rule_mtp
            _flashinfer_gated_delta_rule_mtp_bf16 = gated_delta_rule_mtp_bf16
            _flashinfer_gated_delta_rule_decode = gated_delta_rule_decode_pretranspose
            _flashinfer_gdn_available = (
                is_cuda() and torch.cuda.get_device_capability()[0] >= 9
            )
            if _flashinfer_gdn_available:
                logger.info("FlashInfer GDN kernels loaded successfully")
        except (ImportError, RuntimeError) as e:
            logger.warning(f"FlashInfer GDN kernels not available: {e}")
            _flashinfer_gdn_available = False
            _flashinfer_gated_delta_rule_decode = None
    return (
        _flashinfer_gdn_available,
        _flashinfer_chunk_gated_delta_rule,
        _flashinfer_gated_delta_rule_mtp,
        _flashinfer_gated_delta_rule_decode,
        _flashinfer_gated_delta_rule_mtp_bf16,
    )


def is_flashinfer_gdn_prefill_available() -> bool:
    """Return whether the kernel loader can construct the prefill path."""
    available, prefill_fn, *_ = _get_flashinfer_gdn_kernels()
    return bool(available and prefill_fn is not None)


# ---------------------------------------------------------------------------
# Kernel implementation
# ---------------------------------------------------------------------------


class FlashInferGDNKernel(LinearAttnKernelBase):
    """FlashInfer kernel for GDN with K-last SSM state layout.

    SM90 (Hopper): decode uses gather/scatter; prefill and MTP verify supported.
    SM100 (Blackwell): decode uses gather/scatter; prefill and MTP verify supported.

    Requires flashinfer >= 0.6.14.
    """

    uses_state_checkpoints = True

    def __init__(self):
        (
            available,
            self._prefill_fn,
            self._mtp_fn,
            self._decode_fn,
            mtp_bf16_fn,
        ) = _get_flashinfer_gdn_kernels()

        if not available:
            raise RuntimeError(
                "FlashInfer GDN kernels are not available. "
                "Requires SM90+ and FlashInfer with GDN kernel support."
            )
        if self._decode_fn is None:
            raise RuntimeError("FlashInfer GDN decode kernel is unavailable.")

        sm_major = torch.cuda.get_device_capability()[0]
        self.use_state_pool = sm_major >= 10
        # The SM120 chunked-prefill kernel only accepts float32 initial
        # states; SM100 accepts the state-pool dtype directly.
        self._prefill_needs_fp32_state = sm_major >= 12
        self.supports_target_verify = sm_major in (9, 10)
        self._aligned_input_buffers: dict[tuple, torch.Tensor] = {}
        self._aligned_parameter_cache: dict[
            tuple, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._verify_intermediate_buffers: dict[tuple, torch.Tensor] = {}
        self._alignment_fallback_warned = False
        # FlashInfer writes through mutable state/workspace pointers, so misaligned
        # inputs must fall back to Triton rather than use temporary copies.
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        self._alignment_fallback_kernel = TritonGDNKernel()

        if sm_major == 9 and self._prefill_fn is None:
            raise RuntimeError("FlashInfer GDN prefill kernel is unavailable.")
        if self._mtp_fn is None:
            raise RuntimeError("FlashInfer GDN MTP (verify) kernel is unavailable.")

        if self.use_state_pool and mtp_bf16_fn is not None:
            # Adapt bf16 kernel to fp32 kernel interface so target_verify needs no branching.
            def _mtp_bf16_adapted(
                q,
                k,
                v,
                initial_state,
                initial_state_indices,
                A_log,
                a,
                dt_bias,
                b,
                use_qk_l2norm=True,
                **kw,
            ):
                out = mtp_bf16_fn(
                    A_log=A_log.float(),
                    a=a,
                    dt_bias=dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=q,
                    k=k,
                    v=v,
                    b=b,
                    initial_state_source=initial_state,
                    initial_state_indices=initial_state_indices,
                    use_qk_l2norm_in_kernel=use_qk_l2norm,
                    **kw,
                )
                return out, None

            self._mtp_fn = _mtp_bf16_adapted

        logger.info("Using FlashInfer GDN kernels")

    def _prepare_dynamic_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        # Reuse per-stream aligned scratch for uncommon read-only views; stable
        # addresses avoid allocator churn and remain safe for CUDA graph capture.
        if tensor.data_ptr() % _FLASHINFER_GDN_ALIGNMENT == 0:
            return tensor

        stream_key = (
            torch.cuda.current_stream(tensor.device).cuda_stream
            if tensor.device.type == "cuda"
            else None
        )
        key = (
            name,
            tensor.device,
            tensor.dtype,
            tuple(tensor.shape),
            stream_key,
        )
        aligned = self._aligned_input_buffers.get(key)
        if aligned is None:
            aligned = _empty_aligned_like(tensor)
            self._aligned_input_buffers[key] = aligned
        aligned.copy_(tensor)
        return aligned

    def _prepare_parameter(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        key = (
            name,
            id(tensor),
            dtype,
        )
        cached_entry = self._aligned_parameter_cache.get(key)
        if cached_entry is not None and cached_entry[0] is tensor:
            return cached_entry[1]

        prepared = tensor.detach().reshape(-1)
        if dtype is not None:
            prepared = prepared.to(dtype=dtype, copy=False)
        if (
            not prepared.is_contiguous()
            or prepared.data_ptr() % _FLASHINFER_GDN_ALIGNMENT
        ):
            aligned = _empty_aligned_like(prepared)
            aligned.copy_(prepared)
            prepared = aligned
        # Retaining the source also prevents a recycled Python id from
        # colliding with an older cache entry.
        self._aligned_parameter_cache[key] = (tensor, prepared)
        return prepared

    def _prepare_gate_parameters(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        *,
        A_log_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._prepare_parameter("A_log", A_log, dtype=A_log_dtype),
            self._prepare_parameter("dt_bias", dt_bias),
        )

    def _mutable_inputs_are_aligned(
        self, *named_tensors: tuple[str, Optional[torch.Tensor]]
    ) -> bool:
        for name, tensor in named_tensors:
            if tensor is None or tensor.data_ptr() % _FLASHINFER_GDN_ALIGNMENT == 0:
                continue
            if not self._alignment_fallback_warned:
                logger.warning(
                    "FlashInfer GDN mutable buffer %r has data_ptr %d "
                    "(mod 32 = %d); falling back to Triton for this call.",
                    name,
                    tensor.data_ptr(),
                    tensor.data_ptr() % _FLASHINFER_GDN_ALIGNMENT,
                )
                self._alignment_fallback_warned = True
            return False
        return True

    def _prepare_verify_intermediate_buffer(
        self,
        intermediate_states_buffer: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, bool]:
        # FlashInfer requires exact capture B, which may exceed the pool-scoped
        # buffer; padded tiers use stable scratch and copy owned rows back.
        direct = intermediate_states_buffer[:batch_size]
        if direct.shape[0] == batch_size:
            return direct, False

        stream_key = (
            torch.cuda.current_stream(intermediate_states_buffer.device).cuda_stream
            if intermediate_states_buffer.device.type == "cuda"
            else None
        )
        shape = (batch_size, *intermediate_states_buffer.shape[1:])
        key = (
            intermediate_states_buffer.device,
            intermediate_states_buffer.dtype,
            shape,
            stream_key,
        )
        scratch = self._verify_intermediate_buffers.get(key)
        if scratch is None:
            template = intermediate_states_buffer.new_empty(shape)
            scratch = _empty_aligned_like(template)
            self._verify_intermediate_buffers[key] = scratch
        return scratch, True

    # ---- decode ----

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if not self._mutable_inputs_are_aligned(("ssm_states", ssm_states)):
            return self._alignment_fallback_kernel.decode(
                q,
                k,
                v,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                **kwargs,
            )

        batch_size = cache_indices.shape[0]
        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        query_fi = q.view(batch_size, 1, num_heads, head_k_dim)
        key_fi = k.view(batch_size, 1, num_heads, head_k_dim)
        value_fi = v.view(batch_size, 1, num_v_heads, head_v_dim)
        a_fi = a.view(batch_size, 1, num_v_heads)
        b_fi = b.view(batch_size, 1, num_v_heads)
        query_fi = self._prepare_dynamic_input("decode_q", query_fi)
        key_fi = self._prepare_dynamic_input("decode_k", key_fi)
        value_fi = self._prepare_dynamic_input("decode_v", value_fi)
        a_fi = self._prepare_dynamic_input("decode_a", a_fi)
        b_fi = self._prepare_dynamic_input("decode_b", b_fi)
        A_log_fi, dt_bias_fi = self._prepare_gate_parameters(
            A_log,
            dt_bias,
            # Preserve the original backend contract: the SM100 state-pool
            # kernel consumes float32 A_log, while SM90 uses the source dtype.
            A_log_dtype=torch.float32 if self.use_state_pool else None,
        )

        if self.use_state_pool:
            cache_indices_fi = self._prepare_dynamic_input(
                "decode_cache_indices", cache_indices
            )
            output_fi, _ = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=None,
                A_log=A_log_fi,
                a=a_fi,
                dt_bias=dt_bias_fi,
                b=b_fi,
                use_qk_l2norm=True,
                initial_state=ssm_states,
                initial_state_indices=cache_indices_fi,
            )
        else:
            # TODO: Once FlashInfer PR#2521 is merged for SM90, gather/scatter
            # will no longer be needed here.
            state_batch = ssm_states[cache_indices]
            output_fi, new_state = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=state_batch,
                A_log=A_log_fi,
                a=a_fi,
                dt_bias=dt_bias_fi,
                b=b_fi,
                scale=None,
                output=None,
                use_qk_l2norm=True,
            )
            ssm_states[cache_indices] = new_state

        return output_fi.view(1, batch_size, num_v_heads, head_v_dim)

    # ---- extend (prefill) ----

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
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        **kwargs,
    ) -> tuple:
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        q_fi = l2norm_fwd(q[0].contiguous())
        k_fi = l2norm_fwd(k[0].contiguous())
        v_fi = v[0].contiguous()

        output = kwargs.get("output")
        output_fi = None
        if output is not None:
            expected_output_shape = (
                1,
                total_seq_len,
                num_v_heads,
                head_v_dim,
            )
            if tuple(output.shape) != expected_output_shape:
                raise ValueError(
                    "FlashInfer GDN prefill output shape mismatch: "
                    f"expected {expected_output_shape}, got {tuple(output.shape)}"
                )
            if output.dtype != v.dtype or output.device != v.device:
                raise ValueError(
                    "FlashInfer GDN prefill output must match v dtype/device, "
                    f"got output=({output.dtype}, {output.device}) and "
                    f"v=({v.dtype}, {v.device})"
                )
            output_fi = output[0]
            if not output_fi.is_contiguous():
                raise ValueError("FlashInfer GDN prefill output must be contiguous")

        # g (alpha) and beta: [1, seq, HV] -> [seq, HV], float32 for FlashInfer
        alpha_fi = torch.exp(g[0].to(torch.float32))
        beta_fi = beta[0].to(torch.float32)

        if self.use_state_pool:
            # Negative indices (e.g. -1) are padding markers for slots not yet
            # assigned to a real sequence; clamp them to 0 (the reserved dummy
            # slot) so the FlashInfer kernel never reads out-of-bounds state.
            ssm_cache_indices = cache_indices.clamp(min=0).to(torch.int64)
            initial_state_fi = (
                ssm_states[ssm_cache_indices].to(torch.float32)
                if self._prefill_needs_fp32_state
                else ssm_states[ssm_cache_indices].contiguous()
            )
            cu_seqlens = query_start_loc.to(torch.int64)  # kernel requires int64
        else:
            # SM90: preserve original negative-index handling (remap to last slot).
            ssm_cache_indices = torch.where(
                cache_indices >= 0,
                cache_indices,
                ssm_states.shape[0] - 1,
            ).to(torch.int64)
            # State must be float32; kernel requires int64 cu_seqlens.
            initial_state_fi = ssm_states[ssm_cache_indices].to(torch.float32)
            cu_seqlens = query_start_loc.to(torch.int64)

        # Keep final state and checkpoints in the same kernel state dtype.
        output_state_fi = torch.empty_like(initial_state_fi)
        state_checkpoints = (
            initial_state_fi.new_empty(
                (num_state_checkpoints, *initial_state_fi.shape[1:])
            )
            if num_state_checkpoints > 0
            else None
        )
        output_fi, output_state_fi = self._prefill_fn(
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=alpha_fi,
            beta=beta_fi,
            scale=None,
            initial_state=initial_state_fi,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output_fi,
            output_state=output_state_fi,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=state_checkpoint_cu_starts,
            checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
        )

        # Write back state to pool
        ssm_states.index_copy_(
            0,
            ssm_cache_indices,
            output_state_fi.to(ssm_states.dtype),
        )

        # Output: [seq, HV, V] -> [1, seq, HV, V]
        core_attn_out = output_fi.view(1, total_seq_len, num_v_heads, head_v_dim)

        # Match Triton's [1, checkpoints, H, V, K] intermediate-state layout.
        h = state_checkpoints.unsqueeze(0) if state_checkpoints is not None else None
        return core_attn_out, None, h

    # ---- target_verify (MTP) ----

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # MTP verify using FlashInfer gated_delta_rule_mtp kernel (SM90 + SM100+).
        if retrieve_parent_token is not None:
            raise RuntimeError(
                "FlashInfer GDN verify kernel only supports topk=1 "
                "(retrieve_parent_token must be None)."
            )

        seq_len = q.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size

        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        if a is None or b is None or A_log is None or dt_bias is None:
            raise RuntimeError(
                "FlashInfer GDN MTP kernel requires a, b, A_log, dt_bias."
            )

        intermediate_states_buffer_mtp = intermediate_states_buffer
        copy_verify_intermediate_back = False
        if self.use_state_pool and intermediate_states_buffer is not None:
            # The SM100 bf16 MTP kernel indexes this scratch buffer by the
            # per-call batch id, while SGLang's speculative state cache is
            # Graph padding can exceed the pool-scoped scratch; use exact-B storage
            # and copy owned rows back before post-verify commit reads the pool.
            (
                intermediate_states_buffer_mtp,
                copy_verify_intermediate_back,
            ) = self._prepare_verify_intermediate_buffer(
                intermediate_states_buffer, batch_size
            )
        if not self._mutable_inputs_are_aligned(
            ("ssm_states", ssm_states),
            ("intermediate_states_buffer", intermediate_states_buffer_mtp),
        ):
            return self._alignment_fallback_kernel.target_verify(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=cache_steps,
                retrieve_parent_token=retrieve_parent_token,
                **kwargs,
            )

        query_mtp = self._prepare_dynamic_input(
            "verify_q",
            q.view(batch_size, draft_token_num, num_heads, head_k_dim),
        )
        key_mtp = self._prepare_dynamic_input(
            "verify_k",
            k.view(batch_size, draft_token_num, num_heads, head_k_dim),
        )
        value_mtp = self._prepare_dynamic_input(
            "verify_v",
            v.view(batch_size, draft_token_num, num_v_heads, head_v_dim),
        )

        a_mtp = self._prepare_dynamic_input(
            "verify_a", a.view(batch_size, draft_token_num, num_v_heads)
        )
        b_mtp = self._prepare_dynamic_input(
            "verify_b", b.view(batch_size, draft_token_num, num_v_heads)
        )
        A_log_fi, dt_bias_fi = self._prepare_gate_parameters(A_log, dt_bias)
        cache_indices_fi = self._prepare_dynamic_input(
            "verify_cache_indices", cache_indices
        )

        output_fi, _ = self._mtp_fn(
            q=query_mtp,
            k=key_mtp,
            v=value_mtp,
            initial_state=ssm_states,
            initial_state_indices=cache_indices_fi,
            A_log=A_log_fi,
            a=a_mtp,
            dt_bias=dt_bias_fi,
            b=b_mtp,
            scale=None,
            output=None,
            intermediate_states_buffer=intermediate_states_buffer_mtp,
            disable_state_update=True,
            use_qk_l2norm=True,
        )

        if copy_verify_intermediate_back:
            intermediate_states_buffer.copy_(
                intermediate_states_buffer_mtp[: intermediate_states_buffer.shape[0]]
            )

        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)
