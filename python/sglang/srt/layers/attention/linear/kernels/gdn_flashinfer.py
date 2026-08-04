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
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


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

    checkpoint_every_n_tokens = get_server_args().mamba_cache_chunk_size
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
        self.supports_target_verify = sm_major in (9, 10)
        # See _warm_initial_state_variant: the prefix-free fast path selects a
        # different CuTe compile-cache entry than the gathered-state path, and
        # only the former is exercised before traffic starts.
        self._warmed_initial_state_variant = False

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

        if self.use_state_pool:
            output_fi, _ = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=None,
                A_log=A_log.detach().float(),
                a=a_fi,
                dt_bias=dt_bias.detach(),
                b=b_fi,
                use_qk_l2norm=True,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
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
                A_log=A_log.detach(),
                a=a_fi,
                dt_bias=dt_bias.detach(),
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

        # g (alpha) and beta: [1, seq, HV] -> [seq, HV], float32 for FlashInfer
        alpha_fi = torch.exp(g[0].to(torch.float32))
        beta_fi = beta[0].to(torch.float32)

        if self.use_state_pool:
            # Negative indices (e.g. -1) are padding markers for slots not yet
            # assigned to a real sequence; clamp them to 0 (the reserved dummy
            # slot) so the FlashInfer kernel never reads out-of-bounds state.
            ssm_cache_indices = cache_indices.clamp(min=0).to(torch.int64)
            initial_state_fi = ssm_states[ssm_cache_indices].contiguous()
            cu_seqlens = query_start_loc  # already int32
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

    @staticmethod
    def can_use_fused_prefill(
        mixed_qkv: torch.Tensor,
        *,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        head_q_dim: int,
        head_k_dim: int,
        head_v_dim: int,
    ) -> bool:
        total_dim = (
            num_q_heads * head_q_dim
            + num_k_heads * head_k_dim
            + num_v_heads * head_v_dim
        )
        return (
            mixed_qkv.shape[1] == total_dim
            and mixed_qkv.stride(1) == 1
            and num_q_heads == num_k_heads
            and head_q_dim == head_k_dim == 128
            and num_v_heads > 0
            and num_v_heads & (num_v_heads - 1) == 0
        )

    def _warm_initial_state_variant(
        self,
        q_fi: torch.Tensor,
        k_fi: torch.Tensor,
        v_fi: torch.Tensor,
        alpha_fi: torch.Tensor,
        beta_fi: torch.Tensor,
        ssm_states: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> None:
        """Compile FlashInfer's ``use_initial_state=True`` CuTe variant early.

        ``initial_state=None`` (the prefix-free fast path) is a kernel-codegen
        parameter, so it selects a different entry in FlashInfer's compile
        cache than the gathered-state path. Server warmup and prefill-graph
        capture only ever build prefix-free batches (capture pins
        ``extend_prefix_lens_cpu`` to zeros, and prefix-bearing extends are
        forced eager when the model has MHA companion layers), so without this
        the gathered-state variant would compile on the first chunked
        continuation *during serving* - a ~3.6 s stall that lands in the middle
        of a benchmark or a production ramp.

        The token extent is compile-dynamic, so one short slice is enough to
        populate the entry; every other key component (dtypes, head counts,
        ``store_final_state``) is taken from the real tensors so the warmed key
        matches the one serving will ask for.

        Note: on SM90/SM120 FlashInfer additionally routes between a CP and a
        non-CP kernel on ``num_seqs * num_v_heads`` vs. SM count, and both key
        on the initial state. The single-sequence warm below populates the CP
        entry (what a B=1 chunked continuation uses); large-batch continuations
        still compile their non-CP entry on first use. SM100 has no such
        routing.
        """
        if self._warmed_initial_state_variant:
            return
        # cute.compile does host-side work and allocates: it must never run
        # while a CUDA graph capture is in progress.
        if torch.cuda.is_current_stream_capturing():
            return
        self._warmed_initial_state_variant = True
        try:
            n = min(int(q_fi.shape[0]), 64)
            state = ssm_states.new_zeros((1,) + ssm_states.shape[1:])
            extra = {}
            if self.use_state_pool:
                extra["output_state"] = torch.empty_like(state)
            else:
                state = state.to(torch.float32)
            # The dtype must follow query_start_loc: cu_seqlens' element type is
            # baked into the compiled artifact but is NOT part of FlashInfer's
            # cache key, so a literal dtype here would silently warm the wrong
            # entry.
            cu = n * torch.arange(
                2, device=query_start_loc.device, dtype=query_start_loc.dtype
            )
            self._prefill_fn(
                q=q_fi[:n],
                k=k_fi[:n],
                v=v_fi[:n],
                g=alpha_fi[:n],
                beta=beta_fi[:n],
                scale=None,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu if self.use_state_pool else cu.to(torch.int64),
                use_qk_l2norm_in_kernel=False,
                **extra,
            )
        except Exception as exc:  # pragma: no cover - warm-only, non-fatal
            logger.warning(
                "FlashInfer GDN initial-state variant warm-up failed "
                f"({exc}); it will compile on first use instead."
            )

    def extend_fused(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        num_q_heads: int,
        num_k_heads: int,
        num_v_heads: int,
        head_q_dim: int,
        head_k_dim: int,
        head_v_dim: int,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        no_prefix: bool = False,
        **kwargs,
    ) -> tuple:
        """Run the fused FlashInfer prefill path."""
        from sglang.jit_kernel.triton.gdn_prefill_fused import gdn_prefill_fused

        q_fi, k_fi, v_fi, alpha_fi, beta_fi = gdn_prefill_fused(
            mixed_qkv,
            a,
            b,
            A_log,
            dt_bias,
            num_qk_heads=num_q_heads,
            num_v_heads=num_v_heads,
            head_qk_dim=head_q_dim,
            head_v_dim=head_v_dim,
        )
        q_fi = q_fi[0]
        k_fi = k_fi[0]
        v_fi = v_fi[0]
        alpha_fi = alpha_fi[0]
        beta_fi = beta_fi[0]

        total_seq_len = q_fi.shape[0]
        num_v_heads = v_fi.shape[1]
        head_v_dim = v_fi.shape[2]

        if self.use_state_pool:
            # Negative indices (e.g. -1) are padding markers for slots not yet
            # assigned to a real sequence; clamp them to 0 (the reserved dummy
            # slot) so the FlashInfer kernel never reads out-of-bounds state.
            ssm_cache_indices = cache_indices.clamp(min=0).to(torch.int64)
        else:
            # SM90: negative (pad) indices remap to the last slot, the reserved
            # sentinel.
            ssm_cache_indices = torch.where(
                cache_indices >= 0,
                cache_indices,
                ssm_states.shape[0] - 1,
            ).to(torch.int64)

        if out is not None:
            expected_shape = (1, total_seq_len, num_v_heads, head_v_dim)
            assert (
                out.shape == expected_shape
            ), f"direct-write out buffer {tuple(out.shape)} != expected {expected_shape}"
        output_buf = out.squeeze(0) if out is not None else None

        # When no request in the batch has a prefix, skip the pool gather and
        # let the kernel zero-seed via initial_state=None. Bit-identical: freed
        # pool slots are cleared, so the gather would materialize zeros anyway
        # (and this also insulates fresh prefills from any stale slot content).
        if no_prefix:
            if not self._warmed_initial_state_variant:
                self._warm_initial_state_variant(
                    q_fi=q_fi,
                    k_fi=k_fi,
                    v_fi=v_fi,
                    alpha_fi=alpha_fi,
                    beta_fi=beta_fi,
                    ssm_states=ssm_states,
                    query_start_loc=query_start_loc,
                )
            initial_state_fi = None
        else:
            gathered = ssm_states[ssm_cache_indices]
            # SM90 state must be float32; SM100 keeps the pool's bf16.
            initial_state_fi = (
                gathered.contiguous()
                if self.use_state_pool
                else gathered.to(torch.float32)
            )

        extra = {}
        if self.use_state_pool:
            # Pre-allocate bf16 output_state so the kernel compiles and writes
            # the bf16 state path directly, avoiding a fp32 allocation and a
            # subsequent fp32->bf16 conversion in the scatter step.
            num_seqs = query_start_loc.numel() - 1
            extra["output_state"] = ssm_states.new_empty(
                (num_seqs,) + ssm_states.shape[1:]
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
            cu_seqlens=(
                query_start_loc  # already int32
                if self.use_state_pool
                else query_start_loc.to(torch.int64)
            ),
            use_qk_l2norm_in_kernel=False,
            output=output_buf,
            **extra,
        )

        # Write back state to pool
        ssm_states.index_copy_(
            0,
            ssm_cache_indices,
            output_state_fi.to(ssm_states.dtype),
        )

        # Output: [seq, HV, V] -> [1, seq, HV, V]. When out= was honored this is
        # a view of the caller's buffer, so its data_ptr matches and the caller
        # skips its copy.
        core_attn_out = output_fi.view(1, total_seq_len, num_v_heads, head_v_dim)

        # Return (output, last_recurrent_state, h) to match Triton kernel interface.
        # h=None since FlashInfer doesn't provide intermediate states.
        return core_attn_out, None, None

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

        query_mtp = q.view(batch_size, draft_token_num, num_heads, head_k_dim)
        key_mtp = k.view(batch_size, draft_token_num, num_heads, head_k_dim)
        value_mtp = v.view(batch_size, draft_token_num, num_v_heads, head_v_dim)

        if a is None or b is None or A_log is None or dt_bias is None:
            raise RuntimeError(
                "FlashInfer GDN MTP kernel requires a, b, A_log, dt_bias."
            )

        a_mtp = a.view(batch_size, draft_token_num, num_v_heads)
        b_mtp = b.view(batch_size, draft_token_num, num_v_heads)

        intermediate_states_buffer_mtp = intermediate_states_buffer
        if self.use_state_pool and intermediate_states_buffer is not None:
            # The SM100 bf16 MTP kernel indexes this scratch buffer by the
            # per-call batch id, while SGLang's speculative state cache is
            # pool-scoped and may include an extra dummy slot.
            intermediate_states_buffer_mtp = intermediate_states_buffer[:batch_size]

        output_fi, _ = self._mtp_fn(
            q=query_mtp,
            k=key_mtp,
            v=value_mtp,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            A_log=A_log.detach(),
            a=a_mtp,
            dt_bias=dt_bias.detach(),
            b=b_mtp,
            scale=None,
            output=None,
            intermediate_states_buffer=intermediate_states_buffer_mtp,
            disable_state_update=True,
            use_qk_l2norm=True,
        )

        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)
