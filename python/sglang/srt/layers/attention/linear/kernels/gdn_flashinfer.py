"""FlashInfer-based kernels for GDN (Gated Delta Network) linear attention.

Both SM90 and SM100 use the same pool layout: [pool, HV, V, K] (K-last).

SM90 (Hopper): full support — decode, prefill, MTP.  State dtype: fp32.
SM100 (Blackwell): full support — decode, prefill, MTP.

Requires flashinfer >= 0.6.7.
"""

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd_qk
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
    unwrap_direct_write_out,
)
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import for FlashInfer GDN kernels
# ---------------------------------------------------------------------------
_flashinfer_gdn_available: Optional[bool] = None
_flashinfer_chunk_gated_delta_rule = None
_flashinfer_gated_delta_rule_mtp = None
_flashinfer_gated_delta_rule_decode = None
_flashinfer_gated_delta_rule_mtp_bf16 = None


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

    Requires flashinfer >= 0.6.7.
    """

    # extend() consumes alpha = exp(g), produced by fused_gdn_gating(exp_gate=True).
    extend_expects_exp_gate = True
    supports_prenormed_extend = True

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

    def build_extend_prep(
        self,
        *,
        head_k_dim: int,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        ssm_states: torch.Tensor,
        total_seq_len: int,
    ) -> tuple:
        """Compute the layer-invariant extend metadata once per forward.

        The pool-gather indices depend only on per-request cache slots,
        identical across all GDN layers of a forward, so forward_extend builds
        this once and reuses it for every per-layer extend() call. Must stay
        bit-identical to extend()'s prep=None recompute.
        """
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
        return (ssm_cache_indices,)

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
        out: Optional[torch.Tensor] = None,
        prep: Optional[tuple] = None,
        no_prefix: bool = False,
        **kwargs,
    ) -> tuple:
        q_fi, k_fi = l2norm_fwd_qk(q[0].contiguous(), k[0].contiguous())

        # g is already alpha = exp(g) (see extend_expects_exp_gate); the
        # .to(float32) calls are normally no-op guards.
        return self._extend_core(
            q_fi=q_fi,
            k_fi=k_fi,
            v_fi=v[0].contiguous(),
            alpha_fi=g[0].to(torch.float32),
            beta_fi=beta[0].to(torch.float32),
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            out=out,
            prep=prep,
            no_prefix=no_prefix,
        )

    def extend_prenormed(
        self,
        q_normed: torch.Tensor,
        k_normed: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        prep: Optional[tuple] = None,
        no_prefix: bool = False,
        **kwargs,
    ) -> tuple:
        """Glue-kernel entry point: q/k are ALREADY L2-normalized and ``g`` is
        already alpha = exp(g) fp32 (gdn_prefill_glue with exp_gate=True). No
        normalization or gate transform happens here — passing raw q/k or
        log-space g silently corrupts outputs, hence the loud asserts."""
        assert (
            g.dtype == torch.float32 and beta.dtype == torch.float32
        ), "extend_prenormed expects fp32 alpha/beta straight from the glue kernel"
        q_fi = q_normed[0]
        k_fi = k_normed[0]
        v_fi = v[0]
        assert q_fi.is_contiguous() and k_fi.is_contiguous() and v_fi.is_contiguous()
        return self._extend_core(
            q_fi=q_fi,
            k_fi=k_fi,
            v_fi=v_fi,
            alpha_fi=g[0],
            beta_fi=beta[0],
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            out=out,
            prep=prep,
            no_prefix=no_prefix,
        )

    def _extend_core(
        self,
        *,
        q_fi: torch.Tensor,
        k_fi: torch.Tensor,
        v_fi: torch.Tensor,
        alpha_fi: torch.Tensor,
        beta_fi: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        out: Optional[torch.Tensor],
        prep: Optional[tuple],
        no_prefix: bool,
    ) -> tuple:
        total_seq_len = q_fi.shape[0]
        num_v_heads = v_fi.shape[1]
        head_v_dim = v_fi.shape[2]

        if prep is None:
            # Fallback for direct extend() callers (e.g. unit tests); the hot
            # path passes prep built once per forward.
            prep = self.build_extend_prep(
                head_k_dim=q_fi.shape[-1],
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                ssm_states=ssm_states,
                total_seq_len=total_seq_len,
            )
        (ssm_cache_indices,) = prep

        output_buf = unwrap_direct_write_out(
            out, expected_shape=(1, total_seq_len, num_v_heads, head_v_dim)
        )

        # When no request in the batch has a prefix, skip the pool gather and
        # let the kernel zero-seed via initial_state=None. Bit-identical: freed
        # pool slots are cleared, so the gather would materialize zeros anyway
        # (and this also insulates fresh prefills from any stale slot content).
        if no_prefix:
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
