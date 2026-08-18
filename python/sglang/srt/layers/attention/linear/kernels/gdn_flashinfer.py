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

# ---------------------------------------------------------------------------
# Lazy import for FlashInfer GDN kernels
# ---------------------------------------------------------------------------
_flashinfer_gdn_available: Optional[bool] = None
_flashinfer_chunk_gated_delta_rule = None
_flashinfer_gated_delta_rule_mtp = None
_flashinfer_gated_delta_rule_decode = None
_flashinfer_gated_delta_rule_mtp_bf16 = None
_cake_gdn_decode_api = None
_cake_gdn_decode_api_checked = False


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


def _get_cake_gdn_decode_api():
    """Return the optional public Cake GDN loader without compiling a kernel."""

    global _cake_gdn_decode_api, _cake_gdn_decode_api_checked
    if not _cake_gdn_decode_api_checked:
        _cake_gdn_decode_api_checked = True
        try:
            from flashinfer.jit import cake_gdn_noncp_decode

            _cake_gdn_decode_api = cake_gdn_noncp_decode
        except ImportError:
            _cake_gdn_decode_api = None
    return _cake_gdn_decode_api


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
        self._cake_gdn_api = None
        self._cake_gdn_arch = None
        self._cake_gdn_entries = {}
        self._cake_gdn_outputs = {}
        self._cake_gdn_dt_bias_fp32 = {}
        self._cake_gdn_logged_routes = set()

        if self.use_state_pool:
            cake_gdn_api = _get_cake_gdn_decode_api()
            if cake_gdn_api is not None:
                capability = torch.cuda.get_device_capability()
                try:
                    self._cake_gdn_arch = cake_gdn_api.arch_for_compute_capability(
                        *capability
                    )
                except cake_gdn_api.CakeGDNUnsupportedError:
                    pass
                else:
                    self._cake_gdn_api = cake_gdn_api

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

    @staticmethod
    def _is_cake_strided_layout(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        num_q_heads: int,
        num_v_heads: int,
    ) -> bool:
        """Match the dense-inner, runtime-strided public Cake tensor ABI."""

        return (
            tuple(q.shape) == (batch_size, seq_len, num_q_heads, 128)
            and tuple(k.shape) == (batch_size, seq_len, num_q_heads, 128)
            and tuple(v.shape) == (batch_size, seq_len, num_v_heads, 128)
            and tuple(a.shape) == (batch_size, seq_len, num_v_heads)
            and tuple(b.shape) == (batch_size, seq_len, num_v_heads)
            and state.ndim == 4
            and tuple(state.shape[1:]) == (num_v_heads, 128, 128)
            and tuple(q.stride()[2:]) == (128, 1)
            and tuple(k.stride()[2:]) == (128, 1)
            and tuple(v.stride()[2:]) == (128, 1)
            and q.stride(1) >= num_q_heads * 128
            and k.stride(1) >= num_q_heads * 128
            and v.stride(1) >= num_v_heads * 128
            and q.stride(0) >= seq_len * q.stride(1)
            and k.stride(0) >= seq_len * k.stride(1)
            and v.stride(0) >= seq_len * v.stride(1)
            and a.stride(2) == 1
            and b.stride(2) == 1
            and a.stride(1) >= num_v_heads
            and b.stride(1) >= num_v_heads
            and a.stride(0) >= seq_len * a.stride(1)
            and b.stride(0) >= seq_len * b.stride(1)
            and tuple(state.stride()[1:]) == (128 * 128, 128, 1)
            and state.stride(0) >= num_v_heads * 128 * 128
        )

    def _cake_output_buffer(
        self,
        q: torch.Tensor,
        *,
        layer_id: int,
        batch_size: int,
        seq_len: int,
        num_v_heads: int,
    ) -> torch.Tensor:
        stream_handle = int(torch.cuda.current_stream(q.device).cuda_stream)
        key = (
            q.device.index,
            stream_handle,
            layer_id,
            batch_size,
            seq_len,
            num_v_heads,
        )
        output = self._cake_gdn_outputs.get(key)
        if output is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Cake GDN output buffer was not prepared before CUDA Graph capture"
                )
            output = torch.empty(
                batch_size,
                seq_len,
                num_v_heads,
                128,
                dtype=torch.bfloat16,
                device=q.device,
            )
            self._cake_gdn_outputs[key] = output
        return output

    def _cake_fp32_dt_bias(
        self,
        dt_bias: torch.Tensor,
        *,
        layer_id: int,
    ) -> torch.Tensor:
        """Return persistent FP32 model storage prepared before graph capture."""

        if dt_bias.dtype == torch.float32:
            return dt_bias
        key = (dt_bias.device.index, layer_id, dt_bias.data_ptr())
        converted = self._cake_gdn_dt_bias_fp32.get(key)
        if converted is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Cake GDN FP32 dt_bias was not prepared before CUDA Graph capture"
                )
            converted = dt_bias.detach().float()
            self._cake_gdn_dt_bias_fp32[key] = converted
        return converted

    def _try_cake_decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
        state_indices: torch.Tensor,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        b: torch.Tensor,
        layer_id: Optional[int],
        disable_state_update: bool,
        intermediate_state: Optional[torch.Tensor],
        cache_steps: int,
    ) -> Optional[torch.Tensor]:
        """Launch an exact promoted Cake row, or return None when unsupported."""

        if self._cake_gdn_api is None or layer_id is None:
            return None
        batch_size, seq_len, num_q_heads, head_size = q.shape
        num_v_heads = v.shape[2]
        tensors = (q, k, v, state, state_indices, A_log, a, dt_bias, b)
        if (
            head_size != 128
            or v.shape[-1] != 128
            or any(tensor.device != q.device for tensor in tensors)
            or any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, state, a, b))
            or A_log.dtype != torch.float32
            or dt_bias.dtype not in (torch.bfloat16, torch.float32)
            or tuple(A_log.shape) != (num_v_heads,)
            or tuple(dt_bias.shape) != (num_v_heads,)
            or not A_log.is_contiguous()
            or not dt_bias.is_contiguous()
            or state_indices.dtype != torch.int32
            or tuple(state_indices.shape) != (batch_size,)
            or not state_indices.is_contiguous()
            or not self._is_cake_strided_layout(
                q,
                k,
                v,
                a,
                b,
                state,
                batch_size=batch_size,
                seq_len=seq_len,
                num_q_heads=num_q_heads,
                num_v_heads=num_v_heads,
            )
        ):
            return None

        cache_intermediate_states = intermediate_state is not None
        if cache_intermediate_states:
            expected_cache_shape = (
                batch_size,
                cache_steps,
                num_v_heads,
                128,
                128,
            )
            if (
                tuple(intermediate_state.shape) != expected_cache_shape
                or intermediate_state.dtype != torch.bfloat16
                or intermediate_state.device != q.device
                or not intermediate_state.is_contiguous()
            ):
                return None

        try:
            route = self._cake_gdn_api.select_cake_gdn_decode_variant(
                arch=self._cake_gdn_arch,
                batch_size=batch_size,
                io_dtype="bfloat16",
                state_dtype="bfloat16",
                head_size=128,
                layout="pretranspose",
                num_k_heads=num_q_heads,
                num_q_heads=num_q_heads,
                num_v_heads=num_v_heads,
                scale=128**-0.5,
                seq_len=seq_len,
                use_qk_l2norm=True,
                strided_inputs=True,
                disable_state_update=disable_state_update,
                cache_intermediate_states=cache_intermediate_states,
                cache_steps=cache_steps,
            )
        except self._cake_gdn_api.CakeGDNUnsupportedError:
            return None

        dt_bias = self._cake_fp32_dt_bias(dt_bias, layer_id=layer_id)

        entry = self._cake_gdn_entries.get(route.variant_name)
        if entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Cake GDN kernel was not loaded before CUDA Graph capture"
                )
            entry = self._cake_gdn_api.load_cake_gdn_kernel(
                route.variant_name, self._cake_gdn_arch
            )
            self._cake_gdn_entries[route.variant_name] = entry

        output = self._cake_output_buffer(
            q,
            layer_id=layer_id,
            batch_size=batch_size,
            seq_len=seq_len,
            num_v_heads=num_v_heads,
        )
        state_heads = batch_size * num_v_heads
        tile_v = (
            16
            if route.route_id.endswith(".tile16_fullwarp")
            else 128
            if state_heads >= 1024
            else 64
            if state_heads >= 512
            else 32
        )
        entry(
            q,
            k,
            v,
            state,
            A_log,
            a,
            dt_bias,
            b,
            output,
            intermediate_state if intermediate_state is not None else output,
            state_indices,
            state_indices,
            batch_size * num_v_heads * (128 // tile_v),
            1,
            1,
        )
        if route.route_id not in self._cake_gdn_logged_routes:
            self._cake_gdn_logged_routes.add(route.route_id)
            logger.info("Using %s", route.route_id)
        return output.view(1, batch_size * seq_len, num_v_heads, 128)

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
            output_cake = self._try_cake_decode(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=ssm_states,
                state_indices=cache_indices,
                A_log=A_log.detach(),
                a=a_fi,
                dt_bias=dt_bias.detach(),
                b=b_fi,
                layer_id=kwargs.get("layer_id"),
                disable_state_update=False,
                intermediate_state=None,
                cache_steps=0,
            )
            if output_cake is not None:
                return output_cake
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

        output_cake = self._try_cake_decode(
            q=query_mtp,
            k=key_mtp,
            v=value_mtp,
            state=ssm_states,
            state_indices=cache_indices,
            A_log=A_log.detach(),
            a=a_mtp,
            dt_bias=dt_bias.detach(),
            b=b_mtp,
            layer_id=kwargs.get("layer_id"),
            disable_state_update=True,
            intermediate_state=intermediate_states_buffer_mtp,
            cache_steps=cache_steps,
        )
        if output_cake is not None:
            return output_cake

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
