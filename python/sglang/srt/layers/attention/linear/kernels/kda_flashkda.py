from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

# FlashKDA chunk size. Sequences shorter than this fall back to Triton.
_FLASHKDA_CHUNK_SIZE = 64

# Legacy/default policy ceiling.  Validated device/shape profiles below may
# use a different crossover; this is not a hard limit of the FlashKDA kernel.
_FLASHKDA_MAX_SEQ_LEN = 2048

# H200 serving measurements show that FlashKDA's head-parallel K2 is a win for
# wide local-head shards, while Triton's time-parallel kernel is substantially
# faster at Q=2048 for narrow TP shards.  The wider shard also keeps enough K2
# CTAs resident to extend the useful FlashKDA window through the largest GLM
# chunk-prefill bucket we validate.
#
# Do not use the reported device name for this profile: some environments
# report H200 hardware under a different name.  The tight memory window, exact
# Hopper capability, and SM count distinguish the measured ~140 GiB H200 from
# H100 variants while avoiding extrapolation to other devices.
_GIB = 1 << 30
_FLASHKDA_H200_COMPUTE_CAPABILITY = (9, 0)
_FLASHKDA_H200_SM_COUNT = 132
_FLASHKDA_H200_MIN_TOTAL_MEMORY = 135 * _GIB
_FLASHKDA_H200_MAX_TOTAL_MEMORY = 145 * _GIB
_FLASHKDA_H200_LONG_HEADS = 64
_FLASHKDA_H200_TRITON_HEADS = frozenset({8, 12, 16, 32})
_FLASHKDA_H200_LONG_MAX_SEQ_LEN = 8192


def _is_h200_profile(
    *,
    compute_capability: tuple[int, int],
    multi_processor_count: int,
    total_memory: int,
) -> bool:
    return (
        compute_capability == _FLASHKDA_H200_COMPUTE_CAPABILITY
        and multi_processor_count == _FLASHKDA_H200_SM_COUNT
        and _FLASHKDA_H200_MIN_TOTAL_MEMORY
        <= total_memory
        <= _FLASHKDA_H200_MAX_TOTAL_MEMORY
    )


def _prefer_flashkda_for_shape(
    *,
    compute_capability: tuple[int, int],
    multi_processor_count: int,
    total_memory: int,
    state_dtype: torch.dtype,
    num_heads: int,
    num_sequences: int,
    min_seq_len: int,
    max_seq_len: int,
) -> bool:
    """Pure performance policy; semantic eligibility is checked separately."""
    if min_seq_len < _FLASHKDA_CHUNK_SIZE:
        return False
    if (
        _is_h200_profile(
            compute_capability=compute_capability,
            multi_processor_count=multi_processor_count,
            total_memory=total_memory,
        )
        and num_sequences == 1
        and state_dtype == torch.float32
    ):
        if num_heads == _FLASHKDA_H200_LONG_HEADS:
            return max_seq_len <= _FLASHKDA_H200_LONG_MAX_SEQ_LEN
        if num_heads in _FLASHKDA_H200_TRITON_HEADS:
            # Preserve existing behavior below the measured Q=2048 bucket,
            # but do not select the under-filled K2 at or above that crossover.
            return max_seq_len < _FLASHKDA_MAX_SEQ_LEN
        # Do not extrapolate the profile to unmeasured head counts.
        return max_seq_len <= _FLASHKDA_MAX_SEQ_LEN
    return max_seq_len <= _FLASHKDA_MAX_SEQ_LEN


def _flashkda_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
) -> bool:
    """Check the external kernel's static tensor contract."""
    if A_log is None or dt_bias is None:
        return False
    if not (q.ndim == k.ndim == v.ndim == g.ndim == 4 and beta.ndim == 3):
        return False
    batch, tokens, heads, key_dim = q.shape
    tensors = (
        k,
        v,
        g,
        beta,
        ssm_states,
        cache_indices,
        query_start_loc,
        A_log,
        dt_bias,
    )
    return (
        batch == 1
        and key_dim == 128
        and k.shape == q.shape
        and v.shape == (batch, tokens, heads, 128)
        and g.shape == q.shape
        and beta.shape == (batch, tokens, heads)
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and g.dtype == torch.bfloat16
        and beta.dtype in (torch.bfloat16, torch.float32)
        and ssm_states.ndim == 4
        and ssm_states.shape[1:] == (heads, 128, 128)
        and ssm_states.dtype in (torch.bfloat16, torch.float32)
        and cache_indices.ndim == 1
        and cache_indices.dtype in (torch.int32, torch.int64)
        and cache_indices.numel() > 0
        and cache_indices.numel() == query_start_loc.numel() - 1
        and query_start_loc.ndim == 1
        and query_start_loc.dtype in (torch.int32, torch.int64)
        and A_log.numel() == heads
        and dt_bias.numel() == heads * key_dim
        and all(tensor.device == q.device for tensor in tensors)
    )


def _load_flash_kda():
    """Import the optional ``flash_kda`` CUTLASS module."""
    try:
        import flash_kda
    except ImportError as e:
        raise ImportError(
            "The 'flashkda' KDA prefill backend requires the flash_kda module, "
            "which is not installed. Install it from source:\n"
            "    pip install git+https://github.com/MoonshotAI/FlashKDA.git"
        ) from e
    return flash_kda


def _triton_fallback(
    q,
    k,
    v,
    g,
    beta,
    ssm_states,
    cache_indices,
    query_start_loc,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    beta_is_raw=False,
    return_intermediate_states=False,
):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing).

    `g` is the RAW gate; chunk_kda applies the gate activation internally when
    A_log is provided, so A_log/dt_bias/lower_bound must be threaded through too
    -- otherwise the fallback silently skips activation. chunk_kda updates the
    ssm state in-place via cache_indices and returns only the output tensor
    (or (output, h) when return_intermediate_states is set).
    """
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=query_start_loc,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        beta_is_raw=beta_is_raw,
        output_intermediate_states=return_intermediate_states,
    )


class FlashKDAKernel(LinearAttnKernelBase):
    """FlashKDA (MoonshotAI) fully-fused CUTLASS KDA prefill backend.

    Wraps the external ``flash_kda`` package (https://github.com/MoonshotAI/FlashKDA).

    FlashKDA fuses q/k L2 norm, beta sigmoid, and the KDA gate *inside* the
    kernel, so we pass RAW tensors plus ``A_log``/``dt_bias``/``lower_bound``.
    It is prefill-only, bf16, K == V == 128, HV == H (no GVA), and requires the
    safe (bounded) gate (``lower_bound`` set). The non-safe path and sequences
    outside the selected device/shape performance window fall back to Triton
    ``chunk_kda``.
    Requires an SM90+ GPU with the ``flash_kda`` package.
    """

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
        raise NotImplementedError("FlashKDAKernel only supports prefill (extend)")

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
        extend_seq_lens_cpu: Optional[list] = None,
        is_spec_decode: bool = False,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if q.device.type == "cuda":
            device_properties = torch.cuda.get_device_properties(q.device)
            device_profile = (
                (device_properties.major, device_properties.minor),
                device_properties.multi_processor_count,
                device_properties.total_memory,
            )
        else:
            # CPU wrapper tests replace the external CUDA kernel with a stub.
            device_profile = ((-1, -1), 0, 0)
        # The fused kernel cannot expose per-chunk states (h), which the mamba
        # radix extra_buffer track path needs; route tracked batches through
        # the Triton chunk_kda fallback instead of silently skipping the
        # snapshot (that would corrupt prefix-cache restores).
        if (
            return_intermediate_states
            or not _flashkda_supported(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log,
                dt_bias,
            )
            or self._should_fall_back(
                lower_bound,
                is_spec_decode,
                query_start_loc,
                extend_seq_lens_cpu,
                num_heads=q.shape[2],
                state_dtype=ssm_states.dtype,
                compute_capability=device_profile[0],
                multi_processor_count=device_profile[1],
                total_memory=device_profile[2],
            )
        ):
            return _triton_fallback(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states,
            )

        return self._flashkda_extend(
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
        )

    @staticmethod
    def _should_fall_back(
        lower_bound: Optional[float],
        is_spec_decode: bool,
        query_start_loc: torch.Tensor,
        extend_seq_lens_cpu: Optional[list],
        *,
        num_heads: int,
        state_dtype: torch.dtype,
        compute_capability: tuple[int, int],
        multi_processor_count: int,
        total_memory: int,
    ) -> bool:
        """Whether to use the Triton chunk_kda path instead of the fused kernel."""
        # Safe-gate only: the fused kernel does not support the unbounded gate
        # (-exp(A_log)*softplus); those models leave lower_bound unset.
        if lower_bound is None:
            return True
        # FlashKDA writes the committed recurrent state back in place, so it is
        # unsafe for speculative verify / draft-extend forwards (which must stay
        # rollback-able). Those reach this backend through forward_extend, so
        # gate them here rather than relying on the decode/target_verify stubs.
        if is_spec_decode:
            return True
        # Short sequences (< chunk size) and long sequences (> the crossover
        # where Triton's chunked prefill wins) are faster on Triton. Read the
        # per-request lengths from the CPU-side extend_seq_lens to avoid a
        # GPU->CPU sync on every layer; derive from query_start_loc (one sync)
        # only if they are unavailable.
        if extend_seq_lens_cpu is not None:
            if torch.is_tensor(extend_seq_lens_cpu):
                lo = int(extend_seq_lens_cpu.min())
                hi = int(extend_seq_lens_cpu.max())
            else:
                lo = min(extend_seq_lens_cpu)
                hi = max(extend_seq_lens_cpu)
            num_sequences = len(extend_seq_lens_cpu)
        else:
            seq_lens = query_start_loc[1:] - query_start_loc[:-1]
            lo_t, hi_t = torch.aminmax(seq_lens)
            lo, hi = int(lo_t), int(hi_t)
            num_sequences = query_start_loc.numel() - 1
        return not _prefer_flashkda_for_shape(
            compute_capability=compute_capability,
            multi_processor_count=multi_processor_count,
            total_memory=total_memory,
            state_dtype=state_dtype,
            num_heads=num_heads,
            num_sequences=num_sequences,
            min_seq_len=lo,
            max_seq_len=hi,
        )

    def _flashkda_extend(
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
    ) -> torch.Tensor:
        flash_kda = _load_flash_kda()

        # Input shapes (varlen, B == 1, matching chunk_kda's contract):
        #   q, k = [1, packed_seq, H, K]   v = [1, packed_seq, HV, V]
        #   g    = [1, packed_seq, HV, K]  beta = [1, packed_seq, H]
        # flash_kda wants these 4D tensors directly and RAW (it fuses l2norm /
        # beta sigmoid / gate activation in-kernel).
        num_heads = q.shape[2]
        head_dim = q.shape[3]
        scale = head_dim**-0.5

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        # FlashKDA applies sigmoid internally; invert only the already-activated
        # Kimi beta path.
        if not beta_is_raw:
            beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7))
        beta = beta.to(torch.bfloat16).contiguous()

        # flash_kda wants A_log [H] fp32 and dt_bias [H, K] fp32. The model
        # stores A_log as [1, 1, H, 1] and dt_bias as 1D [H*K], so reshape both.
        A_log = A_log.reshape(-1).float().contiguous()
        if dt_bias is not None:
            dt_bias = dt_bias.reshape(num_heads, -1).float().contiguous()

        # cu_seqlens must be int64 for flash_kda (FLA casts to long).
        cu_seqlens = query_start_loc.to(torch.int64)

        # flash_kda varlen state is [N, H, V, K] -- the SAME layout as sglang's
        # KDA pool, so no transpose is needed. Advanced indexing copies, so the
        # final state is written back in-place below (matching chunk_kda).
        initial_state = ssm_states[cache_indices].contiguous()

        out_buf = torch.empty_like(v)
        final_state = torch.empty_like(initial_state)

        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out_buf,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )

        ssm_states[cache_indices] = final_state

        # out_buf is already [1, packed_seq, HV, V].
        return out_buf
