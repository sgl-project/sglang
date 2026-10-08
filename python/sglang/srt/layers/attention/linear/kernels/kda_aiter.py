from typing import Callable, Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kda_flashkda import _triton_fallback
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_gfx95_supported, is_hip

# AITER's FLASH_KDA_CHUNK and FLASH_KDA_K. Only the two-kernel FlashKDA path
# implements the paged state_cache and the `out` buffer. AITER admits that path
# at chunk 32, with K == V == 128 and no GVA. See flash_kda_supported() in
# aiter/ops/triton/_triton_kernels/kimi_delta_attn/flash_kda.py.
_FLASH_KDA_CHUNK = 32
_FLASH_KDA_HEAD_DIM = 128


def _probe_aiter_kda() -> tuple[Optional[Callable], bool, Optional[str]]:
    """``(chunk_kimi_delta_attn, has_paged_state_cache, unavailable_reason)``."""
    if not is_hip():
        return None, False, "not a ROCm build"
    if not torch.cuda.is_available():
        return None, False, "no GPU visible"
    if not is_gfx95_supported():
        return None, False, "AITER FlashKDA Gluon kernels require gfx950"
    try:
        from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
    except (ImportError, ModuleNotFoundError) as error:
        return None, False, f"aiter.ops.triton.kimi_delta_attn is missing ({error})"

    import inspect

    # ROCm/aiter#5754 added the paged state_cache. An older AITER still works.
    # It uses the gather/scatter tier instead.
    has_paged = "state_cache" in inspect.signature(chunk_kimi_delta_attn).parameters
    return chunk_kimi_delta_attn, has_paged, None


def _paged_state_cache_usable(
    ssm_states: torch.Tensor, *, num_heads: int, head_v_dim: int, head_k_dim: int
) -> bool:
    """Mirror of AITER's ``_check_paged_state_cache``, which raises rather than
    falling back. Each slot must be a dense ``[H, V, K]`` plane. ``stride(0)``
    may be padded. This is the shape of the page-major envelope layout."""
    if ssm_states.dtype != torch.float32 or ssm_states.dim() != 4:
        return False
    if tuple(ssm_states.shape[1:]) != (num_heads, head_v_dim, head_k_dim):
        return False
    return (
        ssm_states.stride(-1) == 1
        and ssm_states.stride(2) == head_k_dim
        and ssm_states.stride(1) == head_v_dim * head_k_dim
        and ssm_states.stride(0) >= num_heads * head_v_dim * head_k_dim
    )


def precompile_aiter_kda_prefill(
    *, num_heads: int, head_k_dim: int, head_v_dim: int, device: torch.device
) -> bool:
    """Compile the FlashKDA PAGED_CACHE specialization before the first request.

    The first call with ``state_cache`` compiles a new Triton specialization.
    AITER's own test discards one launch for the same reason. This returns
    False when another backend will serve, so the caller can stay
    unconditional.
    """
    from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
    from sglang.srt.runtime_context import get_exec

    mamba = get_exec().mamba
    prefill = mamba.linear_attn_prefill_backend or mamba.linear_attn_backend
    if not LinearAttnKernelBackend(prefill).is_aiter():
        return False

    kernel = AiterKDAKernel()
    if not kernel.supports_prefill:
        return False

    # One chunk of one sequence is enough. The paged tier is a constexpr
    # specialization, not a shape-dependent config.
    tokens = _FLASH_KDA_CHUNK
    bf16 = dict(device=device, dtype=torch.bfloat16)
    kernel.extend(
        torch.zeros(1, tokens, num_heads, head_k_dim, **bf16),
        torch.zeros(1, tokens, num_heads, head_k_dim, **bf16),
        torch.zeros(1, tokens, num_heads, head_v_dim, **bf16),
        torch.zeros(1, tokens, num_heads, head_k_dim, **bf16),
        torch.zeros(1, tokens, num_heads, device=device, dtype=torch.float32),
        ssm_states=torch.zeros(
            1, num_heads, head_v_dim, head_k_dim, device=device, dtype=torch.float32
        ),
        cache_indices=torch.zeros(1, device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0, tokens], device=device, dtype=torch.int32),
        A_log=torch.zeros(num_heads, device=device, dtype=torch.float32),
        dt_bias=torch.zeros(num_heads * head_k_dim, device=device, dtype=torch.float32),
        lower_bound=-5.0,
        extend_prefix_lens=torch.zeros(1, device=device, dtype=torch.int32),
        beta_is_raw=True,
    )
    return True


class AiterKDAKernel(LinearAttnKernelBase):
    """AITER FlashKDA KDA prefill backend (ROCm gfx950).

    Wraps ``aiter.ops.triton.kimi_delta_attn.chunk_kimi_delta_attn``. That
    kernel applies the q/k L2 norm, the beta sigmoid and the KDA gate itself.
    This backend therefore passes the raw tensors, plus ``A_log``, ``dt_bias``
    and ``lower_bound``. It serves prefill only, in bf16, with K == V == 128,
    HV == H and a safe gate. Every other case falls back to the Triton
    ``chunk_kda`` reference.

    On an fp32 state pool, AITER's paged ``state_cache`` reads and writes the
    recurrent state in place. No gather or scatter wraps the call. A 2-byte
    pool takes the ``initial_state`` tier instead. That tier is legal since
    ROCm/aiter#5249, but it adds one gather and one scatter.
    """

    # Tracked batches take the Triton fallback. That path forwards the fp32
    # snapshot arguments. AITER rejects return_intermediate_states.
    supports_track_state_snapshot: bool = True

    def __init__(self):
        self._fn, self._has_paged_state_cache, self.unavailable_reason = (
            _probe_aiter_kda()
        )
        self.supports_prefill = self._fn is not None

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
        raise NotImplementedError("AiterKDAKernel only supports prefill (extend)")

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
        extend_prefix_lens: Optional[torch.Tensor] = None,
        is_spec_decode: bool = False,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ):
        # The extra_buffer strategy of the mamba radix cache needs the
        # per-chunk states (h). The fused kernel cannot return them, and AITER
        # rejects the request instead of ignoring it. Route tracked batches to
        # Triton. An unwritten snapshot buffer corrupts prefix-cache restores.
        if (
            return_intermediate_states
            or kwargs.get("track_state") is not None
            or self._falls_back(
                q=q,
                v=v,
                A_log=A_log,
                lower_bound=lower_bound,
                is_spec_decode=is_spec_decode,
                extend_prefix_lens=extend_prefix_lens,
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
                track_state=kwargs.get("track_state"),
                track_chunk_idx=kwargs.get("track_chunk_idx"),
            )

        # Return a bare tensor, not a tuple. forward_extend unpacks only when
        # it asks for intermediate states (kda_backend.py, `if track_ssm`).
        return self._aiter_extend(
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
            extend_prefix_lens=extend_prefix_lens,
            beta_is_raw=beta_is_raw,
        )

    def _falls_back(
        self,
        *,
        q: torch.Tensor,
        v: torch.Tensor,
        A_log: Optional[torch.Tensor],
        lower_bound: Optional[float],
        is_spec_decode: bool,
        extend_prefix_lens: Optional[torch.Tensor],
    ) -> bool:
        """Host-side mirror of AITER's ``flash_kda_supported`` admission check.

        AITER raises when ``state_cache`` or ``out`` reach the default
        pipeline. Divert a call that would miss the FlashKDA path first.
        """
        if not self.supports_prefill:
            return True
        # The fused kernel implements only the safe (bounded) gate. A model with
        # an unbounded gate leaves lower_bound unset. The fused gate needs A_log.
        if lower_bound is None or A_log is None:
            return True
        # draft_extend_v2 must be able to roll back. Both tiers commit the
        # recurrent state, and the paged tier writes the pool in place.
        if is_spec_decode:
            return True
        # extend_prefix_lens produces has_initial_state. Without it, a resumed
        # sequence restarts from zero with no error.
        if extend_prefix_lens is None:
            return True
        if q.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            return True
        num_qk_heads, head_k_dim = q.shape[2], q.shape[3]
        num_v_heads, head_v_dim = v.shape[2], v.shape[3]
        return (
            head_k_dim != _FLASH_KDA_HEAD_DIM
            or head_v_dim != _FLASH_KDA_HEAD_DIM
            or num_v_heads != num_qk_heads
        )

    def _aiter_extend(
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
        dt_bias: Optional[torch.Tensor],
        lower_bound: float,
        extend_prefix_lens: torch.Tensor,
        beta_is_raw: bool,
    ) -> torch.Tensor:
        num_heads, head_k_dim = q.shape[2], q.shape[3]
        head_v_dim = v.shape[3]

        # AITER applies the sigmoid itself. The FlashKDA path has no
        # pre-activated mode, so invert an already-activated beta.
        if not beta_is_raw:
            beta = torch.logit(beta.float().clamp_(1e-7, 1.0 - 1e-7))
        # The kernel rounds the delta-rule write strength to beta's dtype.
        # fp32 keeps the full value.
        beta = beta.float()

        # The model stores A_log as [1, 1, HV, 1] and dt_bias flat as [HV * K].
        # AITER wants [HV] and [HV * K], so only A_log needs the reshape.
        A_log = A_log.reshape(-1).float()
        if dt_bias is not None:
            dt_bias = dt_bias.float()

        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=head_k_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=lower_bound,
            state_v_first=True,
            # None lets AITER pick the chunk size. It picks 32 (FlashKDA)
            # when the call qualifies, and 64 otherwise.
            chunk_size=None,
            cu_seqlens=query_start_loc,
        )
        # K2 stores through packed [B, T, HV, V] strides, so `out` must be dense.
        # Here v is a strided band view, and empty_like would copy its layout.
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        paged = self._has_paged_state_cache and _paged_state_cache_usable(
            ssm_states,
            num_heads=num_heads,
            head_v_dim=head_v_dim,
            head_k_dim=head_k_dim,
        )
        if paged:
            self._fn(
                **common,
                out=out,
                state_cache=ssm_states,
                state_indices=cache_indices.to(torch.int32),
                has_initial_state=extend_prefix_lens > 0,
            )
            return out

        # This tier serves a 2-byte pool, or an AITER without ROCm/aiter#5754.
        # Gather the rows, let the kernel return the final state, then scatter
        # it back. Any float dtype is legal since ROCm/aiter#5249, and
        # final_state follows initial_state's dtype.
        initial_state = ssm_states[cache_indices].contiguous()
        initial_state[~(extend_prefix_lens > 0)] = 0
        _, final_state = self._fn(
            **common,
            out=out,
            initial_state=initial_state,
            output_final_state=True,
        )
        ssm_states[cache_indices] = final_state
        return out
