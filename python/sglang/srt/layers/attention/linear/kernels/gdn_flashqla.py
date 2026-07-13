import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

# Measured crossover on H200 (Qwen3.6-35B GDN head config, TP1): FlashQLA's
# CP preprocessing + gather/scatter overhead loses to Triton below ~16K packed
# tokens and wins 2-3x from 64K up. Partial chunks and seqs < 64 are handled
# correctly by the kernel (validated), so only total batch size gates dispatch.
_FLASHQLA_MIN_TOTAL_TOKENS = 16384

_flash_qla_chunk = None


def _load_flash_qla():
    """Import the optional ``flash_qla`` TileLang module (cached)."""
    global _flash_qla_chunk
    if _flash_qla_chunk is None:
        try:
            from flash_qla import chunk_gated_delta_rule
        except ImportError as e:
            raise ImportError(
                "The 'flashqla' GDN prefill backend requires the flash_qla "
                "package (SM90/SM100, CUDA 12.8+, torch 2.8+). Install it with:\n"
                "    pip install flash-qla"
            ) from e
        _flash_qla_chunk = chunk_gated_delta_rule
    return _flash_qla_chunk


class FlashQLAGDNKernel(TritonGDNKernel):
    """FlashQLA (QwenLM) TileLang GDN chunked-prefill backend.

    Wraps the external ``flash_qla`` package (https://github.com/QwenLM/FlashQLA),
    which fuses the GDN chunked-prefill forward with warp specialization and
    gate-driven intra-card context parallelism (2-3x over the Triton kernel on
    Hopper/Blackwell). Prefill-only: decode / packed_decode / target_verify
    inherit the Triton kernels. bf16/fp16 inputs only; other dtypes and batches
    below the measured crossover point fall back to Triton.

    Unlike the sglang FLA fork, flash_qla takes a dense per-sequence
    ``initial_state`` (no state-pool indexing) and returns the final state
    instead of writing it back in place, so this wrapper gathers
    ``ssm_states[cache_indices]`` on the way in and scatters the returned state
    on the way out. ``state_v_first=True`` matches sglang's ``[N, HV, V, K]``
    pool layout, so no transpose is involved.

    Like the FlashInfer prefill kernel, flash_qla exposes no intermediate chunk
    states (``h``), so extra-buffer mamba radix strategies get no mid-sequence
    checkpoints from this backend; use the default no_buffer strategy.
    """

    def __init__(self):
        # Fail fast at dispatcher construction if flash_qla is not installed.
        _load_flash_qla()

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
        **kwargs,
    ) -> tuple:
        if self._should_fall_back(q):
            return super().extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                **kwargs,
            )

        chunk_gated_delta_rule = _load_flash_qla()

        # Varlen contract ([1, T, ...] + cu_seqlens) matches the Triton path.
        # TileLang's DLPack bridge needs explicit strides, so keep every input
        # contiguous. g stays fp32 out of fused_gdn_gating; flash_qla only
        # dtype-asserts q/k/v.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()

        cu_seqlens = query_start_loc.to(torch.long)

        # flash_qla has no state-pool indexing: gather per-sequence initial
        # states and scatter the final states back after the kernel.
        initial_state = ssm_states[cache_indices].to(q.dtype)

        o, final_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            state_v_first=True,
        )

        ssm_states[cache_indices] = final_state.to(ssm_states.dtype)

        # h=None: same contract as the FlashInfer prefill kernel.
        return o, None, None

    @staticmethod
    def _should_fall_back(q: torch.Tensor) -> bool:
        """Whether to use the Triton chunk path instead of flash_qla.

        Host-side checks only (no GPU sync). q is the packed prefill query
        [1, T_packed, H, K]; index the token dim from the end so a squeezed
        [T_packed, H, K] tensor is handled too.
        """
        if q.dtype not in (torch.bfloat16, torch.float16):
            return True
        return q.shape[-3] < _FLASHQLA_MIN_TOTAL_TOKENS
