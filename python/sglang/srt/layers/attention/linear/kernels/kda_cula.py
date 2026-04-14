import math
from typing import Dict

import torch

# cuLA kernel uses exp2() internally, so gate values must be in log-base-2 space.
# RCP_LN2 converts from natural log space (model output) to log-base-2 space.
RCP_LN2 = 1.0 / math.log(2.0)

# cuLA kernel chunk size. Sequences shorter than this fall back to Triton.
_CULA_CHUNK_SIZE = 64

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)


def _triton_fallback(q, k, v, g, beta, ssm_states, cache_indices, query_start_loc):
    """Fall back to the Triton chunk_kda kernel (handles all preprocessing)."""
    from sglang.srt.layers.attention.fla.kda import chunk_kda

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
    )


class CulaKDAKernel(LinearAttnKernelBase):
    """cuLA SM90 fully-fused kernel for KDA (Kimi Delta Attention) prefill.

    cuLA only supports safe_gate=True mode, where gating values are clamped
    to > -5.  Sequences shorter than the chunk size (64) fall back to Triton.
    """

    def __init__(self):
        super().__init__()
        # Cache workspace buffers per CUDA device
        self._workspace_cache: Dict[int, torch.Tensor] = {}

    def _get_workspace_buffer(self, device: torch.device) -> torch.Tensor:
        """Get or create a workspace buffer for the given device."""
        device_idx = device.index if device.index is not None else 0
        if device_idx not in self._workspace_cache:
            sm_count = torch.cuda.get_device_properties(device).multi_processor_count
            self._workspace_cache[device_idx] = torch.zeros(
                sm_count * 128, dtype=torch.uint8, device=device
            )
        return self._workspace_cache[device_idx]

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
        raise NotImplementedError("CulaKDAKernel only supports prefill (extend)")

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
    ) -> torch.Tensor:
        # Guard: sequences shorter than cuLA chunk size fall back to Triton.
        seq_lens = query_start_loc[1:] - query_start_loc[:-1]
        min_seq_len = seq_lens.min().item()
        if min_seq_len < _CULA_CHUNK_SIZE:
            return _triton_fallback(
                q, k, v, g, beta, ssm_states, cache_indices, query_start_loc
            )

        return self._cula_extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

    def _cula_extend(
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
    ) -> torch.Tensor:
        from sgl_kernel import kda_fwd_prefill

        from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd

        # Input shapes: q, k, v = [1, packed_seq, H, D], g = [1, packed_seq, H, D], beta = [1, packed_seq, H]
        batch_size = q.shape[0]  # should be 1
        packed_seq = q.shape[1]
        num_heads = q.shape[2]
        head_dim = q.shape[3]

        # 1. L2 normalize Q, K (consistent with Triton path use_qk_l2norm_in_kernel=True)
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())

        # 2. Clamp gate values to safe range for cuLA's safe_gate=True mode.
        # This avoids numerical overflow in chunk_local_cumsum when the model
        # produces extreme negative gate values (e.g. < -5).
        g = g.clamp(min=-5.0)

        # 3. Gate cumsum preprocessing (scale=RCP_LN2 for cuLA's exp2-based kernel)
        g = chunk_local_cumsum(
            g, chunk_size=64, scale=RCP_LN2, cu_seqlens=query_start_loc
        )

        # 4. Reshape [1, packed_seq, H, D] -> [packed_seq, H, D], ensure contiguous
        q = q.reshape(packed_seq, num_heads, head_dim).contiguous()
        k = k.reshape(packed_seq, num_heads, head_dim).contiguous()
        v = v.reshape(packed_seq, num_heads, head_dim).contiguous()
        g = g.reshape(packed_seq, num_heads, head_dim).contiguous()
        beta = beta.reshape(packed_seq, num_heads).contiguous()

        # 5. State gather: get per-batch states from the pool (VK layout [N, H, V, K])
        # The kernel natively uses VK layout via CuTe LayoutLeft (K, V, H, N).
        input_state = ssm_states[cache_indices].contiguous()

        # 6. cu_seqlens
        cu_seqlens = query_start_loc.to(torch.int32)

        # 7. Workspace buffer
        workspace_buffer = self._get_workspace_buffer(q.device)

        # 8. Scale
        scale = head_dim**-0.5

        # 9. Call C++ kernel (safe_gate=True)
        output, output_state = kda_fwd_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=cu_seqlens,
            workspace_buffer=workspace_buffer,
            scale=scale,
            safe_gate=True,
            input_state=input_state,
            alpha=g,
            beta=beta,
        )

        # 10. Write output state back (already in VK layout from C++ API)
        ssm_states[cache_indices] = output_state

        # 11. Reshape output: [packed_seq, H, D] -> [1, packed_seq, H, D]
        output = output.reshape(batch_size, packed_seq, num_heads, head_dim)

        return output

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
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("CulaKDAKernel does not support target_verify")
