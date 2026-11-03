import torch

from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel


class AiterGDNKernel(TritonGDNKernel):
    """GDN kernel that runs linear-chain target verify on aiter's FlyDSL kernel.

    Decode and extend stay on Triton; tree verify is routed to Triton by the
    dispatcher before it reaches this kernel.
    """

    supports_strided_target_verify_qkv: bool = True

    def __init__(self):
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_mtp_sglang

        self._gdr_mtp = flydsl_gdr_mtp_sglang

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
        num_tokens = q.shape[1]
        batch_size = num_tokens // cache_steps
        num_k_heads, head_k_dim = q.shape[-2:]
        num_v_heads, head_v_dim = v.shape[-2:]
        out = q.new_empty(batch_size, cache_steps, num_v_heads, head_v_dim)
        self._gdr_mtp(
            q.view(batch_size, cache_steps, num_k_heads, head_k_dim),
            k.view(batch_size, cache_steps, num_k_heads, head_k_dim),
            v.view(batch_size, cache_steps, num_v_heads, head_v_dim),
            a.view(batch_size, cache_steps, num_v_heads),
            b.view(batch_size, cache_steps, num_v_heads),
            dt_bias,
            A_log,
            ssm_states,
            out,
            cache_indices[:batch_size],
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices[:batch_size],
            retrieve_parent_token=retrieve_parent_token,
            disable_state_update=True,
            use_qk_l2norm=True,
        )
        return out.view(1, num_tokens, num_v_heads, head_v_dim)
