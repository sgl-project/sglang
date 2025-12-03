import unittest

import torch
import torch.nn.functional as F
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(
                -1, -2
            )
            @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def chunk_gated_delta_rule_update(
    query,  # [B, T, HK, K]
    key,  # [B, T, HK, K]
    value,  # [B, T, HV, V]
    g,  # [B, T, HV]
    beta,  # [B, T, HV]
    cu_seqlens,  # [N+1]
    initial_state,  # [N, HV, K, V]
    use_qk_l2norm_in_kernel,  # True
):
    num_heads = query.shape[2]
    num_value_heads = value.shape[2]
    batch_size = initial_state.shape[0]
    if num_value_heads // num_heads > 1:
        query = query.repeat_interleave(num_value_heads // num_heads, dim=2)
        key = key.repeat_interleave(num_value_heads // num_heads, dim=2)
    output = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)
    start_q = 0
    for i in range(batch_size):
        end_q = cu_seqlens[i + 1]
        core_attn_outi, last_recurrent_state = torch_chunk_gated_delta_rule(
            query=query[:, start_q:end_q, :, :],
            key=key[:, start_q:end_q, :, :],
            value=value[:, start_q:end_q, :, :],
            g=g[:, start_q:end_q, :],
            beta=beta[:, start_q:end_q, :],
            initial_state=initial_state[i],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        output[:, start_q:end_q, :, :] = core_attn_outi
        final_state[i] = last_recurrent_state
        start_q = end_q
    return output, final_state


class TestMambaAttention(CustomTestCase):
    def test_chunk_gated_delta_rule(self):
        B, L, HK, HV, EK, EV, N = 1, 100, 3, 6, 64, 64, 4
        seqlens = torch.randint(1, L, (N + 1,))
        seqlens[0] = 0
        cu_seqlens_ = torch.cumsum(seqlens, dim=0).to(torch.int32)
        T = cu_seqlens_[-1].item()
        query_ = torch.rand((B, T, HK, EK), dtype=torch.bfloat16) * 0.05
        key_ = torch.rand((B, T, HK, EK), dtype=torch.bfloat16) * 0.05
        value_ = torch.rand((B, T, HV, EV), dtype=torch.bfloat16) * 0.05
        g_ = torch.rand((B, T, HV), dtype=torch.float32) * 0.05
        beta_ = torch.rand((B, T, HV), dtype=torch.bfloat16) * 0.05
        initial_state_ = torch.rand((N, HV, EK, EV), dtype=torch.float32) * 0.05

        for use_qk_l2norm_in_kernel in [True, False]:
            core_attn_out_ref, last_recurrent_state_ref = chunk_gated_delta_rule_update(
                query=query_,
                key=key_,
                value=value_,
                g=g_,
                beta=beta_,
                cu_seqlens=cu_seqlens_,
                initial_state=initial_state_,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

            query = query_.clone()
            key = key_.clone()
            value = value_.clone()
            g = g_.clone()
            beta = beta_.clone()
            cu_seqlens = cu_seqlens_.clone()
            initial_state = initial_state_.clone()

            core_attn_out, last_recurrent_state = (
                torch.ops.sgl_kernel.chunk_gated_delta_rule_cpu(
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=cu_seqlens,
                    head_first=False,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
            )
            atol = rtol = precision[core_attn_out.dtype]
            torch.testing.assert_close(
                core_attn_out, core_attn_out_ref, atol=atol, rtol=rtol
            )
            torch.testing.assert_close(
                last_recurrent_state, last_recurrent_state_ref, atol=atol, rtol=rtol
            )


if __name__ == "__main__":
    unittest.main()
