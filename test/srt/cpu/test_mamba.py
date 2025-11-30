import unittest

import torch
from torch.nn.functional import softplus
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(1234)


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """This function is intended to align with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
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

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(
        value
    )
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
            -1
        ) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def sigmoid_gating_delta_rule_update(
    query,
    key,
    value,
    A_log,
    a,
    dt_bias,
    b,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    beta = b.sigmoid()
    g = -A_log.float().exp() * softplus(a.float() + dt_bias)
    return torch_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g.unsqueeze(0),
        beta.unsqueeze(0),
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def torch_gdn_gating(A_log, a, dt_bias):
    return -A_log.float().exp() * softplus(a.float() + dt_bias)


class TestMambaAttention(CustomTestCase):
    def test_fused_gdn_gating(self):
        dims = [6, 32]
        for dim in dims:
            A_log = torch.rand(dim)
            a = torch.rand(1024, dim, dtype=torch.bfloat16)
            dt_bias = torch.rand(dim, dtype=torch.bfloat16)

            g = torch_gdn_gating(A_log, a, dt_bias)
            g_sgl = torch.ops.sgl_kernel.fused_gdn_gating_cpu(A_log, a, dt_bias)
            atol = rtol = precision[g.dtype]
            torch.testing.assert_close(g, g_sgl, atol=atol, rtol=rtol)

    def test_fused_sigmoid_gating_delta_rule_update(self):
        batch_size = 1
        num_value_heads = 32
        head_k_dim = 128
        head_v_dim = 128
        num_heads = 16
        seq_len = 1
        attn_tp_size = 1
        key_dim = head_k_dim * num_heads
        value_dim = head_v_dim * num_value_heads
        mixed_qkv_dim = (key_dim * 2 + value_dim) // attn_tp_size
        mixed_qkv = torch.rand(
            seq_len * batch_size, mixed_qkv_dim, dtype=torch.bfloat16
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, num_value_heads, head_v_dim)
        A_log = torch.rand(num_value_heads, dtype=torch.float32)
        a = torch.rand(batch_size, num_value_heads, dtype=torch.bfloat16)
        b = torch.rand(batch_size, num_value_heads, dtype=torch.bfloat16)
        dt_bias = torch.rand(num_value_heads, dtype=torch.bfloat16)
        ssm_states = torch.rand(
            513, num_value_heads, head_k_dim, head_v_dim, dtype=torch.float32
        )
        cache_indices = torch.randint(0, 513, (batch_size,), dtype=torch.int32)
        query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        use_qk_l2norm_in_kernel = True
        query_ref = query.clone()
        key_ref = key.clone()
        if num_value_heads // num_heads > 1:
            query_ref = query_ref.repeat_interleave(num_value_heads // num_heads, dim=2)
            key_ref = key_ref.repeat_interleave(num_value_heads // num_heads, dim=2)
        core_attn_out_ref, last_recurrent_state_ref = sigmoid_gating_delta_rule_update(
            query_ref.transpose(0, 1),
            key_ref.transpose(0, 1),
            value.transpose(0, 1),
            A_log,
            a,
            dt_bias,
            b,
            initial_state=ssm_states[cache_indices],
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        core_attn_out = torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu(
            A_log=A_log,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        last_recurrent_state = ssm_states[cache_indices]
        atol = rtol = precision[core_attn_out.dtype]
        torch.testing.assert_close(
            core_attn_out, core_attn_out_ref, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            last_recurrent_state, last_recurrent_state_ref, atol=atol, rtol=rtol
        )


if __name__ == "__main__":
    unittest.main()
