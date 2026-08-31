from types import SimpleNamespace

import torch

from sglang.srt.hardware_backend.npu.attention.ascend_kda_backend import (
    AscendKDAAttnBackend,
    _kda_decode_key_value_state_npu,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def test_kda_decode_updates_key_value_state_layout():
    torch.manual_seed(11)
    batch, q_heads, k_heads, value_heads = 2, 1, 1, 2
    key_dim, value_dim = 5, 3
    q = torch.randn(1, batch, q_heads, key_dim)
    k = torch.randn(1, batch, k_heads, key_dim)
    v = torch.randn(1, batch, value_heads, value_dim)
    a = torch.randn(1, batch, k_heads, key_dim)
    b = torch.randn(1, batch, value_heads)
    A_log = torch.randn(k_heads)
    dt_bias = torch.randn(k_heads, key_dim)
    state_source = torch.randn(4, value_heads, key_dim, value_dim).to(torch.bfloat16)
    state_indices = torch.tensor([1, 3], dtype=torch.int32)

    initial = state_source.index_select(0, state_indices.long()).float()
    actual = _kda_decode_key_value_state_npu(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        state_source=state_source,
        state_indices=state_indices,
        lower_bound=-5.0,
    )

    q_ref = q.squeeze(0)
    k_ref = k.squeeze(0)
    q_ref = q_ref / (q_ref.float().norm(dim=-1, keepdim=True) + 1e-6)
    k_ref = k_ref / (k_ref.float().norm(dim=-1, keepdim=True) + 1e-6)
    q_ref = q_ref.float().repeat_interleave(value_heads // q_heads, dim=1)
    q_ref *= key_dim**-0.5
    k_ref = k_ref.float().repeat_interleave(value_heads // k_heads, dim=1)
    gate_input = a.float().view(batch, k_heads, key_dim) + dt_bias.view(
        k_heads, key_dim
    )
    decay = torch.exp(
        -5.0 * torch.sigmoid(torch.exp(A_log.view(k_heads, 1)) * gate_input)
    ).repeat_interleave(value_heads // k_heads, dim=1)
    beta = torch.sigmoid(b.reshape(batch, value_heads).float())
    expected_state = initial * decay.unsqueeze(-1)
    value = v.squeeze(0).float() - torch.matmul(
        k_ref.unsqueeze(2), expected_state
    ).squeeze(2)
    value = value * beta.unsqueeze(-1)
    expected_state = expected_state + k_ref.unsqueeze(-1) * value.unsqueeze(-2)
    expected = torch.matmul(q_ref.unsqueeze(2), expected_state).squeeze(2).unsqueeze(0)

    torch.testing.assert_close(actual.float(), expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(
        state_source.index_select(0, state_indices.long()).float(),
        expected_state.to(torch.bfloat16).float(),
        rtol=0,
        atol=0,
    )


def test_kda_idle_verify_does_not_require_full_attention_graph_state():
    backend = object.__new__(AscendKDAAttnBackend)
    layer = SimpleNamespace(num_v_heads=2, head_v_dim=3)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        is_speculative_idle_participation=True,
        num_token_non_padded_cpu=0,
    )
    mixed_qkv = torch.randn(4, 11)

    output = backend.forward_extend(
        layer=layer,
        forward_batch=forward_batch,
        mixed_qkv=mixed_qkv,
        a=torch.empty(0),
        b=torch.empty(0),
    )

    assert output.shape == (4, 2, 3)
    assert torch.count_nonzero(output) == 0
