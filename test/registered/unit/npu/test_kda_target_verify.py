import torch
import torch_npu  # noqa: F401

from sglang.srt.hardware_backend.npu.kernels.kda_target_verify import (
    kda_target_verify_npu,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="stage-a-unit-test-npu")


def test_kda_target_verify_accepts_kimi_k3_parameter_layout():
    batch, steps, heads, key_dim, value_dim = 2, 4, 2, 8, 8
    device = torch.device("npu")
    torch.manual_seed(7)

    q = torch.randn(1, batch * steps, heads, key_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, batch * steps, heads, value_dim, device=device)
    a = torch.randn_like(q)
    b = torch.randn(1, batch * steps, heads, device=device)

    # These are the shapes used by KimiK3DeltaAttention after weight loading.
    a_log = torch.randn(1, 1, heads, 1, device=device)
    dt_bias = torch.randn(heads * key_dim, device=device)
    initial_state = torch.randn(
        batch, heads, value_dim, key_dim, device=device
    )
    snapshots = torch.empty(
        batch, steps, heads, value_dim, key_dim, device=device
    )
    indices = torch.arange(batch, dtype=torch.int32, device=device)
    lower_bound = -5.0

    actual = kda_target_verify_npu(
        A_log=a_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=initial_state,
        initial_state_indices=indices,
        intermediate_states_buffer=snapshots,
        intermediate_state_indices=indices,
        cache_steps=steps,
        lower_bound=lower_bound,
        gates_are_preactivated=False,
    )

    state = initial_state.float().clone()
    expected_outputs = []
    expected_snapshots = []
    a_log_flat = a_log.reshape(heads)
    dt_bias_2d = dt_bias.reshape(heads, key_dim)
    scale = key_dim**-0.5
    for step in range(steps):
        token_indices = torch.arange(batch, device=device) * steps + step
        q_step = q[0, token_indices].float()
        q_step /= torch.linalg.vector_norm(
            q_step, dim=-1, keepdim=True
        ) + 1e-6
        k_step = k[0, token_indices].float()
        k_step /= torch.linalg.vector_norm(
            k_step, dim=-1, keepdim=True
        ) + 1e-6
        v_step = v[0, token_indices].float()
        a_step = a[0, token_indices].float()
        b_step = b[0, token_indices].float()

        log_gate = lower_bound * torch.sigmoid(
            torch.exp(a_log_flat)[None, :, None]
            * (a_step + dt_bias_2d[None])
        )
        state *= torch.exp(log_gate).unsqueeze(-2)
        value = v_step - (state * k_step.unsqueeze(-2)).sum(-1)
        value *= torch.sigmoid(b_step).unsqueeze(-1)
        state += value.unsqueeze(-1) * k_step.unsqueeze(-2)
        expected_outputs.append(
            (state * (q_step * scale).unsqueeze(-2)).sum(-1)
        )
        expected_snapshots.append(state.clone())

    expected = torch.stack(expected_outputs, dim=1).reshape_as(actual)
    expected_states = torch.stack(expected_snapshots, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(snapshots, expected_states, rtol=1e-5, atol=1e-5)
