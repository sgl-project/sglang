import pytest
import torch


def jit_fused_share_gate_sigmoid_mul(
    hidden_state,
    share_gate_weight,
    share_expert_output,
) -> torch.Tensor:
    from sglang.jit_kernel.moe import fused_share_gate_sigmoid_mul

    output = fused_share_gate_sigmoid_mul(
        hidden_state, share_gate_weight, share_expert_output
    )
    return output


@torch.inference_mode()
def ref_fused_share_gate_sigmoid_mul(
    hidden_state,
    share_gate_weight,
    share_expert_output,
) -> torch.Tensor:
    import torch.nn.functional as F

    share_expert_gate = torch.nn.Linear(share_gate_weight.shape[-1], 1, bias=False)
    share_expert_gate.weight = torch.nn.Parameter(share_gate_weight)
    x = share_expert_gate(hidden_state)
    x = F.sigmoid(x)
    x = x * share_expert_output
    return x


BS_LIST = [2**n for n in range(0, 16)]
BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]
HIDDEN_SIZE_LIST = [1024, 2048, 4096]
DEVICE_LIST = [
    "cuda",
]
DTYPE_LIST = [torch.bfloat16, torch.half]


@pytest.mark.parametrize("num_tokens", BS_LIST)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZE_LIST)
@pytest.mark.parametrize("device", DEVICE_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_fused_share_gate_sigmoid_mul(
    num_tokens: int,
    hidden_size: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    hidden_state = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    share_gate_weight = torch.randn(1, hidden_size, dtype=dtype, device=device)
    share_expert_output = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )

    ref_out = ref_fused_share_gate_sigmoid_mul(
        hidden_state, share_gate_weight, share_expert_output
    )
    out = jit_fused_share_gate_sigmoid_mul(
        hidden_state, share_gate_weight, share_expert_output
    )
    assert torch.allclose(ref_out, out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
