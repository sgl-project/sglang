import pytest
import torch

from sglang.kernels.ops.qwen4_ple import (
    fused_qwen4_gate_value,
    fused_qwen4_ngram_hash,
    fused_qwen4_short_conv_state,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-b200")


_EOS = 248044
_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]


def _inputs(num_tokens):
    generator = torch.Generator(device="cuda").manual_seed(20260822 + num_tokens)
    contexts = torch.randint(
        0, 248320, (num_tokens, 3), device="cuda", generator=generator
    )
    contexts[0, 0] = _EOS
    if num_tokens > 1:
        contexts[1, 1] = _EOS
    multipliers = torch.tensor(_MULTIPLIERS, dtype=torch.long, device="cuda")
    vocab_sizes = 20000003 + 2 * torch.arange(16, device="cuda")
    offsets = torch.cat([vocab_sizes.new_zeros(1), vocab_sizes.cumsum(0)[:-1]])
    gate = torch.randn(
        num_tokens,
        4,
        1,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        num_tokens,
        2560,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    state = torch.randn(
        num_tokens + 1,
        32,
        9,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    state_indices = torch.arange(1, num_tokens + 1, device="cuda")
    if num_tokens > 1:
        state_indices[-1] = 0
    x = torch.randn(
        num_tokens,
        32,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return (
        contexts,
        multipliers,
        vocab_sizes,
        offsets,
        gate,
        value,
        state,
        state_indices,
        x,
    )


def _hash_reference(contexts, multipliers, vocab_sizes, offsets):
    token_0, token_1, token_2 = contexts.unbind(1)
    mixed_2 = token_2 * multipliers[0] ^ token_1 * multipliers[1]
    token_0 = torch.where((token_0 == _EOS) | (token_1 == _EOS), _EOS, token_0)
    mixed_3 = mixed_2 ^ token_0 * multipliers[2]
    return torch.cat(
        [
            mixed_2[:, None].remainder(vocab_sizes[:8]) + offsets[:8],
            mixed_3[:, None].remainder(vocab_sizes[8:]) + offsets[8:],
        ],
        dim=1,
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 64, 256])
def test_qwen4_ple_fused_kernels_are_bitwise_exact(num_tokens):
    contexts, multipliers, vocab_sizes, offsets, gate, value, state, indices, x = (
        _inputs(num_tokens)
    )
    expected_hash = _hash_reference(contexts, multipliers, vocab_sizes, offsets)
    expected_gate = torch.sigmoid(
        gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    ) * value.unsqueeze(1)
    original_state = state.clone()
    expected_conv_input = torch.cat(
        [original_state.index_select(0, indices), x.unsqueeze(-1)], dim=-1
    )

    assert torch.equal(
        fused_qwen4_ngram_hash(contexts, multipliers, vocab_sizes, offsets, _EOS),
        expected_hash,
    )
    assert torch.equal(fused_qwen4_gate_value(gate, value), expected_gate)
    assert torch.equal(
        fused_qwen4_short_conv_state(state, indices, x), expected_conv_input
    )
    real = indices.ne(0)
    assert torch.equal(
        state[indices[real], :, :-1], original_state[indices[real], :, 1:]
    )
    assert torch.equal(state[indices[real], :, -1], x[real])
    assert torch.equal(state[0], original_state[0])


def test_qwen4_ple_fusion_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_ENABLE_QWEN4_PLE_FUSION", raising=False)
    assert envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get()
