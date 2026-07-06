import pytest
import torch

from sglang.jit_kernel.fused_sampler import fused_topk_sample
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference_topk_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_k: int,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    if top_k == 1:
        return torch.argmax(logits, dim=-1)

    probs = torch.softmax(logits.float() / temperatures.view(-1, 1), dim=-1)
    top_probs, top_idx = torch.topk(probs, k=top_k, dim=-1)
    out = torch.empty((logits.shape[0],), dtype=torch.int64, device=logits.device)

    for row in range(logits.shape[0]):
        cutoff = float(top_ps[row].clamp(0.0, 1.0).item())
        cumulative = 0.0
        accepted = 0
        for i in range(top_k):
            if cumulative > cutoff:
                break
            cumulative += float(top_probs[row, i].item())
            accepted = i + 1

        accepted_sum = float(top_probs[row, :accepted].sum().item())
        target = float(uniforms[row].clamp(0.0, 0.99999994).item()) * accepted_sum
        cumulative = 0.0
        sampled = int(top_idx[row, 0].item())
        for i in range(accepted):
            cumulative += float(top_probs[row, i].item())
            sampled = int(top_idx[row, i].item())
            if cumulative >= target:
                break
        out[row] = sampled

    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("vocab_size", [257, 4099])
@pytest.mark.parametrize("top_k", [1, 4, 8])
@pytest.mark.parametrize("top_p", [1.0, 0.95])
def test_fused_topk_sample_matches_reference(batch_size, vocab_size, top_k, top_p):
    torch.manual_seed(0)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)
    logits += torch.arange(vocab_size, device="cuda", dtype=torch.float32) * 1.0e-7
    temperatures = torch.linspace(0.7, 1.3, batch_size, device="cuda")
    top_ps = torch.full((batch_size,), top_p, device="cuda", dtype=torch.float32)
    uniforms = torch.linspace(0.05, 0.95, batch_size, device="cuda")

    actual = fused_topk_sample(logits, temperatures, top_ps, top_k, uniforms)
    expected = _reference_topk_sample(logits, temperatures, top_ps, top_k, uniforms)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_sample_out_parameter_and_scalar_params():
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [4.0, 1.0, 3.0, 2.0]],
        device="cuda",
        dtype=torch.float32,
    )
    uniforms = torch.tensor([0.01, 0.99], device="cuda", dtype=torch.float32)
    out = torch.empty((2,), device="cuda", dtype=torch.int64)

    result = fused_topk_sample(logits, 1.0, 1.0, 2, uniforms, out=out)
    expected = _reference_topk_sample(
        logits,
        torch.ones((2,), device="cuda"),
        torch.ones((2,), device="cuda"),
        2,
        uniforms,
    )

    assert result is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_topk_sample_rejects_bad_topk():
    logits = torch.randn(2, 16, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="top_k"):
        fused_topk_sample(logits, 1.0, 1.0, 0)
    with pytest.raises(RuntimeError, match="top_k"):
        fused_topk_sample(logits, 1.0, 1.0, 9)
