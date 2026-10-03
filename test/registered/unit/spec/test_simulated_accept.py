from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

import pytest
import torch

from sglang.kernels.ops.speculative.simulated_accept import simulated_accept


@pytest.mark.parametrize("bs", [0, 1, 8, 128])
@pytest.mark.parametrize("width", [1, 4, 7])
@pytest.mark.parametrize("real", [False, True])
def test_simulated(bs, width, real):
    dev = "cuda"
    ids = (
        torch.randperm(bs, device=dev, dtype=torch.int32)[:, None] * width
        + torch.arange(width, device=dev, dtype=torch.int32)[None, :]
    )
    pred = torch.randint(
        0, 248320, (bs * width * 2 + 2,), device=dev, dtype=torch.int32
    )[::2]
    cand = torch.randint(0, 248320, (bs, width * 2), device=dev)[:, ::2]
    target = torch.randint(0, 248320, (bs, width * 2), device=dev)[:, ::2]
    correct = torch.empty(bs * 2, device=dev, dtype=torch.int32)[::2]
    for n in range(1, width + 1):
        expected_pred = pred.clone()
        expected_idx = torch.full((bs, width), -1, device=dev, dtype=torch.int32)
        expected_idx[:, :n] = ids[:, 0, None] + torch.arange(n, device=dev)
        if real:
            expected_pred[expected_idx[:, : n - 1].long()] = cand[:, 1:n].int()
            expected_pred[expected_idx[:, n - 1].long()] = target[:, n - 1].int()
        else:
            expected_pred.fill_(100)
        result = simulated_accept(
            ids,
            pred,
            correct,
            cand if real else None,
            target if real else None,
            width,
            n,
            real,
        )
        assert torch.equal(result, expected_idx)
        assert torch.equal(pred, expected_pred)
        assert torch.equal(correct, torch.full_like(correct, n - 1))
