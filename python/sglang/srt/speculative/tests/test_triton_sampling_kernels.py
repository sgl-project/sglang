"""Tests for Triton sampling kernels (ROCm/HIP).

Validates correctness against PyTorch reference implementations
and replicates upstream sgl-kernel test cases from
sgl-kernel/tests/speculative/test_speculative_sampling.py.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.speculative.triton_sampling_kernels import (
    top_k_renorm_prob,
    top_p_renorm_prob,
    tree_speculative_sampling_target_only,
)

# ---------------------------------------------------------------------------
# PyTorch reference implementations (from bench_renorm.py upstream)
# ---------------------------------------------------------------------------


def torch_top_k_renorm_probs(probs: torch.Tensor, top_k: torch.Tensor):
    probs = probs.float()
    top_k = top_k.to(dtype=torch.int64, device=probs.device)
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    k_indices = (top_k - 1).clamp(min=0).unsqueeze(1)
    thresholds = sorted_probs.gather(1, k_indices)
    renorm = probs.clone()
    renorm[renorm < thresholds] = 0.0
    sums = renorm.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    renorm.div_(sums)
    return renorm


def torch_top_p_renorm_probs(probs: torch.Tensor, top_p: torch.Tensor):
    probs = probs.float()
    top_p = top_p.to(device=probs.device, dtype=torch.float32)
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = (cumsum - sorted_probs) >= top_p.unsqueeze(1)
    sorted_probs[mask] = 0.0
    renorm = torch.zeros_like(probs)
    renorm.scatter_(1, sorted_indices, sorted_probs)
    sums = renorm.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    renorm.div_(sums)
    return renorm


# ---------------------------------------------------------------------------
# top_k_renorm_prob tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("vocab_size", [20, 1024, 32000])
def test_top_k_renorm_prob(bs, vocab_size):
    device = "cuda"
    probs = torch.rand(bs, vocab_size, device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    k = torch.randint(1, min(50, vocab_size), (bs,), device=device, dtype=torch.int64)

    out_triton = top_k_renorm_prob(probs, k)
    out_ref = torch_top_k_renorm_probs(probs, k)

    torch.testing.assert_close(out_triton, out_ref, rtol=1e-4, atol=1e-6)
    sums = out_triton.sum(dim=-1)
    torch.testing.assert_close(
        sums, torch.ones(bs, device=device), rtol=1e-4, atol=1e-4
    )


def test_top_k_renorm_prob_scalar():
    device = "cuda"
    probs = torch.rand(4, 100, device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    out = top_k_renorm_prob(probs, 10)
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(4, device=device), rtol=1e-4, atol=1e-4)


def test_top_k_renorm_prob_k_equals_1():
    device = "cuda"
    probs = torch.rand(2, 50, device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    k = torch.ones(2, device=device, dtype=torch.int64)
    out = top_k_renorm_prob(probs, k)
    for i in range(2):
        nonzero = (out[i] > 0).sum()
        assert nonzero == 1, f"Expected exactly 1 nonzero, got {nonzero}"


# ---------------------------------------------------------------------------
# top_p_renorm_prob tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs", [1, 4, 8])
@pytest.mark.parametrize("vocab_size", [20, 1024, 32000])
def test_top_p_renorm_prob(bs, vocab_size):
    device = "cuda"
    probs = torch.rand(bs, vocab_size, device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    p = torch.rand(bs, device=device, dtype=torch.float32) * 0.9 + 0.1

    out_triton = top_p_renorm_prob(probs, p)
    out_ref = torch_top_p_renorm_probs(probs, p)

    torch.testing.assert_close(out_triton, out_ref, rtol=1e-4, atol=1e-6)
    sums = out_triton.sum(dim=-1)
    torch.testing.assert_close(
        sums, torch.ones(bs, device=device), rtol=1e-4, atol=1e-4
    )


def test_top_p_renorm_prob_scalar():
    device = "cuda"
    probs = torch.rand(4, 100, device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    out = top_p_renorm_prob(probs, 0.9)
    sums = out.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(4, device=device), rtol=1e-4, atol=1e-4)


def test_top_p_renorm_prob_tight():
    device = "cuda"
    probs = torch.rand(2, 50, device=device, dtype=torch.float32)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    p = torch.full((2,), 0.01, device=device, dtype=torch.float32)
    out = top_p_renorm_prob(probs, p)
    for i in range(2):
        nonzero = (out[i] > 0).sum()
        assert nonzero >= 1, f"Expected at least 1 nonzero, got {nonzero}"


# ---------------------------------------------------------------------------
# tree_speculative_sampling_target_only tests
# Replicates test cases from sgl-kernel/tests/speculative/test_speculative_sampling.py
# ---------------------------------------------------------------------------


tree_spec_test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,
        0,
        [1, 2, 18, -1, -1, -1, 11, -1, -1, -1, 12, 18],
        [[0, 1, 2, -1], [6, 10, 11, -1]],
        [2, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    tree_spec_test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    device = "cuda"

    candidates = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
        dtype=torch.int64,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.zeros_like(target_probs)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device).to(torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert (
        predicts.tolist() == expected_predicts
    ), f"Predicts mismatch: got {predicts.tolist()}, expected {expected_predicts}"
    assert (
        accept_index.tolist() == expected_accept_index
    ), f"Accept index mismatch: got {accept_index.tolist()}, expected {expected_accept_index}"
    assert (
        accept_token_num.tolist() == expected_accept_token_num
    ), f"Accept token num mismatch: got {accept_token_num.tolist()}, expected {expected_accept_token_num}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
