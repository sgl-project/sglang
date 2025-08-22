# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/93e1a2634e22355b0856246b032b285ad1d1da6b/tests/test_sampling.py

import pytest
import sgl_kernel
import torch


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [100])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_sampling_from_probs_logits_top_k_first_alignment(
    batch_size, vocab_size, k, p
):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    generator_logits = torch.Generator("cuda:0")
    generator_probs = generator_logits.clone_state()
    samples = sgl_kernel.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="top_k_first", generator=generator_logits
    )
    samples_ref = sgl_kernel.sampling.top_k_top_p_sampling_from_probs(
        torch.softmax(logits, dim=-1),
        k,
        p,
        filter_apply_order="top_k_first",
        generator=generator_probs,
    )
    assert torch.all(samples == samples_ref)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [100])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_sampling_from_probs_logits_joint_alignment(
    batch_size, vocab_size, k, p
):
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * 5
    generator_logits = torch.Generator("cuda:0")
    generator_probs = generator_logits.clone_state()
    samples = sgl_kernel.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, filter_apply_order="joint", generator=generator_logits
    )
    samples_ref = sgl_kernel.sampling.top_k_top_p_sampling_from_probs(
        torch.softmax(logits, dim=-1),
        k,
        p,
        filter_apply_order="joint",
        generator=generator_probs,
    )
    assert torch.all(samples == samples_ref)


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5])
def test_top_k_top_p_joint_sampling_from_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    if p == 0.1:
        k = int(vocab_size * 0.5)
    elif p == 0.5:
        k = int(vocab_size * 0.1)
    else:
        raise ValueError("p not recognized")
    eps = 1e-4
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    # top-p mask
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask_top_p = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask_top_p.scatter_add_(1, indices, (cdf > (1 - p) - eps).int())
    # top-k mask
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask_top_k = (normalized_prob >= pivot.unsqueeze(-1)).int()
    # overall mask
    mask = torch.minimum(mask_top_p, mask_top_k)
    top_p_tensor = torch.full((batch_size,), p, device="cuda:0")
    top_k_tensor = torch.full((batch_size,), k, device="cuda:0")

    num_trails = 1000
    for _ in range(num_trails):
        samples = sgl_kernel.top_k_top_p_sampling_from_probs(
            normalized_prob,
            top_k_tensor,
            top_p_tensor,
            filter_apply_order="joint",
        )
        assert torch.all(samples < vocab_size) and torch.all(samples >= 0)
        assert torch.all(mask[torch.arange(batch_size), samples] == 1), normalized_prob[
            torch.arange(batch_size), samples
        ]


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_renorm_probs(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - p)).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = sgl_kernel.top_p_renorm_prob(normalized_prob, p)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = sgl_kernel.top_k_renorm_prob(normalized_prob, k)
    for i in range(batch_size):
        torch.testing.assert_close(
            renorm_prob_ground_truth[i],
            renorm_prob[i],
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.05, 0.1, 0.2, 0.7, 1])
def test_min_p_sampling(batch_size, vocab_size, p):
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    # scale min-p
    top_probs = sorted_prob[:, -1].unsqueeze(-1)
    scaled_p = p * top_probs
    # min-p mask
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (sorted_prob >= scaled_p).int())
    min_p_tensor = torch.full((batch_size,), p, device="cuda:0")

    num_trails = 1000
    for _ in range(num_trails):
        samples = sgl_kernel.min_p_sampling_from_probs(
            normalized_prob,
            min_p_tensor,
        )

        assert torch.all(mask[torch.arange(batch_size), samples] == 1), samples[
            torch.nonzero(mask[torch.arange(batch_size), samples] == 0)
        ]

        assert torch.all(mask[torch.arange(batch_size), samples] == 1), samples[
            torch.nonzero(mask[torch.arange(batch_size), samples] == 0)
        ]


if __name__ == "__main__":
    pytest.main([__file__])
