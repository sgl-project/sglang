"""A reduced draft vocabulary must not remove target sampling outcomes."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("proposal", [0, 3])
def test_target_distribution_includes_tokens_outside_draft_vocab(proposal):
    from sgl_kernel import tree_speculative_sampling_target_only

    torch.manual_seed(1927 + proposal)
    batch, width, vocab = 32768, 4, 128
    device = "cuda"
    candidates = torch.full((batch, width), proposal, dtype=torch.int64, device=device)
    retrieve = torch.arange(batch * width, device=device, dtype=torch.int64).reshape(
        batch, width
    )
    child = torch.tensor([1, 2, 3, -1], device=device, dtype=torch.int64).repeat(
        batch, 1
    )
    sibling = torch.full_like(child, -1)
    expected = torch.tensor([0.1, 0.2, 0.3, 0.25, 0.15], device=device)
    probs = torch.zeros(batch, width, vocab, device=device)
    probs[:, :, :5] = expected
    predicts = torch.full((batch * width,), -1, device=device, dtype=torch.int32)
    accepted = torch.full((batch, width), -1, device=device, dtype=torch.int32)
    lengths = torch.zeros(batch, device=device, dtype=torch.int32)
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accepted,
        accept_token_num=lengths,
        candidates=candidates,
        retrive_index=retrieve,
        retrive_next_token=child,
        retrive_next_sibling=sibling,
        uniform_samples=torch.rand(batch, width, device=device),
        uniform_samples_for_final_sampling=torch.rand(batch, device=device),
        target_probs=probs,
        draft_probs=torch.zeros_like(probs),
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=True,
    )
    first = predicts[accepted[:, 0].long()]
    assert first.min() >= 0 and first.max() < 5
    frequencies = first.bincount(minlength=5).float() / batch
    # More than six standard errors at the most variable bin; fixed seeds.
    torch.testing.assert_close(frequencies, expected, rtol=0, atol=0.018)
    assert frequencies[2] > 0.27  # token 2 was never proposed by the draft.
