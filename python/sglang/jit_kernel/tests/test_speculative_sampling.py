"""
Tests for the JIT tree_speculative_sampling_target_only kernel.

Correctness is validated by:
1. Comparing against the AOT sgl_kernel implementation (bitwise identical
   for deterministic=True, since the same algorithm + same inputs are used).
2. Smoke tests and boundary checks.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# Helper: build a simple linear-chain tree
# ---------------------------------------------------------------------------


def make_linear_chain(bs, num_draft_tokens, num_spec_step, vocab_size, device):
    """
    Construct tree tensors for a simple linear chain (no branching).

    Tree layout (per batch element):
      slot 0: initial context anchor
      slot 1..num_draft_tokens-1: draft tokens in sequence

    retrive_index[b, i] = i   (slot i writes to predicts[i])
    retrive_next_token[b, i] = i+1  (sequential chain), last = -1
    retrive_next_sibling[b, i] = -1 (no siblings)
    candidates[b, i] = i % vocab_size
    """
    retrive_index = torch.stack(
        [torch.arange(num_draft_tokens, dtype=torch.int64, device=device)] * bs
    )  # [bs, num_draft_tokens]

    next_token = torch.arange(1, num_draft_tokens + 1, dtype=torch.int64, device=device)
    next_token[-1] = -1
    retrive_next_token = next_token.unsqueeze(0).expand(bs, -1).contiguous()

    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int64, device=device
    )

    candidates = (
        torch.arange(num_draft_tokens, dtype=torch.int64, device=device) % vocab_size
    )
    candidates = candidates.unsqueeze(0).expand(bs, -1).contiguous()

    return retrive_index, retrive_next_token, retrive_next_sibling, candidates


# ---------------------------------------------------------------------------
# JIT and AOT wrappers
# ---------------------------------------------------------------------------


def run_jit(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic=True,
):
    from sglang.jit_kernel.speculative_sampling import (
        tree_speculative_sampling_target_only,
    )

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )


def run_aot(
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic=True,
):
    from sgl_kernel import tree_speculative_sampling_target_only

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=deterministic,
    )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

DEVICE = "cuda"


def make_inputs(bs, num_draft_tokens, num_spec_step, vocab_size, seed=42):
    """Create all tensors needed for one kernel call."""
    torch.manual_seed(seed)
    device = DEVICE

    tot_draft = bs * num_draft_tokens
    predicts = torch.zeros(tot_draft, dtype=torch.int32, device=device)
    accept_index = torch.zeros(bs, num_spec_step, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=device)

    retrive_index, retrive_next_token, retrive_next_sibling, candidates = (
        make_linear_chain(bs, num_draft_tokens, num_spec_step, vocab_size, device)
    )

    uniform_samples = torch.rand(
        bs, num_draft_tokens, dtype=torch.float32, device=device
    )
    uniform_samples_for_final_sampling = torch.rand(
        bs, dtype=torch.float32, device=device
    )

    # Uniform probabilities (each token equally likely)
    raw = torch.rand(
        bs, num_draft_tokens, vocab_size, dtype=torch.float32, device=device
    )
    target_probs = raw / raw.sum(dim=-1, keepdim=True)
    draft_probs = target_probs.clone()

    return dict(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
    )


# ---------------------------------------------------------------------------
# Smoke test — just verify it runs without errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bs,num_draft_tokens,num_spec_step,vocab_size",
    [
        (1, 4, 4, 32),
        (2, 8, 5, 64),
        (4, 16, 8, 128),
        (1, 1, 1, 32),
    ],
)
def test_smoke(bs, num_draft_tokens, num_spec_step, vocab_size):
    inputs = make_inputs(bs, num_draft_tokens, num_spec_step, vocab_size)
    run_jit(**inputs, threshold_single=0.9, threshold_acc=0.9)

    # Outputs must be non-negative and in range
    assert inputs["accept_token_num"].min() >= 0
    assert inputs["accept_token_num"].max() < num_spec_step


# ---------------------------------------------------------------------------
# JIT vs AOT cross-validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bs,num_draft_tokens,num_spec_step,vocab_size",
    [
        (1, 4, 4, 32),
        (2, 8, 5, 64),
        (4, 16, 8, 128),
        (1, 1, 1, 32),
        (8, 32, 8, 256),
    ],
)
def test_vs_aot(bs, num_draft_tokens, num_spec_step, vocab_size):
    try:
        from sgl_kernel import (  # noqa: F401
            tree_speculative_sampling_target_only as _aot_fn,
        )
    except ImportError:
        pytest.skip("sgl_kernel not available")

    inputs_jit = make_inputs(bs, num_draft_tokens, num_spec_step, vocab_size, seed=0)
    # Deep-copy all mutable tensors for AOT
    inputs_aot = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in inputs_jit.items()
    }
    # Use same probabilities (from JIT input) for AOT
    inputs_aot["target_probs"] = inputs_jit["target_probs"].clone()
    inputs_aot["draft_probs"] = inputs_jit["draft_probs"].clone()

    run_jit(**inputs_jit, threshold_single=0.9, threshold_acc=0.9, deterministic=True)
    run_aot(**inputs_aot, threshold_single=0.9, threshold_acc=0.9, deterministic=True)

    # Mutable outputs should be identical (deterministic=True, same algorithm)
    assert torch.equal(
        inputs_jit["predicts"], inputs_aot["predicts"]
    ), "predicts mismatch"
    assert torch.equal(
        inputs_jit["accept_index"], inputs_aot["accept_index"]
    ), "accept_index mismatch"
    assert torch.equal(
        inputs_jit["accept_token_num"], inputs_aot["accept_token_num"]
    ), "accept_token_num mismatch"


# ---------------------------------------------------------------------------
# Threshold boundary: threshold_single=1.0 means no token is auto-accepted
# via single-token check (must rely on cumulative).
# threshold_acc=0.0 is floored to 1e-9 by the kernel, so coin <= prob/1e-9
# is almost always true (accept all).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs", [1, 2])
def test_accept_all_when_threshold_acc_zero(bs):
    """With threshold_acc very small, all draft tokens should be accepted."""
    num_draft_tokens = 4
    num_spec_step = 4
    vocab_size = 32

    inputs = make_inputs(bs, num_draft_tokens, num_spec_step, vocab_size, seed=123)
    # Force coins to 0 so coin <= prob / 1e-9 is always satisfied
    inputs["uniform_samples"].fill_(0.0)

    run_jit(**inputs, threshold_single=0.0, threshold_acc=0.0)
    # All num_spec_step - 1 draft tokens should be accepted (+ 1 bonus always)
    assert (inputs["accept_token_num"] == num_spec_step - 1).all()


# ---------------------------------------------------------------------------
# Threshold boundary: threshold_single=0.0 means auto-accept first token
# with prob >= 0, which is always true for non-negative probs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs", [1, 2])
def test_accept_all_when_threshold_single_zero(bs):
    """With threshold_single=0.0 and positive target probs, all tokens accepted."""
    num_draft_tokens = 4
    num_spec_step = 4
    vocab_size = 32

    inputs = make_inputs(bs, num_draft_tokens, num_spec_step, vocab_size, seed=456)
    # threshold_single=0.0 means target_prob_single >= 0 always passes
    run_jit(**inputs, threshold_single=0.0, threshold_acc=1.0)
    assert (inputs["accept_token_num"] == num_spec_step - 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
