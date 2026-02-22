"""
Tests for the JIT eagle_utils kernels:
  - build_tree_kernel_efficient
  - verify_tree_greedy

Correctness is validated by:
1. Comparing against the AOT sgl_kernel implementation (bitwise identical outputs).
2. Smoke tests and boundary checks.
"""

import pytest
import torch

DEVICE = "cuda"

# TreeMaskMode constants (must match eagle_utils.cuh)
FULL_MASK = 0
QLEN_ONLY = 1
QLEN_ONLY_BITPACKING = 2


# ---------------------------------------------------------------------------
# Helper: make inputs for build_tree_kernel_efficient
# ---------------------------------------------------------------------------


def make_build_tree_inputs(
    bs, topk, depth, draft_token_num, seq_len, tree_mask_mode, device=DEVICE
):
    """
    Create a minimal set of tensors for build_tree_kernel_efficient.

    Uses a simple linear-chain tree (each node's parent is the previous one).
    """
    parent_list_size = topk * (depth - 1) + 1
    parent_list = torch.zeros(bs, parent_list_size, dtype=torch.int64, device=device)
    # Simple: parent_list[b, i] = i - 1 (previous node), 0 is root
    for i in range(1, parent_list_size):
        parent_list[:, i] = i - 1

    selected_index = torch.arange(draft_token_num - 1, dtype=torch.int64, device=device)
    selected_index = selected_index.unsqueeze(0).expand(bs, -1).contiguous()

    verified_seq_len = torch.full((bs,), seq_len, dtype=torch.int64, device=device)

    # Allocate tree_mask
    if tree_mask_mode == QLEN_ONLY_BITPACKING:
        if draft_token_num > 16:
            num_bytes = 4
        elif draft_token_num > 8:
            num_bytes = 2
        else:
            num_bytes = 1
        tree_mask = torch.zeros(
            bs * draft_token_num * num_bytes, dtype=torch.uint8, device=device
        )
    elif tree_mask_mode == QLEN_ONLY:
        tree_mask = torch.zeros(
            bs * draft_token_num * draft_token_num, dtype=torch.bool, device=device
        )
    else:  # FULL_MASK
        total_mask_size = (
            sum([seq_len] * bs) * draft_token_num
            + bs * draft_token_num * draft_token_num
        )
        tree_mask = torch.zeros(total_mask_size, dtype=torch.bool, device=device)

    positions = torch.zeros(bs, draft_token_num, dtype=torch.int64, device=device)
    retrive_index = torch.zeros(bs, draft_token_num, dtype=torch.int64, device=device)
    retrive_next_token = torch.full(
        (bs, draft_token_num), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (bs, draft_token_num), -1, dtype=torch.int64, device=device
    )

    return dict(
        parent_list=parent_list,
        selected_index=selected_index,
        verified_seq_len=verified_seq_len,
        tree_mask=tree_mask,
        positions=positions,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
    )


# ---------------------------------------------------------------------------
# Helper: make inputs for verify_tree_greedy
# ---------------------------------------------------------------------------


def make_verify_tree_greedy_inputs(
    bs, num_draft_tokens, num_spec_step, vocab_size, seed=42, device=DEVICE
):
    """Create tensors for verify_tree_greedy with a linear-chain tree.

    retrive_index uses **absolute** flat indices (as produced by build_tree_efficient):
      retrive_index[b, i] = b * num_draft_tokens + i
    target_predict[b, i] contains the token the target model predicts from tree slot (b, i).
    """
    torch.manual_seed(seed)

    tot_draft = bs * num_draft_tokens
    predicts = torch.zeros(tot_draft, dtype=torch.int32, device=device)
    accept_index = torch.zeros(bs, num_spec_step, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=device)

    # Absolute retrive_index: retrive_index[b, i] = b * num_draft_tokens + i
    retrive_index = torch.zeros(bs, num_draft_tokens, dtype=torch.int64, device=device)
    for b in range(bs):
        retrive_index[b] = torch.arange(
            b * num_draft_tokens,
            (b + 1) * num_draft_tokens,
            dtype=torch.int64,
            device=device,
        )

    # retrive_next_token[b, i] = i+1 (linear chain, last = -1)
    next_token = torch.arange(1, num_draft_tokens + 1, dtype=torch.int64, device=device)
    next_token[-1] = -1
    retrive_next_token = next_token.unsqueeze(0).expand(bs, -1).contiguous()

    retrive_next_sibling = torch.full(
        (bs, num_draft_tokens), -1, dtype=torch.int64, device=device
    )

    candidates = (
        (torch.arange(num_draft_tokens, dtype=torch.int64, device=device) % vocab_size)
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )

    # target_predict[b, i] = what target predicts after accepting slot i.
    # For a neutral baseline set it equal to candidates (greedy accepts all but last slot).
    target_predict = candidates.clone()

    return dict(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )


# ---------------------------------------------------------------------------
# Smoke tests for build_tree_kernel_efficient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("tree_mask_mode", [QLEN_ONLY, QLEN_ONLY_BITPACKING])
def test_build_tree_smoke(bs, tree_mask_mode):
    from sglang.jit_kernel.eagle_utils import build_tree_kernel_efficient

    topk = 4
    depth = 5
    draft_token_num = topk * (depth - 1) + 1
    seq_len = 10

    inputs = make_build_tree_inputs(
        bs, topk, depth, draft_token_num, seq_len, tree_mask_mode
    )
    build_tree_kernel_efficient(
        **inputs,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=tree_mask_mode,
    )

    # positions[b, 0] should equal seq_len (root position)
    assert (
        inputs["positions"][:, 0] == seq_len
    ).all(), "Root position should equal seq_len"
    # retrive_index[b, 0] should equal b * draft_token_num
    for b in range(bs):
        assert inputs["retrive_index"][b, 0].item() == b * draft_token_num


# ---------------------------------------------------------------------------
# Smoke tests for verify_tree_greedy
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
def test_verify_tree_greedy_smoke(bs, num_draft_tokens, num_spec_step, vocab_size):
    from sglang.jit_kernel.eagle_utils import verify_tree_greedy

    inputs = make_verify_tree_greedy_inputs(
        bs, num_draft_tokens, num_spec_step, vocab_size
    )
    verify_tree_greedy(**inputs)

    # accept_token_num should be non-negative and < num_spec_step
    assert inputs["accept_token_num"].min() >= 0
    assert inputs["accept_token_num"].max() < num_spec_step


def test_verify_tree_greedy_accept_all():
    """When target_predict[b,i] == candidates[b,i+1], all draft tokens should be accepted."""
    from sglang.jit_kernel.eagle_utils import verify_tree_greedy

    bs = 2
    num_draft_tokens = 4
    num_spec_step = 4
    vocab_size = 32

    inputs = make_verify_tree_greedy_inputs(
        bs, num_draft_tokens, num_spec_step, vocab_size
    )
    # The kernel accepts slot i+1 if target_predict[b, i] == candidates[b, i+1].
    # Set target_predict[b, i] = candidates[b, i+1] to accept every step.
    target_predict = inputs["candidates"].clone()
    target_predict[:, :-1] = inputs["candidates"][:, 1:]
    inputs["target_predict"] = target_predict

    verify_tree_greedy(**inputs)

    # All num_spec_step - 1 draft tokens should be accepted
    assert (inputs["accept_token_num"] == num_spec_step - 1).all()


def test_verify_tree_greedy_accept_none():
    """When target_predict[b,0] != candidates[b,1], no tokens are accepted."""
    from sglang.jit_kernel.eagle_utils import verify_tree_greedy

    bs = 2
    num_draft_tokens = 4
    num_spec_step = 4
    vocab_size = 32

    inputs = make_verify_tree_greedy_inputs(
        bs, num_draft_tokens, num_spec_step, vocab_size
    )
    # candidates[b, i] = i % vocab_size, so candidates[b, 1] = 1.
    # Set target_predict to all zeros: target_predict[b, 0] = 0 != 1 = candidates[b, 1].
    # The first candidate is rejected, sibling is -1, so the loop breaks immediately.
    inputs["target_predict"] = torch.zeros_like(inputs["target_predict"])

    verify_tree_greedy(**inputs)

    # No draft tokens should be accepted
    assert (inputs["accept_token_num"] == 0).all()


# ---------------------------------------------------------------------------
# JIT vs AOT cross-validation: verify_tree_greedy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bs,num_draft_tokens,num_spec_step,vocab_size",
    [
        (1, 4, 4, 32),
        (2, 8, 5, 64),
        (4, 16, 8, 128),
    ],
)
def test_verify_tree_greedy_vs_aot(bs, num_draft_tokens, num_spec_step, vocab_size):
    try:
        from sgl_kernel import verify_tree_greedy as verify_tree_greedy_aot
    except ImportError:
        pytest.skip("sgl_kernel not available")

    from sglang.jit_kernel.eagle_utils import (
        verify_tree_greedy as verify_tree_greedy_jit,
    )

    inputs_jit = make_verify_tree_greedy_inputs(
        bs, num_draft_tokens, num_spec_step, vocab_size, seed=0
    )
    inputs_aot = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in inputs_jit.items()
    }

    verify_tree_greedy_jit(**inputs_jit)
    verify_tree_greedy_aot(**inputs_aot)

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
# JIT vs AOT cross-validation: build_tree_kernel_efficient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("tree_mask_mode", [QLEN_ONLY, QLEN_ONLY_BITPACKING])
def test_build_tree_vs_aot(bs, tree_mask_mode):
    try:
        from sgl_kernel import build_tree_kernel_efficient as build_tree_aot
    except ImportError:
        pytest.skip("sgl_kernel not available")

    from sglang.jit_kernel.eagle_utils import (
        build_tree_kernel_efficient as build_tree_jit,
    )

    topk = 4
    depth = 5
    draft_token_num = topk * (depth - 1) + 1
    seq_len = 8

    inputs_jit = make_build_tree_inputs(
        bs, topk, depth, draft_token_num, seq_len, tree_mask_mode
    )
    inputs_aot = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in inputs_jit.items()
    }

    build_tree_jit(
        **inputs_jit,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=tree_mask_mode,
    )
    build_tree_aot(
        **inputs_aot,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=tree_mask_mode,
    )

    assert torch.equal(
        inputs_jit["positions"], inputs_aot["positions"]
    ), "positions mismatch"
    assert torch.equal(
        inputs_jit["retrive_index"], inputs_aot["retrive_index"]
    ), "retrive_index mismatch"
    assert torch.equal(
        inputs_jit["retrive_next_token"], inputs_aot["retrive_next_token"]
    ), "retrive_next_token mismatch"
    assert torch.equal(
        inputs_jit["retrive_next_sibling"], inputs_aot["retrive_next_sibling"]
    ), "retrive_next_sibling mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
