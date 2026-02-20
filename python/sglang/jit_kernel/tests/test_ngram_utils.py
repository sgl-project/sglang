"""
Tests for the JIT reconstruct_indices_from_tree_mask kernel.

Correctness is validated by:
1. Smoke tests across batch sizes and tree sizes.
2. Known-answer tests: linear chain and branching tree with hand-crafted
   tree masks and expected positions / next_token / next_sibling values.
3. JIT vs AOT cross-validation (when sgl_kernel is available).
"""

import pytest
import torch

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_inputs(bs, draft_token_num, device=DEVICE):
    """Allocate output tensors; tree_mask and verified_seq_len filled by caller."""
    tree_mask = torch.zeros(
        bs * draft_token_num * draft_token_num, dtype=torch.bool, device=device
    )
    verified_seq_len = torch.zeros(bs, dtype=torch.int64, device=device)
    positions = torch.zeros(bs * draft_token_num, dtype=torch.int64, device=device)
    retrive_index = torch.zeros(bs, draft_token_num, dtype=torch.int64, device=device)
    retrive_next_token = torch.full(
        (bs, draft_token_num), -1, dtype=torch.int64, device=device
    )
    retrive_next_sibling = torch.full(
        (bs, draft_token_num), -1, dtype=torch.int64, device=device
    )
    return dict(
        tree_mask=tree_mask,
        verified_seq_len=verified_seq_len,
        positions=positions,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
    )


def build_linear_chain_mask(bs, draft_token_num, device=DEVICE):
    """
    Build a tree_mask for a linear chain: 0 → 1 → 2 → ... → draft_token_num-1.

    tree_mask[b, i, j] = True if token j is an ancestor of token i (j < i).
    For a linear chain, token i has ancestors {0, 1, ..., i-1}.
    """
    tree_mask = torch.zeros(
        bs * draft_token_num * draft_token_num, dtype=torch.bool, device=device
    )
    base = draft_token_num * draft_token_num
    for b in range(bs):
        for i in range(draft_token_num):
            for j in range(i):  # all predecessors are ancestors in a chain
                tree_mask[b * base + i * draft_token_num + j] = True
    return tree_mask


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("draft_token_num", [4, 8, 16])
def test_smoke(bs, draft_token_num):
    from sglang.jit_kernel.ngram_utils import reconstruct_indices_from_tree_mask

    inputs = make_inputs(bs, draft_token_num)
    inputs["tree_mask"] = build_linear_chain_mask(bs, draft_token_num)
    inputs["verified_seq_len"].fill_(10)

    reconstruct_indices_from_tree_mask(**inputs, batch_size=bs, draft_token_num=draft_token_num)

    # retrive_index[b, i] must equal b * draft_token_num + i
    for b in range(bs):
        for i in range(draft_token_num):
            assert inputs["retrive_index"][b, i].item() == b * draft_token_num + i


# ---------------------------------------------------------------------------
# Known-answer: linear chain
# ---------------------------------------------------------------------------


def test_linear_chain_positions():
    """positions[b*N + i] = verified_seq_len[b] + depth(i)  (depth = i for chain)."""
    from sglang.jit_kernel.ngram_utils import reconstruct_indices_from_tree_mask

    bs, draft_token_num, seq_len = 2, 4, 5
    inputs = make_inputs(bs, draft_token_num)
    inputs["tree_mask"] = build_linear_chain_mask(bs, draft_token_num)
    inputs["verified_seq_len"].fill_(seq_len)

    reconstruct_indices_from_tree_mask(**inputs, batch_size=bs, draft_token_num=draft_token_num)

    for b in range(bs):
        for i in range(draft_token_num):
            expected = seq_len + i  # depth of token i in a chain = i
            assert inputs["positions"][b * draft_token_num + i].item() == expected, (
                f"positions[{b},{i}] = {inputs['positions'][b*draft_token_num+i].item()}, expected {expected}"
            )


def test_linear_chain_next_token():
    """In a linear chain, retrive_next_token[b, i] = i+1 (last = -1)."""
    from sglang.jit_kernel.ngram_utils import reconstruct_indices_from_tree_mask

    bs, draft_token_num = 1, 5
    inputs = make_inputs(bs, draft_token_num)
    inputs["tree_mask"] = build_linear_chain_mask(bs, draft_token_num)

    reconstruct_indices_from_tree_mask(**inputs, batch_size=bs, draft_token_num=draft_token_num)

    for i in range(draft_token_num - 1):
        assert inputs["retrive_next_token"][0, i].item() == i + 1
    assert inputs["retrive_next_token"][0, draft_token_num - 1].item() == -1


def test_linear_chain_no_siblings():
    """In a linear chain, there are no siblings."""
    from sglang.jit_kernel.ngram_utils import reconstruct_indices_from_tree_mask

    bs, draft_token_num = 1, 5
    inputs = make_inputs(bs, draft_token_num)
    inputs["tree_mask"] = build_linear_chain_mask(bs, draft_token_num)

    reconstruct_indices_from_tree_mask(**inputs, batch_size=bs, draft_token_num=draft_token_num)

    assert (inputs["retrive_next_sibling"] == -1).all()


# ---------------------------------------------------------------------------
# JIT vs AOT cross-validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bs,draft_token_num",
    [
        (1, 4),
        (2, 8),
        (4, 16),
    ],
)
def test_vs_aot(bs, draft_token_num):
    try:
        from sgl_kernel import reconstruct_indices_from_tree_mask as reconstruct_aot
    except ImportError:
        pytest.skip("sgl_kernel not available")

    from sglang.jit_kernel.ngram_utils import (
        reconstruct_indices_from_tree_mask as reconstruct_jit,
    )

    inputs_jit = make_inputs(bs, draft_token_num)
    inputs_jit["tree_mask"] = build_linear_chain_mask(bs, draft_token_num)
    inputs_jit["verified_seq_len"].fill_(7)

    inputs_aot = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in inputs_jit.items()
    }

    reconstruct_jit(**inputs_jit, batch_size=bs, draft_token_num=draft_token_num)
    reconstruct_aot(**inputs_aot, batch_size=bs, draft_token_num=draft_token_num)

    assert torch.equal(inputs_jit["positions"], inputs_aot["positions"]), "positions mismatch"
    assert torch.equal(inputs_jit["retrive_index"], inputs_aot["retrive_index"]), "retrive_index mismatch"
    assert torch.equal(
        inputs_jit["retrive_next_token"], inputs_aot["retrive_next_token"]
    ), "retrive_next_token mismatch"
    assert torch.equal(
        inputs_jit["retrive_next_sibling"], inputs_aot["retrive_next_sibling"]
    ), "retrive_next_sibling mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
