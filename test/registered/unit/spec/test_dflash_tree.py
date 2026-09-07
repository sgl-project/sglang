"""Test DFLASH beam-tree masks and verification metadata."""

import sys

import pytest
import torch

from sglang.kernels.ops.speculative.dflash import write_dflash_tree_full_mask
from sglang.srt.layers.attention.verify_mask import fill_verify_mask_indptr
from sglang.srt.models.dflash import _beam_walk_torch
from sglang.srt.speculative.dflash_tree import (
    build_ancestor_mask,
    build_dflash_tree_meta,
)

# Reuse the live NGRAM host implementation as an independent link oracle.
from sglang.srt.speculative.ngram_worker import _derive_tree_links
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
# Only the real-kernel comparison needs a device; the mask algebra does not.
register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

GAMMA = 7
PREFIX_LENS = torch.tensor([13, 5], dtype=torch.int64)
LONG_PREFIX_LENS = torch.tensor([13, 2049], dtype=torch.int64)


def _full_mask_reference(*, ancestor_mask, prefix_lens):
    """Reference implementation of the flat full mask."""
    num_nodes = ancestor_mask.shape[1]
    rows = []
    for request, prefix_len in enumerate(prefix_lens.tolist()):
        prefix = torch.ones((num_nodes, int(prefix_len)), dtype=torch.bool)
        rows.append(torch.cat([prefix, ancestor_mask[request].cpu()], dim=1).flatten())
    return torch.cat(rows)


def _beam_tree(*, width, slots=GAMMA, top_k=16, seed=0, max_num_nodes=None):
    """Return parents from the production-shaped beam walk."""
    batch_size = PREFIX_LENS.numel()
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randint(
        -5, 6, (batch_size, slots, top_k, top_k), generator=generator
    ).float()
    candidate_ids = (
        torch.arange(slots * top_k)
        .view(1, slots, top_k)
        .expand(batch_size, slots, top_k)
        .contiguous()
    )
    _, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=torch.arange(batch_size) + 9001,
        beam_width=width,
        max_num_nodes=max_num_nodes,
    )
    return parents


def _expected_depths(*, num_nodes, width):
    """Return BFS layer indices."""
    return torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.arange(num_nodes - 1) // width + 1]
    )


def _parents_from_links(*, next_token, next_sibling, num_nodes):
    """Invert first-child/next-sibling links to parent indices."""
    parents = [-1] * num_nodes
    for node in range(num_nodes):
        child = int(next_token[node])
        while child != -1:
            parents[child] = node
            child = int(next_sibling[child])
    return torch.tensor(parents, dtype=torch.int64)


def test_width_one_closure_is_lower_triangular():
    """Width 1 must produce the causal chain mask."""
    parents = _beam_tree(width=1)
    batch_size, num_nodes = parents.shape
    assert num_nodes == 1 + GAMMA

    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    causal = torch.tril(torch.ones(num_nodes, num_nodes, dtype=torch.bool))
    assert torch.equal(mask, causal.expand(batch_size, num_nodes, num_nodes))


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_closure_row_counts_give_the_layer_index(width):
    """Closure row counts must recover each node's layer."""
    parents = _beam_tree(width=width)
    batch_size, num_nodes = parents.shape
    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    depths = mask.sum(dim=-1) - 1
    expected = _expected_depths(num_nodes=num_nodes, width=width)
    assert torch.equal(depths, expected.expand(batch_size, num_nodes))
    assert int(depths.max()) == GAMMA
    assert mask[:, :, 0].all()


def test_closure_handles_dead_ends_and_uneven_fanout():
    """Closure must handle dead ends and uneven fanout."""
    parents = torch.tensor([[-1, 0, 0, 2, 2, 3]], dtype=torch.int64)

    mask = build_ancestor_mask(node_parents=parents, max_depth=3)

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask[0], expected)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_row_offsets_and_layout_agree_on_the_total_size(width):
    """Row offsets and full-mask size must agree."""
    parents = _beam_tree(width=width)
    batch_size, num_nodes = parents.shape
    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    reference = _full_mask_reference(ancestor_mask=mask, prefix_lens=PREFIX_LENS)
    indptr = fill_verify_mask_indptr(
        mask_indptr=torch.zeros((batch_size + 1,), dtype=torch.int64),
        seq_lens=PREFIX_LENS,
        num_draft_tokens=num_nodes,
        bs=batch_size,
    )

    closed_form = int(PREFIX_LENS.sum()) * num_nodes + num_nodes**2 * batch_size
    assert int(indptr[batch_size]) == closed_form
    assert reference.numel() == closed_form


def test_width_one_mask_row_is_exactly_causal():
    """Width 1 full mask must be exactly causal."""
    parents = _beam_tree(width=1)
    num_nodes = parents.shape[1]
    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    full = _full_mask_reference(ancestor_mask=mask, prefix_lens=PREFIX_LENS)

    causal = torch.tril(torch.ones(num_nodes, num_nodes, dtype=torch.bool))
    expected = torch.cat(
        [
            torch.cat(
                [torch.ones(num_nodes, int(prefix), dtype=torch.bool), causal], dim=1
            ).flatten()
            for prefix in PREFIX_LENS.tolist()
        ]
    )
    assert torch.equal(full, expected)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_links_derived_from_the_mask_recover_the_parents(width):
    """Mask-derived links must recover the original parents."""
    parents = _beam_tree(width=width)
    batch_size, num_nodes = parents.shape
    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    next_token, next_sibling = _derive_tree_links(mask.numpy(), batch_size, num_nodes)

    for request in range(batch_size):
        recovered = _parents_from_links(
            next_token=next_token[request],
            next_sibling=next_sibling[request],
            num_nodes=num_nodes,
        )
        assert torch.equal(recovered, parents[request])


@pytest.mark.parametrize("width,max_num_nodes", [(8, 15), (8, 30), (16, 30)])
def test_links_derived_from_the_mask_recover_a_pruned_tree(width, max_num_nodes):
    """Mask-derived links must also recover pruned trees."""
    parents = _beam_tree(width=width, max_num_nodes=max_num_nodes)
    batch_size, num_nodes = parents.shape
    assert num_nodes == max_num_nodes

    child_counts = torch.zeros(num_nodes, dtype=torch.int64)
    child_counts.scatter_add_(
        0, parents[0, 1:].clamp(min=0), torch.ones(num_nodes - 1, dtype=torch.int64)
    )
    interior = child_counts[: num_nodes - 1]
    assert (interior == 0).any(), "expected a parent the prune left childless"
    assert (interior > 0).any() and int(interior.max()) < width

    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)
    next_token, next_sibling = _derive_tree_links(mask.numpy(), batch_size, num_nodes)

    for request in range(batch_size):
        recovered = _parents_from_links(
            next_token=next_token[request],
            next_sibling=next_sibling[request],
            num_nodes=num_nodes,
        )
        assert torch.equal(recovered, parents[request])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel needs a device")
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_kernel_meta_matches_the_host_derivation(width):
    """Device metadata must match host links and mask depths."""
    parents = _beam_tree(width=width)
    batch_size, num_nodes = parents.shape
    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    positions, retrive_index, next_token, next_sibling = build_dflash_tree_meta(
        ancestor_mask=mask.cuda(), prefix_lens=PREFIX_LENS.cuda()
    )

    expected_token, expected_sibling = _derive_tree_links(
        mask.numpy(), batch_size, num_nodes
    )
    assert torch.equal(next_token.cpu(), expected_token)
    assert torch.equal(next_sibling.cpu(), expected_sibling)
    assert torch.equal(
        retrive_index.cpu(), torch.arange(batch_size * num_nodes).view(batch_size, -1)
    )

    depths = _expected_depths(num_nodes=num_nodes, width=width)
    expected_positions = (PREFIX_LENS[:, None] + depths).flatten()
    assert torch.equal(positions.cpu(), expected_positions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel needs a device")
@pytest.mark.parametrize("width,max_num_nodes", [(8, 15), (8, 30), (16, 30)])
def test_kernel_positions_follow_the_closure_on_a_pruned_tree(width, max_num_nodes):
    """Pruned node positions must follow mask-derived depths."""
    parents = _beam_tree(width=width, max_num_nodes=max_num_nodes)
    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    positions, *_ = build_dflash_tree_meta(
        ancestor_mask=mask.cuda(), prefix_lens=PREFIX_LENS.cuda()
    )

    depths = mask.sum(dim=2) - 1
    assert int(depths.max()) < GAMMA + 1
    assert not torch.equal(
        depths[0], _expected_depths(num_nodes=max_num_nodes, width=width)
    )
    expected_positions = (PREFIX_LENS[:, None] + depths).flatten()
    assert torch.equal(positions.cpu(), expected_positions)


def test_tree_meta_rejects_narrow_prefix_lens():
    """Reject prefix lengths with the wrong dtype for the CUDA kernel."""
    parents = _beam_tree(width=2)
    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)
    with pytest.raises(ValueError, match="int64"):
        build_dflash_tree_meta(
            ancestor_mask=mask, prefix_lens=PREFIX_LENS.to(torch.int32)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel needs a device")
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("poison", [False, True])
@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8])
def test_mask_kernel_rewrites_every_live_cell(width, poison, dtype):
    """The device mask writer must rewrite every live cell."""
    parents = _beam_tree(width=width)
    batch_size, num_nodes = parents.shape
    prefix_lens = LONG_PREFIX_LENS
    ancestor_mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA).cuda()

    expected = _full_mask_reference(
        ancestor_mask=ancestor_mask, prefix_lens=prefix_lens
    )
    buffer = torch.full((expected.numel() + 4096,), poison, dtype=dtype, device="cuda")

    written = write_dflash_tree_full_mask(
        ancestor_mask=ancestor_mask,
        mask_indptr=fill_verify_mask_indptr(
            mask_indptr=torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device="cuda"
            ),
            seq_lens=prefix_lens.cuda(),
            num_draft_tokens=num_nodes,
            bs=batch_size,
        ),
        seq_lens=prefix_lens.cuda(),
        out=buffer,
    )

    assert written.data_ptr() == buffer.data_ptr(), "must write in place"
    assert torch.equal(written[: expected.numel()].cpu().bool(), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
