"""Verify metadata for a DFLASH beam tree.

The mask is the fragile part: attention reads it, and the accept kernel's links are
derived from it, so a transposed or off-by-one closure is silently wrong rather than
loud. These tests pin it from both ends -- structural invariants on the mask itself,
and an independent host-side derivation of the links that has to agree with both the
parents we started from and the device kernel we ship.
"""

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

# The NGRAM path's own host-side reimplementation of the device kernel. Independent
# of everything here -- different author, different algorithm, and live code rather
# than a test fixture (ngram_worker.py:455 uses it on the grammar path).
from sglang.srt.speculative.ngram_worker import _derive_tree_links
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
# Only the real-kernel comparison needs a device; the mask algebra does not.
register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

# block_size 8 is what the DFlash 2 checkpoint resolves to, so gamma is 7.
GAMMA = 7
PREFIX_LENS = torch.tensor([13, 5], dtype=torch.int64)
# Same batch shape, but long enough to make the kernel's prefix loop run several
# times and end on a partial tile (PREFIX_BLOCK is 1024). A prefix that fits in one
# tile would leave the loop's tail mask untested, and the tail is where a mask bug
# spills into the next request's rows.
LONG_PREFIX_LENS = torch.tensor([13, 2049], dtype=torch.int64)


def _full_mask_reference(*, ancestor_mask, prefix_lens):
    """Host derivation of the flat FULL_MASK the backends read.

    Per request: `N` rows of `prefix + N` cells, the leading `prefix` all True (a
    draft node sees the whole committed prefix) and the trailing block the request's
    ancestor closure; requests concatenated. This is what the shipped kernel writes
    on device, kept here as an independent oracle -- it is deliberately the naive
    per-request python loop, which is what the kernel replaced.
    """
    num_nodes = ancestor_mask.shape[1]
    rows = []
    for request, prefix_len in enumerate(prefix_lens.tolist()):
        prefix = torch.ones((num_nodes, int(prefix_len)), dtype=torch.bool)
        rows.append(torch.cat([prefix, ancestor_mask[request].cpu()], dim=1).flatten())
    return torch.cat(rows)


def _beam_tree(*, width, slots=GAMMA, top_k=16, seed=0):
    """`node_parents` from the real beam walk, at production shapes."""
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
    )
    return parents


def _expected_depths(*, num_nodes, width):
    """Node 0 is the root; node `i` sits on layer `(i - 1) // width + 1`."""
    return torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.arange(num_nodes - 1) // width + 1]
    )


def _parents_from_links(*, next_token, next_sibling, num_nodes):
    """Invert the first-child / next-sibling encoding back to a parent array."""
    parents = [-1] * num_nodes
    for node in range(num_nodes):
        child = int(next_token[node])
        while child != -1:
            parents[child] = node
            child = int(next_sibling[child])
    return torch.tensor(parents, dtype=torch.int64)


def test_width_one_closure_is_lower_triangular():
    """W=1 degenerates to a chain, and the chain's closure is exactly what causal
    masking already allows -- the regression gate for every later width."""
    parents = _beam_tree(width=1)
    batch_size, num_nodes = parents.shape
    assert num_nodes == 1 + GAMMA

    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    causal = torch.tril(torch.ones(num_nodes, num_nodes, dtype=torch.bool))
    assert torch.equal(mask, causal.expand(batch_size, num_nodes, num_nodes))


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_closure_row_counts_give_the_layer_index(width):
    """The kernel reads depth off the row population count, so a row holding one bit
    too many or too few shifts that node's position -- and with it its RoPE and its
    KV slot. Also pins that the root is an ancestor of every node."""
    parents = _beam_tree(width=width)
    batch_size, num_nodes = parents.shape
    mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA)

    depths = mask.sum(dim=-1) - 1
    expected = _expected_depths(num_nodes=num_nodes, width=width)
    assert torch.equal(depths, expected.expand(batch_size, num_nodes))
    # The tree is exactly gamma deep: no shallower (a layer went missing) and no
    # deeper (the closure leaked across layers).
    assert int(depths.max()) == GAMMA
    assert mask[:, :, 0].all()


def test_closure_handles_dead_ends_and_uneven_fanout():
    """The beam picks its width globally across parents, so a parent can end up with
    no children at all. A hand-built tree covers the shapes a fixed-width lattice
    cannot produce on demand -- node 1 is a dead end and node 3 is an only child."""
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
    """Three descriptions of the same layout have to land on the same total.

    `fill_verify_mask_indptr` is what the writer and the attention backend index with,
    `_full_mask_reference` is the row-by-row layout, and
    `DFlashVerifyInput.generate_attn_arg_prefill` sizes its padding from the closed form
    `sum(prefix) * N + N**2 * bs`. Any one of the three drifting is a mask written where
    the reader does not look, which is silently wrong tokens rather than an error.
    """
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
    """The mathematical precondition for keeping W=1 on the tree code path: the mask
    we hand attention must permit exactly what running with no mask permits. If the
    numbers still move at W=1, that isolates it to the backend's masked kernel path
    rather than to the closure and layout pinned here."""
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
    """The round trip that makes the mask self-checking: parents -> mask -> links ->
    parents. Transposing the mask, dropping the diagonal, or breaking the BFS order
    the beam walk promises all break it."""
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel needs a device")
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_kernel_meta_matches_the_host_derivation(width):
    """The shipped path is the device kernel, and its only other oracle is NGRAM's
    host reimplementation. `positions` has no host oracle at all, so it is checked
    against the depths the mask itself carries, offset by the committed prefix."""
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
    # The accept kernel indexes `predicts` through this, so anything but the flat
    # identity would send the target's tokens to the wrong nodes.
    assert torch.equal(
        retrive_index.cpu(), torch.arange(batch_size * num_nodes).view(batch_size, -1)
    )

    depths = _expected_depths(num_nodes=num_nodes, width=width)
    expected_positions = (PREFIX_LENS[:, None] + depths).flatten()
    assert torch.equal(positions.cpu(), expected_positions)


def test_tree_meta_rejects_narrow_prefix_lens():
    """The CUDA kernel casts `verified_seq_len` to int64 without checking, so an
    int32 `seq_lens` would be read as garbage positions instead of failing."""
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
    """The shipped writer against the host oracle, in both buffer dtypes it meets.

    The buffer is reused across decode steps and arrives holding the previous step's
    bits, at different offsets, because a request's rows move as its prefix grows. So
    every live cell has to be rewritten: `poison=True` catches a writer that skips
    closure zeros, `poison=False` one that skips the all-True prefix span -- the
    tempting "the prefix never changes" shortcut, which is wrong for exactly that
    reason. `dtype` covers both buffers in production: the triton backend's captured
    mask is uint8, the worker's own fallback is bool.
    """
    parents = _beam_tree(width=width)
    batch_size, num_nodes = parents.shape
    prefix_lens = LONG_PREFIX_LENS
    ancestor_mask = build_ancestor_mask(node_parents=parents, max_depth=GAMMA).cuda()

    expected = _full_mask_reference(
        ancestor_mask=ancestor_mask, prefix_lens=prefix_lens
    )
    # Deliberately larger than needed, like the backend's captured buffer.
    buffer = torch.full(
        (expected.numel() + 4096,), poison, dtype=dtype, device="cuda"
    )

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
