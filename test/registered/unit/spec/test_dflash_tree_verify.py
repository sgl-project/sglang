"""Test DFLASH tree verification wiring and chain equivalence."""

import sys

import pytest
import torch

from sglang.srt.speculative.dflash_tree_verify import (
    accept_tree_greedy,
    build_tree_verify_input,
    commit_positions,
    compact_hidden_to_commit_layout,
)
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.dflash_worker_v2 import _commit_accept
from sglang.srt.speculative.spec_utils import verify_commit_step_indices
from sglang.test.ci.ci_register import register_cuda_ci

# These tests cover CUDA-only verification kernels.
register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the verify kernels need a device"
)

BLOCK_SIZE = 8
PREFIX_LENS = torch.tensor([13, 5], dtype=torch.int64)


def _chain_parents(*, bs, block_size):
    """`node_parents` for a width-1 beam: node i hangs off node i-1."""
    parents = torch.arange(-1, block_size - 1, dtype=torch.int64)
    return parents.unsqueeze(0).expand(bs, block_size).contiguous()


def _mask_scratch(*, prefix_lens, num_nodes):
    """Return test buffers for the device-side full mask writer."""
    return {
        "mask_buffer": torch.zeros(
            num_nodes * int((prefix_lens + num_nodes).sum()),
            dtype=torch.bool,
            device="cuda",
        ),
        "mask_indptr": torch.zeros(
            prefix_lens.numel() + 1, dtype=torch.int64, device="cuda"
        ),
    }


def _logits_from_predictions(target_predict):
    """One-hot logits, so `argmax` inside the accept returns `target_predict`."""
    bs, width = target_predict.shape
    logits = torch.zeros((bs * width, 4096), dtype=torch.float32)
    logits.scatter_(1, target_predict.reshape(-1, 1).to(torch.int64), 1.0)
    return logits.cuda()


def _equivalence_case():
    """Return partial, empty, and full acceptance cases."""
    candidates = torch.tensor(
        [
            [100, 101, 102, 103, 999, 105, 106, 107],
            [200, 777, 202, 203, 204, 205, 206, 207],
            [300, 301, 302, 303, 304, 305, 306, 307],
        ],
        dtype=torch.int64,
    )
    target_predict = torch.tensor(
        [
            [101, 102, 103, 888, 700, 701, 702, 703],
            [201, 800, 801, 802, 803, 804, 805, 806],
            [301, 302, 303, 304, 305, 306, 307, 308],
        ],
        dtype=torch.int64,
    )
    return candidates, target_predict


def test_width_one_tree_accept_matches_the_chain():
    """Width-one tree acceptance must match chain acceptance."""
    candidates, target_predict = _equivalence_case()
    bs = candidates.shape[0]
    prefix_lens = torch.tensor([13, 5, 21], dtype=torch.int64)

    chain_len, chain_bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates.cuda(), target_predict=target_predict.cuda()
    )
    chain_out, chain_commit = _commit_accept(candidates.cuda(), chain_len, chain_bonus)
    assert chain_commit.cpu().tolist() == [4, 1, 8]

    verify_input = build_tree_verify_input(
        node_tokens=candidates.cuda(),
        node_parents=_chain_parents(bs=bs, block_size=BLOCK_SIZE).cuda(),
        block_size=BLOCK_SIZE,
        tree_width=1,
        prefix_lens=prefix_lens.cuda(),
        **_mask_scratch(prefix_lens=prefix_lens, num_nodes=BLOCK_SIZE),
    )
    accepted = accept_tree_greedy(
        verify_input=verify_input,
        next_token_logits=_logits_from_predictions(target_predict),
        bs=bs,
    )

    assert torch.equal(accepted.commit_lens.cpu(), chain_commit.cpu())
    assert torch.equal(
        accepted.bonus_tokens.cpu().to(torch.int64), chain_bonus.cpu().to(torch.int64)
    )
    for request, length in enumerate(chain_commit.cpu().tolist()):
        assert torch.equal(
            accepted.out_tokens[request, :length].cpu().to(torch.int64),
            chain_out[request, :length].cpu(),
        ), f"request {request} committed a different run"


def test_width_one_accept_index_is_the_identity_chain():
    """Width-one accept indices must be the identity chain."""
    candidates, target_predict = _equivalence_case()
    bs = candidates.shape[0]
    prefix_lens = torch.tensor([13, 5, 21], dtype=torch.int64)

    verify_input = build_tree_verify_input(
        node_tokens=candidates.cuda(),
        node_parents=_chain_parents(bs=bs, block_size=BLOCK_SIZE).cuda(),
        block_size=BLOCK_SIZE,
        tree_width=1,
        prefix_lens=prefix_lens.cuda(),
        **_mask_scratch(prefix_lens=prefix_lens, num_nodes=BLOCK_SIZE),
    )
    accepted = accept_tree_greedy(
        verify_input=verify_input,
        next_token_logits=_logits_from_predictions(target_predict),
        bs=bs,
    )

    accept_index = accepted.accept_index.cpu()
    for request, length in enumerate(accepted.commit_lens.cpu().tolist()):
        expected = request * BLOCK_SIZE + torch.arange(length, dtype=torch.int32)
        assert torch.equal(accept_index[request, :length], expected)
        assert (accept_index[request, length:] == -1).all()


def test_predict_stays_in_vocabulary_where_accept_index_pads():
    """Padded accept indices must gather valid prediction tokens."""
    candidates, target_predict = _equivalence_case()
    bs = candidates.shape[0]
    prefix_lens = torch.tensor([13, 5, 21], dtype=torch.int64)
    logits = _logits_from_predictions(target_predict)

    verify_input = build_tree_verify_input(
        node_tokens=candidates.cuda(),
        node_parents=_chain_parents(bs=bs, block_size=BLOCK_SIZE).cuda(),
        block_size=BLOCK_SIZE,
        tree_width=1,
        prefix_lens=prefix_lens.cuda(),
        **_mask_scratch(prefix_lens=prefix_lens, num_nodes=BLOCK_SIZE),
    )
    accepted = accept_tree_greedy(
        verify_input=verify_input, next_token_logits=logits, bs=bs
    )

    assert (accepted.accept_index == -1).any()
    padded_reads = accepted.predict[accepted.accept_index.to(torch.int64).reshape(-1)]
    assert int(padded_reads.min()) >= 0
    assert int(padded_reads.max()) < logits.shape[-1]


# An off-spine accepted path exercises node/depth differences.
TREE_WIDTH = 2
VERIFY_WIDTH = 1 + (BLOCK_SIZE - 1) * TREE_WIDTH
OFF_SPINE_ACCEPT = [0, 2, 4] + [-1] * (BLOCK_SIZE - 3)


def _off_spine_accept_index(*, bs):
    row = torch.tensor(OFF_SPINE_ACCEPT, dtype=torch.int32)
    offsets = torch.arange(bs, dtype=torch.int32) * VERIFY_WIDTH
    # Flat node ids, per request, with the -1 pad preserved.
    return torch.where(row.unsqueeze(0) < 0, row.unsqueeze(0), row + offsets[:, None])


def _fixed_width_parents(*, block_size, tree_width):
    """Build a valid BFS-ordered fixed-width tree."""
    parents = [-1]
    for depth in range(1, block_size):
        prev_first = 1 + (depth - 2) * tree_width if depth >= 2 else 0
        prev_count = tree_width if depth >= 2 else 1
        parents.extend(prev_first + slot % prev_count for slot in range(tree_width))
    return torch.tensor(parents, dtype=torch.int64)


class _StubBatch:
    """Minimal batch stub for verify commit-index arithmetic."""

    def __init__(self, seq_lens):
        self.seq_lens = seq_lens
        self.mamba_track_indices = None


def test_mamba_step_index_follows_the_node_not_the_depth():
    """Mamba commit index must follow the last accepted node."""
    bs = 2
    accept_index = _off_spine_accept_index(bs=bs).cuda()
    commit_lens = torch.full((bs,), 3, dtype=torch.int32, device="cuda")

    last_step, track_step = verify_commit_step_indices(
        batch=_StubBatch(seq_lens=torch.tensor([13, 5], device="cuda")),
        accept_index=accept_index,
        accept_lens=commit_lens,
        draft_token_num=VERIFY_WIDTH,
    )

    assert last_step.cpu().tolist() == [4, 4]
    assert last_step.cpu().tolist() != (commit_lens.cpu() - 1).tolist()
    assert track_step is None  # tracking off -> no interval-crossing step


def test_commit_layout_gathers_source_rows_and_chain_positions():
    """Compaction must map accepted nodes to chain slots and positions."""
    bs = 2
    accept_index = _off_spine_accept_index(bs=bs).cuda()
    hidden = (
        torch.arange(bs * VERIFY_WIDTH, dtype=torch.float32)
        .unsqueeze(1)
        .expand(bs * VERIFY_WIDTH, 8)
        .contiguous()
        .cuda()
    )

    compacted = compact_hidden_to_commit_layout(
        target_hidden=hidden,
        accept_index=accept_index,
        bs=bs,
        verify_width=VERIFY_WIDTH,
    ).view(bs, VERIFY_WIDTH, 8)

    for request in range(bs):
        for depth, node in enumerate(OFF_SPINE_ACCEPT[:3]):
            expected = request * VERIFY_WIDTH + node
            assert compacted[request, depth, 0].item() == expected, (
                f"depth {depth} of request {request} holds row "
                f"{compacted[request, depth, 0].item()}, not node {expected}"
            )

    positions = commit_positions(
        prefix_lens=PREFIX_LENS.cuda(), verify_width=VERIFY_WIDTH
    ).view(bs, VERIFY_WIDTH)
    for request, prefix in enumerate(PREFIX_LENS.tolist()):
        assert torch.equal(
            positions[request].cpu(),
            prefix + torch.arange(VERIFY_WIDTH, dtype=torch.int64),
        )


def test_tree_meta_links_match_the_beam_parents():
    """Tree metadata links must match beam parents."""
    bs = 2
    node_parents = (
        _fixed_width_parents(block_size=BLOCK_SIZE, tree_width=TREE_WIDTH)
        .unsqueeze(0)
        .expand(bs, -1)
        .contiguous()
    )
    assert node_parents.shape[1] == VERIFY_WIDTH

    verify_input = build_tree_verify_input(
        node_tokens=torch.arange(bs * VERIFY_WIDTH, dtype=torch.int64)
        .view(bs, VERIFY_WIDTH)
        .cuda(),
        node_parents=node_parents.cuda(),
        block_size=BLOCK_SIZE,
        tree_width=TREE_WIDTH,
        prefix_lens=PREFIX_LENS.cuda(),
        **_mask_scratch(prefix_lens=PREFIX_LENS, num_nodes=VERIFY_WIDTH),
    )

    recovered = [-1] * VERIFY_WIDTH
    next_token = verify_input.retrieve_next_token[0].cpu().tolist()
    next_sibling = verify_input.retrieve_next_sibling[0].cpu().tolist()
    for node in range(VERIFY_WIDTH):
        child = next_token[node]
        while child != -1:
            recovered[child] = node
            child = next_sibling[child]
    assert recovered == node_parents[0].tolist()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
