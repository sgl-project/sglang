import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    _beam_walk_torch,
    _grouped_conv,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
# The triton beam walk needs a device; the rest of this file needs no kernel.
register_cuda_ci(est_time=3, stage="base-b", runner_config="1-gpu-small")


def test_dflash_unary_logit_transform():
    logits = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.bfloat16)
    for fields in ({}, {"output_multiplier": 0.2, "final_logit_softcapping": 20.0}):
        config = parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    **fields,
                },
            }
        )
        actual = DFlash2DraftModel._transform_unary_logits(
            SimpleNamespace(draft_config=config), logits
        )
        expected = logits.float() * config.output_multiplier
        if config.final_logit_softcapping is not None:
            expected = torch.tanh(expected / config.final_logit_softcapping)
            expected *= config.final_logit_softcapping
        torch.testing.assert_close(actual, expected)


def test_selector_greedy_row_walk_is_deterministic_in_a_mixed_batch():
    """A greedy row walks the argmax, so the q it hands verify has to be the point
    mass there. Greedy reaches the selector as top_k=1 with the temperature reset
    to 1.0, so a softmax q stays a real distribution and verify would
    rejection-sample a deterministic request against it. The row must also not
    depend on who else is in the batch."""
    selector = CandidateSelector(hidden_size=4, vocab_size=16, state_rank=2, top_k=4)
    torch.manual_seed(1)
    candidate_ids = torch.randint(0, 16, (2, 3, 4))
    scores = torch.randn(2, 3, 4, 4)
    uniforms = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.6]])
    temperatures = torch.tensor([1.0, 0.7])
    greedy_mask = torch.tensor([True, False])

    mixed_tokens, mixed_q = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    assert torch.all((mixed_q[0] == 0) | (mixed_q[0] == 1))
    for row in range(2):
        tokens, q_rows = selector.sample_path(
            candidate_ids=candidate_ids[row : row + 1],
            scores=scores[row : row + 1],
            uniforms=uniforms[row : row + 1],
            temperatures=temperatures[row : row + 1],
            greedy_mask=greedy_mask[row : row + 1],
        )
        torch.testing.assert_close(mixed_tokens[row], tokens[0])
        torch.testing.assert_close(mixed_q[row], q_rows[0])


def test_selector_rejects_a_quantized_target_lm_head():
    """The candidate matmuls read the lm_head weight directly, so a packed or
    absent weight would be read as if it were dense."""
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(8, 4, dtype=torch.int8)),
        candidate_selector=SimpleNamespace(top_k=4),
    )
    with pytest.raises(RuntimeError, match="requires a dense"):
        DFlash2DraftModel.compute_candidates(model, torch.randn(2, 4))


def _lattice(*, bs, slots, top_k, seed=0, spread=8.0):
    """A lattice whose scores are well separated, so index selection is stable.

    Beam picks compare fp32 sums across a flattened axis; the triton kernel and the
    torch reference normalize with different reduction orders and differ by ~1e-7.
    Integer-valued scores keep every comparison gap far above that, which is what
    makes elementwise equality a non-flaky assertion.
    """
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randint(
        -5, 6, (bs, slots, top_k, top_k), generator=generator
    ).float()
    # Distinct ids per slot, so "same-parent siblings differ" is about the walk
    # rather than about the candidate set happening to repeat a token.
    candidate_ids = (
        torch.arange(slots * top_k).view(1, slots, top_k).expand(bs, slots, top_k)
    )
    anchor = torch.arange(bs) + 9001
    return candidate_ids.contiguous(), scores * spread, anchor


@pytest.mark.parametrize("slots,top_k", [(3, 4), (7, 16)])
def test_beam_width_one_reproduces_the_greedy_chain(slots, top_k):
    """Width 1 is the regression gate for every later phase: the tree must degenerate
    to the chain `sample_path` already produces, root included at index 0."""
    bs = 3
    candidate_ids, scores, anchor = _lattice(bs=bs, slots=slots, top_k=top_k)
    selector = CandidateSelector(
        hidden_size=4, vocab_size=64, state_rank=2, top_k=top_k
    )

    chain, _ = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=torch.zeros(bs, slots),
        temperatures=torch.ones(bs),
        greedy_mask=torch.ones(bs, dtype=torch.bool),
    )
    tokens, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=1,
    )

    assert torch.equal(tokens[:, 0], anchor)
    assert torch.equal(tokens[:, 1:], chain)
    expected_parents = torch.cat([torch.tensor([-1]), torch.arange(slots)])
    assert torch.equal(parents, expected_parents.expand(bs, slots + 1))


@pytest.mark.parametrize("beam_width", [1, 2, 3, 4])
@pytest.mark.parametrize("slots,top_k", [(3, 4), (7, 16)])
def test_tree_is_fixed_width_and_bfs_ordered(beam_width, slots, top_k):
    """`parent < index` is the precondition for the single-pass ancestor closure and
    for `reconstruct_indices_from_tree_mask`; the fixed node count is what keeps the
    verify window a constant length. Width 3 also covers the padded-lane path."""
    bs = 2
    candidate_ids, scores, anchor = _lattice(bs=bs, slots=slots, top_k=top_k)
    tokens, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=beam_width,
    )

    num_nodes = 1 + slots * beam_width
    assert tokens.shape == (bs, num_nodes)
    assert parents.shape == (bs, num_nodes)
    assert torch.equal(parents[:, 0], torch.full((bs,), -1))
    assert (parents[:, 1:] < torch.arange(1, num_nodes)).all()
    # Every parent lives on the depth immediately above.
    for slot in range(slots):
        base = 1 + slot * beam_width
        block = parents[:, base : base + beam_width]
        lower = 1 + (slot - 1) * beam_width if slot else 0
        assert (block >= lower).all()
        assert (block < base).all()


def test_spine_is_independent_of_the_beam_width():
    """The spine is beam 0's own argmax chain and never reads the accumulated score,
    so one lattice must yield one spine at every width. Miscomputing `cum`,
    normalizing on the wrong axis, or shuffling the beam order all break this while
    leaving the shape invariants intact."""
    slots, top_k = 7, 16
    candidate_ids, scores, anchor = _lattice(bs=2, slots=slots, top_k=top_k, seed=3)

    spines = []
    for beam_width in (1, 2, 4, 8):
        tokens, _ = _beam_walk_torch(
            candidate_ids=candidate_ids,
            scores=scores,
            anchor_token_ids=anchor,
            beam_width=beam_width,
        )
        spines.append(tokens[:, [1 + slot * beam_width for slot in range(slots)]])

    for spine in spines[1:]:
        assert torch.equal(spine, spines[0])


def test_same_parent_siblings_carry_distinct_tokens():
    """`verify_tree_greedy` sweeps a sibling chain and takes the first match, so a
    repeated token under one parent could send the walk into another subtree and stop
    earlier -- which would break "tree acceptance >= chain acceptance". Distinctness
    holds structurally because the width picks are distinct (beam, candidate) pairs;
    switching to per-state dedup, or letting predecessors share candidate numbering,
    loses it."""
    slots, top_k, beam_width = 7, 16, 4
    candidate_ids, scores, anchor = _lattice(bs=3, slots=slots, top_k=top_k, seed=5)
    tokens, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=beam_width,
    )

    for row in range(tokens.shape[0]):
        for slot in range(slots):
            base = 1 + slot * beam_width
            groups = {}
            for beam in range(beam_width):
                parent = int(parents[row, base + beam])
                groups.setdefault(parent, []).append(int(tokens[row, base + beam]))
            for siblings in groups.values():
                assert len(siblings) == len(set(siblings))


def test_rows_are_normalized_before_the_scores_accumulate():
    """Two predecessors with identical conditional distributions but rows offset by a
    constant must compete equally: a logit row is only defined up to a per-row
    additive constant, so accumulating raw scores lets the offset alone decide which
    subtree eats the budget. Here the +10 row would take both free slots and starve
    node 2 into a dead end."""
    beam_width = top_k = 3
    scores = torch.zeros(1, 2, top_k, top_k)
    scores[0, 1, 0] = torch.tensor([0.0, -1.0, -2.0])
    scores[0, 1, 1] = torch.tensor([0.0, -1.0, -2.0])
    scores[0, 1, 2] = torch.tensor([10.0, 9.0, 8.0])
    candidate_ids = torch.arange(2 * top_k).view(1, 2, top_k)

    _, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=torch.tensor([7]),
        beam_width=beam_width,
    )

    # Depth 1 is nodes 1..3; every one of them must have kept a child.
    assert sorted(int(p) for p in parents[0, 1 + beam_width :]) == [1, 2, 3]


def test_exact_ties_break_beam_major():
    """Flattening the pool as `beam * top_k + candidate` is a convention the torch
    reference and the triton kernel have to share; c-major would pick a different
    parent on a tie. Within-row ties are reproducible in both (equal inputs give equal
    `log_softmax` outputs), so this pins the flattening without relying on luck."""
    beam_width = top_k = 2
    scores = torch.zeros(1, 2, top_k, top_k)
    candidate_ids = torch.arange(2 * top_k).view(1, 2, top_k)

    _, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=torch.tensor([7]),
        beam_width=beam_width,
    )

    # Depth 2 is nodes 3 and 4. Node 3 is the spine, under node 1. Every remaining
    # pool entry ties, so beam-major hands node 4 to beam 0 -- node 1, not node 2.
    assert parents[0, 3] == 1
    assert parents[0, 4] == 1


def test_beam_walk_rejects_a_width_above_the_candidate_count():
    """Depth 1 only has top_k candidates, so a wider beam cannot fill a fixed width;
    the resolved server args promise this never happens, and the walk says so rather
    than silently emitting duplicates."""
    selector = CandidateSelector(hidden_size=4, vocab_size=64, state_rank=2, top_k=4)
    candidate_ids, scores, anchor = _lattice(bs=1, slots=3, top_k=4)
    with pytest.raises(ValueError, match="selector_top_k"):
        selector.beam_walk(
            candidate_ids=candidate_ids,
            scores=scores,
            anchor_token_ids=anchor,
            beam_width=8,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton needs a GPU")
@pytest.mark.parametrize("beam_width", [1, 2, 3, 4, 8])
def test_triton_beam_walk_matches_the_reference(beam_width):
    """The kernel's first-max and register reductions are new code with no other
    oracle. Separated scores keep every comparison gap far above fp32 noise, so
    elementwise equality is meaningful; cross-row ties are explicitly not promised."""
    slots, top_k = 7, 16
    candidate_ids, scores, anchor = _lattice(bs=4, slots=slots, top_k=top_k, seed=11)
    selector = CandidateSelector(
        hidden_size=4, vocab_size=slots * top_k, state_rank=2, top_k=top_k
    )

    expected = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=beam_width,
    )
    actual = selector.beam_walk(
        candidate_ids=candidate_ids.cuda(),
        scores=scores.cuda(),
        anchor_token_ids=anchor.cuda(),
        beam_width=beam_width,
    )

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])


def test_grouped_conv_supports_runtime_block_sizes():
    """The conv indexes a position inside the block, so it must follow whatever
    block size the worker resolved -- including one that is not a power of two."""
    torch.manual_seed(0)
    groups, group_size, taps = 3, 2, 2
    hidden_size = groups * group_size
    batch_size = 2

    for block_size in (5, 8, 16):
        hidden = torch.randn(batch_size * block_size, hidden_size)
        delta = torch.randn(batch_size * block_size, taps, groups)
        base = torch.randn(taps, hidden_size)

        actual = _grouped_conv(
            hidden, delta, base, block_size, groups, group_size, taps
        )

        expected = torch.empty_like(hidden)
        hidden_3d = hidden.view(batch_size, block_size, groups, group_size)
        delta_4d = delta.view(batch_size, block_size, taps, groups)
        base_3d = base.view(taps, groups, group_size)
        for batch in range(batch_size):
            for position in range(block_size):
                value = torch.zeros(groups, group_size)
                for tap in range(min(taps, position + 1)):
                    coefficient = base_3d[tap] + delta_4d[batch, position, tap, :, None]
                    value += coefficient * hidden_3d[batch, position - tap]
                expected[batch * block_size + position] = value.flatten()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
