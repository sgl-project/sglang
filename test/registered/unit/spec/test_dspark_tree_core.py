"""CPU tests for the backend-neutral DSpark tree core.

This is the single-file port of the reference ``tree`` package CPU suite:
shape, layout, scoring, builder, acceptance, output, metadata, sparse, commit,
config and round. The module under test (``dspark_tree``) must stay
torch/dataclasses-only so it can be imported and exercised without torch_npu,
ACL, or the sglang runtime.

Import setup mirrors the reference ``conftest.py``: the pure ``dspark_tree``
module lives under ``dspark_components/`` and is imported standalone (by
inserting that directory on ``sys.path``) instead of via ``import sglang``,
so no ``sglang/__init__.py`` (and no vendor backend) is ever executed.
"""

import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

# Pure module lives under dspark_components and must be importable without
# executing sglang/__init__.py (no torch_npu on CPU CI).
_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "python" / "sglang" / "srt" / "speculative" / "dspark_components"))

from dspark_tree import (  # noqa: E402
    CandidateLayer,
    DSparkMarkovScorer,
    IndependentScorer,
    TreeBuilder,
    TreeCommitPlan,
    TreeConfig,
    TreeLayout,
    TreeShape,
    build_tree_accept_outs,
    build_tree_metadata,
    build_tree_ori_sparse_indices,
    finalize_tree_layout_torch,
    gather_accepted_rows,
    greedy_tree_accept,
    plan_tree_commit,
    plan_tree_post,
    plan_tree_pre,
    resolve_draft_query_width,
    resolve_dspark_widths,
    resolve_tree_config,
    resolve_tree_for_server,
)


def _layout_buf(bs: int, budget: int) -> TreeLayout:
    return TreeLayout(
        tokens=torch.full((bs, budget), -1, dtype=torch.long),
        depths=torch.zeros((bs, budget), dtype=torch.long),
        parents=torch.full((bs, budget), -1, dtype=torch.long),
        num_nodes=torch.zeros(bs, dtype=torch.int32),
        visibility=torch.zeros((bs, budget, budget), dtype=torch.bool),
        first_child=torch.full((bs, budget + 1), -1, dtype=torch.long),
        next_sibling=torch.full((bs, budget + 1), -1, dtype=torch.long),
    )


def _branch_layout():
    return finalize_tree_layout_torch(
        _layout_buf(1, 4),
        tokens=torch.tensor([[5, 6, 7, 8]]),
        depths=torch.tensor([[1, 1, 2, 2]]),
        parent_ids=torch.tensor([[0, 0, 1, 1]]),
        num_nodes=4,
    )


class TestTreeShape:
    def test_topk1_width_is_depth_plus_one(self):
        assert TreeShape(depth=7, budget=7, topk=1).target_query_len == 8

    def test_topk_gt1_width_is_budget_plus_one(self):
        assert TreeShape(depth=7, budget=10, topk=2).target_query_len == 11

    def test_rejects_budget_below_depth(self):
        with pytest.raises(ValueError):
            TreeShape(depth=7, budget=6, topk=2)

    def test_rejects_zero_topk(self):
        with pytest.raises(ValueError):
            TreeShape(depth=7, budget=7, topk=0)

    def test_validate_selected_nodes(self):
        shape = TreeShape(depth=7, budget=10, topk=2)
        assert shape.validate_selected_nodes(10) == 10
        with pytest.raises(ValueError):
            shape.validate_selected_nodes(11)


class TestTreeLayout:
    def test_links_and_visibility(self):
        tokens = torch.tensor([[11, 12, 13, 14]])
        depths = torch.tensor([[1, 1, 2, 2]])
        parents = torch.tensor([[0, 0, 1, 1]])
        out = finalize_tree_layout_torch(_layout_buf(1, 6), tokens, depths, parents, 4)
        # Children are linked in reverse insertion order; traversal still visits
        # every sibling exactly once.
        assert out.first_child[0, 0].item() == 2
        assert out.next_sibling[0, 2].item() == 1
        assert out.first_child[0, 1].item() == 4
        assert out.next_sibling[0, 4].item() == 3
        vis = out.visibility[0]
        assert bool(vis[0, 0]) and not bool(vis[0, 1])
        assert bool(vis[1, 1]) and not bool(vis[1, 0])
        assert bool(vis[2, 0]) and bool(vis[2, 2]) and not bool(vis[2, 1])
        assert bool(vis[3, 0]) and bool(vis[3, 3]) and not bool(vis[3, 1])

    def test_padding_rows_are_ignored(self):
        tokens = torch.tensor([[11, 12]])
        depths = torch.tensor([[1, 2]])
        parents = torch.tensor([[0, 1]])
        out = finalize_tree_layout_torch(_layout_buf(1, 4), tokens, depths, parents, 2)
        assert out.tokens[0, 2:].tolist() == [-1, -1]
        assert out.depths[0, 2:].tolist() == [0, 0]
        assert out.parents[0, 2:].tolist() == [-1, -1]
        assert not bool(out.visibility[0, 2:].any())



class _BiasHead:
    """Stateless fake Markov head: token t biases vocabulary index t % vocab."""

    def __init__(self, vocab: int, *, stateful: bool = False, shard=None):
        self.vocab = vocab
        self.stateful = stateful
        self.vocab_shard = shard

    def step_bias(self, prev_tokens, *, hidden_states=None, state=None):
        bias = torch.zeros(*prev_tokens.shape, self.vocab)
        flat_tokens = prev_tokens.reshape(-1)
        flat_bias = bias.reshape(-1, self.vocab)
        for i, tok in enumerate(flat_tokens.tolist()):
            flat_bias[i, tok % self.vocab] = 10.0
        return bias, (torch.zeros(1) if self.stateful else None)


class TestMarkovScorer:
    def test_bias_is_applied_per_parent(self):
        scorer = DSparkMarkovScorer(_BiasHead(4), topk=2)
        base = torch.zeros(1, 4)
        frontier = torch.tensor([[0, 1]])
        layer = scorer.score(base_logits=base, frontier_tokens=frontier, width=2)
        assert isinstance(layer, CandidateLayer)
        assert layer.token_ids[0, 0, 0].item() == 0
        assert layer.token_ids[0, 1, 0].item() == 1

    def test_rejects_vocab_shard(self):
        with pytest.raises(NotImplementedError, match="TP-sharded"):
            DSparkMarkovScorer(_BiasHead(4, shard=object()), topk=2)

    def test_rejects_stateful_head(self):
        scorer = DSparkMarkovScorer(_BiasHead(4, stateful=True), topk=2)
        with pytest.raises(NotImplementedError, match="stateless"):
            scorer.score(torch.zeros(1, 4), torch.tensor([[0]]), width=2)

    def test_independent_scorer_selects_from_logits(self):
        scorer = IndependentScorer(topk=2)
        logits = torch.tensor([[0.0, 5.0, 1.0, 0.0]])
        layer = scorer.score(logits, torch.tensor([[0]]), width=2)
        assert layer.token_ids[0, 0, 0].item() == 1


class TestTreeBuilder:
    def test_topk1_is_a_chain(self):
        builder = TreeBuilder(budget=3, topk=1, depth=3)
        logits = torch.zeros(1, 3, 8)
        logits[:, 0, 4] = 10.0
        logits[:, 1, 5] = 10.0
        logits[:, 2, 6] = 10.0
        out = builder.build(logits, root_token_ids=torch.tensor([1]))
        assert out.parents[0, :3].tolist() == [0, 1, 2]
        assert out.depths[0, :3].tolist() == [1, 2, 3]
        assert out.num_nodes[0].item() == 3

    def test_budget_caps_nodes_and_is_ancestor_closed(self):
        builder = TreeBuilder(budget=4, topk=2, depth=3)
        logits = torch.zeros(1, 3, 8)
        out = builder.build(logits, root_token_ids=torch.tensor([1]))
        n = int(out.num_nodes[0])
        assert n <= 4
        parents = out.parents[0, :n].tolist()
        for slot, parent in enumerate(parents):
            # parent id is either root (0) or an earlier node id
            assert parent == 0 or 1 <= parent <= slot

    def test_budget_above_depth_builds_a_full_budget_tree(self):
        builder = TreeBuilder(budget=8, topk=2, depth=5)
        logits = torch.zeros(1, 5, 8)
        out = builder.build(logits, root_token_ids=torch.tensor([1]))
        n = int(out.num_nodes[0])
        assert n == 8
        assert out.tokens.shape[1] == 8
        depths = out.depths[0, :n].tolist()
        assert all(1 <= d <= 5 for d in depths)
        parents = out.parents[0, :n].tolist()
        for slot, parent in enumerate(parents):
            assert parent == 0 or 1 <= parent <= slot

    def test_budget_above_capacity_is_rejected(self):
        builder = TreeBuilder(budget=8, topk=2, depth=2)
        with pytest.raises(ValueError, match="builder capacity"):
            builder.build(torch.zeros(1, 2, 8), root_token_ids=torch.tensor([1]))

    def test_deterministic_on_ties(self):
        builder = TreeBuilder(budget=4, topk=2, depth=3)
        logits = torch.zeros(2, 3, 8)
        a = builder.build(logits, root_token_ids=torch.tensor([1, 1]))
        b = builder.build(logits, root_token_ids=torch.tensor([1, 1]))
        assert torch.equal(a.tokens, b.tokens)
        assert torch.equal(a.parents, b.parents)


class TestGreedyAccept:
    def _tree(self):
        layout = _layout_buf(1, 4)
        return finalize_tree_layout_torch(
            layout,
            tokens=torch.tensor([[5, 6, 7, 8]]),
            depths=torch.tensor([[1, 1, 2, 2]]),
            parent_ids=torch.tensor([[0, 0, 1, 1]]),
            num_nodes=4,
        )

    def test_full_path_with_bonus(self):
        # root target 5 -> node1; node1 target 7 -> node3; node3 target 9 -> bonus.
        target = torch.tensor([[5, 7, 0, 9, 0]])
        result = greedy_tree_accept(self._tree(), target)
        assert result.path_node_ids[0, :2].tolist() == [1, 3]
        assert int(result.num_accepted[0]) == 2
        assert int(result.bonus[0]) == 9

    def test_root_only(self):
        target = torch.tensor([[99, 0, 0, 0, 0]])
        result = greedy_tree_accept(self._tree(), target)
        assert int(result.num_accepted[0]) == 0
        assert result.path_node_ids[0].tolist() == [-1, -1, -1, -1]
        assert int(result.bonus[0]) == 99

    def test_sibling_is_not_followed(self):
        # root target 6 -> node2 (sibling of node1). node2 has no child.
        target = torch.tensor([[6, 0, 0, 0, 0]])
        result = greedy_tree_accept(self._tree(), target)
        assert result.path_node_ids[0, 0].item() == 2
        assert int(result.num_accepted[0]) == 1
        assert int(result.bonus[0]) == 0


class TestOutput:
    def test_path_gather_and_accept_index(self):
        verify_ids = torch.tensor([[100, 5, 6, 7, 8]])
        path = torch.tensor([[1, 3, -1, -1]])
        out = build_tree_accept_outs(
            verify_ids=verify_ids,
            path_node_ids=path,
            bonus=torch.tensor([9]),
            prefix_lens=torch.tensor([20]),
        )
        assert out.out_tokens[0, :3].tolist() == [5, 7, 9]
        assert out.out_tokens[0, 3:].tolist() == [0, 0]
        assert out.accept_index[0].tolist() == [1, 3, -1, -1, -1]
        assert out.commit_lens[0].item() == 3
        assert out.new_seq_lens[0].item() == 23

    def test_root_only_output(self):
        verify_ids = torch.tensor([[100, 5, 6]])
        path = torch.tensor([[-1, -1]])
        out = build_tree_accept_outs(
            verify_ids=verify_ids,
            path_node_ids=path,
            bonus=torch.tensor([7]),
            prefix_lens=torch.tensor([10]),
        )
        assert out.out_tokens[0, 0].item() == 7
        assert out.out_tokens[0, 1:].tolist() == [0, 0]
        assert out.commit_lens[0].item() == 1
        assert out.new_seq_lens[0].item() == 11

    def test_gather_accepted_rows(self):
        tensor = torch.tensor([[[10.0], [11.0], [12.0]]])
        accept_index = torch.tensor([[1, 2, -1]])
        gathered = gather_accepted_rows(tensor, accept_index)
        assert gathered[0, :2, 0].tolist() == [11.0, 12.0]
        # padding clamps to column 0; the commit-length gate drops it
        assert gathered[0, 2, 0].item() == 10.0


class TestDSparkWidths:
    def test_tree_budget_above_gamma_uses_budget_width(self):
        widths = resolve_dspark_widths(
            explicit_block_size=5,
            checkpoint_gamma=None,
            num_draft_tokens=9,
            tree_enable=True,
            tree_topk=2,
            tree_budget=8,
        )
        assert widths.gamma == 5
        assert widths.verify_width == 9
        assert widths.tree is not None
        assert widths.tree.depth == 5
        assert widths.tree.budget == 8
        assert widths.tree.verify_width == 9

    def test_tree_width_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match=r"1 \+ tree budget"):
            resolve_dspark_widths(
                explicit_block_size=5,
                checkpoint_gamma=None,
                num_draft_tokens=6,
                tree_enable=True,
                tree_topk=2,
                tree_budget=8,
            )

    def test_tree_requires_known_gamma(self):
        with pytest.raises(ValueError, match="requires a known draft depth"):
            resolve_dspark_widths(
                explicit_block_size=None,
                checkpoint_gamma=None,
                num_draft_tokens=9,
                tree_enable=True,
                tree_topk=2,
                tree_budget=8,
            )

    def test_tree_capacity_is_topk_times_depth(self):
        with pytest.raises(ValueError, match="builder capacity"):
            resolve_dspark_widths(
                explicit_block_size=5,
                checkpoint_gamma=None,
                num_draft_tokens=11,
                tree_enable=True,
                tree_topk=2,
                tree_budget=11,
            )

    def test_tree_budget_equal_gamma_is_still_supported(self):
        cfg = resolve_tree_config(enable=True, topk=2, budget=5, depth=5)
        assert cfg is not None
        assert cfg.verify_width == 6

    def test_chain_requires_budget_equal_depth(self):
        cfg = resolve_tree_config(enable=True, topk=1, budget=5, depth=5)
        assert cfg is not None
        assert cfg.verify_width == 6
        with pytest.raises(ValueError, match="budget == depth"):
            resolve_tree_config(enable=True, topk=1, budget=6, depth=5)

    def test_non_tree_keeps_gamma_plus_one(self):
        widths = resolve_dspark_widths(
            explicit_block_size=5,
            checkpoint_gamma=None,
            num_draft_tokens=6,
            tree_enable=False,
            tree_topk=None,
            tree_budget=None,
        )
        assert widths.gamma == 5
        assert widths.verify_width == 6
        assert widths.tree is None

        legacy = resolve_dspark_widths(
            explicit_block_size=None,
            checkpoint_gamma=None,
            num_draft_tokens=9,
            tree_enable=False,
            tree_topk=None,
            tree_budget=None,
        )
        assert legacy.gamma == 8
        assert legacy.verify_width == 9

    def test_draft_query_width_anchor_rule(self):
        assert resolve_draft_query_width(gamma=5, sample_from_anchor=True) == 5
        assert resolve_draft_query_width(gamma=5, sample_from_anchor=False) == 6


class TestTreeMetadata:
    def test_positions_and_seqused(self):
        meta = build_tree_metadata(
            _branch_layout(),
            prefix_lens=torch.tensor([10]),
            anchor_tokens=torch.tensor([100]),
        )
        # root at 10; nodes 1/2 at 11; nodes 3/4 at 12
        assert meta.positions.tolist() == [10, 11, 11, 12, 12]
        assert meta.node_seqused_kv.tolist() == [11, 12, 12, 13, 13]
        assert meta.verify_ids.tolist() == [100, 5, 6, 7, 8]

    def test_ancestors_are_root_first(self):
        meta = build_tree_metadata(
            _branch_layout(),
            prefix_lens=torch.tensor([10]),
            anchor_tokens=torch.tensor([100]),
        )
        # rows: root, node1, node2, node3, node4
        assert meta.tree_ancestors.tolist() == [
            [0, -1, -1],
            [0, 1, -1],
            [0, 2, -1],
            [0, 1, 3],
            [0, 1, 4],
        ]

    def test_request_and_node_geometry(self):
        meta = build_tree_metadata(
            _branch_layout(),
            prefix_lens=torch.tensor([10]),
            anchor_tokens=torch.tensor([100]),
        )
        assert meta.node_cu_seqlens_q.tolist() == [0, 1, 2, 3, 4, 5]
        assert meta.request_cu_seqlens_q.tolist() == [0, 5]
        assert meta.request_seqused_kv.tolist() == [15]
        assert meta.tree_prefix.tolist() == [10, 10, 10, 10, 10]
        assert meta.node_req_ids.tolist() == [0, 0, 0, 0, 0]



class TestSparseIndices:
    def test_prefix_window_plus_ancestors_and_no_sibling(self):
        layout = _branch_layout()
        block_table = torch.tensor([[0, 1, 2, 3]])
        slots = build_tree_ori_sparse_indices(
            layout,
            prefix_lens=torch.tensor([3]),
            block_table=block_table,
            block_size=4,
            window=4,
            budget=4,
        )
        # width is 128-aligned
        assert slots.shape == (5, 1, 128)
        root = set(int(x) for x in slots[0, 0] if x >= 0)
        node1 = set(int(x) for x in slots[1, 0] if x >= 0)
        node2 = set(int(x) for x in slots[2, 0] if x >= 0)
        node3 = set(int(x) for x in slots[3, 0] if x >= 0)
        # prefix slots 0,1,2 + root slot 3 are visible to every node
        assert {0, 1, 2, 3}.issubset(root)
        # node1 sees itself (slot 4), not sibling node2 (slot 5)
        assert 4 in node1 and 5 not in node1
        # node2 sees itself (slot 5), not sibling node1 (slot 4) or node3
        assert 5 in node2 and 4 not in node2 and 6 not in node2
        # node3 sees ancestor node1 (slot 4) + itself (slot 6), not sibling node4
        assert 4 in node3 and 6 in node3 and 7 not in node3





class TestCommitPlan:
    def test_root_and_path_move_bonus_does_not(self):
        plan = plan_tree_commit(
            path_node_ids=torch.tensor([[1, 3, -1, -1]]),
            verify_cache_loc=torch.tensor([[100, 101, 102, 103, 104]]),
            prefix_lens=torch.tensor([10]),
            block_table=torch.tensor([[0, 1, 2, 3, 4]]),
            block_size=4,
        )
        assert plan.source_rows.tolist() == [[100, 101, 103, -1, -1]]
        assert plan.dest_positions.tolist() == [[10, 11, 12, -1, -1]]
        assert plan.dest_slots.tolist() == [[100, 101, 102, -1, -1]]
        assert plan.commit_lens.tolist() == [3]
        assert plan.new_seq_lens.tolist() == [13]

    def test_root_only(self):
        plan = plan_tree_commit(
            path_node_ids=torch.tensor([[-1, -1]]),
            verify_cache_loc=torch.tensor([[100, 101, 102]]),
            prefix_lens=torch.tensor([7]),
            block_table=torch.tensor([[0, 1, 2]]),
            block_size=4,
        )
        assert plan.source_rows.tolist() == [[100, -1, -1]]
        assert plan.dest_positions.tolist() == [[7, -1, -1]]
        assert plan.commit_lens.tolist() == [1]





class TestTreeConfig:
    def test_disabled_returns_none(self):
        assert resolve_tree_config(enable=False, topk=2, budget=8, depth=7) is None

    def test_requires_budget(self):
        with pytest.raises(ValueError):
            resolve_tree_config(enable=True, topk=2, budget=None, depth=7)

    def test_budget_must_be_at_least_depth(self):
        with pytest.raises(ValueError):
            resolve_tree_config(enable=True, topk=2, budget=6, depth=7)

    def test_topk1_requires_chain_budget(self):
        with pytest.raises(ValueError):
            resolve_tree_config(enable=True, topk=1, budget=8, depth=7)

    def test_verify_width(self):
        cfg = resolve_tree_config(enable=True, topk=2, budget=8, depth=7)
        assert isinstance(cfg, TreeConfig)
        assert cfg.verify_width == 9

    def test_resolve_from_server_args(self):
        off = SimpleNamespace(
            speculative_dspark_tree_enable=False,
            speculative_eagle_topk=2,
            speculative_dspark_tree_budget=5,
        )
        assert resolve_tree_for_server(off, gamma=5) is None

        on = SimpleNamespace(
            speculative_dspark_tree_enable=True,
            speculative_eagle_topk=2,
            speculative_dspark_tree_budget=5,
        )
        cfg = resolve_tree_for_server(on, gamma=5)
        assert cfg.verify_width == 6

        default_topk = SimpleNamespace(
            speculative_dspark_tree_enable=True,
            speculative_eagle_topk=None,
            speculative_dspark_tree_budget=5,
        )
        assert resolve_tree_for_server(default_topk, gamma=5).topk == 2

        with pytest.raises(ValueError, match="static"):
            resolve_tree_for_server(on, gamma=5, ragged_mode="compact")


def test_chain_identity_end_to_end():
    builder = TreeBuilder(budget=3, topk=1, depth=3)
    logits = torch.zeros(1, 3, 8)
    logits[0, 0, 5] = 10.0
    logits[0, 1, 6] = 10.0
    logits[0, 2, 7] = 10.0
    layout = builder.build(logits, root_token_ids=torch.tensor([1]))
    assert layout.parents[0, :3].tolist() == [0, 1, 2]

    meta = build_tree_metadata(
        layout, prefix_lens=torch.tensor([4]), anchor_tokens=torch.tensor([1])
    )
    assert meta.verify_ids.tolist() == [1, 5, 6, 7]
    assert meta.positions.tolist() == [4, 5, 6, 7]
    assert meta.node_seqused_kv.tolist() == [5, 6, 7, 8]

    target = torch.tensor([[5, 6, 7, 99]])
    acc = greedy_tree_accept(layout, target)
    assert acc.path_node_ids[0, :3].tolist() == [1, 2, 3]
    assert int(acc.bonus[0]) == 99

    outs = build_tree_accept_outs(
        verify_ids=meta.verify_ids.reshape(1, -1),
        path_node_ids=acc.path_node_ids,
        bonus=acc.bonus,
        prefix_lens=torch.tensor([4]),
    )
    assert outs.out_tokens[0, :4].tolist() == [5, 6, 7, 99]
    assert outs.commit_lens[0].item() == 4
    assert outs.new_seq_lens[0].item() == 8

    plan = plan_tree_commit(
        path_node_ids=acc.path_node_ids,
        verify_cache_loc=(torch.arange(4).unsqueeze(0) + 100),
        prefix_lens=torch.tensor([4]),
        block_table=torch.tensor([[0, 1]]),
        block_size=4,
    )
    assert plan.source_rows[0, :4].tolist() == [100, 101, 102, 103]
    assert plan.dest_positions[0, :4].tolist() == [4, 5, 6, 7]
    assert plan.commit_lens[0].item() == 4


class _BiasToSeven:
    vocab_shard = None

    def step_bias(self, prev_tokens, *, hidden_states=None, state=None):
        bias = torch.zeros(*prev_tokens.shape, 8)
        bias[..., 7] = 100.0
        return bias, None


def test_markov_scorer_changes_root_candidate():
    logits = torch.zeros(1, 3, 8)
    independent = TreeBuilder(budget=4, topk=2, depth=3, scorer=IndependentScorer(2))
    markov = TreeBuilder(
        budget=4, topk=2, depth=3, scorer=DSparkMarkovScorer(_BiasToSeven(), 2)
    )
    ind = independent.build(logits, root_token_ids=torch.tensor([1]))
    mar = markov.build(logits, root_token_ids=torch.tensor([1]))
    assert 7 not in ind.tokens[0].tolist()
    assert 7 in mar.tokens[0].tolist()


def test_tree_pipeline_shapes():
    builder = TreeBuilder(budget=4, topk=2, depth=3)
    logits = torch.zeros(1, 3, 8)
    layout = builder.build(logits, root_token_ids=torch.tensor([1]))
    budget = layout.tokens.shape[1]
    assert int(layout.num_nodes[0]) <= 4

    meta = build_tree_metadata(
        layout, prefix_lens=torch.tensor([3]), anchor_tokens=torch.tensor([1])
    )
    assert meta.query_width == budget + 1
    assert meta.tree_ancestors.shape == (budget + 1, int(layout.depths.max()) + 1)

    slots = build_tree_ori_sparse_indices(
        layout,
        prefix_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
        window=4,
        budget=budget,
    )
    assert slots.shape == (budget + 1, 1, 128)

    target = torch.zeros(1, budget + 1, dtype=torch.long)
    acc = greedy_tree_accept(layout, target)
    outs = build_tree_accept_outs(
        verify_ids=meta.verify_ids.reshape(1, -1),
        path_node_ids=acc.path_node_ids,
        bonus=acc.bonus,
        prefix_lens=torch.tensor([3]),
    )
    assert outs.out_tokens.shape == (1, budget + 1)
    assert outs.accept_index.shape == (1, budget + 1)
    plan = plan_tree_commit(
        path_node_ids=acc.path_node_ids,
        verify_cache_loc=torch.arange(budget + 1).unsqueeze(0),
        prefix_lens=torch.tensor([3]),
        block_table=torch.tensor([[0, 1, 2, 3]]),
        block_size=4,
    )
    assert plan.commit_lens[0].item() == int(acc.num_accepted[0]) + 1
