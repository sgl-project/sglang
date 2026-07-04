"""CPU unit + integration tests for DFlash tree construction (SGLang #29524).

No GPU required. Covers the host-side tree core (builders, budget, buffer/mask
materialization, the greedy accept oracle), boundary/edge cases, the server-arg
flags + validation, the chain->tree buffer seam, DFlashVerifyInput wiring, and the
keystone integration check: the retrieve_* pointer walk (what the CUDA
verify_tree_greedy kernel does) matches the parent-based oracle.
"""

import random
import unittest

import torch

from sglang.srt.speculative.dflash_tree import (
    DFlashDraftTree,
    build_ancestor_mask,
    build_tree,
    build_tree_custom_mask,
    build_tree_depth_first,
    build_tree_opt_prefix,
    compute_tree_budget,
    tree_to_verify_buffers,
    verify_tree_greedy_cpu,
)
from sglang.test.ci.ci_register import register_cpu_ci

# Fast CPU-only unit + integration suite: no server, no GPU kernels, no model.
register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _invariants(tc, t):
    tc.assertEqual(t.parent_indices[0], -1)
    tc.assertEqual(t.depths[0], 0)
    for i in range(1, t.num_nodes):
        tc.assertLess(t.parent_indices[i], i)
        tc.assertEqual(t.depths[i], t.depths[t.parent_indices[i]] + 1)


class TestTreeBudget(unittest.TestCase):
    def test_width_one_is_block_size(self):
        self.assertEqual(compute_tree_budget(block_size=5, tree_width=1), 5)

    def test_full_binary_tree_size(self):
        self.assertEqual(compute_tree_budget(block_size=3, tree_width=2), 7)  # 1+2+4
        self.assertEqual(compute_tree_budget(block_size=2, tree_width=3), 4)  # 1+3

    def test_budget_cap(self):
        self.assertEqual(
            compute_tree_budget(block_size=4, tree_width=4, max_budget=10), 10
        )
        self.assertEqual(
            compute_tree_budget(block_size=2, tree_width=2, max_budget=99), 3
        )


class TestDraftTree(unittest.TestCase):
    def test_root_only(self):
        t = DFlashDraftTree(tokens=[42], parent_indices=[-1], depths=[0], num_nodes=1)
        self.assertEqual(t.num_nodes, 1)
        self.assertEqual(t.longest_chain_len(), 1)

    def test_longest_chain(self):
        t = DFlashDraftTree(
            tokens=[0, 1, 2, 3],
            parent_indices=[-1, 0, 1, 0],
            depths=[0, 1, 2, 1],
            num_nodes=4,
        )
        self.assertEqual(t.longest_chain_len(), 3)


class TestOptPrefix(unittest.TestCase):
    def test_width_one_is_chain(self):
        t = build_tree_opt_prefix(
            5, [[10], [20], [30]], [[-0.1], [-0.2], [-0.3]], budget=4
        )
        self.assertEqual(t.tokens, [5, 10, 20, 30])
        self.assertEqual(t.parent_indices, [-1, 0, 1, 2])
        self.assertEqual(t.depths, [0, 1, 2, 3])
        _invariants(self, t)

    def test_budget_exact(self):
        t = build_tree_opt_prefix(
            0,
            [[1, 2], [3, 4], [5, 6]],
            [[-0.1, -0.9], [-0.1, -0.9], [-0.1, -0.9]],
            budget=4,
        )
        self.assertEqual(t.num_nodes, 4)
        _invariants(self, t)

    def test_best_first_order(self):
        t = build_tree_opt_prefix(
            0, [[1, 2], [3, 4]], [[-0.01, -5.0], [-0.01, -5.0]], budget=3
        )
        self.assertEqual(t.tokens[:3], [0, 1, 3])
        _invariants(self, t)

    def test_root_only_budgets(self):
        t = build_tree_opt_prefix(7, [[1]], [[-0.1]], budget=1)
        self.assertEqual(t.num_nodes, 1)
        self.assertEqual(t.tokens, [7])


class TestDepthFirst(unittest.TestCase):
    def test_contains_full_top1_spine(self):
        t = build_tree_depth_first(
            0,
            [[1, 2], [3, 4], [5, 6]],
            [[-0.1, -0.9], [-0.1, -0.9], [-0.1, -0.9]],
            budget=4,
        )
        spine = [t.tokens[0]]
        cur = 0
        for _ in range(3):
            kids = [i for i in range(t.num_nodes) if t.parent_indices[i] == cur]
            self.assertTrue(kids, "spine broken")
            cur = min(kids)
            spine.append(t.tokens[cur])
        self.assertEqual(spine, [0, 1, 3, 5])

    def test_budget_respected(self):
        t = build_tree_depth_first(
            0, [[1, 2, 9], [3, 4, 9], [5, 6, 9]], [[-0.1, -0.5, -9.0]] * 3, budget=5
        )
        self.assertEqual(t.num_nodes, 5)
        _invariants(self, t)


class TestBuildTreeDispatch(unittest.TestCase):
    def test_width_one_chain_regardless_of_construction(self):
        tk, lp = [[1], [2]], [[-0.1], [-0.2]]
        for c in ("depth_first", "opt_prefix"):
            t = build_tree(9, tk, lp, tree_width=1, budget=3, construction=c)
            self.assertEqual(t.tokens, [9, 1, 2])
            self.assertEqual(t.parent_indices, [-1, 0, 1])

    def test_construction_selects_builder(self):
        tk, lp = [[1, 2], [3, 4]], [[-0.1, -0.9], [-0.1, -0.9]]
        self.assertEqual(
            build_tree(
                0, tk, lp, tree_width=2, budget=4, construction="opt_prefix"
            ).num_nodes,
            4,
        )
        self.assertEqual(
            build_tree(
                0, tk, lp, tree_width=2, budget=4, construction="depth_first"
            ).num_nodes,
            4,
        )

    def test_bad_construction_raises(self):
        with self.assertRaises(ValueError):
            build_tree(0, [[1]], [[-0.1]], tree_width=2, budget=2, construction="nope")


class TestVerifyBuffers(unittest.TestCase):
    def test_branching_pointers(self):
        tree = DFlashDraftTree(
            tokens=[100, 101, 102, 103],
            parent_indices=[-1, 0, 0, 1],
            depths=[0, 1, 1, 2],
            num_nodes=4,
        )
        b = tree_to_verify_buffers([tree], num_verify_tokens=4, base_positions=[7])
        self.assertEqual(b["retrieve_next_token"][0].tolist(), [1, 3, -1, -1])
        self.assertEqual(b["retrieve_next_sibling"][0].tolist(), [-1, 2, -1, -1])
        self.assertEqual(b["positions"][0].tolist(), [7, 8, 8, 9])
        self.assertEqual(b["draft_token"][0].tolist(), [100, 101, 102, 103])

    def test_padding_short_tree(self):
        tree = DFlashDraftTree(
            tokens=[1, 2], parent_indices=[-1, 0], depths=[0, 1], num_nodes=2
        )
        b = tree_to_verify_buffers([tree], num_verify_tokens=4, base_positions=[0])
        self.assertEqual(b["retrieve_next_token"][0].tolist(), [1, -1, -1, -1])
        self.assertEqual(b["retrieve_next_sibling"][0].tolist(), [-1, -1, -1, -1])

    def test_chain_matches_legacy_layout(self):
        t = build_tree(0, [[1], [2], [3]], [[-0.1]] * 3, tree_width=1, budget=4)
        b = tree_to_verify_buffers([t], num_verify_tokens=4, base_positions=[0])
        self.assertEqual(b["retrieve_next_token"][0].tolist(), [1, 2, 3, -1])
        self.assertEqual(b["retrieve_next_sibling"][0].tolist(), [-1, -1, -1, -1])
        self.assertEqual(b["retrieve_index"][0].tolist(), [0, 1, 2, 3])


class TestTreeMask(unittest.TestCase):
    def test_ancestor_semantics(self):
        m = build_ancestor_mask([-1, 0, 0, 1], 4).tolist()
        self.assertEqual(m[0], [True, False, False, False])
        self.assertEqual(m[1], [True, True, False, False])
        self.assertEqual(m[2], [True, False, True, False])
        self.assertEqual(m[3], [True, True, False, True])

    def test_flattened_size_formula(self):
        tree = DFlashDraftTree([0, 1, 2], [-1, 0, 1], [0, 1, 2], 3)
        N, kv_lens = 3, [5]
        mask = build_tree_custom_mask([tree], num_verify_tokens=N, kv_lens=kv_lens)
        self.assertEqual(mask.numel(), sum(kv_lens) * N + N * N * 1)
        self.assertEqual(mask.dtype, torch.bool)

    def test_context_all_allowed(self):
        tree = DFlashDraftTree([0, 1], [-1, 0], [0, 1], 2)
        n, kv = 2, 3
        mask = build_tree_custom_mask([tree], num_verify_tokens=n, kv_lens=[kv])
        # Per-row layout is [kv context cols | N block cols] (row-major, matching the
        # Triton reader's row*kv_len+col indexing). The context columns of every row
        # must be all-allowed; the block columns carry the ancestor mask.
        self.assertTrue(bool(mask.view(n, kv + n)[:, :kv].all()))


class TestAcceptOracle(unittest.TestCase):
    def test_accept_full_spine(self):
        tree = DFlashDraftTree([0, 11, 22], [-1, 0, 1], [0, 1, 2], 3)
        out = verify_tree_greedy_cpu(tree, target_predict=[11, 22, 99])
        self.assertEqual(out["accept_indices"], [0, 1, 2])
        self.assertEqual(out["num_accept_tokens"], 3)
        self.assertEqual(out["bonus_token"], 99)

    def test_pick_longest_branch(self):
        tree = DFlashDraftTree([0, 10, 20, 30], [-1, 0, 0, 1], [0, 1, 1, 2], 4)
        out = verify_tree_greedy_cpu(tree, target_predict=[10, 30, 0, 7])
        self.assertEqual(out["accept_indices"], [0, 1, 3])
        self.assertEqual(out["num_accept_tokens"], 3)
        self.assertEqual(out["bonus_token"], 7)

    def test_reject_all(self):
        tree = DFlashDraftTree([0, 10], [-1, 0], [0, 1], 2)
        out = verify_tree_greedy_cpu(tree, target_predict=[999, 0])
        self.assertEqual(out["accept_indices"], [0])
        self.assertEqual(out["num_accept_tokens"], 1)
        self.assertEqual(out["bonus_token"], 999)


class TestEdgeCases(unittest.TestCase):
    def test_budget_max_zero_means_full(self):
        self.assertEqual(compute_tree_budget(3, 2, max_budget=0), 7)

    def test_budget_block_size_one(self):
        self.assertEqual(compute_tree_budget(1, 5), 1)

    def test_empty_depth_is_root_only(self):
        for fn in (build_tree_opt_prefix, build_tree_depth_first):
            t = fn(7, topk_tokens=[], topk_logprobs=[], budget=10)
            self.assertEqual(t.num_nodes, 1)
            self.assertEqual(t.tokens, [7])

    def test_budget_zero_and_one_are_root_only(self):
        for fn in (build_tree_opt_prefix, build_tree_depth_first):
            for B in (0, 1):
                self.assertEqual(fn(0, [[1, 2]], [[-0.1, -0.9]], budget=B).num_nodes, 1)

    def test_budget_exceeds_full_tree_caps_at_full(self):
        for fn in (build_tree_opt_prefix, build_tree_depth_first):
            t = fn(0, [[1, 2], [3, 4]], [[-0.1, -0.9], [-0.1, -0.9]], budget=999)
            self.assertEqual(t.num_nodes, 7)

    def test_single_depth(self):
        t = build_tree_opt_prefix(0, [[1, 2, 3]], [[-0.1, -0.2, -0.3]], budget=3)
        self.assertEqual(sorted(t.tokens), [0, 1, 2])
        self.assertTrue(all(t.parent_indices[i] == 0 for i in range(1, t.num_nodes)))

    def test_tied_logprobs_are_deterministic_and_valid(self):
        tk, lp = [[1, 2], [3, 4]], [[-0.5, -0.5], [-0.5, -0.5]]
        for fn in (build_tree_opt_prefix, build_tree_depth_first):
            a = fn(0, tk, lp, 5)
            b = fn(0, tk, lp, 5)
            self.assertEqual(a.parent_indices, b.parent_indices)
            _invariants(self, a)

    def test_root_only_buffers(self):
        t = DFlashDraftTree([5], [-1], [0], 1)
        b = tree_to_verify_buffers([t], num_verify_tokens=4, base_positions=[0])
        self.assertEqual(b["retrieve_next_token"][0].tolist(), [-1, -1, -1, -1])
        self.assertEqual(b["retrieve_next_sibling"][0].tolist(), [-1, -1, -1, -1])

    def test_oversize_tree_raises(self):
        t = DFlashDraftTree([0, 1, 2], [-1, 0, 1], [0, 1, 2], 3)
        with self.assertRaises(AssertionError):
            tree_to_verify_buffers([t], num_verify_tokens=2, base_positions=[0])

    def test_batch_mixed_sizes(self):
        t1 = DFlashDraftTree([0, 1], [-1, 0], [0, 1], 2)
        t2 = DFlashDraftTree([0, 1, 2, 3], [-1, 0, 0, 1], [0, 1, 1, 2], 4)
        b = tree_to_verify_buffers(
            [t1, t2], num_verify_tokens=4, base_positions=[10, 20]
        )
        self.assertEqual(b["draft_token"].shape, (2, 4))
        self.assertEqual(b["positions"][1].tolist(), [20, 21, 21, 22])
        self.assertEqual(b["retrieve_next_token"][0].tolist(), [1, -1, -1, -1])

    def test_single_node_mask(self):
        self.assertEqual(build_ancestor_mask([-1], 1).tolist(), [[True]])

    def test_chain_mask_is_ancestor_closure(self):
        m = build_ancestor_mask([-1, 0, 1], 3).tolist()
        self.assertEqual(
            m, [[True, False, False], [True, True, False], [True, True, True]]
        )

    def test_oracle_root_only(self):
        out = verify_tree_greedy_cpu(
            DFlashDraftTree([0], [-1], [0], 1), target_predict=[55]
        )
        self.assertEqual(out["accept_indices"], [0])
        self.assertEqual(out["num_accept_tokens"], 1)
        self.assertEqual(out["bonus_token"], 55)

    def test_zero_width_rows_are_root_only(self):
        # Empty per-depth rows (width 0) must not crash either builder (regression
        # for the depth_first IndexError flagged in review).
        for fn in (build_tree_opt_prefix, build_tree_depth_first):
            self.assertEqual(fn(0, [[], []], [[], []], budget=5).num_nodes, 1)
        self.assertEqual(
            build_tree(0, [[], []], [[], []], tree_width=2, budget=5).num_nodes, 1
        )


def _accept_via_pointers(bufs, b, target_predict_row):
    """Faithful CPU simulation of the CUDA verify_tree_greedy traversal: from the
    root, follow first-child / next-sibling pointers, accepting the child whose token
    equals the parent's target prediction; stop at the first level with no match."""
    nt = bufs["retrieve_next_token"][b].tolist()
    ns = bufs["retrieve_next_sibling"][b].tolist()
    dt = bufs["draft_token"][b].tolist()
    node, path = 0, [0]
    while True:
        pred = target_predict_row[node]
        child, matched = nt[node], -1
        while child != -1:
            if dt[child] == pred:
                matched = child
                break
            child = ns[child]
        if matched == -1:
            break
        path.append(matched)
        node = matched
    return path


class TestIntegration(unittest.TestCase):
    def _random_logits(self, depth, width, seed):
        rng = random.Random(seed)
        tk, lp = [], []
        for _ in range(depth):
            tk.append(rng.sample(range(1, 10000), width))  # distinct per depth
            lp.append(sorted((-rng.random() * 5 for _ in range(width)), reverse=True))
        return tk, lp

    def test_pointer_walk_matches_oracle(self):
        """KEYSTONE: the path the CUDA kernel would walk (retrieve_* pointers) equals
        the parent-based oracle path, for random trees, budgets, and targets."""
        for seed in range(200):
            tk, lp = self._random_logits(depth=4, width=3, seed=seed)
            for construction in ("opt_prefix", "depth_first"):
                tree = build_tree(
                    0, tk, lp, tree_width=3, budget=15, construction=construction
                )
                n = tree.num_nodes
                bufs = tree_to_verify_buffers(
                    [tree], num_verify_tokens=n, base_positions=[0]
                )
                rng = random.Random(seed + 777)
                target = []
                for node in range(n):
                    kids = [i for i in range(n) if tree.parent_indices[i] == node]
                    if kids and rng.random() < 0.6:
                        target.append(tree.tokens[rng.choice(kids)])
                    else:
                        target.append(rng.randrange(1, 10000))
                ptr_path = _accept_via_pointers(bufs, 0, target)
                oracle = verify_tree_greedy_cpu(tree, target)
                self.assertEqual(
                    ptr_path, oracle["accept_indices"], f"seed={seed} {construction}"
                )

    def test_full_pipeline_accepts_spine_when_target_agrees(self):
        tk, lp = [[1, 2], [3, 4], [5, 6]], [[-0.1, -0.9]] * 3
        tree = build_tree(0, tk, lp, tree_width=2, budget=8, construction="depth_first")
        n = tree.num_nodes
        mask = build_tree_custom_mask([tree], num_verify_tokens=n, kv_lens=[4])
        self.assertEqual(mask.numel(), 4 * n + n * n)
        spine, cur = [0], 0
        for _ in range(3):
            cur = min(i for i in range(n) if tree.parent_indices[i] == cur)
            spine.append(cur)
        target = [0] * n
        for k in range(len(spine) - 1):
            target[spine[k]] = tree.tokens[spine[k + 1]]
        out = verify_tree_greedy_cpu(tree, target)
        self.assertEqual(out["accept_indices"], spine)

    def test_batch_shapes(self):
        tk, lp = self._random_logits(depth=3, width=3, seed=1)
        trees = [
            build_tree(0, tk, lp, tree_width=3, budget=10, construction="depth_first")
            for _ in range(3)
        ]
        n = 10
        bufs = tree_to_verify_buffers(
            trees, num_verify_tokens=n, base_positions=[0, 5, 9]
        )
        self.assertEqual(bufs["draft_token"].shape, (3, n))
        kv_lens = [4, 6, 8]
        mask = build_tree_custom_mask(trees, num_verify_tokens=n, kv_lens=kv_lens)
        self.assertEqual(mask.numel(), sum(kv_lens) * n + n * n * 3)

    def test_width_one_pipeline_equals_legacy_chain(self):
        from sglang.srt.speculative.dflash_utils import (
            _get_or_create_chain_verify_buffers,
        )

        trees = [
            build_tree(0, [[1], [2], [3]], [[-0.1]] * 3, tree_width=1, budget=4)
            for _ in range(2)
        ]
        tree_bufs = tree_to_verify_buffers(
            trees, num_verify_tokens=4, base_positions=[0, 0]
        )
        legacy = _get_or_create_chain_verify_buffers(
            bs=2, draft_token_num=4, device=torch.device("cpu")
        )
        legacy_nt = (
            legacy[1] if isinstance(legacy, tuple) else legacy["retrieve_next_token"]
        )
        legacy_ns = (
            legacy[2] if isinstance(legacy, tuple) else legacy["retrieve_next_sibling"]
        )
        self.assertTrue(torch.equal(tree_bufs["retrieve_next_token"], legacy_nt))
        self.assertTrue(torch.equal(tree_bufs["retrieve_next_sibling"], legacy_ns))


class TestVerifyInput(unittest.TestCase):
    def test_tree_fields_populated(self):
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        dt = torch.zeros(8, dtype=torch.int64)
        pos = torch.zeros(8, dtype=torch.int64)
        mask = torch.ones(40, dtype=torch.bool)
        vi = DFlashVerifyInput.from_tree(
            draft_token=dt, positions=pos, custom_mask=mask, tree_width=2, num_nodes=4
        )
        self.assertEqual(vi.topk, 2)
        self.assertIs(vi.custom_mask, mask)
        self.assertEqual(vi.draft_token_num, 4)

    def test_chain_default_unchanged(self):
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        vi = DFlashVerifyInput(
            draft_token=torch.zeros(4, dtype=torch.int64),
            positions=torch.zeros(4, dtype=torch.int64),
            draft_token_num=4,
        )
        self.assertEqual(vi.topk, 1)
        self.assertIsNone(vi.custom_mask)


class TestTreeStartupGuard(unittest.TestCase):
    """The live verify-loop integration is the #29524 GPU follow-up, so tree mode is
    guarded off at startup. Chain mode (width<=1) is unaffected. Tested against the
    extracted _handle_dflash_tree with a lightweight stub (no heavy ServerArgs)."""

    def _sa(self, width):
        from types import SimpleNamespace

        return SimpleNamespace(speculative_dflash_tree_width=width)

    def test_tree_width_gt_one_is_guarded(self):
        from sglang.srt.arg_groups.speculative_hook import _handle_dflash_tree

        with self.assertRaises(ValueError):
            _handle_dflash_tree(self._sa(2))

    def test_width_one_and_none_are_noop(self):
        from sglang.srt.arg_groups.speculative_hook import _handle_dflash_tree

        _handle_dflash_tree(self._sa(1))  # chain: no raise
        _handle_dflash_tree(self._sa(None))  # default: no raise


class TestTreeBridge(unittest.TestCase):
    """The verify-input bridge the deferred GPU worker integration will call."""

    def test_build_verify_input_packs_tree(self):
        from sglang.srt.speculative.dflash_info import build_dflash_tree_verify_input

        vi, bufs = build_dflash_tree_verify_input(
            root_tokens=[0],
            topk_tokens=[[[1, 2], [3, 4]]],
            topk_logprobs=[[[-0.1, -0.9], [-0.1, -0.9]]],
            tree_width=2,
            budget=4,
            construction="depth_first",
            base_positions=[5],
            kv_lens=[3],
            device="cpu",
        )
        self.assertEqual(vi.topk, 2)
        self.assertEqual(vi.draft_token_num, 4)
        self.assertEqual(vi.custom_mask.numel(), 3 * 4 + 4 * 4 * 1)
        self.assertEqual(bufs["draft_token"].shape, (1, 4))


class TestTensorInputs(unittest.TestCase):
    """Draft output arrives on-device; build_tree / verify_tree_greedy_cpu must accept
    tensors and produce the same result as the equivalent host lists."""

    def test_build_tree_accepts_tensors(self):
        tok = [[1, 2], [3, 4]]
        lp = [[-0.1, -0.9], [-0.1, -0.9]]
        from_lists = build_tree(0, tok, lp, tree_width=2, budget=4)
        from_tensors = build_tree(
            0,
            torch.tensor(tok, dtype=torch.int64),
            torch.tensor(lp, dtype=torch.float32),
            tree_width=2,
            budget=4,
        )
        self.assertEqual(from_tensors.tokens, from_lists.tokens)
        self.assertEqual(from_tensors.parent_indices, from_lists.parent_indices)

    def test_oracle_accepts_tensor_target(self):
        tree = DFlashDraftTree([0, 11, 22], [-1, 0, 1], [0, 1, 2], 3)
        out = verify_tree_greedy_cpu(
            tree, torch.tensor([11, 22, 99], dtype=torch.int64)
        )
        self.assertEqual(out["accept_indices"], [0, 1, 2])
        self.assertEqual(out["bonus_token"], 99)
        self.assertIsInstance(out["bonus_token"], int)


if __name__ == "__main__":
    unittest.main()
