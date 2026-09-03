"""Unit tests for ``ngram_worker._linearize_chain`` (CPU, no server, no model).

Inputs are built in the layout ``NgramCorpus.batch_get`` returns: flat int64
tokens ``(bs*D,)`` and a flat int64 ancestor mask ``(bs*D*D,)``, produced by a
Python replica of ``trie.cpp`` ``buildRecency`` (bfs breadth 1: every anchor
re-enters from the root and inserts one chain, shared prefixes merge, node
budget D-1) followed by ``result.cpp`` ``fillResult`` (BFS order, zero-padding
nodes as ROOT children, mask row i = copy of the parent row + the diagonal).

One thing the replica cannot pin down: ``Node.next`` is a
``std::unordered_map`` (``result.h``), so the BFS order AMONG SIBLINGS -- and
therefore which chain is "first" on a length tie -- is hash order in the real
corpus. The replica takes a ``child_order`` hook and every star / partial case
is run under several sibling orders; the assertions only require the result to
be ONE of the depth-maximal root chains, never a specific sibling.

Checks
  1. a single chain is preserved (tokens and tril mask unchanged);
  2. a star (several anchors -> several depth-1 root children) collapses to ONE
     depth-1 child, padded, under every sibling order; the unlinearized star has
     duplicate positions;
  3. a partial match (one 2-chain + one 1-chain) keeps the longest chain under
     every sibling order;
  4. padding nodes (root children with token 0) never displace a real chain;
     no-match -> [root, 0, 0, 0];
  5. branching below the root (D=6) keeps the longest path, not the first child;
  6. batched (bs=3) inputs: per-request results, output dtypes / shapes match
     ``batch_get``, inputs untouched;
  7. what the verify code derives from the mask: a replica of
     ``reconstruct_indices_from_tree_mask`` (``ngram_utils.cu``) gives positions
     seq_len..seq_len+D-1, retrieve_next_token [1..D-1, -1], retrieve_next_sibling
     all -1, parent(row j) == row j-1; the host-side ``_derive_tree_links`` agrees;
  8. a replica of ``verify_tree_greedy`` (``kernels/aot/csrc/cpu/spec.cpp``) on
     the chain: accept_index is always a row prefix b*D + arange(n), -1 after;
     accept_lens in [1, D]; a padding-0 node is accepted only when the target
     argmax at its parent row is 0;
  9. the real JIT ``NgramCorpus`` at breadth 1 / D=4 fed with a history that
     provably creates a multi-anchor star (the raw mask of request 0 is asserted
     to be non-tril); its ``batch_get`` output is linearized and re-checked
     (skipped when the corpus extension cannot be built on the host).
"""

import unittest
from collections import deque

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.speculative.ngram_worker import (  # noqa: E402
    _derive_tree_links,
    _linearize_chain,
)

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

D = 4
ROOT = 1234


# ---------------------------------------------------------------------- corpus replicas
def fill_result(last_token, d, tree, child_order=None):
    """``result.cpp`` ``fillResult``: ``tree[node] = {token: child_node}``; BFS from
    root 0; the root row holds ``last_token``; zero padding to ``d`` with
    ``prevs = 0`` (root children); ``mask[i] = mask[parent][:parent+1]`` then the
    diagonal. ``child_order(items) -> items`` models the ``std::unordered_map``
    iteration order of ``Node.next``: insertion order by default here."""
    child_order = child_order or (lambda items: items)
    tokens, prevs = [last_token], [-1]
    queue = deque((tok, nxt, 0) for tok, nxt in child_order(list(tree[0].items())))
    while queue:
        tok, nxt, prev = queue.popleft()
        tokens.append(tok)
        prevs.append(prev)
        for t, n in child_order(list(tree[nxt].items())):
            queue.append((t, n, len(tokens) - 1))
    while len(tokens) < d:
        tokens.append(0)
        prevs.append(0)
    n = len(tokens)
    mask = np.zeros((n, n), dtype=np.int64)
    mask[0, 0] = 1
    for i in range(n):
        if prevs[i] != -1:
            mask[i, : prevs[i] + 1] = mask[prevs[i], : prevs[i] + 1]
        mask[i, i] = 1
    return np.array(tokens, dtype=np.int64), mask


# sibling orders a std::unordered_map could produce
SIBLING_ORDERS = [
    lambda items: items,
    lambda items: list(reversed(items)),
    lambda items: items[1:] + items[:1],  # rotate: the middle child first
    lambda items: sorted(items, key=lambda kv: kv[0]),
    lambda items: sorted(items, key=lambda kv: (kv[0] * 2654435761) % 97),
]


def build_recency(last_token, d, anchor_paths, child_order=None):
    """``trie.cpp`` ``buildRecency`` at bfs breadth 1: each anchor (deepest first)
    re-enters from the root and inserts its chain (an existing token at the
    parent is reused); node ids 1..d-1."""
    n_real = d - 1
    tree = [dict() for _ in range(d)]
    cursor = 1
    for path in anchor_paths:
        parent = 0
        for tok in path:
            if cursor > n_real:
                break
            if tok in tree[parent]:
                parent = tree[parent][tok]
            else:
                tree[parent][tok] = cursor
                parent = cursor
                cursor += 1
    return fill_result(last_token, d, tree, child_order)


def maximal_root_chains(toks, mask, d):
    """All root->leaf token paths of maximal depth in one request's tree
    (padding nodes included: they are root children with token 0)."""
    toks = np.asarray(toks).reshape(d)
    m = np.asarray(mask).reshape(d, d)
    parents = [-1] + [max(j for j in range(i) if m[i, j]) for i in range(1, d)]
    children = [[i for i in range(d) if parents[i] == p] for p in range(d)]

    def paths(i):
        if not children[i]:
            return [[]]
        return [[c] + rest for c in children[i] for rest in paths(c)]

    all_paths = paths(0)
    depth = max(len(p) for p in all_paths)
    return depth, {tuple(int(toks[k]) for k in p) for p in all_paths if len(p) == depth}


def batch_get_like(results):
    """Flatten per-request (tokens, mask) exactly like ``NgramCorpus.batch_get``."""
    toks = np.concatenate([t for t, _ in results]).astype(np.int64)
    mask = np.concatenate([m.reshape(-1) for _, m in results]).astype(np.int64)
    return toks, mask


def tril(d):
    return np.tril(np.ones((d, d), dtype=np.int64))


# ---------------------------------------------------------------------- verify-side replicas
def reconstruct_ref(mask, seq_lens, bs, d):
    """Replica of sgl_kernel ``reconstructIndicesFromTreeMask`` (``ngram_utils.cu``)."""
    tree = mask.reshape(bs, d, d).astype(bool)
    positions = np.empty(bs * d, dtype=np.int64)
    ri = np.empty((bs, d), dtype=np.int64)
    nt = np.full((bs, d), -1, dtype=np.int64)
    ns = np.full((bs, d), -1, dtype=np.int64)
    parents = np.full((bs, d), -1, dtype=np.int64)
    for b in range(bs):
        for t in range(d):
            depth, parent = 0, -1
            for i in range(t - 1, -1, -1):
                if tree[b, t, i]:
                    depth += 1
                    if parent == -1:
                        parent = i
            ri[b, t] = b * d + t
            positions[b * d + t] = depth + seq_lens[b]
            parents[b, t] = parent
            for i in range(t + 1, d):
                if tree[b, i, t]:
                    nt[b, t] = i
                    break
            if parent != -1:
                for i in range(t + 1, d):
                    if tree[b, i, parent] and not tree[b, i, parent + 1 : i].any():
                        ns[b, t] = i
                        break
    return positions, ri, nt, ns, parents


def verify_greedy_ref(candidates, ri, nt, ns, target_predict, bs, d):
    """Replica of ``verify_tree_greedy_kernel_impl`` (``kernels/aot/csrc/cpu/spec.cpp``);
    accept_index has ``num_spec_step = d`` columns."""
    predicts = np.zeros(bs * d, dtype=np.int64)
    accept_index = np.full((bs, d), -1, dtype=np.int64)
    accept_token_num = np.zeros(bs, dtype=np.int64)
    cand, rif, ntf, nsf, tp = (
        x.reshape(-1) for x in (candidates, ri, nt, ns, target_predict)
    )
    for bx in range(bs):
        off = bx * d
        last = rif[off]
        accept_index[bx, 0] = last
        n_correct, cur = 0, 0
        for _ in range(1, d):
            cur = ntf[off + cur]
            while cur != -1:
                draft_idx, draft_tok = rif[off + cur], cand[off + cur]
                if draft_tok == tp[last]:
                    predicts[last] = tp[last]
                    n_correct += 1
                    accept_index[bx, n_correct] = draft_idx
                    last = draft_idx
                    break
                cur = nsf[off + cur]
            if cur == -1:
                break
        accept_token_num[bx] = n_correct
        predicts[last] = tp[last]
    return predicts, accept_index, accept_token_num + 1  # accept_lens incl. bonus


class TestLinearizeChain(CustomTestCase):
    def assert_linearized(self, toks, mask, d, expect_depth=None):
        """One request: result is a depth-maximal root chain + zero padding, mask tril."""
        depth, chains = maximal_root_chains(toks, mask, d)
        if expect_depth is not None:
            self.assertEqual(depth, expect_depth, chains)
        t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, d)
        self.assertTrue((m.reshape(d, d) == tril(d)).all(), m)
        self.assertEqual(t[0], toks[0])
        self.assertIn(tuple(t[1 : 1 + depth].tolist()), chains, t.tolist())
        self.assertTrue((t[1 + depth :] == 0).all(), t)
        return t, m, chains

    def check_chain_derivations(self, toks, mask, bs, d, seq_lens=None):
        """Checks 7 and 8 on a linearized (toks, mask) pair."""
        if seq_lens is None:
            seq_lens = np.array([100 + 7 * b for b in range(bs)], dtype=np.int64)
        m = mask.reshape(bs, d, d)
        self.assertTrue((m == tril(d)[None]).all(), f"mask is not tril:\n{m}")
        positions, ri, nt, ns, parents = reconstruct_ref(mask, seq_lens, bs, d)
        exp_pos = (seq_lens[:, None] + np.arange(d)[None, :]).reshape(-1)
        self.assertTrue((positions == exp_pos).all(), (positions, exp_pos))
        self.assertTrue(
            (ri == (np.arange(bs)[:, None] * d + np.arange(d)[None, :])).all()
        )
        exp_nt = np.tile(np.append(np.arange(1, d), -1), (bs, 1))
        self.assertTrue((nt == exp_nt).all(), (nt, exp_nt))
        self.assertTrue((ns == -1).all(), ns)
        exp_par = np.tile(np.arange(-1, d - 1), (bs, 1))
        self.assertTrue((parents == exp_par).all(), parents)
        h_nt, h_ns = _derive_tree_links(mask, bs, d)
        self.assertTrue(torch.equal(h_nt, torch.from_numpy(exp_nt)), h_nt)
        self.assertTrue(torch.equal(h_ns, torch.from_numpy(ns)), h_ns)
        # greedy verify: accept 0..d-1 drafts by crafting target_predict per request
        cand = toks.reshape(bs, d)
        for n_acc in range(d):
            tp = np.full((bs, d), -7, dtype=np.int64)  # -7: never equal to a draft
            for b in range(bs):
                tp[b, :n_acc] = cand[b, 1 : n_acc + 1]  # row j predicts draft row j+1
                tp[b, n_acc:] = 999_999 + b  # bonus token(s)
            predicts, ai, al = verify_greedy_ref(cand, ri, nt, ns, tp, bs, d)
            for b in range(bs):
                n = int(al[b])
                self.assertEqual(n, n_acc + 1)
                self.assertTrue((ai[b, :n] == b * d + np.arange(n)).all(), ai[b])
                self.assertTrue((ai[b, n:] == -1).all(), ai[b])
                acc = predicts[ai[b, :n]]
                self.assertTrue((acc[:-1] == cand[b, 1:n]).all(), acc)
                self.assertEqual(acc[-1], 999_999 + b)
        return positions, ri, nt, ns

    def test_single_chain_preserved(self):
        toks, mask = build_recency(ROOT, D, [[11, 12, 13]])
        self.assertEqual(toks.tolist(), [ROOT, 11, 12, 13])
        self.assertTrue((mask == tril(D)).all())
        t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
        self.assertEqual(t.tolist(), [ROOT, 11, 12, 13])
        self.assertTrue((m.reshape(D, D) == tril(D)).all())

    def test_star_collapses_to_one_chain(self):
        # three anchors, each one depth-1 child -> ONE depth-1 child kept, padded with 0.
        toks, mask = build_recency(ROOT, D, [[21], [22], [23]])
        self.assertEqual(toks.tolist(), [ROOT, 21, 22, 23])
        self.assertEqual(
            mask.tolist(),
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]],
        )
        pos_star = reconstruct_ref(mask.reshape(-1), np.array([50]), 1, D)[0]
        self.assertEqual(pos_star.tolist(), [50, 51, 51, 51])  # the collision
        seen = []
        for order in SIBLING_ORDERS:
            toks_o, mask_o = build_recency(ROOT, D, [[21], [22], [23]], order)
            self.assertTrue((mask_o == mask).all())
            self.assertEqual(sorted(toks_o.tolist()), sorted(toks.tolist()))
            t, _, chains = self.assert_linearized(toks_o, mask_o, D, expect_depth=1)
            self.assertEqual(chains, {(21,), (22,), (23,)})
            # lowest node index wins the tie
            self.assertEqual(t.tolist(), [ROOT, int(toks_o[1]), 0, 0])
            seen.append(int(t[1]))
        self.assertEqual(set(seen), {21, 22, 23})  # the orders really differed

    def test_partial_star_keeps_longest_chain(self):
        # root->a->b and root->c (deepest anchor first, as getExpandableAnchors_ orders them)
        toks, mask = build_recency(ROOT, D, [[31, 32], [33]])
        self.assertEqual(toks.tolist(), [ROOT, 31, 33, 32])  # BFS: depth-1 nodes first
        self.assertEqual(
            mask.tolist(),
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 1]],
        )
        t, _, chains = self.assert_linearized(toks, mask, D, expect_depth=2)
        self.assertEqual(chains, {(31, 32)})
        self.assertEqual(t.tolist(), [ROOT, 31, 32, 0])
        for order in SIBLING_ORDERS:
            for anchors in ([[31, 32], [33]], [[33], [31, 32]]):
                toks_o, mask_o = build_recency(ROOT, D, anchors, order)
                t, _, _ = self.assert_linearized(toks_o, mask_o, D, expect_depth=2)
                self.assertEqual(t.tolist(), [ROOT, 31, 32, 0], (toks_o, mask_o))

    def test_padding_and_no_match(self):
        toks, mask = build_recency(ROOT, D, [[41]])
        self.assertEqual(toks.tolist(), [ROOT, 41, 0, 0])
        self.assertEqual(  # pads are root children
            mask.tolist(),
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]],
        )
        t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
        self.assertEqual(t.tolist(), [ROOT, 41, 0, 0])
        self.assertTrue((m.reshape(D, D) == tril(D)).all())
        toks, mask = build_recency(ROOT, D, [])
        self.assertEqual(toks.tolist(), [ROOT, 0, 0, 0])
        t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
        self.assertEqual(t.tolist(), [ROOT, 0, 0, 0])
        self.assertTrue((m.reshape(D, D) == tril(D)).all())
        # a real 2-chain listed AFTER a padding-like real child keeps the 2-chain
        toks, mask = build_recency(ROOT, D, [[0], [42, 43]])
        t, _ = _linearize_chain(*batch_get_like([(toks, mask)]), 1, D)
        self.assertEqual(t.tolist(), [ROOT, 42, 43, 0])

    def test_branch_below_root_keeps_longest_path(self):
        d6 = 6
        toks, mask = build_recency(ROOT, d6, [[51, 52], [51, 53, 54]])
        self.assertEqual(toks.tolist(), [ROOT, 51, 52, 53, 54, 0])
        t, m = _linearize_chain(*batch_get_like([(toks, mask)]), 1, d6)
        self.assertEqual(t.tolist(), [ROOT, 51, 53, 54, 0, 0])
        self.assertTrue((m.reshape(d6, d6) == tril(d6)).all())
        self.check_chain_derivations(t, m, 1, d6)
        for order in SIBLING_ORDERS:
            toks_o, mask_o = build_recency(ROOT, d6, [[51, 52], [51, 53, 54]], order)
            t, _, _ = self.assert_linearized(toks_o, mask_o, d6, expect_depth=3)
            self.assertEqual(t.tolist(), [ROOT, 51, 53, 54, 0, 0], toks_o)

    def _batched(self):
        rs = [
            build_recency(1, D, [[11, 12, 13]]),
            build_recency(2, D, [[21], [22], [23]]),
            build_recency(3, D, []),
        ]
        return batch_get_like(rs)

    def test_batched_shapes_dtypes_inputs_untouched(self):
        toks, mask = self._batched()
        toks0, mask0 = toks.copy(), mask.copy()
        self.assertEqual((toks.dtype, mask.dtype), (np.int64, np.int64))
        self.assertEqual((toks.shape, mask.shape), ((3 * D,), (3 * D * D,)))
        t, m = _linearize_chain(toks, mask, 3, D)
        self.assertEqual((t.dtype, m.dtype), (np.int64, np.int64))
        self.assertEqual((t.shape, m.shape), ((3 * D,), (3 * D * D,)))
        self.assertTrue(t.flags.c_contiguous and m.flags.c_contiguous)
        self.assertTrue(t.flags.writeable and m.flags.writeable)
        self.assertEqual(
            t.reshape(3, D).tolist(),
            [[1, 11, 12, 13], [2, 21, 0, 0], [3, 0, 0, 0]],
        )
        self.assertTrue((toks == toks0).all() and (mask == mask0).all())

    def test_verify_side_derivations(self):
        toks, mask = self._batched()
        t, m = _linearize_chain(toks, mask, 3, D)
        _, ri, nt, ns = self.check_chain_derivations(t, m, 3, D)
        # padding-0 semantics: accepted only when the parent row's argmax is 0
        cand = t.reshape(3, D)
        tp = np.array([[11, 12, 13, 5], [21, 0, 0, 5], [0, 0, 7, 5]])
        predicts, ai, al = verify_greedy_ref(cand, ri, nt, ns, tp, 3, D)
        self.assertEqual(al.tolist(), [4, 4, 3])
        self.assertEqual(ai.tolist(), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, -1]])
        self.assertEqual(predicts[ai[2, :3]].tolist(), [0, 0, 7])
        tp = np.array([[11, 12, 13, 5], [21, 9, 0, 5], [9, 0, 7, 5]])
        _, ai, al = verify_greedy_ref(cand, ri, nt, ns, tp, 3, D)
        self.assertEqual(al.tolist(), [4, 2, 1])
        self.assertEqual(ai.tolist(), [[0, 1, 2, 3], [4, 5, -1, -1], [8, -1, -1, -1]])

    def test_real_corpus_star(self):
        try:
            from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

            # max_trie_depth=4: the deepest anchor (7,8) of query [7,8] yields the
            # chain 20->5 and the depth-1 anchor (8) inserts its most recent child
            # 30 as a SECOND root child -> a star, raw mask not tril.
            corpus = NgramCorpus(
                max_trie_depth=4,
                min_bfs_breadth=1,
                max_bfs_breadth=1,
                draft_token_num=D,
            )
        except Exception as e:  # the JIT extension cannot be built on this host
            self.skipTest(f"NgramCorpus unavailable: {type(e).__name__}: {e}")
        corpus.batch_put([[7, 8, 20, 5, 8, 30]])
        corpus.synchronize()
        q = [[7, 8], [5, 8], [99, 98]]
        raw_t, raw_m = corpus.batch_get(
            [f"r{i}" for i in range(3)], q, [len(x) for x in q]
        )
        self.assertEqual((raw_t.dtype, raw_m.dtype), (np.int64, np.int64))
        self.assertEqual(raw_t.shape, (3 * D,))
        raw_m3 = raw_m.reshape(3, D, D)
        self.assertFalse((raw_m3[0] == tril(D)).all(), raw_m3[0])
        star_pos = reconstruct_ref(raw_m3[0].reshape(-1), np.array([50]), 1, D)[0]
        self.assertLess(len(set(star_pos.tolist())), D)  # duplicate positions
        self.assertEqual(sorted(raw_t.reshape(3, D)[0, 1:].tolist()), [5, 20, 30])
        depth0, chains0 = maximal_root_chains(raw_t.reshape(3, D)[0], raw_m3[0], D)
        self.assertEqual((depth0, chains0), (2, {(20, 5)}))
        lt, lm = _linearize_chain(raw_t, raw_m, 3, D)
        self.assertEqual(lt.reshape(3, D)[0].tolist(), [8, 20, 5, 0])
        self.check_chain_derivations(lt, lm, 3, D)
        for b in range(3):
            self.assert_linearized(raw_t.reshape(3, D)[b], raw_m3[b], D)
            chain = [x for x in lt.reshape(3, D)[b, 1:].tolist() if x != 0]
            leaf_paths = corpus.leaf_paths_from_mask(
                raw_t.reshape(3, D)[b].tolist(), raw_m3[b].tolist()
            )
            leaf_paths = [p[1:] for p in leaf_paths]  # drop the root token
            self.assertTrue(
                any(p[: len(chain)] == chain for p in leaf_paths) or not chain,
                (chain, leaf_paths),
            )


if __name__ == "__main__":
    unittest.main()
