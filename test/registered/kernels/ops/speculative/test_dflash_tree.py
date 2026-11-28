"""DFlash tree build and accept against a python best-first build and accept walk."""

import sys
import types

import pytest
import torch

from sglang.srt.speculative import dflash_tree as dt
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
N, B, S, K = dt.TREE_NODES, 4, 7, 4


@pytest.mark.skipif(not is_sm90_supported(), reason="SM90 kernels")
def test_tree_build_and_branching_accept():
    """The accepted path takes a later sibling at a branching node, and accept
    copies each of its K and V rows into the committed slots."""
    torch.manual_seed(0)
    cand = torch.rand(B, S, 64, device="cuda").argsort(-1)[..., :K]
    sc = torch.randn(B, S, K, K, device="cuda") * 3
    t = dt.TreeBuffers(B, "cuda")
    dt.build_tree(candidate_ids=cand, scores=sc, anchor=cand[:, 0, 0], out=t)
    lsm, c = sc.double().log_softmax(-1).tolist(), cand.tolist()
    for b in range(B):  # node: (token, parent, depth, ancestor bits)
        tree, front, lp, s, k, n = [(c[b][0][0], -1, 0, 1)], [], 0, -1, 0, 0
        while len(tree) < N:
            if s + 1 < S:  # the last pop's top 4 children
                r = lsm[b][s + 1][k]
                front += sorted((lp + r[i], s + 1, i, n) for i in range(K))[-4:]
            front.remove(e := max(front))
            (lp, s, k, p), n = e, len(tree)
            tree.append((c[b][s][k], p, s + 1, 1 << n | tree[p][3]))
        got = [x[b].tolist() for x in (t.tok, t.parent, t.depth, t.anc)]
        assert got == [list(x) for x in zip(*tree)]

    # The target follows each tree to its first branching node, takes that node's
    # last child (not the first sibling), then first children.
    tok, par = t.tok.tolist(), t.parent.tolist()
    pred = torch.full((B, N), 7777, dtype=torch.int64, device="cuda")
    branch = []
    for b in range(B):
        kids = [[j for j in range(N) if par[b][j] == i] for i in range(N)]
        fork = next(i for i in range(N) if len(kids[i]) > 1)
        path = [fork]
        while path[0]:
            path.insert(0, par[b][path[0]])
        nxt = kids[fork][-1]
        branch.append(nxt)
        while nxt is not None:
            path.append(nxt)
            nxt = kids[nxt][0] if kids[nxt] else None
        for u, v in zip(path, path[1:]):
            pred[b, u] = tok[b][v]
    loc = torch.arange(B * N, device="cuda").view(B, N)
    # two layers with separate K and V: row i of buffer n holds i + 1000 * n
    kv = [loc.view(-1, 1).repeat(1, 2) + 1000 * n for n in range(4)]
    pool = types.SimpleNamespace(k_buffer=kv[0::2], v_buffer=kv[1::2])
    mv = dt.TreeKVMove(pool, "cuda")
    res = dt.accept_tree(tree=t, target_predict=pred, verify_cache_loc=loc, kv_move=mv)
    for b in range(B):
        tok_b, par_b, p = tok[b], par[b], pred[b].tolist()
        w = [0]
        while q := [j for j in range(N) if par_b[j] == w[-1] and tok_b[j] == p[w[-1]]]:
            w.append(q[0])
        a, w = len(w) - 1, w + [j for j in range(N) if j not in w]
        assert branch[b] in w[1 : a + 1]  # accepted a later sibling
        out = [tok_b[j] for j in w[1 : a + 1]] + [p[w[a]]] + [0] * (N - 1 - a)
        assert [x[b].tolist() for x in res] == [out, a + 1, out[a]]
        assert t.perm[b].tolist() == [b * N + j for j in w]
        for n, buf in enumerate(kv):  # each K and V row of the path moved to its slot
            moved = buf[b * N : b * N + a + 1, 0] - 1000 * n
            assert moved.tolist() == t.perm[b, : a + 1].tolist(), ("KV"[n % 2], n)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
