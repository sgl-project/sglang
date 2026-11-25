"""DFlash tree kernels against a python best-first build and accept walk."""

import sys
import types

import pytest
import torch

from sglang.srt.speculative import dflash_tree as dt
from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
N, B, S = dt.TREE_NODES, 37, 7


@pytest.mark.skipif(not is_sm90_supported(), reason="SM90 kernels")
@pytest.mark.parametrize("K", [4, 32])
def test_tree(K):
    torch.manual_seed(K)
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

    # the target's argmax at each parent is a child's token, or knocked out
    pred = t.tok.scatter(1, t.parent[:, 1:].long(), t.tok[:, 1:])
    pred[torch.rand(B, N, device="cuda") < 0.3] = -1
    loc = torch.arange(B * N, device="cuda").view(B, N)
    # two layers with separate K and V: row i of buffer n holds i + 1000 * n
    kv = [loc.view(-1, 1).repeat(1, 2) + 1000 * n for n in range(4)]
    pool = types.SimpleNamespace(k_buffer=kv[0::2], v_buffer=kv[1::2])
    mv = dt.TreeKVMove(pool, "cuda")
    res = dt.accept_tree(tree=t, target_predict=pred, verify_cache_loc=loc, kv_move=mv)
    for b in range(B):
        tok, par, p = (x[b].tolist() for x in (t.tok, t.parent, pred))
        w = [0]
        while q := [j for j in range(N) if par[j] == w[-1] and tok[j] == p[w[-1]]]:
            w.append(q[0])
        a, w = len(w) - 1, w + [j for j in range(N) if j not in w]
        out = [tok[j] for j in w[1 : a + 1]] + [p[w[a]]] + [0] * (N - 1 - a)
        assert [x[b].tolist() for x in res] == [out, a + 1, out[a]]
        assert t.perm[b].tolist() == [b * N + j for j in w]
        for n, buf in enumerate(kv):  # each K and V row of the path moved to its slot
            moved = buf[b * N : b * N + a + 1, 0] - 1000 * n
            assert moved.tolist() == t.perm[b, : a + 1].tolist()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
