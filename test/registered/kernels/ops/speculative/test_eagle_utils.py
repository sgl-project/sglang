import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

# CUDA and ROCm both build this kernel through the JIT launcher; MUSA still
# gets it from the AOT extension.
if torch.version.cuda is not None or torch.version.hip is not None:
    from sglang.kernels.ops.speculative.tree import verify_tree_greedy
else:
    from sgl_kernel import verify_tree_greedy

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_verify_tree_greedy():
    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device="cuda")
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10
    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i][j]) < 10:
                target_logits[i][j][18] = 10

    target_predict = torch.argmax(target_logits, dim=-1)
    predict_shape = (12,)

    bs = candidates.shape[0]
    num_spec_step = 4

    predicts = torch.full(
        predict_shape, -1, dtype=torch.int32, device="cuda"
    )  # mutable
    accept_index = torch.full(
        (bs, num_spec_step), -1, dtype=torch.int32, device="cuda"
    )  # mutable
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device="cuda")  # mutable

    verify_tree_greedy(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        target_predict=target_predict,
    )

    # Check the expected output.
    assert predicts.tolist() == [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
    assert accept_index.tolist() == [
        [0, 3, 4, 5],
        [6, 10, 11, -1],
    ]
    assert accept_token_num.tolist() == [3, 2]


def _chain_accept(linear, cand, ridx, rnext, rsib, target, sim_len, real_draft):
    from sglang.kernels.ops.speculative.eagle import _chain_accept_kernel
    from sglang.srt.utils import next_power_of_2

    bs, width = cand.shape
    out = dict(
        predict=torch.full((bs * width,), -7, dtype=torch.int32, device="cuda"),
        accept_index=torch.full((bs, width), -7, dtype=torch.int32, device="cuda"),
        num_correct=torch.full((bs,), -7, dtype=torch.int32, device="cuda"),
        accept_lens=torch.full((bs,), -7, dtype=torch.int32, device="cuda"),
        sim_index=torch.full((bs, width), -7, dtype=torch.int32, device="cuda"),
    )
    _chain_accept_kernel[(bs,)](
        out["predict"],
        out["accept_index"],
        out["num_correct"],
        out["accept_lens"],
        out["sim_index"] if sim_len is not None else None,
        cand,
        ridx,
        rnext,
        rsib,
        target,
        target,
        sim_len,
        WIDTH=width,
        BLOCK=next_power_of_2(width),
        VERIFY=True,
        FINISH=True,
        SIMULATE=sim_len is not None,
        REAL_DRAFT_TOKENS=real_draft,
        LINEAR=linear,
        num_warps=1,
    )
    return out


@pytest.mark.parametrize("mode", ["real", "simulated", "simulated-draft-tokens"])
def test_chain_accept_linear_is_the_walk(mode):
    """SGLANG_PERFLL_GLUE_LINEAR_ACCEPT: the vector prefix match writes every output the
    tree walk writes, for canonical chains with every accepted length, and a tree that is
    not a canonical chain takes the walk."""
    g = torch.Generator(device="cuda").manual_seed(2026)
    bs, width = 8, 7
    target = torch.randint(0, 50000, (bs, width), device="cuda", generator=g)
    cand = torch.randint(0, 50000, (bs, width), device="cuda", generator=g)
    for b in range(bs):
        k = b % width  # drafts 1..k equal the previous node's target
        cand[b, 1 : k + 1] = target[b, :k]
        if k + 1 < width:
            cand[b, k + 1] = target[b, k] + 1
    rows = torch.arange(bs * width, device="cuda").view(bs, width)
    col = torch.arange(width, device="cuda").expand(bs, width)
    rnext = torch.where(col < width - 1, col + 1, -1)
    rsib = torch.full_like(rnext, -1)
    # request 3 is a tree: node 1 has a sibling, node 2 is the second child of node 0
    rnext_tree, rsib_tree = rnext.clone(), rsib.clone()
    rnext_tree[3, 1], rsib_tree[3, 1] = 3, 2
    sim_len = (
        None if mode == "real" else torch.tensor([4], dtype=torch.int64, device="cuda")
    )
    for nxt, sib in ((rnext, rsib), (rnext_tree, rsib_tree)):
        args = (cand, rows, nxt, sib, target, sim_len, mode == "simulated-draft-tokens")
        walk = _chain_accept(False, *args)
        linear = _chain_accept(True, *args)
        torch.cuda.synchronize()
        for key in walk:
            assert torch.equal(walk[key], linear[key]), key


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
