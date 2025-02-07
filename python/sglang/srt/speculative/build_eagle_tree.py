# NOTE: Please run this file to make sure the test cases are correct.

from typing import List

import torch

from sglang.srt.utils import is_cuda_available

if is_cuda_available():
    from sgl_kernel import build_tree_kernel as sgl_build_tree_kernel
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def build_tree_kernel_efficient_preprocess(
    verified_id: torch.Tensor,
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_verify_tokens: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(
        1
    )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
    ss_token_list = torch.cat(
        token_list, dim=1
    )  # b, (self.topk + (num_steps-1) * self.topk)
    top_scores = torch.topk(score_list, num_verify_tokens - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values

    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()
    parent_list = torch.cat(parents_list[:-1], dim=1)

    return parent_list, top_scores_index, draft_tokens


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
):
    parent_list, top_scores_index, draft_tokens = (
        build_tree_kernel_efficient_preprocess(
            verified_id,
            score_list,
            token_list,
            parents_list,
            num_verify_tokens,
        )
    )

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    tree_mask = torch.full(
        (
            seq_lens_sum * num_verify_tokens
            + num_verify_tokens * num_verify_tokens * bs,
        ),
        True,
        device=device,
    )
    retrive_index = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_next_token = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_next_sibling = torch.full(
        (bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    positions = torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.long)

    sgl_build_tree_kernel_efficient(
        parent_list,
        top_scores_index,
        seq_lens.to(torch.int32),
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        spec_steps,
        num_verify_tokens,
    )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def build_tree_kernel(
    verified_id: torch.Tensor,
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
):
    parent_list, top_scores_index, draft_tokens = (
        build_tree_kernel_efficient_preprocess(
            verified_id,
            score_list,
            token_list,
            parents_list,
            num_verify_tokens,
        )
    )

    bs = seq_lens.numel()
    device = seq_lens.device

    tree_mask = torch.full(
        (
            seq_lens_sum * num_verify_tokens
            + num_verify_tokens * num_verify_tokens * bs,
        ),
        True,
        device=device,
    )
    retrive_index = torch.full(
        (bs, num_verify_tokens, spec_steps + 2), -1, device=device, dtype=torch.long
    )
    positions = torch.empty((bs * num_verify_tokens,), device=device, dtype=torch.long)

    sgl_build_tree_kernel(
        parent_list,
        top_scores_index,
        seq_lens.to(torch.int32),
        tree_mask,
        positions,
        retrive_index,
        topk,
        spec_steps,
        num_verify_tokens,
    )

    index = retrive_index.sum(dim=-1) != -spec_steps - 2
    cum_len = torch.cumsum(torch.sum(index, dim=-1), dim=-1)
    retrive_cum_len = torch.zeros(
        (cum_len.numel() + 1,), dtype=torch.int32, device="cuda"
    )
    retrive_cum_len[1:] = cum_len
    # TODO: this indexing cause a synchronization, optimize this
    retrive_index = retrive_index[index]
    return tree_mask, positions, retrive_index, retrive_cum_len, draft_tokens


def test_build_tree_kernel():
    def findp(p_i, index, parent_list):
        pos = index // 10
        index_list = index.tolist()
        parent_list = parent_list.tolist()
        res = [p_i]
        while True:
            p = pos[p_i]
            if p == 0:
                break
            token_idx = parent_list[p]
            p_i = index_list.index(token_idx)
            res.append(p_i)
        return res

    def create_mask(seq_len, draft_token, index, parent_list, max_depth):
        mask = []
        positions = []
        retrive_index = []
        for i, lens in enumerate(seq_len.tolist()):
            first_mask = torch.full((lens + draft_token,), True)
            first_mask[-(draft_token - 1) :] = False
            positions.append(lens)
            mask.append(first_mask)
            seq_order = []
            first_index = torch.Tensor([0] + [-1] * (depth + 1)).cuda().to(torch.long)
            r_index = [first_index]
            for j in range(draft_token - 1):
                mask.append(torch.full((lens + 1,), True))
                idx = findp(j, index, parent_list)

                seq_order.append(idx)
                positions.append(len(idx) + seq_len)
                t = torch.full((draft_token - 1,), False)
                t[idx] = True
                mask.append(t)

            for i in range(1, draft_token - 1):
                is_leaf = 0
                for j in range(draft_token - 1):
                    if i in seq_order[j]:
                        is_leaf += 1

                if is_leaf == 1:
                    order_list = [0] + [x + 1 for x in seq_order[i][::-1]]
                    for _ in range(max_depth + 1 - len(seq_order[i])):
                        order_list.append(-1)
                    order = torch.Tensor(order_list).cuda().to(torch.long)
                    r_index.append(order)
            retrive_index.append(torch.stack(r_index))

        return (
            torch.cat(mask).cuda(),
            torch.Tensor(positions).cuda().to(torch.long),
            torch.stack(retrive_index),
        )

    index = (
        torch.Tensor(
            [
                0,
                1,
                2,
                3,
                10,
                11,
                12,
                13,
                20,
                21,
                22,
                30,
                110,
                130,
                150,
                160,
                210,
                211,
                212,
                213,
                214,
                215,
                216,
                217,
                218,
                219,
                220,
                230,
                310,
                311,
                312,
                313,
                314,
                315,
                316,
                317,
                320,
                321,
                322,
                330,
                360,
                380,
                390,
                410,
                411,
                412,
                413,
                414,
                415,
                416,
                417,
                418,
                419,
                420,
                421,
                422,
                423,
                430,
                431,
                440,
                441,
                460,
                470,
            ]
        )
        .to(torch.long)
        .cuda()
    )

    parent_list = (
        torch.Tensor(
            [
                -1,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                20,
                30,
                21,
                13,
                22,
                40,
                23,
                110,
                130,
                160,
                150,
                190,
                120,
                111,
                121,
                200,
                180,
                210,
                211,
                212,
                213,
                214,
                215,
                216,
                220,
                230,
                217,
                310,
                311,
                312,
                313,
                320,
                314,
                321,
                315,
                316,
                317,
            ]
        )
        .to(torch.long)
        .cuda()
    )

    verified_seq_len = torch.Tensor([47]).to(torch.long).cuda()
    bs = verified_seq_len.shape[0]
    topk = 10
    depth = 5  # depth <= 10
    num_draft_token = 64

    tree_mask = torch.full(
        (
            torch.sum(verified_seq_len).item() * num_draft_token
            + num_draft_token * num_draft_token * bs,
        ),
        True,
    ).cuda()
    retrive_index = torch.full(
        (bs, num_draft_token, depth + 2), -1, device="cuda", dtype=torch.long
    )
    positions = torch.empty((bs * num_draft_token,), device="cuda", dtype=torch.long)

    sgl_build_tree_kernel(
        parent_list.unsqueeze(0),
        index.unsqueeze(0),
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        topk,
        depth,
        num_draft_token,
    )

    retrive_index = retrive_index[retrive_index.sum(dim=-1) != -depth - 2]

    c_mask, c_positions, c_retive_index = create_mask(
        verified_seq_len, num_draft_token, index, parent_list, depth
    )

    assert torch.allclose(tree_mask, c_mask), "tree mask has error."
    assert torch.allclose(positions, c_positions), "positions has error."
    assert torch.allclose(retrive_index, c_retive_index), "retrive_index has error."


def test_build_tree_kernel_efficient():
    verified_id = torch.tensor([29974, 13], device="cuda", dtype=torch.int32)
    score_list = [
        torch.tensor(
            [
                [[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],
                [[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.9142e-01, 1.2863e-02, 1.6873e-03, 1.1871e-03],
                    [2.4787e-01, 1.8818e-02, 1.4204e-02, 9.2235e-04],
                    [2.2971e-03, 1.6700e-06, 1.8737e-07, 8.3146e-08],
                    [1.2771e-03, 2.4374e-04, 1.7832e-04, 1.1947e-05],
                ],
                [
                    [8.4832e-02, 6.6068e-02, 5.8304e-02, 5.7851e-02],
                    [2.3616e-03, 1.1243e-03, 5.4368e-04, 2.7768e-04],
                    [2.5286e-04, 1.5578e-04, 2.8817e-05, 1.2888e-05],
                    [1.2834e-04, 2.5417e-06, 1.1279e-06, 1.6088e-08],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.6438e-01, 2.6997e-02, 2.4236e-05, 4.0821e-06],
                    [2.4402e-01, 2.8409e-03, 5.0935e-04, 2.9022e-04],
                    [1.6178e-02, 2.0567e-03, 4.5892e-04, 3.0034e-05],
                    [1.3023e-02, 5.0497e-04, 3.6371e-04, 8.7750e-05],
                ],
                [
                    [2.3263e-02, 2.0054e-02, 9.3990e-03, 2.7783e-03],
                    [6.4156e-02, 5.5506e-04, 1.0429e-04, 9.7211e-05],
                    [4.9950e-02, 5.0630e-03, 9.0068e-04, 3.3656e-04],
                    [7.5817e-03, 8.5731e-04, 6.9972e-04, 6.0793e-04],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    [6.6420e-01, 1.0525e-04, 6.5864e-05, 1.2253e-06],
                    [1.3019e-01, 1.0461e-01, 5.2083e-03, 1.6777e-03],
                    [2.0103e-02, 6.7335e-03, 1.2625e-04, 1.0364e-05],
                    [1.5142e-02, 7.0819e-04, 9.6595e-05, 8.7951e-05],
                ],
                [
                    [5.8608e-02, 1.8840e-03, 7.8535e-04, 4.4400e-04],
                    [1.2185e-02, 2.0684e-03, 1.7418e-03, 1.4327e-03],
                    [6.2455e-03, 6.1487e-03, 2.6862e-03, 1.8034e-03],
                    [1.8590e-03, 1.6151e-03, 1.2481e-03, 3.6038e-04],
                ],
            ],
            dtype=torch.float32,
            device="cuda",
        ),
    ]
    token_list = [
        torch.tensor(
            [[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]],
            dtype=torch.int64,
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29889,
                    29974,
                    29945,
                    29900,
                    29974,
                    29922,
                    29930,
                    29958,
                    29889,
                    29974,
                    29930,
                    29945,
                    29974,
                    29922,
                    29930,
                    29958,
                ],
                [
                    22550,
                    4136,
                    16492,
                    8439,
                    29871,
                    2,
                    3001,
                    13,
                    2,
                    13,
                    29906,
                    29946,
                    2,
                    13,
                    29871,
                    259,
                ],
            ],
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29946,
                    29945,
                    29953,
                    29906,
                    29896,
                    29945,
                    29900,
                    29906,
                    29896,
                    29945,
                    29906,
                    29953,
                    29896,
                    29945,
                    29906,
                    29946,
                ],
                [
                    29871,
                    2,
                    29901,
                    29889,
                    29871,
                    2,
                    395,
                    259,
                    29901,
                    29871,
                    2,
                    29889,
                    3001,
                    1234,
                    7146,
                    2186,
                ],
            ],
            device="cuda",
        ),
        torch.tensor(
            [
                [
                    29946,
                    29974,
                    29945,
                    29930,
                    29889,
                    29922,
                    29974,
                    29930,
                    29974,
                    29946,
                    29930,
                    29922,
                    29889,
                    29974,
                    29945,
                    29922,
                ],
                [
                    29941,
                    29906,
                    2,
                    29946,
                    29871,
                    450,
                    319,
                    14990,
                    29946,
                    29941,
                    2,
                    29906,
                    29871,
                    2,
                    3001,
                    13,
                ],
            ],
            device="cuda",
        ),
    ]
    parents_list = [
        torch.tensor(
            [[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype=torch.int64, device="cuda"
        ),
        torch.tensor([[4, 8, 9, 10], [4, 5, 6, 7]], dtype=torch.int64, device="cuda"),
        torch.tensor(
            [[20, 24, 21, 28], [24, 28, 20, 21]], dtype=torch.int64, device="cuda"
        ),
        torch.tensor(
            [[36, 40, 41, 44], [36, 40, 44, 45]], dtype=torch.int64, device="cuda"
        ),
    ]
    seq_lens = torch.tensor([5, 10], dtype=torch.int64, device="cuda")
    topk = 4
    depth = 4
    num_draft_token = 8

    tree_mask, position, retrive_index, retrive_cum_len, draft_tokens = (
        build_tree_kernel(
            verified_id=verified_id,
            score_list=score_list,
            token_list=token_list,
            parents_list=parents_list,
            seq_lens=seq_lens,
            seq_lens_sum=torch.sum(seq_lens).item(),
            topk=topk,
            spec_steps=depth,
            num_verify_tokens=num_draft_token,
        )
    )

    from sglang.srt.utils import first_rank_print

    first_rank_print("=========== build tree kernel ==========")
    # first_rank_print(f"{tree_mask=}", flush=True)
    first_rank_print(f"{position=}", flush=True)
    first_rank_print(f"{retrive_index=}", flush=True)
    first_rank_print(f"{retrive_cum_len=}", flush=True)
    first_rank_print(f"{draft_tokens=}", flush=True)
    assert position.tolist() == [5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14]
    assert retrive_index.tolist() == [
        [0, -1, -1, -1, -1, -1],
        [0, 2, 4, 6, -1, -1],
        [0, 1, 3, 5, 7, -1],
        [8, -1, -1, -1, -1, -1],
        [8, 9, 10, -1, -1, -1],
        [8, 9, 12, -1, -1, -1],
        [8, 9, 13, -1, -1, -1],
        [8, 9, 11, 14, 15, -1],
    ]
    assert retrive_cum_len.tolist() == [0, 3, 8]
    assert draft_tokens.tolist() == [
        29974,
        29896,
        29906,
        29889,
        29974,
        29946,
        29896,
        29946,
        13,
        13,
        22550,
        4136,
        16492,
        8439,
        29871,
        29941,
    ]

    (
        tree_mask,
        position,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    ) = build_tree_kernel_efficient(
        verified_id=verified_id,
        score_list=score_list,
        token_list=token_list,
        parents_list=parents_list,
        seq_lens=seq_lens,
        seq_lens_sum=torch.sum(seq_lens).item(),
        topk=topk,
        spec_steps=depth,
        num_verify_tokens=num_draft_token,
    )

    first_rank_print("=========== build tree kernel efficient ==========")
    # first_rank_print(f"{tree_mask=}", flush=True)
    first_rank_print(f"{position=}", flush=True)
    first_rank_print(f"{retrive_index=}", flush=True)
    first_rank_print(f"{retrive_next_token=}", flush=True)
    first_rank_print(f"{retrive_next_sibling=}", flush=True)
    first_rank_print(f"{draft_tokens=}", flush=True)
    assert position.tolist() == [5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14]
    assert retrive_index.tolist() == [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ]
    assert retrive_next_token.tolist() == [
        [1, 3, 4, 5, 6, 7, -1, -1],
        [1, 2, -1, 6, -1, -1, 7, -1],
    ]
    assert retrive_next_sibling.tolist() == [
        [-1, 2, -1, -1, -1, -1, -1, -1],
        [-1, -1, 3, 4, 5, -1, -1, -1],
    ]
    assert draft_tokens.tolist() == [
        29974,
        29896,
        29906,
        29889,
        29974,
        29946,
        29896,
        29946,
        13,
        13,
        22550,
        4136,
        16492,
        8439,
        29871,
        29941,
    ]


if __name__ == "__main__":
    test_build_tree_kernel_efficient()
    test_build_tree_kernel()
