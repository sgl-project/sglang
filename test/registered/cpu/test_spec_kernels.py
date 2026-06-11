import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from torch.nn.functional import softplus
from utils import precision

from sglang.srt.speculative.eagle_utils import TreeMaskMode, organize_draft_results
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-b-test-cpu")

torch.manual_seed(1234)

PAD_SLOT_ID = -1


def _topk1_chain_inputs(bs, num_steps):
    parent_width = num_steps if num_steps > 1 else 0
    parent_list = torch.arange(-1, parent_width - 1, dtype=torch.int64).repeat(bs, 1)
    selected_index = torch.arange(num_steps, dtype=torch.int64).repeat(bs, 1)
    return parent_list, selected_index


def _gen_draft_tree(bs, topk, num_steps, draft_token_num):
    """Simulate the EAGLE draft loop (select_top_k_tokens + organize_draft_results)
    with random probabilities to obtain a valid random (parent_list, selected_index).
    """
    scores = torch.rand(bs, topk, dtype=torch.float32)
    score_chunks = [scores]
    parents_chunks = [torch.arange(-1, topk, dtype=torch.int64).expand(bs, -1)]
    cum_scores = scores
    for i in range(1, num_steps):
        # Probabilities in (0, 1): a child cumulative score is strictly smaller
        # than its parent's, so the global topk always keeps full ancestor chains.
        step_p = torch.rand(bs, topk, topk, dtype=torch.float32)
        expand_scores = cum_scores.unsqueeze(2) * step_p
        cum_scores, topk_cs_index = torch.topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )
        score_chunks.append(expand_scores.flatten(start_dim=1))
        parents_chunks.append(topk_cs_index + (topk * topk * (i - 1) + topk))
    score_flat = torch.cat(score_chunks, dim=1)
    selected_index = torch.sort(
        torch.topk(score_flat, draft_token_num - 1, dim=-1).indices, dim=-1
    ).values
    parent_list = torch.cat(parents_chunks[:-1], dim=1)
    return parent_list, selected_index


def _ref_build_tree(parent_list, selected_index, seq_lens, topk, draft_token_num):
    bs = selected_index.shape[0]
    retrieve_index = torch.full((bs, draft_token_num), -1, dtype=torch.int64)
    retrieve_next_token = torch.full((bs, draft_token_num), -1, dtype=torch.int64)
    retrieve_next_sibling = torch.full((bs, draft_token_num), -1, dtype=torch.int64)
    positions = torch.zeros(bs * draft_token_num, dtype=torch.int64)
    tree_mask = torch.zeros(bs, draft_token_num, draft_token_num, dtype=torch.bool)

    for bid in range(bs):
        off = bid * draft_token_num
        sel = selected_index[bid].tolist()
        parents = parent_list[bid].tolist()
        retrieve_index[bid] = torch.arange(off, off + draft_token_num)

        # parent position (in tree-node numbering) of each node i >= 1
        parent_pos = [0] * draft_token_num
        for i in range(1, draft_token_num):
            parent_tb_idx = sel[i - 1] // topk
            if parent_tb_idx == 0:
                parent_pos[i] = 0
            else:
                parent_token_idx = parents[parent_tb_idx]
                parent_pos[i] = sel.index(parent_token_idx) + 1

        # head-insertion linking, iterating from the last node (kernel order)
        next_token = [-1] * draft_token_num
        next_sibling = [-1] * draft_token_num
        for i in range(draft_token_num - 1, 0, -1):
            p = parent_pos[i]
            if next_token[p] == -1:
                next_token[p] = i
            else:
                next_sibling[i] = next_token[p]
                next_token[p] = i
        retrieve_next_token[bid] = torch.tensor(next_token, dtype=torch.int64)
        retrieve_next_sibling[bid] = torch.tensor(next_sibling, dtype=torch.int64)

        seq_len = int(seq_lens[bid])
        positions[off] = seq_len
        tree_mask[bid, :, 0] = True
        for i in range(1, draft_token_num):
            ancestors = []
            j = i
            while j != 0:
                ancestors.append(j)
                j = parent_pos[j]
            positions[off + i] = seq_len + len(ancestors)
            for j in ancestors:
                tree_mask[bid, i, j] = True

    return (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        positions,
        tree_mask,
    )


def _run_build_tree_kernel(
    parent_list, selected_index, seq_lens, topk, num_steps, draft_token_num, mode
):
    bs = seq_lens.numel()
    if mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (bs * draft_token_num * draft_token_num,), True, dtype=torch.bool
        )
    else:  # FULL_MASK carries the (all-true) prefix columns explicitly
        seq_lens_sum = int(seq_lens.sum())
        tree_mask = torch.full(
            (seq_lens_sum * draft_token_num + bs * draft_token_num * draft_token_num,),
            True,
            dtype=torch.bool,
        )
    positions = torch.zeros(bs * draft_token_num, dtype=torch.int64)
    retrieve_buf = torch.full((3, bs, draft_token_num), -1, dtype=torch.int64)
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    torch.ops.sgl_kernel.build_tree_kernel_efficient_cpu(
        parent_list,
        selected_index,
        seq_lens.to(torch.int32),
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        topk,
        num_steps,
        draft_token_num,
        mode,
    )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def _ref_verify_tree_greedy(
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    num_spec_step,
):
    bs, draft_token_num = candidates.shape
    predicts = torch.full((bs * draft_token_num,), -1, dtype=torch.int32)
    accept_indices = torch.full((bs, num_spec_step), -1, dtype=torch.int32)
    num_correct_drafts = torch.zeros(bs, dtype=torch.int32)
    target_flat = target_predict.reshape(-1)

    for bx in range(bs):
        last_accept_idx = int(retrieve_index[bx, 0])
        accept_indices[bx, 0] = last_accept_idx
        num_correct = 0
        cur = 0
        for _ in range(1, num_spec_step):
            cur = int(retrieve_next_token[bx, cur])
            while cur != -1:
                draft_idx = int(retrieve_index[bx, cur])
                draft_token = int(candidates[bx, cur])
                target_token = int(target_flat[last_accept_idx])
                if draft_token == target_token:
                    predicts[last_accept_idx] = target_token
                    num_correct += 1
                    accept_indices[bx, num_correct] = draft_idx
                    last_accept_idx = draft_idx
                    break
                cur = int(retrieve_next_sibling[bx, cur])
            if cur == -1:
                break
        num_correct_drafts[bx] = num_correct
        predicts[last_accept_idx] = int(target_flat[last_accept_idx])
    return predicts, accept_indices, num_correct_drafts


class TestVerifyTreeGreedy(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def _run_and_check(
        self,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        num_spec_step,
    ):
        bs, draft_token_num = candidates.shape
        predicts = torch.full((bs * draft_token_num,), -1, dtype=torch.int32)
        accept_indices = torch.full((bs, num_spec_step), -1, dtype=torch.int32)
        num_correct_drafts = torch.empty((bs,), dtype=torch.int32)
        torch.ops.sgl_kernel.verify_tree_greedy_cpu(
            predicts,
            accept_indices,
            num_correct_drafts,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
        )
        predicts_ref, accept_indices_ref, num_correct_drafts_ref = (
            _ref_verify_tree_greedy(
                candidates,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                target_predict,
                num_spec_step,
            )
        )
        torch.testing.assert_close(predicts, predicts_ref, atol=0, rtol=0)
        torch.testing.assert_close(accept_indices, accept_indices_ref, atol=0, rtol=0)
        torch.testing.assert_close(
            num_correct_drafts, num_correct_drafts_ref, atol=0, rtol=0
        )
        return predicts, accept_indices, num_correct_drafts

    def test_verify_tree_greedy_chain_topk1(self):
        # MTP config: topk=1, steps=3, draft_token_num=4 -> linear chain.
        # Per row, force exactly `m` leading drafts to match the target so every
        # accept count in [0, num_steps] is exercised.
        num_steps, draft_token_num = 3, 4
        vocab = 100
        for bs in [1, 3]:
            with self.subTest(bs=bs):
                offs = (
                    torch.arange(bs, dtype=torch.int64).unsqueeze(1) * draft_token_num
                )
                retrieve_index = (
                    torch.arange(draft_token_num, dtype=torch.int64).repeat(bs, 1)
                    + offs
                )
                retrieve_next_token = torch.tensor(
                    [[1, 2, 3, -1]], dtype=torch.int64
                ).repeat(bs, 1)
                retrieve_next_sibling = torch.full(
                    (bs, draft_token_num), -1, dtype=torch.int64
                )
                target_predict = torch.randint(
                    0, vocab, (bs, draft_token_num), dtype=torch.int64
                )
                candidates = torch.randint(
                    0, vocab, (bs, draft_token_num), dtype=torch.int64
                )
                forced_num_correct = [
                    bx % (num_steps + 1) if bs > 1 else num_steps for bx in range(bs)
                ]
                for bx, m in enumerate(forced_num_correct):
                    for i in range(m):
                        candidates[bx, i + 1] = target_predict[bx, i]
                    for i in range(m, draft_token_num - 1):
                        candidates[bx, i + 1] = (target_predict[bx, i] + 1) % vocab
                _, _, num_correct_drafts = self._run_and_check(
                    candidates,
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                    target_predict,
                    num_steps + 1,
                )
                self.assertEqual(num_correct_drafts.tolist(), forced_num_correct)

    def test_verify_tree_greedy_chain_hand_case(self):
        # Deterministic chain: drafts 3, 4 match the target, draft 5 does not.
        retrieve_index = torch.arange(4, dtype=torch.int64).unsqueeze(0)
        retrieve_next_token = torch.tensor([[1, 2, 3, -1]], dtype=torch.int64)
        retrieve_next_sibling = torch.full((1, 4), -1, dtype=torch.int64)
        candidates = torch.tensor([[10, 3, 4, 5]], dtype=torch.int64)
        target_predict = torch.tensor([[3, 4, 9, 9]], dtype=torch.int64)
        predicts, accept_indices, num_correct_drafts = self._run_and_check(
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
            4,
        )
        self.assertEqual(predicts.tolist(), [3, 4, 9, -1])
        self.assertEqual(accept_indices.tolist(), [[0, 1, 2, -1]])
        self.assertEqual(num_correct_drafts.tolist(), [2])

    def test_verify_tree_greedy_upstream_golden(self):
        # Golden fixture ported from the CUDA kernel UT
        # sgl-kernel/tests/speculative/test_eagle_utils.py::test_verify_tree_greedy
        # (device swapped to CPU); expected outputs are the CUDA kernel's.
        candidates = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [7, 8, 9, 10, 11, 12],
            ],
            dtype=torch.int64,
        )
        retrieve_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10, 11],
            ],
            dtype=torch.int64,
        )
        retrieve_next_token = torch.tensor(
            [
                [1, 2, -1, 4, 5, -1],
                [4, 2, 3, -1, 5, -1],
            ],
            dtype=torch.int64,
        )
        retrieve_next_sibling = torch.tensor(
            [
                [-1, 3, -1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1],
            ],
            dtype=torch.int64,
        )

        target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32)
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

        predicts, accept_indices, num_correct_drafts = self._run_and_check(
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
            4,  # num_spec_step
        )
        self.assertEqual(
            predicts.tolist(), [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
        )
        self.assertEqual(accept_indices.tolist(), [[0, 3, 4, 5], [6, 10, 11, -1]])
        self.assertEqual(num_correct_drafts.tolist(), [3, 2])

    def test_verify_tree_greedy_tree_topk4(self):
        # EAGLE config: topk=4, steps=3, draft_token_num=16
        topk, num_steps, draft_token_num = 4, 3, 16
        vocab = 3  # small vocab so accepts, rejects and sibling hops all happen
        for bs in [1, 3]:
            with self.subTest(bs=bs):
                parent_list, selected_index = _gen_draft_tree(
                    bs, topk, num_steps, draft_token_num
                )
                seq_lens = torch.randint(4, 32, (bs,), dtype=torch.int64)
                _, _, retrieve_index, retrieve_next_token, retrieve_next_sibling = (
                    _run_build_tree_kernel(
                        parent_list,
                        selected_index,
                        seq_lens,
                        topk,
                        num_steps,
                        draft_token_num,
                        TreeMaskMode.QLEN_ONLY,
                    )
                )
                # the random trees must branch, otherwise sibling traversal
                # would go untested
                self.assertGreater(int((retrieve_next_sibling != -1).sum()), 0)
                candidates = torch.randint(
                    0, vocab, (bs, draft_token_num), dtype=torch.int64
                )
                target_predict = torch.randint(
                    0, vocab, (bs, draft_token_num), dtype=torch.int64
                )
                _, _, num_correct_drafts = self._run_and_check(
                    candidates,
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                    target_predict,
                    num_steps + 1,
                )
                # deterministic with the fixed seed: the accept path is taken
                self.assertGreater(int(num_correct_drafts.sum()), 0)


class TestBuildTreeKernelEfficient(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def _check_against_reference(
        self, parent_list, selected_index, seq_lens, topk, num_steps, draft_token_num
    ):
        bs = seq_lens.numel()
        (
            retrieve_index_ref,
            retrieve_next_token_ref,
            retrieve_next_sibling_ref,
            positions_ref,
            tree_mask_ref,
        ) = _ref_build_tree(
            parent_list, selected_index, seq_lens, topk, draft_token_num
        )

        # QLEN_ONLY
        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _run_build_tree_kernel(
            parent_list,
            selected_index,
            seq_lens,
            topk,
            num_steps,
            draft_token_num,
            TreeMaskMode.QLEN_ONLY,
        )
        torch.testing.assert_close(retrieve_index, retrieve_index_ref, atol=0, rtol=0)
        torch.testing.assert_close(
            retrieve_next_token, retrieve_next_token_ref, atol=0, rtol=0
        )
        torch.testing.assert_close(
            retrieve_next_sibling, retrieve_next_sibling_ref, atol=0, rtol=0
        )
        torch.testing.assert_close(positions, positions_ref, atol=0, rtol=0)
        torch.testing.assert_close(
            tree_mask.view(bs, draft_token_num, draft_token_num),
            tree_mask_ref,
            atol=0,
            rtol=0,
        )

        # FULL_MASK: same retrieve_*/positions; the mask additionally carries the
        # (all-true) committed-prefix columns per row.
        (
            full_mask,
            positions_f,
            retrieve_index_f,
            retrieve_next_token_f,
            retrieve_next_sibling_f,
        ) = _run_build_tree_kernel(
            parent_list,
            selected_index,
            seq_lens,
            topk,
            num_steps,
            draft_token_num,
            TreeMaskMode.FULL_MASK,
        )
        torch.testing.assert_close(retrieve_index_f, retrieve_index_ref, atol=0, rtol=0)
        torch.testing.assert_close(
            retrieve_next_token_f, retrieve_next_token_ref, atol=0, rtol=0
        )
        torch.testing.assert_close(
            retrieve_next_sibling_f, retrieve_next_sibling_ref, atol=0, rtol=0
        )
        torch.testing.assert_close(positions_f, positions_ref, atol=0, rtol=0)
        offset = 0
        for bid in range(bs):
            seq_len = int(seq_lens[bid])
            chunk = full_mask[
                offset : offset + draft_token_num * (seq_len + draft_token_num)
            ].view(draft_token_num, seq_len + draft_token_num)
            self.assertTrue(chunk[:, :seq_len].all().item())
            torch.testing.assert_close(
                chunk[:, seq_len:], tree_mask_ref[bid], atol=0, rtol=0
            )
            offset += draft_token_num * (seq_len + draft_token_num)

        return tree_mask_ref, positions_ref

    def test_build_tree_chain_topk1(self):
        # MTP config: topk=1, steps=3, draft_token_num=4
        num_steps, draft_token_num = 3, 4
        bs = 2
        seq_lens = torch.tensor([7, 12], dtype=torch.int64)
        parent_list, selected_index = _topk1_chain_inputs(bs, num_steps)
        tree_mask_ref, positions_ref = self._check_against_reference(
            parent_list, selected_index, seq_lens, 1, num_steps, draft_token_num
        )
        # A chain must yield a causal (lower-triangular) mask and consecutive positions.
        tril = torch.tril(
            torch.ones(draft_token_num, draft_token_num, dtype=torch.bool)
        )
        for bid in range(bs):
            torch.testing.assert_close(tree_mask_ref[bid], tril, atol=0, rtol=0)
            self.assertEqual(
                positions_ref[
                    bid * draft_token_num : (bid + 1) * draft_token_num
                ].tolist(),
                [int(seq_lens[bid]) + i for i in range(draft_token_num)],
            )

    def test_build_tree_topk2_hand_case(self):
        # topk=2, steps=2, draft_token_num=4. Nodes: 1 and 2 are children of the
        # root; node 3 is a child of node 1.
        topk, num_steps, draft_token_num = 2, 2, 4
        parent_list = torch.tensor([[-1, 0, 1]], dtype=torch.int64)
        selected_index = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        seq_lens = torch.tensor([5], dtype=torch.int64)
        (
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _run_build_tree_kernel(
            parent_list,
            selected_index,
            seq_lens,
            topk,
            num_steps,
            draft_token_num,
            TreeMaskMode.QLEN_ONLY,
        )
        self.assertEqual(retrieve_index.tolist(), [[0, 1, 2, 3]])
        self.assertEqual(retrieve_next_token.tolist(), [[1, 3, -1, -1]])
        self.assertEqual(retrieve_next_sibling.tolist(), [[-1, 2, -1, -1]])
        self.assertEqual(positions.tolist(), [5, 6, 6, 7])
        self.assertEqual(
            tree_mask.view(draft_token_num, draft_token_num).int().tolist(),
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
            ],
        )
        # Cross-check the python reference on the same hand-built case.
        self._check_against_reference(
            parent_list, selected_index, seq_lens, topk, num_steps, draft_token_num
        )

    def test_build_tree_upstream_golden(self):
        # Captured EAGLE trace ported from the CUDA UT
        # test/registered/spec/utils/test_build_eagle_tree.py::TestBuildEagleTree::
        # test_build_tree_kernel_efficient (device swapped to CPU). The inputs go
        # through the same organize_draft_results; the expected positions and
        # retrieve_* vectors are the CUDA kernel's.
        score_list = [
            torch.tensor(
                [
                    [[7.1127e-01, 2.8292e-01, 2.2995e-03, 1.7357e-03]],
                    [[9.7476e-01, 2.2219e-02, 6.5031e-04, 1.3212e-04]],
                ],
                dtype=torch.float32,
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
            ),
        ]
        token_list = [
            torch.tensor(
                [[29896, 29906, 29900, 29945], [13, 2, 29871, 28956]],
                dtype=torch.int64,
            ),
            # fmt: off
            torch.tensor(
                [
                    [29889, 29974, 29945, 29900, 29974, 29922, 29930, 29958,
                     29889, 29974, 29930, 29945, 29974, 29922, 29930, 29958],
                    [22550, 4136, 16492, 8439, 29871, 2, 3001, 13,
                     2, 13, 29906, 29946, 2, 13, 29871, 259],
                ],
            ),
            torch.tensor(
                [
                    [29946, 29945, 29953, 29906, 29896, 29945, 29900, 29906,
                     29896, 29945, 29906, 29953, 29896, 29945, 29906, 29946],
                    [29871, 2, 29901, 29889, 29871, 2, 395, 259,
                     29901, 29871, 2, 29889, 3001, 1234, 7146, 2186],
                ],
            ),
            torch.tensor(
                [
                    [29946, 29974, 29945, 29930, 29889, 29922, 29974, 29930,
                     29974, 29946, 29930, 29922, 29889, 29974, 29945, 29922],
                    [29941, 29906, 2, 29946, 29871, 450, 319, 14990,
                     29946, 29941, 2, 29906, 29871, 2, 3001, 13],
                ],
            ),
            # fmt: on
        ]
        parents_list = [
            torch.tensor([[-1, 0, 1, 2, 3], [-1, 0, 1, 2, 3]], dtype=torch.int64),
            torch.tensor([[4, 8, 9, 10], [4, 5, 6, 7]], dtype=torch.int64),
            torch.tensor([[20, 24, 21, 28], [24, 28, 20, 21]], dtype=torch.int64),
            torch.tensor([[36, 40, 41, 44], [36, 40, 44, 45]], dtype=torch.int64),
        ]
        seq_lens = torch.tensor([5, 10], dtype=torch.int64)
        topk, depth, draft_token_num = 4, 4, 8

        parent_list, selected_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, draft_token_num
        )
        # The CUDA golden interleaves the bonus tokens [29974, 13] at positions
        # 0 and 8; on CPU that concat happens outside the kernel, so compare the
        # drafts only.
        self.assertEqual(
            draft_tokens.tolist(),
            [
                [29896, 29906, 29889, 29974, 29946, 29896, 29946],
                [13, 22550, 4136, 16492, 8439, 29871, 29941],
            ],
        )

        (
            _,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
        ) = _run_build_tree_kernel(
            parent_list,
            selected_index,
            seq_lens,
            topk,
            depth,
            draft_token_num,
            TreeMaskMode.QLEN_ONLY,
        )
        self.assertEqual(
            positions.tolist(),
            [5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 12, 12, 13, 14],
        )
        self.assertEqual(
            retrieve_index.tolist(),
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
            ],
        )
        self.assertEqual(
            retrieve_next_token.tolist(),
            [
                [1, 3, 4, 5, 6, 7, -1, -1],
                [1, 2, -1, 6, -1, -1, 7, -1],
            ],
        )
        self.assertEqual(
            retrieve_next_sibling.tolist(),
            [
                [-1, 2, -1, -1, -1, -1, -1, -1],
                [-1, -1, 3, 4, 5, -1, -1, -1],
            ],
        )
        # Cross-check the python reference (incl. tree masks, which the CUDA UT
        # does not assert) on the same trace.
        self._check_against_reference(
            parent_list, selected_index, seq_lens, topk, depth, draft_token_num
        )

    def test_build_tree_topk4_random(self):
        # EAGLE config: topk=4, steps=3, draft_token_num=16
        topk, num_steps, draft_token_num = 4, 3, 16
        bs = 3
        seq_lens = torch.tensor([9, 3, 21], dtype=torch.int64)
        parent_list, selected_index = _gen_draft_tree(
            bs, topk, num_steps, draft_token_num
        )
        tree_mask_ref, positions_ref = self._check_against_reference(
            parent_list, selected_index, seq_lens, topk, num_steps, draft_token_num
        )
        # Invariants: causality (mask support only on self + earlier nodes) and
        # node depth bounded by the number of speculative steps.
        for bid in range(bs):
            mask = tree_mask_ref[bid]
            self.assertTrue(mask.diagonal().all().item())
            self.assertTrue(torch.triu(mask.int(), diagonal=1).sum().item() == 0)
            depths = positions_ref[
                bid * draft_token_num : (bid + 1) * draft_token_num
            ] - int(seq_lens[bid])
            self.assertEqual(int(depths[0]), 0)
            self.assertTrue(bool((depths[1:] >= 1).all()))
            self.assertTrue(bool((depths <= num_steps).all()))
            # row depth == number of visible draft tokens (self + ancestors) - 1
            torch.testing.assert_close(
                depths, mask.sum(dim=1).to(torch.int64) - 1, atol=0, rtol=0
            )


class TestFillKernels(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_fill_bonus_tokens(self):
        bs, accept_stride = 4, 4
        accept_tokens = torch.randint(0, 1000, (bs, accept_stride), dtype=torch.int32)
        accept_lens = torch.tensor([1, 4, 2, 3], dtype=torch.int32)
        bonus_tokens = torch.empty((bs,), dtype=torch.int32)
        torch.ops.sgl_kernel.fill_bonus_tokens_cpu(
            accept_tokens, accept_lens, bonus_tokens, accept_stride
        )
        bonus_tokens_ref = accept_tokens[
            torch.arange(bs), accept_lens.to(torch.int64) - 1
        ]
        torch.testing.assert_close(bonus_tokens, bonus_tokens_ref, atol=0, rtol=0)

    def test_fill_accept_out_cache_loc(self):
        bs, num_spec_step, draft_token_num = 3, 4, 16
        accept_indices = torch.tensor(
            [
                [0, 1, -1, -1],
                [16, 18, 21, -1],
                [32, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        out_cache_loc = torch.randperm(4096, dtype=torch.int64)[: bs * draft_token_num]
        size = bs * num_spec_step
        accept_out_cache_loc = torch.zeros(size, dtype=torch.int64)
        torch.ops.sgl_kernel.fill_accept_out_cache_loc_cpu(
            accept_indices, out_cache_loc, accept_out_cache_loc, size
        )
        valid = accept_indices.flatten()[accept_indices.flatten() > -1].to(torch.int64)
        ref = torch.zeros(size, dtype=torch.int64)
        ref[: valid.numel()] = out_cache_loc[valid]
        torch.testing.assert_close(accept_out_cache_loc, ref, atol=0, rtol=0)


class TestAssignCacheLocKernels(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_assign_draft_cache_locs_contiguous(self):
        bs, topk, num_steps, pool_len, num_reqs = 3, 2, 3, 32, 5
        req_pool_indices = torch.tensor([4, 0, 2], dtype=torch.int64)
        req_to_token = torch.randint(0, 10000, (num_reqs, pool_len), dtype=torch.int32)
        seq_lens = torch.tensor([5, 0, 17], dtype=torch.int64)
        out_cache_loc = torch.empty((bs * topk * num_steps,), dtype=torch.int64)
        torch.ops.sgl_kernel.assign_draft_cache_locs_contiguous_cpu(
            req_pool_indices,
            req_to_token,
            seq_lens,
            out_cache_loc,
            pool_len,
            topk,
            num_steps,
        )
        copy_len = topk * num_steps
        ref = torch.empty_like(out_cache_loc)
        for i in range(bs):
            start = int(seq_lens[i])
            ref[i * copy_len : (i + 1) * copy_len] = req_to_token[
                int(req_pool_indices[i]), start : start + copy_len
            ].to(torch.int64)
        torch.testing.assert_close(out_cache_loc, ref, atol=0, rtol=0)

        # dtype contract: int64 req_pool_indices/seq_lens/out_cache_loc, int32
        # req_to_token (TORCH_CHECK)
        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.assign_draft_cache_locs_contiguous_cpu(
                req_pool_indices,
                req_to_token,
                seq_lens.to(torch.int32),
                out_cache_loc,
                pool_len,
                topk,
                num_steps,
            )
        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.assign_draft_cache_locs_contiguous_cpu(
                req_pool_indices,
                req_to_token.to(torch.int64),
                seq_lens,
                out_cache_loc,
                pool_len,
                topk,
                num_steps,
            )
        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.assign_draft_cache_locs_contiguous_cpu(
                req_pool_indices,
                req_to_token,
                seq_lens,
                out_cache_loc[:-1],  # wrong numel
                pool_len,
                topk,
                num_steps,
            )

    def test_assign_extend_cache_locs(self):
        bs, pool_len, num_reqs = 3, 32, 5
        req_pool_indices = torch.tensor([1, 3, 0], dtype=torch.int64)
        req_to_token = torch.randint(0, 10000, (num_reqs, pool_len), dtype=torch.int32)
        start_offset = torch.tensor([4, 0, 11], dtype=torch.int64)
        end_offset = torch.tensor([9, 7, 12], dtype=torch.int64)
        total = int((end_offset - start_offset).sum())
        out_cache_loc = torch.empty((total,), dtype=torch.int64)
        torch.ops.sgl_kernel.assign_extend_cache_locs_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            pool_len,
        )
        ref = torch.cat(
            [
                req_to_token[
                    int(req_pool_indices[i]),
                    int(start_offset[i]) : int(end_offset[i]),
                ].to(torch.int64)
                for i in range(bs)
            ]
        )
        torch.testing.assert_close(out_cache_loc, ref, atol=0, rtol=0)

    def test_assign_req_to_token_pool(self):
        bs, pool_len, num_reqs = 3, 32, 5
        req_pool_indices = torch.tensor([2, 4, 1], dtype=torch.int32)
        req_to_token = torch.zeros((num_reqs, pool_len), dtype=torch.int32)
        req_to_token_ref = req_to_token.clone()
        start_offset = torch.tensor([0, 3, 8], dtype=torch.int32)
        end_offset = torch.tensor([6, 3, 15], dtype=torch.int32)
        total = int((end_offset - start_offset).sum())
        out_cache_loc = torch.randperm(4096, dtype=torch.int64)[:total]
        torch.ops.sgl_kernel.assign_req_to_token_pool_cpu(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            pool_len,
            4,  # bs_upper = next_power_of_2(bs)
        )
        offset = 0
        for i in range(bs):
            length = int(end_offset[i]) - int(start_offset[i])
            req_to_token_ref[
                int(req_pool_indices[i]),
                int(start_offset[i]) : int(end_offset[i]),
            ] = out_cache_loc[offset : offset + length].to(torch.int32)
            offset += length
        torch.testing.assert_close(req_to_token, req_to_token_ref, atol=0, rtol=0)


class TestExtendAttentionTreeMask(CustomTestCase):
    H_Q, H_KV, D, DV = 4, 2, 64, 64

    def setUp(self):
        torch.manual_seed(1234)

    def _build_target_verify_inputs(self, prefix_lens, draft_token_num):
        # Mimic intel_amx_backend.forward_extend in TARGET_VERIFY mode: kv cache
        # for the draft tokens is written first (set_kv_buffer), seq_lens are the
        # committed lens + draft_token_num and extend lens are inferred uniform.
        dtype = torch.bfloat16
        bs = len(prefix_lens)
        prefix_lens = torch.tensor(prefix_lens, dtype=torch.int64)
        seq_lens = prefix_lens + draft_token_num  # seq_lens += draft_token_num
        total_tokens = int(seq_lens.sum())

        req_to_token = torch.zeros((bs, int(seq_lens.max())), dtype=torch.int32)
        start = 0
        for i in range(bs):
            req_to_token[i, : int(seq_lens[i])] = torch.arange(
                start, start + int(seq_lens[i]), dtype=torch.int32
            )
            start += int(seq_lens[i])

        k_buffer = torch.randn((total_tokens, self.H_KV, self.D), dtype=dtype)
        v_buffer = torch.randn((total_tokens, self.H_KV, self.DV), dtype=dtype)

        # extend (draft) tokens were already written into the buffers
        extend_token_num = bs * draft_token_num
        k_extend = torch.empty((extend_token_num, self.H_KV, self.D), dtype=dtype)
        v_extend = torch.empty((extend_token_num, self.H_KV, self.DV), dtype=dtype)
        q_extend = torch.randn((extend_token_num, self.H_Q, self.D), dtype=dtype)
        for i in range(bs):
            buf_start = int(seq_lens[:i].sum()) + int(prefix_lens[i])
            buf_end = buf_start + draft_token_num
            k_extend[i * draft_token_num : (i + 1) * draft_token_num] = k_buffer[
                buf_start:buf_end
            ]
            v_extend[i * draft_token_num : (i + 1) * draft_token_num] = v_buffer[
                buf_start:buf_end
            ]

        req_pool_indices = torch.arange(bs, dtype=torch.int64)
        extend_seq_lens = torch.full((bs,), draft_token_num, dtype=torch.int32)
        extend_start_loc = torch.zeros((bs,), dtype=torch.int32)
        if bs > 1:
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
        return (
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
        )

    def _run_kernel(self, inputs, draft_token_num, tree_mask, with_trailing_arg=True):
        (
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
        ) = inputs
        o_extend = torch.empty(
            (q_extend.shape[0], self.H_Q, self.DV), dtype=q_extend.dtype
        )
        sm_scale = 1.0 / (self.D**0.5)
        args = [
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            draft_token_num,  # max_len_extend
            sm_scale,
            0.0,  # logit_cap
            False,  # is_cross_attn
            0,  # sliding_window_size (layer.sliding_window_size + 1)
            None,  # encoder_lens
            None,  # sinks
        ]
        if with_trailing_arg:
            args.append(tree_mask)
        torch.ops.sgl_kernel.extend_attention_cpu(*args)
        return o_extend

    def _ref_sdpa(self, inputs, draft_token_num, qlen_masks):
        (
            q_extend,
            _,
            _,
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            _,
            _,
        ) = inputs
        bs = seq_lens.shape[0]
        sm_scale = 1.0 / (self.D**0.5)
        out = torch.empty(
            (bs * draft_token_num, self.H_Q, self.DV), dtype=torch.float32
        )
        for i in range(bs):
            kv_len = int(seq_lens[i])
            tokens = req_to_token[int(req_pool_indices[i]), :kv_len].to(torch.int64)
            keys = k_buffer[tokens].float().movedim(0, 1)  # [H_KV, kv_len, D]
            values = v_buffer[tokens].float().movedim(0, 1)
            queries = (
                q_extend[i * draft_token_num : (i + 1) * draft_token_num]
                .float()
                .movedim(0, 1)
            )  # [H_Q, qlen, D]
            attn_mask = torch.cat(
                [
                    torch.ones(
                        draft_token_num, kv_len - draft_token_num, dtype=torch.bool
                    ),
                    qlen_masks[i],
                ],
                dim=1,
            )
            o = F.scaled_dot_product_attention(
                queries.unsqueeze(0),
                keys.unsqueeze(0),
                values.unsqueeze(0),
                attn_mask=attn_mask,
                scale=sm_scale,
                enable_gqa=True,
            ).squeeze(0)
            out[i * draft_token_num : (i + 1) * draft_token_num] = o.movedim(0, 1)
        return out

    def test_no_tree_mask_matches_legacy_call(self):
        # (a) with and without the trailing optional argument must be bitwise equal
        draft_token_num = 16
        inputs = self._build_target_verify_inputs([13, 7], draft_token_num)
        o_legacy = self._run_kernel(
            inputs, draft_token_num, None, with_trailing_arg=False
        )
        o_none = self._run_kernel(inputs, draft_token_num, None, with_trailing_arg=True)
        self.assertTrue(torch.equal(o_legacy, o_none))

    def test_chain_topk1_matches_causal_sdpa(self):
        # (b) topk=1 chain (no tree mask passed): plain causal masking
        draft_token_num = 4
        inputs = self._build_target_verify_inputs([9, 5], draft_token_num)
        o_extend = self._run_kernel(inputs, draft_token_num, None)
        tril = torch.tril(
            torch.ones(draft_token_num, draft_token_num, dtype=torch.bool)
        )
        o_ref = self._ref_sdpa(inputs, draft_token_num, [tril] * len(inputs[7]))
        atol = rtol = precision[torch.bfloat16]
        torch.testing.assert_close(o_extend.float(), o_ref, atol=atol, rtol=rtol)

    def test_tree_topk4_matches_tree_sdpa(self):
        # (c) topk=4 tree with a QLEN_ONLY mask from build_tree_kernel_efficient_cpu
        topk, num_steps, draft_token_num = 4, 3, 16
        bs = 2
        prefix_lens = [13, 7]
        inputs = self._build_target_verify_inputs(prefix_lens, draft_token_num)
        parent_list, selected_index = _gen_draft_tree(
            bs, topk, num_steps, draft_token_num
        )
        tree_mask, _, _, _, _ = _run_build_tree_kernel(
            parent_list,
            selected_index,
            torch.tensor(prefix_lens, dtype=torch.int64),
            topk,
            num_steps,
            draft_token_num,
            TreeMaskMode.QLEN_ONLY,
        )
        o_extend = self._run_kernel(inputs, draft_token_num, tree_mask)
        qlen_masks = tree_mask.view(bs, draft_token_num, draft_token_num)
        o_ref = self._ref_sdpa(
            inputs, draft_token_num, [qlen_masks[i] for i in range(bs)]
        )
        # 2x the shared bf16 tolerance: the sparse tree mask leaves few terms
        # per row, so single-element bf16 rounding differences dominate.
        torch.testing.assert_close(o_extend.float(), o_ref, atol=2e-2, rtol=2e-2)


class TestFusedSigmoidGatingDeltaRuleVerify(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_target_verify_multi_token(self):
        # varlen multi-token verify: B=2 sequences of T=4 draft tokens each
        B, T = 2, 4
        HK, HV, K, V = 2, 4, 32, 32
        total_tokens = B * T
        num_slots, cache_size = 6, 4
        dtype = torch.bfloat16

        q = torch.rand(1, total_tokens, HK, K, dtype=dtype) * 0.1
        k = torch.rand(1, total_tokens, HK, K, dtype=dtype) * 0.1
        v = torch.rand(1, total_tokens, HV, V, dtype=dtype) * 0.1
        # a/b are 2-D [tokens, HV] in verify mode (per-token gating)
        a = torch.rand(total_tokens, HV, dtype=dtype) * 0.1
        b = torch.rand(total_tokens, HV, dtype=dtype) * 0.1
        dt_bias = torch.rand(HV, dtype=dtype) * 0.1
        cu_seqlens = torch.tensor([0, T, 2 * T], dtype=torch.int32)
        ssm_states_init = torch.rand(num_slots, HV, K, V, dtype=torch.float32) * 0.05
        cache_indices = torch.tensor([5, 1], dtype=torch.int32)
        intermediate_state_indices = torch.tensor([2, 0], dtype=torch.int32)

        for A_log_dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(A_log_dtype=A_log_dtype):
                A_log = (torch.rand(HV, dtype=torch.float32) * 0.1).to(A_log_dtype)
                ssm_states = ssm_states_init.clone()
                intermediate_states_buffer = torch.zeros(
                    cache_size, T, HV, K, V, dtype=torch.float32
                )
                core_attn_out = (
                    torch.ops.sgl_kernel.fused_sigmoid_gating_delta_rule_update_cpu(
                        A_log=A_log,
                        dt_bias=dt_bias,
                        q=q,
                        k=k,
                        v=v,
                        a=a,
                        b=b,
                        initial_state_source=ssm_states,
                        initial_state_indices=cache_indices,
                        cu_seqlens=cu_seqlens,
                        use_qk_l2norm_in_kernel=True,
                        softplus_beta=1.0,
                        softplus_threshold=20.0,
                        is_kda=False,
                        disable_state_update=True,
                        intermediate_states_buffer=intermediate_states_buffer,
                        intermediate_state_indices=intermediate_state_indices,
                        cache_steps=T,
                    )
                )

                # disable_state_update must leave the ssm state pool untouched
                self.assertTrue(torch.equal(ssm_states, ssm_states_init))

                out_ref, buffer_ref = self._ref_delta_rule_verify(
                    q,
                    k,
                    v,
                    a,
                    b,
                    A_log,
                    dt_bias,
                    ssm_states_init,
                    cache_indices,
                    cu_seqlens,
                    intermediate_state_indices,
                    cache_size,
                )
                atol = rtol = precision[torch.bfloat16]
                torch.testing.assert_close(
                    core_attn_out.float(), out_ref, atol=atol, rtol=rtol
                )
                torch.testing.assert_close(
                    intermediate_states_buffer, buffer_ref, atol=1e-3, rtol=1e-3
                )

    def _ref_delta_rule_verify(
        self,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        cu_seqlens,
        intermediate_state_indices,
        cache_size,
    ):
        # Sequential pure-python delta rule with sigmoid gating, fp32 math.
        _, total_tokens, HK, K = q.shape
        HV, V = v.shape[2], v.shape[3]
        T = int(cu_seqlens[1])
        group_size = HV // HK
        scale = 1.0 / (K**0.5)
        eps = 1e-5  # kernel l2norm epsilon

        qf = q[0].float()
        kf = k[0].float()
        qn = qf * (qf.pow(2).sum(-1, keepdim=True) + eps).rsqrt()
        kn = kf * (kf.pow(2).sum(-1, keepdim=True) + eps).rsqrt()
        g = -A_log.float().exp() * softplus(a.float() + dt_bias.float())  # [tokens, HV]
        beta = b.float().sigmoid()

        out = torch.zeros(1, total_tokens, HV, V, dtype=torch.float32)
        buffer_ref = torch.zeros(cache_size, T, HV, K, V, dtype=torch.float32)
        num_sequences = cu_seqlens.numel() - 1
        for n in range(num_sequences):
            bos = int(cu_seqlens[n])
            seq_len = int(cu_seqlens[n + 1]) - bos
            for hv in range(HV):
                h = ssm_states[int(cache_indices[n]), hv].clone()  # [K, V]
                hk = hv // group_size
                for t in range(seq_len):
                    pos = bos + t
                    h = h * g[pos, hv].exp()
                    kv_mem = (h * kn[pos, hk].unsqueeze(-1)).sum(dim=0)  # [V]
                    delta = (v[0, pos, hv].float() - kv_mem) * beta[pos, hv]
                    h = h + kn[pos, hk].unsqueeze(-1) * delta.unsqueeze(0)
                    out[0, pos, hv] = (h * qn[pos, hk].unsqueeze(-1)).sum(0) * scale
                    buffer_ref[int(intermediate_state_indices[n]), t, hv] = h
        return out, buffer_ref


class TestCausalConv1dUpdateMultiToken(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_multi_token_with_intermediate_conv_window(self):
        batch, dim, width, seqlen = 3, 32, 4, 4
        num_entries, cache_size = 6, 5
        dtype = torch.bfloat16
        state_len = width - 1

        x = torch.randn(batch, dim, seqlen, dtype=dtype)
        conv_states = torch.randn(num_entries, dim, state_len, dtype=dtype)
        weight = torch.randn(dim, width, dtype=dtype)
        bias = torch.randn(dim, dtype=dtype)
        conv_state_indices = torch.tensor([4, 0, 2], dtype=torch.int32)
        intermediate_state_indices = torch.tensor([1, 3, 0], dtype=torch.int32)
        intermediate_conv_window = torch.zeros(
            cache_size, seqlen, dim, state_len, dtype=dtype
        )
        packed_weight = torch.ops.sgl_kernel.causal_conv1d_weight_pack(weight)

        # reference: sequential single-token updates with state caching
        conv_state_ref = conv_states[conv_state_indices.to(torch.int64)].clone()
        window_ref = torch.zeros_like(intermediate_conv_window)
        out_ref = torch.empty_like(x)
        for t in range(seqlen):
            x_t = x[:, :, t]
            x_cat = torch.cat([conv_state_ref, x_t.unsqueeze(-1)], dim=-1)
            conv_state_ref = x_cat[:, :, -state_len:].clone()
            out_t = F.conv1d(
                x_cat.float(),
                weight.float().unsqueeze(1),
                bias.float(),
                padding=0,
                groups=dim,
            )[:, :, -1]
            out_ref[:, :, t] = F.silu(out_t).to(dtype)
            for i in range(batch):
                window_ref[int(intermediate_state_indices[i]), t] = conv_state_ref[i]

        out = torch.ops.sgl_kernel.causal_conv1d_update_cpu(
            x,
            conv_states,
            packed_weight,
            bias,
            True,  # silu_activation
            None,  # cache_seqlens
            conv_state_indices,
            PAD_SLOT_ID,
            True,  # is_vnni
            intermediate_conv_window,
            intermediate_state_indices,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            conv_states[conv_state_indices.to(torch.int64)],
            conv_state_ref,
            atol=atol,
            rtol=rtol,
        )
        torch.testing.assert_close(
            intermediate_conv_window, window_ref, atol=atol, rtol=rtol
        )


class TestMultiLayerEagleStateKernels(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_rotate_input_ids(self):
        extend_seq_lens = torch.tensor([4, 1, 3], dtype=torch.int64)
        extend_start_loc = torch.tensor([0, 4, 5], dtype=torch.int64)
        topk_index = torch.tensor([100, 200, 300], dtype=torch.int64)
        total = int(extend_seq_lens.sum())
        input_ids_init = torch.arange(10, 10 + total, dtype=torch.int64)

        # without select_index: shift each request left by 1, append the new token
        input_ids = input_ids_init.clone()
        torch.ops.sgl_kernel.rotate_input_ids_cpu(
            input_ids, extend_start_loc, extend_seq_lens, topk_index, None
        )
        ref = input_ids_init.clone()
        for i in range(extend_seq_lens.numel()):
            start = int(extend_start_loc[i])
            seq_len = int(extend_seq_lens[i])
            ref[start : start + seq_len - 1] = input_ids_init[
                start + 1 : start + seq_len
            ]
            ref[start + seq_len - 1] = topk_index[i]
        torch.testing.assert_close(input_ids, ref, atol=0, rtol=0)

        # with select_index: the new token lands at the given global slots
        select_index = torch.tensor([2, 4, 7], dtype=torch.int64)
        input_ids = input_ids_init.clone()
        torch.ops.sgl_kernel.rotate_input_ids_cpu(
            input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index
        )
        ref = input_ids_init.clone()
        for i in range(extend_seq_lens.numel()):
            start = int(extend_start_loc[i])
            seq_len = int(extend_seq_lens[i])
            ref[start : start + seq_len - 1] = input_ids_init[
                start + 1 : start + seq_len
            ]
        for i in range(extend_seq_lens.numel()):
            ref[int(select_index[i])] = topk_index[i]
        torch.testing.assert_close(input_ids, ref, atol=0, rtol=0)

        # int64 contract: int32 input_ids must be rejected
        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.rotate_input_ids_cpu(
                input_ids_init.to(torch.int32),
                extend_start_loc,
                extend_seq_lens,
                topk_index,
                None,
            )

    def test_assign_new_state(self):
        num_seqs, hidden_dim = 2, 8
        num_reqs, pool_len, pool_size = 3, 16, 4
        step = 1
        old_extend_seq_lens = torch.tensor([2, 3], dtype=torch.int64)
        old_extend_start_loc = torch.tensor([0, 2], dtype=torch.int64)
        old_total = 5
        new_total = old_total + num_seqs

        next_token_ids = torch.tensor([111, 222], dtype=torch.int64)
        old_input_ids = torch.arange(10, 10 + old_total, dtype=torch.int64)
        old_positions = torch.tensor([3, 4, 7, 8, 9], dtype=torch.int64)
        old_out_cache_loc = torch.arange(50, 50 + old_total, dtype=torch.int64)
        old_hidden_states = torch.randn(old_total, hidden_dim, dtype=torch.float32)

        input_ids = torch.full((new_total,), -1, dtype=torch.int64)
        positions = torch.full((new_total,), -1, dtype=torch.int64)
        out_cache_loc = torch.full((new_total,), -1, dtype=torch.int64)
        extend_seq_lens = torch.zeros(num_seqs, dtype=torch.int64)
        extend_start_loc = torch.zeros(num_seqs, dtype=torch.int64)
        hidden_states = torch.zeros(new_total, hidden_dim, dtype=torch.float32)

        seq_lens = torch.tensor([10, 9], dtype=torch.int64)
        padding_lens = torch.tensor([0, 1], dtype=torch.int64)
        req_pool_indices = torch.tensor([1, 0], dtype=torch.int64)
        req_to_token = torch.randint(0, 10000, (num_reqs, pool_len), dtype=torch.int32)
        req_to_hidden_states_pool = torch.randn(
            num_reqs, pool_size, hidden_dim, dtype=torch.float32
        )

        torch.ops.sgl_kernel.assign_new_state_cpu(
            next_token_ids,
            old_input_ids,
            old_positions,
            old_out_cache_loc,
            old_extend_seq_lens,
            old_extend_start_loc,
            input_ids,
            positions,
            out_cache_loc,
            extend_seq_lens,
            extend_start_loc,
            seq_lens,
            padding_lens,
            req_pool_indices,
            req_to_token,
            num_seqs,
            step,
            hidden_states,
            old_hidden_states,
            req_to_hidden_states_pool,
        )

        # pure-torch reference (mirrors assign_new_state_triton)
        ids_ref = torch.full((new_total,), -1, dtype=torch.int64)
        pos_ref = torch.full((new_total,), -1, dtype=torch.int64)
        ocl_ref = torch.full((new_total,), -1, dtype=torch.int64)
        esl_ref = torch.zeros(num_seqs, dtype=torch.int64)
        esloc_ref = torch.zeros(num_seqs, dtype=torch.int64)
        hs_ref = torch.zeros(new_total, hidden_dim, dtype=torch.float32)
        for pid in range(num_seqs):
            old_len = int(old_extend_seq_lens[pid])
            old_start = int(old_extend_start_loc[pid])
            new_start = old_start + pid
            esl_ref[pid] = old_len + 1
            esloc_ref[pid] = new_start

            ids_ref[new_start : new_start + old_len] = old_input_ids[
                old_start : old_start + old_len
            ]
            ids_ref[new_start + old_len - int(padding_lens[pid])] = next_token_ids[pid]

            pos_ref[new_start + 1 : new_start + 1 + old_len] = old_positions[
                old_start : old_start + old_len
            ]
            pos_ref[new_start] = max(int(old_positions[old_start]) - 1, 0)

            ocl_ref[new_start + 1 : new_start + 1 + old_len] = old_out_cache_loc[
                old_start : old_start + old_len
            ]
            token_idx_col = int(seq_lens[pid]) - old_len - 1
            if token_idx_col >= 0:
                ocl_ref[new_start] = int(
                    req_to_token[int(req_pool_indices[pid]), token_idx_col]
                )

            hs_ref[new_start + 1 : new_start + 1 + old_len] = old_hidden_states[
                old_start : old_start + old_len
            ]
            hs_ref[new_start] = req_to_hidden_states_pool[
                int(req_pool_indices[pid]), pool_size - (step + 1)
            ]

        torch.testing.assert_close(input_ids, ids_ref, atol=0, rtol=0)
        torch.testing.assert_close(positions, pos_ref, atol=0, rtol=0)
        torch.testing.assert_close(out_cache_loc, ocl_ref, atol=0, rtol=0)
        torch.testing.assert_close(extend_seq_lens, esl_ref, atol=0, rtol=0)
        torch.testing.assert_close(extend_start_loc, esloc_ref, atol=0, rtol=0)
        torch.testing.assert_close(hidden_states, hs_ref, atol=0, rtol=0)

        # int64 contract: int32 next_token_ids must be rejected
        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.assign_new_state_cpu(
                next_token_ids.to(torch.int32),
                old_input_ids,
                old_positions,
                old_out_cache_loc,
                old_extend_seq_lens,
                old_extend_start_loc,
                input_ids,
                positions,
                out_cache_loc,
                extend_seq_lens,
                extend_start_loc,
                seq_lens,
                padding_lens,
                req_pool_indices,
                req_to_token,
                num_seqs,
                step,
                hidden_states,
                old_hidden_states,
                req_to_hidden_states_pool,
            )


class TestBuildDraftDecodeMetadata(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def _ref_build_metadata(
        self, req_to_token, req_pool_indices, seq_lens, topk, num_steps
    ):
        """Pure-python reference. Draft row b*topk+tk holds the committed prefix
        followed by candidate tk's draft slots, which live contiguously in the
        source row at sl + tk*num_steps + s (assign_draft_cache_locs_contiguous
        layout); only the first sl + num_steps entries per row are defined.
        """
        rows = []
        for b in range(seq_lens.numel()):
            src_row = req_to_token[int(req_pool_indices[b])]
            sl = int(seq_lens[b])
            for tk in range(topk):
                draft_start = sl + tk * num_steps
                rows.append(
                    torch.cat(
                        [src_row[:sl], src_row[draft_start : draft_start + num_steps]]
                    )
                )
        return rows

    def _run_and_check(self, seq_lens, topk, num_steps, pool_len, num_reqs):
        num_seqs = seq_lens.numel()
        # distinct slot values so any misplaced copy is caught exactly
        req_to_token = (
            torch.randperm(num_reqs * pool_len).to(torch.int32).view(num_reqs, pool_len)
        )
        req_pool_indices = torch.randperm(num_reqs, dtype=torch.int64)[:num_seqs]
        req_to_token_draft = torch.ops.sgl_kernel.build_draft_decode_metadata_cpu(
            req_to_token,
            req_pool_indices,
            seq_lens,
            topk,
            num_steps,
            pool_len,
        )
        self.assertEqual(
            req_to_token_draft.shape, torch.Size([num_seqs * topk, pool_len])
        )
        self.assertEqual(req_to_token_draft.dtype, req_to_token.dtype)
        ref_rows = self._ref_build_metadata(
            req_to_token, req_pool_indices, seq_lens, topk, num_steps
        )
        for b in range(num_seqs):
            sl = int(seq_lens[b])
            for tk in range(topk):
                flat = b * topk + tk
                # Only the first sl + num_steps entries are the kernel's
                # contract: the output is allocated uninitialized and the
                # draft-decode consumer reads at most seq_len + num_steps
                # slots per candidate, so the row tail stays unasserted.
                torch.testing.assert_close(
                    req_to_token_draft[flat, : sl + num_steps],
                    ref_rows[flat],
                    atol=0,
                    rtol=0,
                )

    def test_build_metadata_topk1_chain(self):
        # MTP config: topk=1 -> the single candidate's drafts sit right after
        # the prefix, so each draft row is the source row's first
        # sl + num_steps slots verbatim.
        seq_lens = torch.tensor([5, 0, 17], dtype=torch.int64)
        for num_steps in [1, 2, 3]:
            with self.subTest(num_steps=num_steps):
                self._run_and_check(seq_lens, 1, num_steps, pool_len=32, num_reqs=5)

    def test_build_metadata_topk4(self):
        # EAGLE config: topk=4 -> candidate tk's drafts come from the strided
        # source range [sl + tk*num_steps, sl + (tk+1)*num_steps) but always
        # land right after the prefix in the expanded row.
        seq_lens = torch.tensor([9, 0, 3, 21], dtype=torch.int64)
        for num_steps in [1, 2, 3]:
            with self.subTest(num_steps=num_steps):
                self._run_and_check(seq_lens, 4, num_steps, pool_len=64, num_reqs=6)


if __name__ == "__main__":
    unittest.main()
