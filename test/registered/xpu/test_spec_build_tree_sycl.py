"""SGLANG_ENABLE_SPEC_OPTIMIZED routes XPU EAGLE tree building and greedy tree
verify to the sgl-kernel-xpu SYCL kernels, which must stay bit-identical to Triton."""

import itertools
import unittest
from unittest import mock

import sgl_kernel
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.verify_mask import maybe_create_verify_mask
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    organize_draft_results,
    verify_tree_greedy_func,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=40, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"


def _draft_tree(bs, topk, steps, num_verify, gen):
    """Random draft outputs laid out like select_top_k_tokens produces them.

    Scores are cumulative path probabilities (child = beam * p), so every selected
    node's parent is selected too and the tree is connected.
    """
    beam_scores = torch.rand(bs, topk, generator=gen)
    score_list = [beam_scores.unsqueeze(1)]
    token_list = [torch.randint(0, 32000, (bs, topk), generator=gen)]
    parents_list = [torch.arange(-1, topk).repeat(bs, 1)]
    for i in range(1, steps):
        expand_scores = beam_scores.unsqueeze(2) * torch.rand(
            bs, topk, topk, generator=gen
        )
        score_list.append(expand_scores)
        token_list.append(torch.randint(0, 32000, (bs, topk * topk), generator=gen))
        beam_scores, topk_cs_index = expand_scores.flatten(1).topk(topk, dim=-1)
        # Same offset as select_top_k_tokens: topk_cs_index + topk^2*(i-1) + topk.
        parents_list.append(topk_cs_index + (topk * topk * (i - 1) + topk))
    parents_list.append(torch.zeros(bs, topk, dtype=torch.long))

    def to_device(xs):
        return [x.to(DEVICE) for x in xs]

    parent_list, top_scores_index, draft_tokens = organize_draft_results(
        score_list=to_device(score_list),
        token_list=to_device(token_list),
        parents_list=to_device(parents_list),
        num_draft_token=num_verify,
    )
    seq_lens = torch.randint(1, 300, (bs,), generator=gen).to(DEVICE)
    bonus_tokens = torch.randint(0, 32000, (bs,), generator=gen).to(DEVICE)
    return bonus_tokens, parent_list, top_scores_index, draft_tokens, seq_lens


def _build(use_sycl, tree, topk, steps, num_verify, mode, tree_mask_buf=None):
    bonus_tokens, parent_list, top_scores_index, draft_tokens, seq_lens = tree
    with envs.SGLANG_ENABLE_SPEC_OPTIMIZED.override(use_sycl):
        return build_tree_kernel_efficient(
            bonus_tokens=bonus_tokens,
            parent_list=parent_list,
            top_scores_index=top_scores_index,
            draft_tokens=draft_tokens,
            seq_lens=seq_lens,
            seq_lens_sum=int(seq_lens.sum().item()),
            topk=topk,
            spec_steps=steps,
            num_verify_tokens=num_verify,
            tree_mask_mode=mode,
            tree_mask_buf=tree_mask_buf,
        )


def _target_predict(candidates, retrieve_next_token, retrieve_next_sibling, gen):
    """Per node, predict a random child's draft token 70% of the time, so verify
    walks accept chains, sibling hops and early stops."""
    bs, num_verify = candidates.shape
    cand, nxt, sib = (
        candidates.cpu(),
        retrieve_next_token.cpu(),
        retrieve_next_sibling.cpu(),
    )
    target = torch.randint(0, 32000, (bs, num_verify), generator=gen)
    for b, j in itertools.product(range(bs), range(num_verify)):
        children, k = [], int(nxt[b, j])
        while k != -1:
            children.append(k)
            k = int(sib[b, k])
        if children and torch.rand(1, generator=gen).item() < 0.7:
            pick = int(torch.randint(len(children), (1,), generator=gen))
            target[b, j] = cand[b, children[pick]]
    return target.to(DEVICE)


def _verify(use_sycl, candidates, retrieve, target_predict, depth):
    bs, num_verify = candidates.shape
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve
    with envs.SGLANG_ENABLE_SPEC_OPTIMIZED.override(use_sycl):
        return verify_tree_greedy_func(
            predicts=torch.zeros(bs * num_verify, dtype=torch.int32, device=DEVICE),
            accept_index=torch.full((bs, depth), -1, dtype=torch.int32, device=DEVICE),
            accept_token_num=torch.empty((bs,), dtype=torch.int32, device=DEVICE),
            candidates=candidates,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )


def _verify_inputs(topk, steps, num_verify, bs, gen):
    tree = _draft_tree(bs, topk, steps, num_verify, gen)
    _, _, *retrieve, draft_tokens = _build(
        False, tree, topk, steps, num_verify, TreeMaskMode.FULL_MASK
    )
    candidates = draft_tokens.reshape(bs, num_verify)
    target_predict = _target_predict(candidates, retrieve[1], retrieve[2], gen)
    return candidates, retrieve, target_predict


class TestSpecBuildTreeSycl(CustomTestCase):
    # (topk, spec_steps, num_verify_tokens, bs): chain, small tree, the
    # B60-tuned topk=4 tree, and topk=8 where num_verify < nodes in the tree.
    CONFIGS = [(1, 3, 4, 3), (2, 3, 6, 3), (4, 5, 8, 8), (8, 5, 16, 4)]
    MODES = [TreeMaskMode.FULL_MASK, TreeMaskMode.QLEN_ONLY]

    def test_sycl_matches_triton(self):
        gen = torch.Generator().manual_seed(0)
        for (topk, steps, num_verify, bs), mode in itertools.product(
            self.CONFIGS, self.MODES
        ):
            with self.subTest(topk=topk, steps=steps, bs=bs, mode=mode.name):
                tree = _draft_tree(bs, topk, steps, num_verify, gen)
                triton_out = _build(False, tree, topk, steps, num_verify, mode)
                sycl_out = _build(True, tree, topk, steps, num_verify, mode)
                for name, ref, out in zip(
                    (
                        "tree_mask",
                        "positions",
                        "retrieve_index",
                        "retrieve_next_token",
                        "retrieve_next_sibling",
                        "draft_tokens",
                    ),
                    triton_out,
                    sycl_out,
                ):
                    self.assertTrue(torch.equal(ref, out), name)

    def test_sycl_fills_uint8_verify_mask_buffer(self):
        """The triton attention backend hands build_tree its preallocated uint8
        verify mask; the SYCL path must accept it and write Triton's bytes."""
        topk, steps, num_verify, bs = 2, 3, 6, 3
        tree = _draft_tree(
            bs, topk, steps, num_verify, torch.Generator().manual_seed(2)
        )
        outs = []
        for use_sycl in (False, True):
            verify_mask = maybe_create_verify_mask(
                is_draft_runner=False,
                skip_prefill=False,
                max_bs=bs,
                max_context_len=512,
                num_draft_tokens=num_verify,
                device=DEVICE,
                is_read=True,
                dtype=torch.uint8,
            )
            outs.append(
                _build(
                    use_sycl,
                    tree,
                    topk,
                    steps,
                    num_verify,
                    verify_mask.mode,
                    tree_mask_buf=verify_mask.buffer,
                )
            )
        self.assertEqual(outs[1][0].dtype, torch.uint8)
        for ref, out in zip(*outs):
            self.assertTrue(torch.equal(ref, out))

    def test_sycl_verify_matches_triton(self):
        gen = torch.Generator().manual_seed(3)
        for topk, steps, num_verify, bs in self.CONFIGS:
            with self.subTest(topk=topk, steps=steps, bs=bs):
                candidates, retrieve, target_predict = _verify_inputs(
                    topk, steps, num_verify, bs, gen
                )
                # verify_input.max_tree_depth is spec_steps + 1.
                triton_out = _verify(
                    False, candidates, retrieve, target_predict, steps + 1
                )
                sycl_out = _verify(
                    True, candidates, retrieve, target_predict, steps + 1
                )
                for name, ref, out in zip(
                    ("predicts", "accept_index", "accept_token_num"),
                    triton_out,
                    sycl_out,
                ):
                    self.assertTrue(torch.equal(ref, out), name)

    def test_flag_selects_kernel(self):
        gen = torch.Generator().manual_seed(1)
        tree = _draft_tree(2, 2, 3, 6, gen)
        candidates, retrieve, target_predict = _verify_inputs(2, 3, 6, 2, gen)
        for use_sycl in (False, True):
            with (
                self.subTest(use_sycl=use_sycl),
                mock.patch.object(
                    sgl_kernel,
                    "build_tree_kernel_efficient",
                    wraps=sgl_kernel.build_tree_kernel_efficient,
                ) as sycl_build_tree,
                mock.patch.object(
                    sgl_kernel,
                    "verify_tree_greedy",
                    wraps=sgl_kernel.verify_tree_greedy,
                ) as sycl_verify,
            ):
                _build(use_sycl, tree, 2, 3, 6, TreeMaskMode.FULL_MASK)
                _verify(use_sycl, candidates, retrieve, target_predict, 4)
                self.assertEqual(sycl_build_tree.called, use_sycl)
                self.assertEqual(sycl_verify.called, use_sycl)


if __name__ == "__main__":
    unittest.main()
