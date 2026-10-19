"""SGLANG_ENABLE_SPEC_OPTIMIZED routes XPU EAGLE tree building to the
sgl-kernel-xpu SYCL kernel, which must stay bit-identical to the Triton kernel."""

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
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"


def _draft_tree(bs, topk, steps, num_verify, gen):
    """Random draft outputs laid out like EagleWorker.draft_forward produces them."""
    score_list = [torch.rand(bs, 1, topk, generator=gen)]
    token_list = [torch.randint(0, 32000, (bs, topk), generator=gen)]
    parents_list = [torch.arange(-1, topk).repeat(bs, 1)]
    for i in range(1, steps):
        score_list.append(torch.rand(bs, topk, topk, generator=gen))
        token_list.append(torch.randint(0, 32000, (bs, topk * topk), generator=gen))
        # Same offset as select_top_k_tokens: topk_cs_index + topk^2*(i-1) + topk.
        topk_cs_index = torch.stack(
            [torch.randperm(topk * topk, generator=gen)[:topk] for _ in range(bs)]
        )
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

    def test_flag_selects_kernel(self):
        tree = _draft_tree(2, 2, 3, 6, torch.Generator().manual_seed(1))
        for use_sycl in (False, True):
            with (
                self.subTest(use_sycl=use_sycl),
                mock.patch.object(
                    sgl_kernel,
                    "build_tree_kernel_efficient",
                    wraps=sgl_kernel.build_tree_kernel_efficient,
                ) as sycl_kernel,
            ):
                _build(use_sycl, tree, 2, 3, 6, TreeMaskMode.FULL_MASK)
                self.assertEqual(sycl_kernel.called, use_sycl)


if __name__ == "__main__":
    unittest.main()
