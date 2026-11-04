from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")

import unittest

import torch

from sglang.kernels.ops.speculative.chain_verify import greedy_chain_verify
from sglang.kernels.ops.speculative.row_argmax import row_argmax
from sglang.kernels.ops.speculative.spec_tree import verify_tree_greedy_func
from sglang.test.test_utils import CustomTestCase


def chain_case(bs, width, vocab, g, dev, ties=False):
    """Logits rows and top-1 chains whose drafts match the targets for a random
    number of steps per request (the retrieve arrays build_chain_tree gives)."""
    logits = torch.randn((bs * width, vocab), generator=g, device=dev) * 3
    if ties:
        logits = logits.to(torch.bfloat16).float()
    target = torch.argmax(logits, dim=-1).view(bs, width)
    candidates = torch.randint(0, vocab, (bs, width), generator=g, device=dev)
    for b in range(bs):
        n = int(torch.randint(0, width, (1,), generator=g, device=dev))
        candidates[b, 1 : n + 1] = target[b, :n]
    base = torch.arange(bs, device=dev).view(-1, 1) * width
    retrieve_index = base + torch.arange(width, device=dev)
    retrieve_next_token = torch.arange(1, width + 1, device=dev).repeat(bs, 1)
    retrieve_next_token[:, -1] = -1
    retrieve_next_sibling = torch.full_like(retrieve_next_token, -1)
    return (
        logits,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    )


def reference(logits, candidates, ri, rn, rs, depth):
    bs, width = candidates.shape
    predict = torch.zeros((bs * width,), dtype=torch.int32, device=logits.device)
    accept_index = torch.full((bs, depth), -1, dtype=torch.int32, device=logits.device)
    num = torch.empty((bs,), dtype=torch.int32, device=logits.device)
    target = row_argmax(logits).reshape(bs, width)
    predict, accept_index, num = verify_tree_greedy_func(
        predicts=predict,
        accept_index=accept_index,
        accept_token_num=num,
        candidates=candidates,
        retrieve_index=ri,
        retrieve_next_token=rn,
        retrieve_next_sibling=rs,
        target_predict=target,
        topk=1,
    )
    return predict, accept_index, num, target


@unittest.skipUnless(torch.cuda.is_available(), "a GPU is required for this test.")
class TestGreedyChainVerify(CustomTestCase):
    def test_matches_argmax_then_tree_verify(self):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(0)
        for vocab in (154880, 32000, 8191):
            for bs in (1, 2, 3, 16):
                for width in (2, 4):
                    for ties in (False, True):
                        with self.subTest(vocab=vocab, bs=bs, width=width, ties=ties):
                            case = chain_case(bs, width, vocab, g, dev, ties)
                            want = reference(*case, width)
                            got = greedy_chain_verify(*case[:4], width)[:4]
                            for w, o in zip(want, got):
                                torch.testing.assert_close(o, w, rtol=0, atol=0)

    def test_accept_lens(self):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(1)
        for bs in (1, 2, 16):
            with self.subTest(bs=bs):
                case = chain_case(bs, 4, 154880, g, dev, ties=True)
                lens = torch.full((bs,), -7, dtype=torch.int32, device=dev)
                got = greedy_chain_verify(*case[:4], 4, accept_lens=lens)
                plain = greedy_chain_verify(*case[:4], 4)
                for w, o in zip(plain[:4], got[:4]):
                    torch.testing.assert_close(o, w, rtol=0, atol=0)
                torch.testing.assert_close(lens, got[2] + 1, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
