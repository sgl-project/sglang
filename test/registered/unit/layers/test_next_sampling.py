import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("rows", [0, 1, 4, 128])
@pytest.mark.parametrize("cols", [32771, 151936, 248320])
def test_speculative_argmax_graph(rows, cols):
    from sglang.kernels.ops.speculative.row_argmax import speculative_argmax

    logits = torch.randn((rows * 2, cols), device="cuda")[::2]
    speculative_argmax(logits)
    speculative_argmax(logits, with_probs=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids = speculative_argmax(logits)
        probabilities, ids2 = speculative_argmax(logits, with_probs=True)
    for case in range(4):
        if case == 1:
            logits.fill_(-float("inf"))
        elif case == 2:
            logits[:, 2049] = float("inf")
            logits[:, -1] = float("inf")
        elif case == 3:
            logits[:, 15] = float("nan")
            logits[:, -17] = float("nan")
        graph.replay()
        expected = logits.argmax(dim=-1)
        torch.testing.assert_close(ids, expected, rtol=0, atol=0)
        torch.testing.assert_close(ids2[:, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            probabilities, torch.ones_like(probabilities), rtol=0, atol=0
        )


@pytest.mark.parametrize("bs", [1, 3, 128])
@pytest.mark.parametrize("width", [1, 2, 4, 7])
@pytest.mark.parametrize("vocab", [131073, 248320])
def test_greedy_verify_chain_graph(bs, width, vocab):
    from sglang.kernels.ops.speculative.row_argmax import greedy_verify_chain
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func

    logits = torch.randn((bs * width, vocab), device="cuda")
    candidates = torch.empty((bs * 2, width), device="cuda", dtype=torch.int64)[::2]
    indices = torch.arange(bs * width, device="cuda").reshape(bs, width)
    next_tokens = torch.arange(1, width + 1, device="cuda").repeat(bs, 1)
    next_tokens[:, -1] = -1
    siblings = torch.full_like(indices, -1)
    greedy_verify_chain(logits, candidates)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = greedy_verify_chain(logits, candidates)
    for case in range(3):
        if case == 1:
            logits.fill_(-float("inf"))
            logits[:, -2:] = float("inf")
        elif case == 2:
            logits[:, 3] = float("nan")
            logits[:, -1] = float("nan")
        target = logits.argmax(-1).reshape(bs, width)
        for accepted in range(width):
            candidates[:, 0] = 17
            candidates[:, 1:] = target[:, :-1]
            if accepted < width - 1:
                candidates[:, accepted + 1].add_(1).remainder_(vocab)
            graph.replay()
            expected = verify_tree_greedy_func(
                predicts=torch.zeros(bs * width, device="cuda", dtype=torch.int32),
                accept_index=torch.full(
                    (bs, width), -1, device="cuda", dtype=torch.int32
                ),
                accept_token_num=torch.empty(bs, device="cuda", dtype=torch.int32),
                candidates=candidates.contiguous(),
                retrieve_index=indices,
                retrieve_next_token=next_tokens,
                retrieve_next_sibling=siblings,
                target_predict=target,
                topk=1,
            )
            for a, e in zip(actual[:3], expected):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            torch.testing.assert_close(actual[3], target, rtol=0, atol=0)
