"""ROCm EAGLE must honor the request distribution instead of forcing argmax."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.speculative.eagle_utils import eagle_sample
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")
pytestmark = pytest.mark.skipif(not is_hip(), reason="ROCm verification regression")


def sample(
    *,
    temperature=1.0,
    top_p=1.0,
    top_k=32,
    min_p=0.0,
    greedy=False,
    seed=None,
    mixed=False,
    grammar=None,
):
    bs, slots, vocab = 4096, 3, 32
    device = "cuda"
    info = SimpleNamespace(
        is_all_greedy=greedy,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        temperatures=torch.full((bs, 1), temperature, device=device),
        top_ks=torch.full((bs,), top_k, dtype=torch.int32, device=device),
        top_ps=torch.full((bs,), top_p, device=device),
        min_ps=torch.full((bs,), min_p, device=device),
        need_top_k_sampling=top_k < vocab,
        need_top_p_sampling=top_p < 1.0,
        need_min_p_sampling=min_p > 0.0,
        sampling_seed=(
            torch.arange(bs, device=device) + seed if seed is not None else None
        ),
    )
    if mixed:
        info.top_ks[: bs // 2] = 1
        info.need_top_k_sampling = True
    indices = torch.arange(bs * slots, device=device).reshape(bs, slots)
    verify = SimpleNamespace(
        draft_token_num=slots,
        max_tree_depth=slots,
        tree_topk=1,
        draft_token=torch.zeros(bs * slots, dtype=torch.int64, device=device),
        retrieve_index=indices,
        retrieve_next_token=torch.tensor([1, 2, -1], device=device).repeat(bs, 1),
        retrieve_next_sibling=torch.full(
            (bs, slots), -1, dtype=torch.int64, device=device
        ),
        positions=torch.arange(slots, device=device).repeat(bs) + 100,
    )
    batch = SimpleNamespace(
        device=device,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.full((bs,), 100, device=device),
        sampling_info=info,
    )
    logits = torch.full((bs * slots, vocab), -float("inf"), device=device)
    logits[:, 0] = 0.4
    logits[:, 1] = 0.0
    with (
        patch(
            "sglang.srt.distributed.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False
        ),
    ):
        predict, lengths, accepted = eagle_sample(
            verify, batch, SimpleNamespace(next_token_logits=logits), grammar
        )
    torch.cuda.synchronize()
    roots = predict[indices[:, 0]]
    assert torch.all((lengths >= 1) & (lengths <= slots))
    assert torch.equal(accepted[:, 0].long(), indices[:, 0])
    assert torch.all((roots == 0) | (roots == 1))
    return roots, lengths


def test_nongreedy_distribution_and_acceptance():
    roots, lengths = sample()
    expected = torch.softmax(torch.tensor([0.4, 0.0]), 0)[1].item()
    assert abs((roots == 1).float().mean().item() - expected) < 0.04
    assert torch.any(lengths == 1) and torch.any(lengths == 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"greedy": True},
        {"top_k": 1},
        {"top_p": 0.5},
        {"min_p": 0.9},
        {"temperature": 0.01},
    ],
)
def test_restricted_distributions(kwargs):
    roots, _ = sample(**kwargs)
    assert torch.all(roots == 0)


def test_seed_reproducibility():
    a, _ = sample(seed=123)
    b, _ = sample(seed=123)
    c, _ = sample(seed=456)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_mixed_greedy_and_sampling_requests():
    roots, _ = sample(mixed=True)
    assert torch.all(roots[:2048] == 0)
    assert 0.35 < (roots[2048:] == 1).float().mean().item() < 0.45


def test_grammar_is_applied_before_sampling():
    def apply(logits):
        logits[:, 0] = -float("inf")

    roots, _ = sample(grammar=SimpleNamespace(apply=apply))
    assert torch.all(roots == 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
