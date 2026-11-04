"""ROCm's Triton tree_speculative_sampling_target_only must reproduce the CUDA kernel,
sample the target distribution, and replay under a CUDA graph."""

import sys

import pytest
import torch

from sglang.kernels.ops.speculative.target_only_sampling import (
    tree_speculative_sampling_target_only_triton,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: the Triton port is ROCm's sampled-verify kernel
register_amd_ci(est_time=25, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
# Probabilities are multiples of 2**-16, so every partial sum is exact in fp32
# and no reduction order can move a CDF boundary.
_DYADIC_DENOM = 2**16


def _reference(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
):
    """Torch port of TreeSpeculativeSamplingTargetOnly (sgl_kernel/speculative/
    sampling.cuh) over CPU tensors; updates the outputs and draft_probs in place."""
    threshold_single = torch.tensor(threshold_single, dtype=torch.float32)
    threshold_acc = torch.tensor(max(threshold_acc, 1e-9), dtype=torch.float32)
    num_spec_tokens = accept_index.shape[1]
    vocab_size = target_probs.shape[-1]
    for b in range(candidates.shape[0]):
        prob_acc = torch.zeros((), dtype=torch.float32)
        row = 0
        coin = uniform_samples[b, 0]
        last = int(retrive_index[b, 0])
        accept_index[b, 0] = last
        num_accepted = 0
        cur = 0
        for _ in range(1, num_spec_tokens):
            cur = int(retrive_next_token[b, cur])
            while cur != -1:
                token = int(candidates[b, cur])
                prob = target_probs[b, row, token].clone()
                prob_acc = prob_acc + prob
                if prob > 0 and (
                    coin < prob_acc / threshold_acc or prob >= threshold_single
                ):
                    prob_acc = torch.zeros((), dtype=torch.float32)
                    row = cur
                    coin = uniform_samples[b, cur]
                    predicts[last] = token
                    num_accepted += 1
                    last = int(retrive_index[b, cur])
                    accept_index[b, num_accepted] = last
                    break
                draft_probs[b, row, token] = prob
                cur = int(retrive_next_sibling[b, cur])
            if cur == -1:
                break
        accept_token_num[b] = num_accepted

        # The bonus token after the last slot has no draft distribution.
        residual = target_probs[b, row]
        if num_accepted != num_spec_tokens - 1:
            residual = residual - draft_probs[b, row]
        residual = residual.clamp(min=0)
        u = uniform_samples_for_final_sampling[b] * residual.sum()
        above = ((residual.cumsum(0) > u) & (residual > 0)).nonzero().flatten()
        positive = (residual > 0).nonzero().flatten()
        if len(above):
            predicts[last] = int(above[0])
        elif len(positive):
            predicts[last] = int(positive[-1])
        else:
            predicts[last] = vocab_size - 1


def _dyadic_rows(shape, vocab, generator):
    rows = torch.zeros(*shape, vocab)
    for row in rows.view(-1, vocab):
        support = int(torch.randint(1, min(vocab, 48) + 1, (1,), generator=generator))
        tokens = torch.randperm(vocab, generator=generator)[:support]
        cuts = torch.randint(0, _DYADIC_DENOM + 1, (support - 1,), generator=generator)
        bounds = torch.cat(
            [torch.tensor([0]), cuts.sort().values, torch.tensor([_DYADIC_DENOM])]
        )
        row[tokens] = bounds.diff().float() / _DYADIC_DENOM
    return rows


def _make_inputs(*, bs, num_nodes, vocab, chain, random_draft, seed):
    """Random draft trees (node 0 is the root) as first-child / next-sibling
    links. Candidates mostly come from the parent's target support, so levels
    accept as well as reject."""
    generator = torch.Generator().manual_seed(seed)
    next_token = torch.full((bs, num_nodes), -1, dtype=torch.int64)
    next_sibling = torch.full((bs, num_nodes), -1, dtype=torch.int64)
    target = _dyadic_rows((bs, num_nodes), vocab, generator)
    if random_draft:
        draft = _dyadic_rows((bs, num_nodes), vocab, generator)
    else:
        draft = torch.zeros_like(target)
    candidates = torch.zeros((bs, num_nodes), dtype=torch.int64)
    for b in range(bs):
        last_child = {}
        for node in range(1, num_nodes):
            if chain:
                parent = node - 1
            else:
                parent = int(torch.randint(0, node, (1,), generator=generator))
            if parent in last_child:
                next_sibling[b, last_child[parent]] = node
            else:
                next_token[b, parent] = node
            last_child[parent] = node
            support = target[b, parent].nonzero().flatten()
            if torch.rand((), generator=generator) < 0.75:
                pick = support[torch.randint(len(support), (1,), generator=generator)]
            else:
                pick = torch.randint(vocab, (1,), generator=generator)
            candidates[b, node] = int(pick)
    return {
        "candidates": candidates,
        # Scattered global slots exercise the predicts / accept_index indirection.
        "retrive_index": torch.randperm(bs * num_nodes, generator=generator).view(
            bs, num_nodes
        ),
        "retrive_next_token": next_token,
        "retrive_next_sibling": next_sibling,
        "uniform_samples": torch.rand((bs, num_nodes), generator=generator),
        "uniform_samples_for_final_sampling": torch.rand((bs,), generator=generator),
        "target_probs": target,
        "draft_probs": draft,
    }


def _outputs(bs, num_nodes, num_spec_tokens, device):
    return {
        "predicts": torch.full((bs * num_nodes,), -1, dtype=torch.int32, device=device),
        "accept_index": torch.full(
            (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
        ),
        "accept_token_num": torch.full((bs,), -1, dtype=torch.int32, device=device),
    }


def _run(inputs, num_spec_tokens, threshold_single=1.0, threshold_acc=1.0):
    """Run the torch reference and the kernel on copies of ``inputs``."""
    bs, num_nodes = inputs["candidates"].shape
    expected = {k: v.clone() for k, v in inputs.items()}
    expected.update(_outputs(bs, num_nodes, num_spec_tokens, "cpu"))
    _reference(
        **expected, threshold_single=threshold_single, threshold_acc=threshold_acc
    )
    actual = {k: v.to(DEVICE) for k, v in inputs.items()}
    actual.update(_outputs(bs, num_nodes, num_spec_tokens, DEVICE))
    tree_speculative_sampling_target_only_triton(
        **actual,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )
    return expected, actual


def _assert_same(expected, actual):
    for name in ("predicts", "accept_index", "accept_token_num", "draft_probs"):
        assert torch.equal(actual[name].cpu(), expected[name]), name


@pytest.mark.parametrize("random_draft", [False, True])
@pytest.mark.parametrize(
    "threshold_single,threshold_acc", [(1, 1), (0.5, 0.25), (0, 0)]
)
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("vocab", [11, 9001])
@pytest.mark.parametrize("chain", [True, False])
def test_matches_reference(
    chain, vocab, truncated, threshold_single, threshold_acc, random_draft
):
    num_nodes = 6
    inputs = _make_inputs(
        bs=16,
        num_nodes=num_nodes,
        vocab=vocab,
        chain=chain,
        random_draft=random_draft,
        seed=vocab + 2 * chain,
    )
    # A truncated walk stops below the leaves and samples the bonus token.
    num_spec_tokens = 3 if truncated else num_nodes
    expected, actual = _run(inputs, num_spec_tokens, threshold_single, threshold_acc)
    _assert_same(expected, actual)


def _chain(target_rows, draft_rows, drafts, coins, final_coin):
    num_nodes = len(target_rows)
    next_token = torch.arange(1, num_nodes + 1).unsqueeze(0)
    next_token[0, -1] = -1
    return {
        "candidates": torch.tensor([[0, *drafts]]),
        "retrive_index": torch.arange(num_nodes).unsqueeze(0),
        "retrive_next_token": next_token,
        "retrive_next_sibling": torch.full((1, num_nodes), -1),
        "uniform_samples": torch.tensor([coins], dtype=torch.float32),
        "uniform_samples_for_final_sampling": torch.tensor([final_coin]),
        "target_probs": torch.tensor([target_rows], dtype=torch.float32),
        "draft_probs": torch.tensor([draft_rows], dtype=torch.float32),
    }


def test_bonus_ignores_draft_row_after_last_slot():
    # Both drafts are certain; a draft row at the bonus slot would empty the
    # residual if it were read.
    inputs = _chain(
        target_rows=[[0, 1, 0], [0, 0, 1], [0.25, 0.75, 0]],
        draft_rows=[[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        drafts=[1, 2],
        coins=[0.5, 0.5, 0.5],
        final_coin=0.5,
    )
    expected, actual = _run(inputs, num_spec_tokens=3)
    _assert_same(expected, actual)
    assert actual["accept_token_num"].item() == 2
    assert actual["predicts"].tolist() == [1, 2, 1]


def test_rejected_draft_is_excluded_from_residual():
    # Draft 0 holds 1/2 of the mass and is rejected at coin 0.75. The residual
    # [0, 1/4, 1/4] sums to 1/2, so u = 0.75 * 1/2 = 3/8 falls on token 2.
    inputs = _chain(
        target_rows=[[0.5, 0.25, 0.25], [1, 0, 0]],
        draft_rows=[[0, 0, 0], [0, 0, 0]],
        drafts=[0],
        coins=[0.75, 0.5],
        final_coin=0.75,
    )
    expected, actual = _run(inputs, num_spec_tokens=2)
    _assert_same(expected, actual)
    assert actual["accept_token_num"].item() == 0
    assert actual["predicts"][0].item() == 2
    assert actual["draft_probs"][0, 0].tolist() == [0.5, 0, 0]


@pytest.mark.parametrize(
    "draft_row,final_coin,expected_token",
    [
        # Draft mass covers the target everywhere: an empty residual takes the
        # last vocabulary id.
        ([1, 1, 1, 1, 1], 0.5, 4),
        # u at the full residual mass: no CDF entry exceeds it, so the last
        # positive token is taken.
        ([0, 0, 0, 0, 0], 1.0, 3),
    ],
    ids=["empty-residual", "coin-at-total-mass"],
)
def test_final_sampling_fallbacks(draft_row, final_coin, expected_token):
    inputs = _chain(
        target_rows=[[0.25, 0.25, 0, 0.5, 0], [1, 0, 0, 0, 0]],
        draft_rows=[draft_row, [0] * 5],
        drafts=[0],
        coins=[0.75, 0.5],
        final_coin=final_coin,
    )
    expected, actual = _run(inputs, num_spec_tokens=2)
    _assert_same(expected, actual)
    assert actual["predicts"][0].item() == expected_token


def test_output_matches_target_distribution():
    """Every committed token, accepted draft or final sample, is distributed
    as the target row of its slot, and each draft is accepted with its target
    probability."""
    batch, vocab = 65536, 8
    target_rows = torch.tensor(
        [
            [0.5, 0.25, 0.125, 0.125, 0, 0, 0, 0],
            [0, 0.375, 0, 0.25, 0.25, 0.125, 0, 0],
            [0.0625, 0.0625, 0.125, 0.25, 0.5, 0, 0, 0],
            [0.125] * 8,
        ],
        device=DEVICE,
    )
    slots = target_rows.shape[0]
    generator = torch.Generator(device=DEVICE).manual_seed(1234)
    # Proposals put a quarter of their mass off the target support.
    proposal = 0.75 * target_rows[:-1] + 0.25 / vocab
    drafts = torch.multinomial(proposal, batch, replacement=True, generator=generator)
    candidates = torch.cat(
        [torch.zeros(batch, 1, dtype=torch.int64, device=DEVICE), drafts.T], dim=1
    )
    next_token = torch.arange(1, slots + 1, device=DEVICE).repeat(batch, 1)
    next_token[:, -1] = -1
    inputs = {
        "candidates": candidates.contiguous(),
        "retrive_index": torch.arange(batch * slots, device=DEVICE).view(batch, slots),
        "retrive_next_token": next_token,
        "retrive_next_sibling": torch.full_like(next_token, -1),
        "uniform_samples": torch.rand(batch, slots, device=DEVICE, generator=generator),
        "uniform_samples_for_final_sampling": torch.rand(
            batch, device=DEVICE, generator=generator
        ),
        "target_probs": target_rows.expand(batch, -1, -1).contiguous(),
        "draft_probs": torch.zeros(batch, slots, vocab, device=DEVICE),
    }
    reference_rows = 256
    expected, _ = _run({k: v[:reference_rows].cpu() for k, v in inputs.items()}, slots)
    actual = dict(inputs, **_outputs(batch, slots, slots, DEVICE))
    tree_speculative_sampling_target_only_triton(**actual)
    for name in ("accept_index", "accept_token_num", "draft_probs"):
        assert torch.equal(actual[name][:reference_rows].cpu(), expected[name]), name
    assert torch.equal(
        actual["predicts"][: reference_rows * slots].cpu(), expected["predicts"]
    )

    lengths = actual["accept_token_num"].long()
    # With retrive_index = arange, slot k of a request holds the token after node k.
    committed = actual["predicts"].view(batch, slots).long()
    accept_probs = target_rows[:-1].gather(1, drafts).T  # [batch, slots - 1]
    reach = torch.ones(batch, device=DEVICE)
    for slot in range(slots):
        if slot > 0:
            reach = reach * accept_probs[:, slot - 1]
            observed = (lengths >= slot).float().mean().item()
            expected_reach = reach.mean().item()
            std = (expected_reach * (1 - expected_reach) / batch) ** 0.5
            assert abs(observed - expected_reach) < 5 * std + 1e-3, slot
        reached = lengths >= slot
        count = int(reached.sum())
        freqs = torch.bincount(committed[reached, slot], minlength=vocab) / count
        std = (target_rows[slot] * (1 - target_rows[slot]) / count).sqrt()
        assert (freqs - target_rows[slot]).abs().le(5 * std + 1e-3).all(), slot


def test_graph_replay_matches_eager():
    bs, num_nodes = 8, 6
    inputs = _make_inputs(
        bs=bs, num_nodes=num_nodes, vocab=9001, chain=True, random_draft=False, seed=5
    )
    static = {k: v.to(DEVICE) for k, v in inputs.items()}
    static.update(_outputs(bs, num_nodes, num_nodes, DEVICE))
    initial_draft = static["draft_probs"].clone()
    # Compile outside the capture.
    tree_speculative_sampling_target_only_triton(**static)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tree_speculative_sampling_target_only_triton(**static)

    generator = torch.Generator(device=DEVICE).manual_seed(9)
    for _ in range(3):
        static["uniform_samples"].copy_(
            torch.rand(bs, num_nodes, device=DEVICE, generator=generator)
        )
        static["uniform_samples_for_final_sampling"].copy_(
            torch.rand(bs, device=DEVICE, generator=generator)
        )
        static["draft_probs"].copy_(initial_draft)
        for name in ("predicts", "accept_index", "accept_token_num"):
            static[name].fill_(-1)
        eager = {k: v.clone() for k, v in static.items()}
        tree_speculative_sampling_target_only_triton(**eager)
        graph.replay()
        for name in ("predicts", "accept_index", "accept_token_num", "draft_probs"):
            assert torch.equal(static[name], eager[name]), name


@pytest.mark.skipif(torch.version.cuda is None, reason="needs the CUDA JIT kernel")
@pytest.mark.parametrize("chain", [True, False])
def test_matches_cuda_kernel(chain):
    from sglang.kernels.ops.speculative.sampling import (
        tree_speculative_sampling_target_only,
    )

    inputs = _make_inputs(
        bs=32, num_nodes=6, vocab=9001, chain=chain, random_draft=False, seed=11
    )
    results = []
    for kernel in (
        tree_speculative_sampling_target_only,
        tree_speculative_sampling_target_only_triton,
    ):
        run = {k: v.to(DEVICE) for k, v in inputs.items()}
        run.update(_outputs(32, 6, 6, DEVICE))
        kernel(**run, threshold_single=1.0, threshold_acc=1.0, deterministic=True)
        results.append(run)
    for name in ("predicts", "accept_index", "accept_token_num", "draft_probs"):
        assert torch.equal(results[0][name], results[1][name]), name


@pytest.mark.skipif(torch.version.hip is None, reason="ROCm dispatch")
def test_rocm_exports_triton_kernel():
    from sglang.kernels.ops.speculative import verify_sampling
    from sglang.srt.speculative.dflash_utils import (
        is_dflash_sampling_verify_available,
    )

    assert (
        verify_sampling.tree_speculative_sampling_target_only
        is tree_speculative_sampling_target_only_triton
    )
    assert is_dflash_sampling_verify_available()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
