import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dflash_utils import build_speculative_verify_target_probs
from sglang.srt.speculative.eagle_utils import _apply_joint_top_k_top_p
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

if torch.version.cuda is not None:
    from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

    from sglang.kernels.ops.speculative.sampling import (
        tree_speculative_sampling_target_only,
    )
    from sglang.srt.speculative.uno_utils import _build_sparse_target_support_tensors
else:
    if torch.version.hip is not None:
        # ROCm resolves the name to the Triton port of the CUDA kernel.
        from sglang.kernels.ops.speculative.verify_sampling import (
            tree_speculative_sampling_target_only,
        )
    else:
        from sgl_kernel import tree_speculative_sampling_target_only

    from sglang.kernels.ops.sampling.renorm_triton import (
        top_k_renorm_probs_triton as top_k_renorm_probs,
    )
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_p_renorm_probs_triton as top_p_renorm_probs,
    )

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")
# backend-specific: ROCm resolves the kernel to its Triton port
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

test_cases = [
    (
        1,
        1,
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
    (
        0,  # threshold_single
        0,  # threshold_acc
        [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18],
        [[0, 3, 4, 5], [6, 10, 11, -1]],
        [3, 2],
    ),
]


@pytest.mark.parametrize(
    "threshold_single, threshold_acc, expected_predicts, expected_accept_index, expected_accept_token_num",
    test_cases,
)
def test_tree_speculative_sampling_target_only(
    threshold_single,
    threshold_acc,
    expected_predicts,
    expected_accept_index,
    expected_accept_token_num,
):
    """
    Tests the tree_speculative_sampling_target_only function using Pytest parameterization.
    """
    device = "cuda"

    candidates = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_token = torch.tensor(
        [
            [1, 2, -1, 4, 5, -1],
            [4, 2, 3, -1, 5, -1],
        ],
        dtype=torch.int64,
        device=device,
    )
    retrive_next_sibling = torch.tensor(
        [
            [-1, 3, -1, -1, -1, -1],
            [-1, -1, -1, -1, 1, -1],
        ],
        dtype=torch.int64,
        device=device,
    )

    target_logits = torch.full((2, 6, 20), 1, dtype=torch.float32, device=device)
    target_logits[0, 0, 3] = 10
    target_logits[0, 3, 4] = 10
    target_logits[0, 4, 5] = 10
    target_logits[1, 0, 11] = 10
    target_logits[1, 4, 12] = 10

    for i in range(target_logits.shape[0]):
        for j in range(target_logits.shape[1]):
            if torch.max(target_logits[i, j]) < 10:
                target_logits[i, j, 18] = 10

    temperatures = torch.tensor([0.01, 0.01], dtype=torch.float32, device=device)
    bs, num_draft_tokens = candidates.shape
    num_spec_step = len(expected_accept_index[0])
    predict_shape = (len(expected_predicts),)

    predicts = torch.full(predict_shape, -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.full((bs,), 0, dtype=torch.int32, device=device)

    expanded_temperature = temperatures.unsqueeze(1).unsqueeze(1)
    target_probs = F.softmax(target_logits / expanded_temperature, dim=-1)
    draft_probs = torch.full_like(target_probs, 0, dtype=torch.float32, device=device)
    coins = torch.rand(bs, num_draft_tokens, device=device, dtype=torch.float32)
    coins_for_final_sampling = torch.rand(bs, device=device).to(torch.float32)

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=coins_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )

    assert predicts.tolist() == expected_predicts, (
        f"Predicts mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )
    assert accept_index.tolist() == expected_accept_index, (
        f"Accept index mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )
    assert accept_token_num.tolist() == expected_accept_token_num, (
        f"Accept token num mismatch for thresholds ({threshold_single}, {threshold_acc})"
    )


@pytest.mark.parametrize(
    "candidate_tokens,target_probabilities,coin,threshold_single,expected_token,expected_accept_token_num",
    [
        ([2], [0.0, 1.0, 0.0], 0.0, 0.0, 1, 0),
        ([2], [0.0, 1.0, 0.0], 0.0, 1.0, 1, 0),
        ([1, 2, 3], [0.0, 0.25, 0.0, 0.75], 0.25, 1.0, 3, 1),
        ([1], [0.0, 0.25, 0.75], 0.0, 1.0, 1, 1),
        ([1], [0.0, 1.0], 1.0 - 2**-24, 1.0, 1, 1),
    ],
    ids=[
        "zero-mass-zero-threshold",
        "zero-mass-default-threshold",
        "interior-boundary",
        "lower-endpoint",
        "upper-endpoint",
    ],
)
def test_target_only_sampling_cdf_boundaries(
    candidate_tokens,
    target_probabilities,
    coin,
    threshold_single,
    expected_token,
    expected_accept_token_num,
):
    num_candidates = len(candidate_tokens)
    candidates = torch.tensor(
        [[0, *candidate_tokens]], dtype=torch.int64, device="cuda"
    )
    num_draft_tokens = candidates.shape[1]
    retrive_index = torch.arange(
        num_draft_tokens, dtype=torch.int64, device="cuda"
    ).unsqueeze(0)
    retrive_next_token = torch.full_like(retrive_index, -1)
    retrive_next_token[0, 0] = 1
    retrive_next_sibling = torch.full_like(retrive_index, -1)
    if num_candidates > 1:
        retrive_next_sibling[0, 1:num_candidates] = torch.arange(
            2, num_candidates + 1, dtype=torch.int64, device="cuda"
        )

    target_probs = torch.zeros(
        (1, num_draft_tokens, len(target_probabilities)),
        dtype=torch.float32,
        device="cuda",
    )
    target_probs[0, 0] = torch.tensor(
        target_probabilities, dtype=torch.float32, device="cuda"
    )
    target_probs[0, 1:, 0] = 1.0
    draft_probs = torch.zeros_like(target_probs)
    predicts = torch.full((num_draft_tokens,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((1, 2), -1, dtype=torch.int32, device="cuda")
    accept_token_num = torch.zeros((1,), dtype=torch.int32, device="cuda")
    coins = torch.zeros((1, num_draft_tokens), dtype=torch.float32, device="cuda")
    coins[0, 0] = coin

    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=candidates,
        retrive_index=retrive_index,
        retrive_next_token=retrive_next_token,
        retrive_next_sibling=retrive_next_sibling,
        uniform_samples=coins,
        uniform_samples_for_final_sampling=torch.zeros(
            (1,), dtype=torch.float32, device="cuda"
        ),
        target_probs=target_probs,
        draft_probs=draft_probs,
        threshold_single=threshold_single,
        threshold_acc=1.0,
        deterministic=True,
    )

    assert predicts[0].item() == expected_token
    assert accept_token_num.item() == expected_accept_token_num


@pytest.fixture
def joint_top_k_top_p_case(request):
    override = get_context().override_server_args(tp_size=1)
    override.install()
    request.addfinalizer(override.restore)
    original_probs = torch.tensor(
        [[0.35, 0.25, 0.20, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03]],
        device="cuda",
    )
    sampling_info = SimpleNamespace(
        temperatures=torch.ones((1, 1), device="cuda"),
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        top_ks=torch.tensor([4], dtype=torch.int32, device="cuda"),
        top_ps=torch.tensor([0.82], device="cuda"),
    )
    expected_probs = torch.zeros_like(original_probs)
    expected_probs[0, :4] = original_probs[0, :4] / original_probs[0, :4].sum()
    return original_probs, sampling_info, expected_probs


def test_eagle_preserves_original_top_p_nucleus(joint_top_k_top_p_case):
    """Production renormalizers must retain token 3: its original preceding
    mass is 0.80 < top_p, while its top-k-normalized mass exceeds top_p."""
    original_probs, sampling_info, expected_probs = joint_top_k_top_p_case
    filtered_probs = _apply_joint_top_k_top_p(
        original_probs,
        sampling_info,
        draft_token_num=1,
        top_k_renorm_prob=top_k_renorm_probs,
        top_p_renorm_prob=top_p_renorm_probs,
    )
    torch.testing.assert_close(filtered_probs, expected_probs)


@pytest.mark.parametrize("use_sparse_topk", [False, True], ids=["dense", "sparse"])
def test_speculative_target_preserves_original_top_p_nucleus(
    joint_top_k_top_p_case, use_sparse_topk
):
    """Dense and sparse targets must retain token 3 using its original
    preceding mass of 0.80, not its top-k-normalized mass of 0.80 / 0.85."""
    original_probs, sampling_info, expected_probs = joint_top_k_top_p_case
    filtered_probs = build_speculative_verify_target_probs(
        next_token_logits=original_probs.log(),
        sampling_info=sampling_info,
        draft_token_num=1,
        bs=1,
        max_top_k=4,
        uniform_top_k_value=4,
        use_sparse_topk=use_sparse_topk,
    ).view_as(original_probs)
    torch.testing.assert_close(filtered_probs, expected_probs)


@pytest.mark.skipif(torch.version.cuda is None, reason="UNO compact path requires CUDA")
def test_uno_compact_preserves_original_top_p_nucleus(joint_top_k_top_p_case):
    """Compact support must retain token 3 using its original preceding
    mass of 0.80, not its top-k-normalized mass of 0.80 / 0.85."""
    original_probs, sampling_info, expected_probs = joint_top_k_top_p_case
    support_ids, support_probs = _build_sparse_target_support_tensors(
        next_token_logits=original_probs.log(),
        temperatures=sampling_info.temperatures,
        top_ks=sampling_info.top_ks,
        top_ps=sampling_info.top_ps,
        batch_size=1,
        forward_width=1,
        max_top_k=4,
    )
    torch.testing.assert_close(
        support_ids,
        torch.tensor([[[0, 1, 2, 3]]], dtype=support_ids.dtype, device="cuda"),
    )
    torch.testing.assert_close(support_probs, expected_probs[:, :4].view(1, 1, 4))


rocm_only = pytest.mark.skipif(torch.version.hip is None, reason="ROCm sampled verify")
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


def _make_chain_inputs(*, bs, num_nodes, vocab, seed):
    """Random draft chains (node 0 is the root) with random draft rows. Candidates
    mostly come from the parent's target support, so levels accept as well as
    reject."""
    generator = torch.Generator().manual_seed(seed)
    next_token = torch.full((bs, num_nodes), -1, dtype=torch.int64)
    next_token[:, :-1] = torch.arange(1, num_nodes)
    target = _dyadic_rows((bs, num_nodes), vocab, generator)
    draft = _dyadic_rows((bs, num_nodes), vocab, generator)
    candidates = torch.zeros((bs, num_nodes), dtype=torch.int64)
    for b in range(bs):
        for node in range(1, num_nodes):
            support = target[b, node - 1].nonzero().flatten()
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
        "retrive_next_sibling": torch.full((bs, num_nodes), -1, dtype=torch.int64),
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
    tree_speculative_sampling_target_only(
        **actual,
        threshold_single=threshold_single,
        threshold_acc=threshold_acc,
        deterministic=True,
    )
    return expected, actual


def _assert_same(expected, actual):
    for name in ("predicts", "accept_index", "accept_token_num", "draft_probs"):
        assert torch.equal(actual[name].cpu(), expected[name]), name


@rocm_only
def test_matches_reference():
    """Random chains over a vocabulary of several 4096-wide blocks with fractional
    thresholds and random draft rows; the base known-answer cases cover one block
    and thresholds 1 and 0."""
    inputs = _make_chain_inputs(bs=16, num_nodes=6, vocab=9001, seed=9003)
    # A walk truncated below the leaves samples the bonus token after its last
    # slot, where the draft row must not be read.
    expected, actual = _run(inputs, 3, threshold_single=0.5, threshold_acc=0.25)
    _assert_same(expected, actual)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
