import sys

import pytest
import torch

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.kernels.ops.sampling.textseal_selector import (
    force_watermark_tokens_triton,
    prepare_watermark_contexts_triton,
    select_watermark_tokens_triton,
)
from sglang.srt.sampling.watermark import (
    WatermarkState,
    _hash_contexts,
    _truncate_probabilities,
    _watermark_hash32_torch,
    select_watermark_tokens_torch,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_hash_matches_detector_vectors():
    contexts = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.int64)
    lengths = torch.tensor([4, 4], dtype=torch.int32)
    keys = torch.tensor(
        [0x0123456789ABCDEF, 0xFEDCBA9876543210 - (1 << 64)],
        dtype=torch.int64,
    )
    token_ids = torch.tensor([0, 1, 17, 8191, 8192, 16396], dtype=torch.int64)
    expected_context_hashes = torch.tensor([1145416960, 47748951])
    expected = torch.tensor(
        [
            [1293512163, 858402549, 2305555132, 2450309311, 227333036, 2684237202],
            [221641642, 4015047244, 843143906, 1076944989, 3882127500, 2413234263],
        ],
        dtype=torch.int64,
    )

    context_hashes = _hash_contexts(contexts, lengths)
    torch.testing.assert_close(context_hashes, expected_context_hashes)
    torch.testing.assert_close(
        _watermark_hash32_torch(keys, context_hashes, token_ids), expected
    )
    actual = murmur_hash32(keys.cuda(), context_hashes.cuda(), token_ids.cuda())
    torch.testing.assert_close(actual.cpu().to(torch.int64), expected)


def test_selector_matches_torch_across_split_boundaries():
    generator = torch.Generator().manual_seed(7)
    batch_size = 5
    vocab_size = 16397
    probabilities = torch.rand(batch_size, vocab_size, generator=generator)
    support_lengths = torch.tensor([1, 8191, 8192, 8193, vocab_size])
    support = torch.arange(vocab_size).view(1, -1) < support_lengths.view(-1, 1)
    probabilities = torch.where(support, probabilities, 0.0)
    probabilities /= probabilities.sum(dim=-1, keepdim=True)
    context_hashes = torch.tensor([0, 1, 2**31 - 1, 2**32 - 1, 1145416960])
    keys = torch.tensor([0, 1, -1, -(2**63), 0x0123456789ABCDEF])

    expected = select_watermark_tokens_torch(probabilities, context_hashes, keys)
    actual = select_watermark_tokens_triton(
        probabilities.cuda(), context_hashes.cuda(), keys.cuda()
    ).cpu()

    torch.testing.assert_close(actual.to(torch.int64), expected)


@pytest.mark.parametrize(
    ("dtype", "vocab_size"),
    [
        (torch.bfloat16, 8191),
        (torch.bfloat16, 8192),
        (torch.bfloat16, 8193),
        (torch.bfloat16, 151936),
        (torch.float16, 16397),
        (torch.float32, 16397),
    ],
)
def test_fused_force_matches_torch_truncation(dtype, vocab_size):
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn((5, vocab_size), generator=generator, device="cuda").to(dtype)
    temperatures = torch.tensor([[0.7], [1.0], [1.3], [0.9], [1.1]], device="cuda")
    top_ks = torch.tensor(
        [1, 7, min(100, vocab_size), vocab_size, vocab_size],
        dtype=torch.int32,
        device="cuda",
    )
    top_ps = torch.tensor([1.0, 0.95, 0.5, 0.01, 1.0], device="cuda")
    min_ps = torch.tensor([0.0, 0.0, 0.05, 0.0, 0.2], device="cuda")
    context_hashes = torch.tensor(
        [0, 1, 2**31 - 1, 2**32 - 1, 1145416960],
        dtype=torch.int64,
        device="cuda",
    )
    keys = torch.tensor(
        [0, 1, -1, -(2**63), 0x0123456789ABCDEF],
        dtype=torch.int64,
        device="cuda",
    )
    eligible = torch.tensor([False, True, True, True, True], device="cuda")

    expected_logits = logits.clone()
    probabilities = _truncate_probabilities(
        expected_logits, temperatures, top_ks, top_ps, min_ps
    )
    rows = eligible.nonzero(as_tuple=True)[0]
    expected = torch.full((5,), -1, dtype=torch.int64, device="cuda")
    expected[rows] = select_watermark_tokens_torch(
        probabilities[rows].float(), context_hashes[rows], keys[rows]
    )
    expected_logits[rows] = -torch.inf
    expected_logits[rows, expected[rows]] = 0.0

    actual_logits = logits.clone()
    actual = force_watermark_tokens_triton(
        actual_logits,
        context_hashes,
        eligible,
        temperatures,
        top_ks,
        top_ps,
        min_ps,
        keys,
    )

    assert torch.equal(actual.to(torch.int64), expected)
    assert torch.equal(actual_logits, expected_logits)


def test_fused_context_state_matches_torch():
    max_contexts = 2048
    reference = WatermarkState(
        max_num_reqs=8,
        context_window=4,
        max_contexts_per_req=max_contexts,
        key="0123456789abcdef",
        device="cuda",
    )
    actual = WatermarkState(
        max_num_reqs=8,
        context_window=4,
        max_contexts_per_req=max_contexts,
        key="0123456789abcdef",
        device="cuda",
    )
    req_pool_indices = torch.tensor([1, 3, 5, 7], dtype=torch.int32, device="cuda")
    tails = [[10, 11, 12, 13], [20, 21], [30, 31, 32, 33], [40, 41, 42]]
    reference.init_from_prompt(req_pool_indices, tails)
    actual.init_from_prompt(req_pool_indices, tails)
    counts = torch.tensor([0, 1, 1025, 1500], dtype=torch.int32, device="cuda")
    reference.num_watermarked_contexts[req_pool_indices.long()] = counts
    actual.num_watermarked_contexts.copy_(reference.num_watermarked_contexts)
    generator = torch.Generator(device="cuda").manual_seed(9)
    history = torch.randint(
        -(2**31),
        2**31 - 1,
        reference.watermarked_context_hashes.shape,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    reference.watermarked_context_hashes.copy_(history)
    actual.watermarked_context_hashes.copy_(history)
    context_windows = torch.tensor([4, 1, 3, 2], dtype=torch.int32, device="cuda")
    watermark_enabled = torch.tensor([True, True, True, True], device="cuda")
    top_ks = torch.tensor([64, 64, 64, 1], dtype=torch.int32, device="cuda")

    contexts, context_lengths = reference.contexts_tail(
        req_pool_indices, context_windows
    )
    expected_hashes = _hash_contexts(contexts, context_lengths)
    reference.watermarked_context_hashes[req_pool_indices[1], 0] = expected_hashes[
        1
    ].to(torch.int32)
    reference.watermarked_context_hashes[req_pool_indices[2], 1024] = expected_hashes[
        2
    ].to(torch.int32)
    actual.watermarked_context_hashes.copy_(reference.watermarked_context_hashes)
    initial_counts = reference.num_watermarked_contexts.clone()
    expected_eligible = reference._new_context_mask(
        req_pool_indices,
        expected_hashes,
        watermark_enabled & (top_ks > 1) & (context_lengths > 0),
    )
    reference._record_contexts(req_pool_indices, expected_hashes, expected_eligible)

    actual_hashes = torch.empty(4, dtype=torch.int64, device="cuda")
    actual_eligible = torch.empty(4, dtype=torch.bool, device="cuda")
    prepare_watermark_contexts_triton(
        actual.token_ids,
        actual.lengths,
        actual.write_positions,
        actual.watermarked_context_hashes,
        actual.num_watermarked_contexts,
        req_pool_indices,
        context_windows,
        watermark_enabled,
        top_ks,
        actual_hashes,
        actual_eligible,
    )

    torch.testing.assert_close(actual_hashes, expected_hashes)
    torch.testing.assert_close(actual_eligible, expected_eligible)
    torch.testing.assert_close(
        actual.num_watermarked_contexts, reference.num_watermarked_contexts
    )
    for row, pool_index in enumerate(req_pool_indices.tolist()):
        if expected_eligible[row]:
            position = initial_counts[pool_index]
            assert (
                actual.watermarked_context_hashes[pool_index, position]
                == reference.watermarked_context_hashes[pool_index, position]
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
