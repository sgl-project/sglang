import sys

import pytest
import torch

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.kernels.ops.sampling.textseal_selector import (
    select_watermark_tokens_triton,
)
from sglang.srt.sampling.watermark import (
    _hash_contexts,
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
