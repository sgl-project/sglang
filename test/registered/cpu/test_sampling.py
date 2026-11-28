import sys

import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="extra-test-cpu-intel")


@pytest.mark.parametrize("positions_dtype", [torch.int64, torch.uint64])
def test_murmur_hash32_cpu_known_values(positions_dtype):
    seed = torch.tensor([0, 1, 42, 0x123456789ABCDEF0], dtype=torch.uint64)

    positions = torch.tensor([0, 7, 123, 456], dtype=positions_dtype)

    col_indices = torch.tensor([0, 1, 2, 17], dtype=torch.int64)

    actual = murmur_hash32(seed, positions, col_indices)

    expected = torch.tensor(
        [
            [2167721464, 10027521, 2423355346, 2203067026],
            [3755322398, 4196701286, 1002451629, 183234019],
            [772287619, 548237471, 2740678348, 3656549299],
            [3746406971, 2891010872, 104055988, 3372550890],
        ],
        dtype=torch.uint32,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("positions_dtype", [torch.int64, torch.uint64])
def test_murmur_hash32_cpu_shape_and_dtype(positions_dtype):
    seed = torch.tensor([1, 2, 3], dtype=torch.uint64)

    positions = torch.tensor([10, 20, 30], dtype=positions_dtype)

    col_indices = torch.arange(128, dtype=torch.int64)

    actual = murmur_hash32(
        seed,
        positions,
        col_indices,
    )

    assert actual.device.type == "cpu"
    assert actual.dtype == torch.uint32
    assert actual.shape == (3, 128)


def test_watermark_state_folds_wide_multimodal_padding_ids():
    from sglang.srt.sampling.watermarking.gumbel.algorithm import hash_contexts
    from sglang.srt.sampling.watermarking.gumbel.state import WatermarkState

    # _compute_pad_values() builds padding ids from a 62-bit mask, which does
    # not fit the int32 context buffer used by the watermark state.
    wide_id = (1 << 62) | 0x2A5B
    state = WatermarkState(
        max_num_reqs=1,
        context_window=4,
        max_contexts_per_req=2,
        device="cpu",
    )
    state.init_from_prompt(torch.tensor([0], dtype=torch.int32), [[wide_id, 123]])

    folded = ((wide_id & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
    assert state.token_ids[0, :2].tolist() == [folded, 123]
    assert state.lengths[0].item() == 2

    # The watermark hashes the low 32 bits of each id, so folding keeps the
    # context hash identical to hashing the original wide id.
    lengths = torch.tensor([2], dtype=torch.int32)
    original_hash = hash_contexts(
        torch.tensor([[wide_id, 123]], dtype=torch.int64), lengths
    )
    stored_hash = hash_contexts(
        state.token_ids[0, :2].to(torch.int64).view(1, -1), lengths
    )
    assert original_hash.item() == stored_hash.item()


def test_watermark_state_folds_negative_int32_wide_ids_across_rollover():
    from sglang.srt.sampling.watermarking.gumbel.algorithm import hash_contexts
    from sglang.srt.sampling.watermarking.gumbel.state import WatermarkState

    # The low 32 bits set the int32 sign bit (0x8000007B == -2147483525), so
    # the folded value is negative on this platform.
    wide_id = (1 << 62) | 0x8000007B
    folded = ((wide_id & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
    assert folded == -2147483525

    context_window = 4
    state = WatermarkState(
        max_num_reqs=1,
        context_window=context_window,
        max_contexts_per_req=2,
        device="cpu",
    )
    state.init_from_prompt(torch.tensor([0], dtype=torch.int32), [[7, wide_id, 123]])
    state.append(torch.tensor([0], dtype=torch.int32), torch.tensor([10]))
    state.append(torch.tensor([0], dtype=torch.int32), torch.tensor([11]))

    # Two appends roll the ring buffer past the window; the tail now starts at
    # write position 1 and the folded negative id is still present.
    contexts, lengths = state.contexts_tail(torch.tensor([0], dtype=torch.int32))
    assert lengths.item() == context_window
    assert contexts[0].tolist() == [folded, 123, 10, 11]

    # Folding still preserves the original low-32-bit hash after rollover.
    lengths = torch.tensor([context_window], dtype=torch.int32)
    original_hash = hash_contexts(
        torch.tensor([[wide_id, 123, 10, 11]], dtype=torch.int64), lengths
    )
    stored_hash = hash_contexts(contexts.to(torch.int64), lengths)
    assert original_hash.item() == stored_hash.item()


def test_watermark_state_folds_negative_wide_id_in_speculative_contexts():
    from sglang.srt.sampling.watermarking.gumbel.algorithm import hash_contexts
    from sglang.srt.sampling.watermarking.gumbel.state import WatermarkState

    wide_id = (1 << 62) | 0x8000007B
    folded = ((wide_id & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
    assert folded == -2147483525

    context_window = 9
    state = WatermarkState(
        max_num_reqs=1,
        context_window=context_window,
        max_contexts_per_req=2,
        device="cpu",
    )
    req_pool_indices = torch.tensor([0], dtype=torch.int32)
    state.init_from_prompt(req_pool_indices, [[7, wide_id, 123]])
    for token in range(10, 16):
        state.append(req_pool_indices, torch.tensor([token]))

    # Six appends fill the ring and wrap the write position once.
    tail, tail_lengths = state.contexts_tail(req_pool_indices)
    assert tail_lengths.item() == context_window
    assert tail[0].tolist() == [7, folded, 123, 10, 11, 12, 13, 14, 15]

    # Row 0 keeps only the committed tail; row 1 chains draft token 21, which
    # pushes the oldest committed token out of the window but keeps the
    # folded wide id.
    contexts, context_lengths = state.speculative_contexts(
        req_pool_indices,
        torch.tensor([[20, 21]], dtype=torch.int32),
        torch.tensor([[True, False], [True, True]], dtype=torch.bool),
        torch.tensor([[9, 10]], dtype=torch.int32),
        2,
        False,
        torch.tensor([context_window], dtype=torch.int32),
    )
    assert context_lengths.tolist() == [context_window, context_window]
    assert contexts[0].tolist() == [7, folded, 123, 10, 11, 12, 13, 14, 15]
    assert contexts[1].tolist() == [folded, 123, 10, 11, 12, 13, 14, 15, 21]

    # Every speculative row hashes exactly as if the original wide id had been
    # stored instead of its folded low-32-bit view.
    for row in range(2):
        stored = contexts[row].to(torch.int64).view(1, -1)
        lengths = context_lengths[row : row + 1]
        original = torch.where(
            stored == folded, torch.full_like(stored, wide_id), stored
        )
        assert (
            hash_contexts(stored, lengths).item()
            == hash_contexts(original, lengths).item()
        )

    # Accept only the first draft token: the ring wraps once more and the
    # folded wide id remains in the committed tail.
    state.append_speculative(
        req_pool_indices,
        torch.tensor([[20, 21]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    tail, tail_lengths = state.contexts_tail(req_pool_indices)
    assert tail_lengths.item() == context_window
    assert tail[0].tolist() == [folded, 123, 10, 11, 12, 13, 14, 15, 20]
    stored = tail.to(torch.int64)
    original = torch.where(stored == folded, torch.full_like(stored, wide_id), stored)
    lengths = torch.tensor([context_window], dtype=torch.int32)
    assert (
        hash_contexts(stored, lengths).item() == hash_contexts(original, lengths).item()
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
