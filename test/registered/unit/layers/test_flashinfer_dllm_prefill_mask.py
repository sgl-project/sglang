import unittest
from math import log, sqrt

import torch

from sglang.srt.dllm.attention import build_dllm_prefill_blockwise_mask
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

try:
    from flashinfer import (
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state

    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False

register_cpu_ci(est_time=1, suite="base-c-test-cpu")
register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _dense_blockwise_reference(q, k, v, prefix_len, block_size):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        repeats = num_q_heads // num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / sqrt(q.shape[-1])
    query_positions = prefix_len + torch.arange(q.shape[0], device=q.device)
    key_positions = torch.arange(k.shape[0], device=q.device)
    query_blocks = torch.div(query_positions, block_size, rounding_mode="floor")
    key_blocks = torch.div(key_positions, block_size, rounding_mode="floor")
    mask = key_blocks[None, :] <= query_blocks[:, None]
    scores.masked_fill_(~mask[None, :, :], -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", probs, v.float())
    # FlashInfer 0.6.x returns LSE in log2 scale.
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1).contiguous() / log(2.0)
    return output, lse


def _make_paged_cache(k, v, page_size):
    num_tokens, num_kv_heads, head_dim = k.shape
    num_pages = (num_tokens + page_size - 1) // page_size
    cache = torch.zeros(
        num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=k.dtype,
        device=k.device,
    )
    for page_id in range(num_pages):
        start = page_id * page_size
        end = min(start + page_size, num_tokens)
        cache[page_id, 0, : end - start] = k[start:end]
        cache[page_id, 1, : end - start] = v[start:end]
    kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=k.device)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=k.device)
    last_page_len = torch.tensor(
        [num_tokens - (num_pages - 1) * page_size],
        dtype=torch.int32,
        device=k.device,
    )
    return cache, kv_indptr, kv_indices, last_page_len


def _run_ragged_prefix_cascade(q, k_ext, v_ext, k_prefix, v_prefix, block_size):
    device = q.device
    extend_len = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k_ext.shape[1]
    head_dim = q.shape[-1]
    qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
    mask = build_dllm_prefill_blockwise_mask(
        [k_prefix.shape[0]],
        [extend_len],
        block_size,
        device,
        include_prefix=False,
    )
    assert mask is not None

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    ragged = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend="fa2")
    ragged.begin_forward(
        qo_indptr,
        qo_indptr,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q_data_type=q.dtype,
        custom_mask=mask,
    )
    current_out, current_lse = ragged.forward_return_lse(q, k_ext, v_ext, causal=False)

    page_size = 16
    cache, kv_indptr, kv_indices, last_page_len = _make_paged_cache(
        k_prefix, v_prefix, page_size
    )
    paged = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    paged.begin_forward(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q.dtype,
    )
    prefix_out, prefix_lse = paged.forward_return_lse(q, cache, causal=False)
    return merge_state(current_out, current_lse, prefix_out, prefix_lse)


class TestDllmPrefillBlockwiseMask(unittest.TestCase):
    def test_masked_and_unmasked_select_different_wrappers(self):
        backend = object.__new__(FlashInferAttnBackend)
        unmasked_wrapper = object()
        masked_wrapper = object()
        backend.prefill_wrapper_ragged = unmasked_wrapper
        backend.prefill_wrapper_ragged_custom_mask = masked_wrapper

        self.assertIs(backend._select_prefill_ragged_wrapper(None), unmasked_wrapper)
        self.assertIs(
            backend._select_prefill_ragged_wrapper(torch.ones(1, dtype=torch.bool)),
            masked_wrapper,
        )

    def test_ragged_multi_block_mask(self):
        mask = build_dllm_prefill_blockwise_mask(
            [0], [6], 2, torch.device("cpu"), include_prefix=False
        )
        expected = torch.tensor(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(mask.view(6, 6), expected)

    def test_paged_mask_includes_fully_visible_prefix(self):
        mask = build_dllm_prefill_blockwise_mask(
            [4], [4], 2, torch.device("cpu"), include_prefix=True
        )
        expected = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(mask.view(4, 8), expected)

    def test_heterogeneous_batch_contains_all_request_segments(self):
        mask = build_dllm_prefill_blockwise_mask(
            [0, 4], [2, 4], 2, torch.device("cpu"), include_prefix=False
        )
        self.assertEqual(mask.numel(), 2 * 2 + 4 * 4)
        self.assertTrue(mask[:4].all())
        expected_second = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(mask[4:].view(4, 4), expected_second)

    def test_single_block_batch_needs_no_mask(self):
        mask = build_dllm_prefill_blockwise_mask(
            [0, 8], [2, 1], 2, torch.device("cpu"), include_prefix=False
        )
        self.assertIsNone(mask)

    def test_unaligned_prefix_crossing_block_needs_mask(self):
        mask = build_dllm_prefill_blockwise_mask(
            [1], [2], 2, torch.device("cpu"), include_prefix=False
        )
        expected = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        torch.testing.assert_close(mask.view(2, 2), expected)

    def test_invalid_lengths(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            build_dllm_prefill_blockwise_mask(
                [0], [2, 4], 2, torch.device("cpu"), include_prefix=False
            )

    @unittest.skipUnless(
        _HAS_FLASHINFER and torch.cuda.is_available(),
        "FlashInfer and CUDA are required",
    )
    def test_flashinfer_ragged_output_matches_dense_reference(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        extend_len = 128
        block_size = 32
        num_q_heads = 16
        num_kv_heads = 4
        head_dim = 128
        dtype = torch.bfloat16

        mask = build_dllm_prefill_blockwise_mask(
            [0], [extend_len], block_size, device, include_prefix=False
        )
        qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
        q = torch.randn(extend_len, num_q_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(extend_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn_like(k)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        # Plan an unmasked auto wrapper first, matching server warmup order.
        # The masked path must use a separate FA2 wrapper and remain unaffected.
        unmasked_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "NHD", backend="auto"
        )
        unmasked_indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
        unmasked_wrapper.begin_forward(
            unmasked_indptr,
            unmasked_indptr,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_data_type=q.dtype,
        )
        unmasked_wrapper.forward(q[:2], k[:2], v[:2], causal=False)

        wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, "NHD", backend="fa2")
        wrapper.begin_forward(
            qo_indptr,
            qo_indptr,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_data_type=q.dtype,
            custom_mask=mask,
        )
        self.assertEqual(wrapper._backend, "fa2")
        actual = wrapper.forward(q, k, v, causal=False)

        expected, _ = _dense_blockwise_reference(
            q, k, v, prefix_len=0, block_size=block_size
        )
        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)

        # Re-plan the unmasked wrapper after the masked run; its backend state
        # must remain independent from the dedicated masked wrapper.
        unmasked_wrapper.begin_forward(
            unmasked_indptr,
            unmasked_indptr,
            num_q_heads,
            num_kv_heads,
            head_dim,
            q_data_type=q.dtype,
        )
        unmasked_wrapper.forward(q[:2], k[:2], v[:2], causal=False)

    @unittest.skipUnless(
        _HAS_FLASHINFER and torch.cuda.is_available(),
        "FlashInfer and CUDA are required",
    )
    def test_prefix_cascade_matches_dense_reference(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        prefix_len, extend_len, block_size = 64, 128, 32
        num_heads, head_dim = 4, 64
        q = torch.randn(
            extend_len, num_heads, head_dim, dtype=torch.float16, device=device
        )
        k_ext = torch.randn_like(q)
        v_ext = torch.randn_like(q)
        k_prefix = torch.randn(
            prefix_len, num_heads, head_dim, dtype=q.dtype, device=device
        )
        v_prefix = torch.randn_like(k_prefix)

        actual, actual_lse = _run_ragged_prefix_cascade(
            q, k_ext, v_ext, k_prefix, v_prefix, block_size
        )
        expected, expected_lse = _dense_blockwise_reference(
            q,
            torch.cat([k_prefix, k_ext]),
            torch.cat([v_prefix, v_ext]),
            prefix_len,
            block_size,
        )
        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            actual_lse.float(), expected_lse, rtol=2e-2, atol=2e-2
        )

    @unittest.skipUnless(
        _HAS_FLASHINFER and torch.cuda.is_available(),
        "FlashInfer and CUDA are required",
    )
    def test_future_block_isolation(self):
        torch.manual_seed(2)
        device = torch.device("cuda")
        prefix_len, block_size, num_blocks = 64, 32, 4
        extend_len = block_size * num_blocks
        q = torch.randn(extend_len, 4, 64, dtype=torch.float16, device=device)
        k_ext = torch.randn_like(q)
        v_ext = torch.randn_like(q)
        k_prefix = torch.randn(prefix_len, 4, 64, dtype=q.dtype, device=device)
        v_prefix = torch.randn_like(k_prefix)

        before, _ = _run_ragged_prefix_cascade(
            q, k_ext, v_ext, k_prefix, v_prefix, block_size
        )
        last_block_start = extend_len - block_size
        k_mutated = k_ext.clone()
        v_mutated = v_ext.clone()
        k_mutated[last_block_start:] = torch.randn_like(k_mutated[last_block_start:])
        v_mutated[last_block_start:] = torch.randn_like(v_mutated[last_block_start:])
        after, _ = _run_ragged_prefix_cascade(
            q, k_mutated, v_mutated, k_prefix, v_prefix, block_size
        )

        torch.testing.assert_close(
            before[:last_block_start], after[:last_block_start], rtol=0, atol=0
        )
        self.assertGreater(
            (before[last_block_start:].float() - after[last_block_start:].float())
            .abs()
            .max()
            .item(),
            1e-3,
        )

    @unittest.skipUnless(
        _HAS_FLASHINFER and torch.cuda.is_available(),
        "FlashInfer and CUDA are required",
    )
    def test_paged_fallback_matches_dense_reference(self):
        torch.manual_seed(3)
        device = torch.device("cuda")
        prefix_len, extend_len, block_size = 64, 128, 32
        total_len, num_heads, head_dim = prefix_len + extend_len, 4, 64
        q = torch.randn(
            extend_len, num_heads, head_dim, dtype=torch.float16, device=device
        )
        k = torch.randn(total_len, num_heads, head_dim, dtype=q.dtype, device=device)
        v = torch.randn_like(k)
        mask = build_dllm_prefill_blockwise_mask(
            [prefix_len],
            [extend_len],
            block_size,
            device,
            include_prefix=True,
        )
        self.assertIsNotNone(mask)

        page_size = 16
        cache, kv_indptr, kv_indices, last_page_len = _make_paged_cache(k, v, page_size)
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        qo_indptr = torch.tensor([0, extend_len], dtype=torch.int32, device=device)
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
        wrapper.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
            q_data_type=q.dtype,
            custom_mask=mask,
        )
        actual = wrapper.forward(q, cache, causal=False)
        expected, _ = _dense_blockwise_reference(
            q, k, v, prefix_len=prefix_len, block_size=block_size
        )
        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
