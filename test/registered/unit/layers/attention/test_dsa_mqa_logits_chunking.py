"""Contract tests for the MQA-logits chunk decision shared by the DSA and DSV4
indexers.

On ROCm the `[num_q x num_k]` fp32 logits tensor goes to aiter's
`fp8_mqa_logits`, which only compiles below 2 GiB, so the budget that decides
chunking is a correctness bound there and not only an out-of-memory guard.

`Indexer` decides that with a budget; `IndexerKPool` splits the query rows
around its aiter call. CPU fixtures also exercise the real paged indexer entry
to check DeepGEMM chunk assembly, native next-n rows, and chunk lifetime.
"""

import sys
import types
import weakref
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.attention.dsa import (  # noqa: E402
    dsa_indexer,
    dsa_indexer_kpool,
)
from sglang.srt.layers.attention.mqa_logits_utils import (  # noqa: E402
    MQA_LOGITS_MAX_BYTES_ROCM,
    mqa_logits_row_bytes,
    mqa_logits_should_chunk,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=16, suite="base-a-test-cpu")

CEILING = MQA_LOGITS_MAX_BYTES_ROCM
# More than any single logits tensor here needs, so it never decides a case.
HUGE_MEM_BUDGET = 64 * 2**30


def _decide(num_q, num_k, mem_budget=HUGE_MEM_BUDGET, is_hip=True):
    return mqa_logits_should_chunk(
        num_rows=num_q,
        num_cols=num_k,
        get_budget_bytes=lambda: mem_budget,
        rocm=is_hip,
    )


def test_the_ceiling_is_the_largest_logits_aiter_still_takes():
    # 16384 x 32768 x 4 bytes is exactly 2 GiB, and aiter compares `bytes <
    # 2 GiB`, so that shape has to chunk and one KV token less must not.
    assert _decide(16_384, 32_768) == (True, CEILING)
    assert _decide(16_384, 32_767) == (False, CEILING)


def test_a_smaller_memory_budget_still_wins():
    one_gib = 2**30
    assert _decide(16_384, 32_767, mem_budget=one_gib) == (True, one_gib)


def test_off_rocm_the_budget_is_untouched():
    # Elsewhere the logits go to DeepGEMM, which has no such limit.
    assert _decide(16_384, 32_768, is_hip=False) == (False, HUGE_MEM_BUDGET)


def _kpool_mqa_logits(num_q, num_k, cap_bytes):
    """Run `IndexerKPool._fp8_mqa_logits` over a stub that returns each row's
    index plus its `starts`, so a misordered reassembly or an unsliced per-row
    argument shows up in the logits."""
    rows_per_call = []

    def _kernel(q_fp8, k_fp8, k_scale, weights, starts, ends, *, clean_logits):
        rows = q_fp8.shape[0]
        assert weights.shape[0] == rows
        assert starts.shape[0] == rows
        assert ends.shape[0] == rows
        rows_per_call.append(rows)
        row_value = q_fp8[:, 0, 0] + starts
        return row_value.unsqueeze(1).expand(-1, k_fp8.shape[0]).contiguous()

    aiter_kernel_module = types.ModuleType("aiter.ops.triton.fp8_mqa_logits")
    aiter_kernel_module.fp8_mqa_logits = _kernel
    stub_modules = {
        name: types.ModuleType(name)
        for name in ("aiter", "aiter.ops", "aiter.ops.triton")
    }
    stub_modules["aiter.ops.triton.fp8_mqa_logits"] = aiter_kernel_module

    q_fp8 = torch.arange(num_q, dtype=torch.float32).view(num_q, 1, 1).expand(-1, 4, 8)
    k_fp8 = torch.zeros(num_k, 8, dtype=torch.float32)
    with (
        mock.patch.dict(sys.modules, stub_modules),
        mock.patch.object(dsa_indexer_kpool, "is_hip", lambda: True),
        mock.patch.object(dsa_indexer_kpool, "MQA_LOGITS_MAX_BYTES_ROCM", cap_bytes),
    ):
        logits = dsa_indexer_kpool.IndexerKPool._fp8_mqa_logits(
            q_fp8,
            k_fp8,
            torch.ones(num_k),
            torch.ones(num_q, 4),
            torch.arange(num_q, dtype=torch.float32) * 10,
            torch.full((num_q,), num_k, dtype=torch.int32),
            clean_logits=True,
        )
    return logits, rows_per_call


def test_kpool_splits_the_query_rows_to_stay_under_the_aiter_cap():
    """Above 2 GiB of logits the aiter kernel abort()s the process instead of
    raising, so the k-pool wrapper must never hand it a larger tensor. aiter
    pads each row to 256 columns, so the cap counts padded rows."""
    num_q, num_k = 8, 4
    row_bytes = mqa_logits_row_bytes(num_k)
    logits, rows_per_call = _kpool_mqa_logits(num_q, num_k, 3 * row_bytes + 1)

    assert rows_per_call == [3, 3, 2]
    rows = torch.arange(num_q, dtype=torch.float32)
    assert torch.equal(logits, (rows + rows * 10).unsqueeze(1).expand(-1, num_k))


class TestPagedMQAChunkAssembly(CustomTestCase):
    def test_paged_chunks_release_storage_and_preserve_query_rows(self):
        """Large speculative batches must not retain every dense logits chunk.

        The old assembly kept all chunks until a second full-size allocation.
        A weak reference checks that the previous returned tensor is released
        before the next kernel starts. Distinct row inputs expose wrong native
        next-n slicing; padded views expose incorrect copying of row strides.
        """
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        cases = [
            (ForwardMode.DECODE, 0, 1, 0),
            (ForwardMode.DECODE, 1, 1, 0),
            (ForwardMode.DECODE, 4, 1, 0),
            (ForwardMode.DECODE, 5, 1, 0),
            (ForwardMode.DECODE, 9, 1, 2),
            (ForwardMode.TARGET_VERIFY, 1, 6, 0),
            (ForwardMode.TARGET_VERIFY, 4, 6, 0),
            (ForwardMode.TARGET_VERIFY, 5, 6, 2),
            (ForwardMode.TARGET_VERIFY, 9, 6, 0),
            (ForwardMode.DRAFT_EXTEND_V2, 1, 6, 0),
            (ForwardMode.DRAFT_EXTEND_V2, 3, 6, 2),
        ]
        for mode, batch_size, next_n, padding in cases:
            with self.subTest(mode=mode, batch_size=batch_size, next_n=next_n):
                self._check_paged_assembly(mode, batch_size, next_n, padding)

    def _check_paged_assembly(self, mode, batch_size, next_n, padding):
        from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
            DSAPagedMQALogitsBackend,
        )

        rows, width, heads, dim, sms = batch_size * next_n, 192, 2, 3, 4
        native = mode.is_target_verify() and next_n > 1
        query = torch.arange((rows + padding) * heads * dim, dtype=torch.float32)
        query = query.view(rows + padding, heads, dim)
        weights = (
            torch.arange((rows + padding) * heads, dtype=torch.float32)
            .view(rows + padding, heads, 1)
            .add_(0.25)
        )
        seq_lens = 180 - torch.arange(batch_size, dtype=torch.int32) % 3
        contexts = (
            seq_lens[:, None] - next_n + torch.arange(1, next_n + 1, dtype=torch.int32)
        )
        if batch_size > 1:
            contexts[-1].zero_()  # Padded/idle graph companion rows.
            seq_lens[-1] = 0
        expanded = contexts.flatten().to(torch.int32)
        pages = torch.arange(batch_size * 3, dtype=torch.int32).view(batch_size, 3)
        expanded_pages = pages.repeat_interleave(next_n, dim=0)
        kv = torch.zeros(2, 64 * 132, dtype=torch.uint8)
        pool = types.SimpleNamespace(index_page_size=64)
        observed, weak_chunks, calls = {}, [], []

        row_values = (
            query[:rows, 0, 0]
            + weights[:rows, :, 0].sum(dim=1)
            + expanded.float()
            + expanded_pages.sum(dim=1)
        )
        expected_logits = row_values[:, None] + torch.arange(width)[None, :]

        def select(logits, topk):
            observed["logits"] = logits.clone()
            observed["stride"] = logits.stride()
            self.assertEqual(logits.dtype, torch.float32)
            valid = torch.arange(width)[None, :] < expanded[:, None]
            result = logits.masked_fill(~valid, -torch.inf).topk(topk, dim=1).indices
            result[expanded == 0] = -1
            return result.to(torch.int32)

        metadata = types.SimpleNamespace(
            get_page_table_64=lambda: expanded_pages,
            get_seqlens_int32=lambda: seq_lens,
            get_seqlens_expanded=lambda: expanded,
            get_dsa_extend_len_cpu=lambda: [next_n] * batch_size,
            paged_mqa_schedule_metadata=contexts.clone()
            if native
            else expanded[:, None].clone(),
            paged_mqa_ctx_lens_2d=contexts if native else expanded[:, None],
            topk_transform=select,
        )
        indexer = types.SimpleNamespace(
            sm_count=sms,
            index_topk=3,
            paged_mqa_logits_backend=DSAPagedMQALogitsBackend.DEEPGEMM,
            num_init_tokens=0,
            num_local_tokens=0,
            _get_index_k_read_buffer=lambda _pool, _layer: kv,
        )
        indexer._mask_init_and_local_tokens = types.MethodType(
            dsa_indexer.Indexer._mask_init_and_local_tokens, indexer
        )

        def schedule(ctx, page_size, sm_count):
            self.assertEqual((page_size, sm_count), (64, sms))
            return ctx.clone()

        def kernel(q, cache, w, ctx, table, plan, max_len, *, clean_logits):
            self.assertTrue(
                all(ref() is None for ref in weak_chunks),
                "Previous dense logits chunk is still live at the next kernel call",
            )
            start = sum(calls)
            count = q.shape[0] * q.shape[1]
            calls.append(count)
            self.assertLessEqual(q.shape[0], sms)
            self.assertEqual(q.shape[1], next_n if native else 1)
            self.assertEqual(max_len, width)
            self.assertFalse(clean_logits)
            self.assertEqual(cache.data_ptr(), kv.data_ptr())
            torch.testing.assert_close(
                q.flatten(0, 1), query[start : start + count], rtol=0, atol=0
            )
            torch.testing.assert_close(
                w, weights[start : start + count, :, 0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                ctx.flatten(), expanded[start : start + count], rtol=0, atol=0
            )
            torch.testing.assert_close(plan, ctx, rtol=0, atol=0)
            expected_pages = expanded_pages[start : start + count]
            if native:
                expected_pages = expected_pages[::next_n]
            torch.testing.assert_close(table, expected_pages, rtol=0, atol=0)
            padded = torch.full((count, 256), float("nan"))
            chunk = padded[:, :width]
            chunk.copy_(expected_logits[start : start + count])
            weak_chunks.append(weakref.ref(chunk))
            return chunk

        with (
            mock.patch.object(dsa_indexer, "_is_cuda", True),
            mock.patch.object(dsa_indexer, "_is_hip", False),
            mock.patch.object(dsa_indexer, "_is_xpu", False),
            mock.patch.object(dsa_indexer, "get_token_to_kv_pool", return_value=pool),
            mock.patch.object(
                dsa_indexer, "resolve_fp8_paged_mqa_logits_fn", return_value=kernel
            ),
            mock.patch.object(
                dsa_indexer,
                "resolve_paged_mqa_logits_metadata_fn",
                return_value=schedule,
            ),
        ):
            actual = dsa_indexer.Indexer._get_topk_paged(
                indexer,
                types.SimpleNamespace(forward_mode=mode),
                0,
                query,
                weights,
                metadata,
            )
        torch.testing.assert_close(observed["logits"], expected_logits, rtol=0, atol=0)
        effective_batch = batch_size if native else rows
        if effective_batch > sms:
            self.assertEqual(observed["stride"], (width, 1))
        else:
            self.assertEqual(observed["stride"][1], 1)
            self.assertEqual(observed["stride"][0] % 4, 0)
        expected = expanded[:, None] - torch.arange(1, 4)[None, :]
        expected[expanded == 0] = -1
        if padding:
            expected = torch.cat([expected, torch.full((padding, 3), -1)], dim=0)
        torch.testing.assert_close(actual, expected.to(torch.int32), rtol=0, atol=0)
        self.assertEqual(sum(calls), rows)
        self.assertTrue(all(ref() is None for ref in weak_chunks))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
