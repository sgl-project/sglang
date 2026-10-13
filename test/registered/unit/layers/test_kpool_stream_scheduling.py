"""CPU kpool indexer checks: streams, cache contracts, gate math, logits chunking."""

import sys
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsa import triton_kernel
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as indexer_module
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _eager(method):
    return getattr(method, "_torchdynamo_orig_callable", method)


class _Stream:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace

    def wait_stream(self, other):
        self.trace.append(("wait", self.name, other.name))


class _Streams:
    def __init__(self):
        self.trace = []
        self.current = _Stream("main", self.trace)
        self.alt = _Stream("alt", self.trace)
        self.gate = _Stream("gate", self.trace)

    @contextmanager
    def use(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    def record(self, operation):
        self.trace.append((operation, self.current.name))


class TestKPoolStreamScheduling(unittest.TestCase):
    def test_prefill_overlap_excludes_cp_and_graph_capture(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            for has_stream in (False, True):
                for capture, breakable, cp in (
                    (False, False, False),
                    (True, False, False),
                    (False, True, False),
                    (False, False, True),
                ):
                    with (
                        self.subTest(
                            mode=mode,
                            stream=has_stream,
                            capture=capture,
                            breakable=breakable,
                            cp=cp,
                        ),
                        patch.object(
                            indexer_module, "get_is_capture_mode", return_value=capture
                        ),
                        patch.object(
                            indexer_module,
                            "is_in_breakable_cuda_graph",
                            return_value=breakable,
                        ),
                        patch.object(
                            indexer_module, "dsa_use_prefill_cp", return_value=cp
                        ),
                    ):
                        indexer = SimpleNamespace(
                            alt_stream=object() if has_stream else None
                        )
                        actual = IndexerKPool._can_overlap_prefill(
                            indexer,
                            SimpleNamespace(forward_mode=mode),
                            return_indices=True,
                        )
                        self.assertEqual(
                            actual,
                            mode == ForwardMode.EXTEND
                            and has_stream
                            and not (capture or breakable or cp),
                        )

    def test_precomputed_head_gate_matches_original_math(self):
        torch.manual_seed(19)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(11, 32).to(dtype)
            matrix = torch.randn(8, 32)
            q_scale = torch.rand(11, 8, 1)
            indexer = SimpleNamespace(
                weights_proj=lambda value: (F.linear(value, matrix), None),
                n_heads=8,
                softmax_scale=128**-0.5,
            )
            expected = _eager(IndexerKPool._get_logits_head_gate)(indexer, x, q_scale)
            projected = _eager(IndexerKPool._project_and_scale_head_gates)(indexer, x)
            actual = _eager(IndexerKPool._apply_q_scale_and_softmax_scale)(
                indexer, projected, q_scale
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_projection_reordering_preserves_rope_and_third_stream(self):
        for skip_rope in (False, True):
            streams = _Streams()
            x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
            indexer = SimpleNamespace(
                alt_stream=streams.alt,
                compress_gate_stream=streams.gate,
                half_device_sm_count=8,
                head_dim=4,
                rope_head_dim=2,
                skip_rope=skip_rope,
                index_kpool_compress_gate=torch.ones(4, 8),
            )

            def project_q(value):
                streams.record("project_q")
                return value.clone(), None

            def project_k(value):
                streams.record("project_k")
                return value[:, :4].clone(), None

            def head_gate(value):
                streams.record("head_gate")
                return value[:, :2].clone()

            def rotate(value):
                streams.record("rotate")
                return value.flip(-1)

            def rope(positions, q, k):
                streams.record("rope")
                return q + 1, k + 2

            indexer.wq_b = project_q
            indexer.wk = project_k
            indexer.k_norm = lambda value: value
            indexer.rotary_emb = rope
            indexer._project_and_scale_head_gates = head_gate
            with (
                patch.object(
                    torch.cuda, "current_stream", side_effect=lambda: streams.current
                ),
                patch.object(torch.cuda, "stream", side_effect=streams.use),
                patch.object(
                    indexer_module.deep_gemm_wrapper,
                    "configure_deep_gemm_num_sms",
                    return_value=nullcontext(),
                ),
                patch.object(indexer_module, "rotate_activation", side_effect=rotate),
            ):
                actual = IndexerKPool._get_q_k_bf16(
                    indexer,
                    x,
                    x,
                    torch.arange(4),
                    True,
                    None,
                    precompute_compress_gate=True,
                    precompute_head_gate=True,
                )
                trace = streams.trace[:]
                expected = IndexerKPool._get_q_k_bf16(
                    indexer, x, x, torch.arange(4), False, None
                )
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])
            torch.testing.assert_close(
                actual[2], F.linear(x, indexer.index_kpool_compress_gate)
            )
            self.assertIsNotNone(actual[3])
            self.assertIn(("wait", "gate", "main"), trace)
            join = trace.index(("wait", "main", "alt"))
            self.assertLess(trace.index(("head_gate", "main")), join)
            if skip_rope:
                self.assertLess(trace.index(("rotate", "main")), join)
                self.assertNotIn(("rope", "main"), trace)
            else:
                self.assertGreater(trace.index(("rope", "main")), join)
                self.assertGreater(trace.index(("rotate", "main")), join)

    def test_prefill_waits_before_topk_and_preserves_deferred_cache(self):
        for has_plan in (False, True):
            for return_indices in (False, True):
                for cp in (False, True):
                    for num_tokens in (0, 128, 8192):
                        with self.subTest(
                            plan=has_plan,
                            indices=return_indices,
                            cp=cp,
                            tokens=num_tokens,
                        ):
                            self._run_prefill(has_plan, return_indices, cp, num_tokens)

    def _run_prefill(self, has_plan, return_indices, cp, num_tokens):
        streams = _Streams()
        x = torch.ones(num_tokens, 8)
        compressed = object()
        calls = []
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                kpool_extend_plan=object() if has_plan else None
            )
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([8192 + num_tokens]),
        )
        prepare_qk = Mock(return_value=(x, x, None, None))
        indexer = SimpleNamespace(
            alt_stream=streams.alt,
            compress_gate_stream=streams.gate,
            index_topk=16,
            index_kpool=4,
            index_kpool_compress=True,
            block_size=128,
            scale_fmt=None,
            _get_q_k_bf16=prepare_qk,
        )
        indexer._can_overlap_prefill = MethodType(
            IndexerKPool._can_overlap_prefill, indexer
        )

        def compress(**kwargs):
            streams.record("compress")
            calls.append(kwargs)
            return compressed

        def quant(*args):
            streams.record("quant")
            return x, torch.ones(num_tokens, 1, 1)

        def head(*args):
            streams.record("head")
            return x

        def topk(*args, **kwargs):
            streams.record("topk")
            if not has_plan:
                self.assertIs(kwargs["kpool_extend_cache"], compressed)
            return x

        indexer._compress_write = compress
        indexer._resolve_head_gate_weights = head
        indexer._get_topk_ragged = topk
        indexer._get_topk_ragged_kpool_plan = topk
        with (
            patch.object(indexer_module, "is_cuda", return_value=True),
            patch.object(indexer_module, "is_hip", return_value=False),
            patch.object(indexer_module, "is_npu", return_value=False),
            patch.object(indexer_module, "get_is_capture_mode", return_value=False),
            patch.object(
                indexer_module, "is_in_breakable_cuda_graph", return_value=False
            ),
            patch.object(indexer_module, "dsa_use_prefill_cp", return_value=cp),
            patch.object(
                indexer_module,
                "get_attn_backend",
                return_value=SimpleNamespace(
                    get_indexer_metadata=lambda *args: metadata
                ),
            ),
            patch.object(
                torch.cuda, "current_stream", side_effect=lambda: streams.current
            ),
            patch.object(torch.cuda, "stream", side_effect=streams.use),
            patch.object(triton_kernel, "act_quant", side_effect=quant),
        ):
            actual = IndexerKPool._forward_cuda_impl(
                indexer, x, x, torch.arange(num_tokens), batch, 0, return_indices
            )
        self.assertEqual(calls[0]["return_compressed"], return_indices)
        self.assertEqual(calls[0]["write_cache"], has_plan or not return_indices)
        self.assertFalse(prepare_qk.call_args.args[3])
        self.assertFalse(prepare_qk.call_args.kwargs["precompute_head_gate"])
        overlap = not cp and return_indices
        self.assertIn(("compress", "alt" if overlap else "main"), streams.trace)
        if overlap:
            self.assertLess(
                streams.trace.index(("wait", "alt", "main")),
                streams.trace.index(("compress", "alt")),
            )
            join = streams.trace.index(("wait", "main", "alt"))
            self.assertGreater(join, streams.trace.index(("compress", "alt")))
            self.assertGreater(join, streams.trace.index(("quant", "main")))
            self.assertGreater(join, streams.trace.index(("head", "main")))
            self.assertLess(join, streams.trace.index(("topk", "main")))
        else:
            self.assertFalse(any(event[0] == "wait" for event in streams.trace))
        if return_indices:
            self.assertIs(actual, x)
            self.assertIn(("quant", "main"), streams.trace)
        else:
            self.assertIsNone(actual)
            self.assertNotIn(("quant", "main"), streams.trace)
            self.assertNotIn(("head", "main"), streams.trace)
            self.assertNotIn(("topk", "main"), streams.trace)


# Over the 8M-element skip threshold, so the budget decides.
_LARGE_ROWS, _LARGE_COLS = 16384, 229376


def _plan_chunks(*, num_rows, num_cols, budget_bytes, capture_mode=False):
    with (
        patch.object(
            indexer_module, "mqa_logits_budget_bytes", return_value=budget_bytes
        ) as budget,
        patch.object(indexer_module, "is_hip", return_value=False),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(indexer_module.capture_mode, "is_capture_mode", capture_mode),
    ):
        chunks = indexer_module._mqa_logits_row_chunks(
            num_rows=num_rows, num_cols=num_cols, device=torch.device("cuda", 0)
        )
    return chunks, budget


class TestKPoolMqaLogitsRowChunks(CustomTestCase):
    def test_small_matrices_skip_the_budget_query(self):
        chunks, budget = _plan_chunks(num_rows=64, num_cols=1024, budget_bytes=1 << 20)
        self.assertEqual(chunks, (slice(0, 64),))
        budget.assert_not_called()

    def test_chunks_cover_every_row_once_and_fit_the_budget(self):
        budget_bytes = 6 << 30
        chunks, _ = _plan_chunks(
            num_rows=_LARGE_ROWS, num_cols=_LARGE_COLS, budget_bytes=budget_bytes
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].stop, _LARGE_ROWS)
        for prev, nxt in zip(chunks, chunks[1:]):
            self.assertEqual(prev.stop, nxt.start)
        for rows in chunks:
            self.assertGreater(rows.stop, rows.start)
            self.assertLessEqual(
                (rows.stop - rows.start) * _LARGE_COLS * 4, budget_bytes
            )

    def test_a_matrix_within_budget_runs_in_one_call(self):
        chunks, _ = _plan_chunks(
            num_rows=_LARGE_ROWS, num_cols=_LARGE_COLS, budget_bytes=64 << 30
        )
        self.assertEqual(chunks, (slice(0, _LARGE_ROWS),))

    def test_real_capture_keeps_one_call_without_a_budget_query(self):
        chunks, budget = _plan_chunks(
            num_rows=_LARGE_ROWS,
            num_cols=_LARGE_COLS,
            budget_bytes=6 << 30,
            capture_mode=True,
        )
        self.assertEqual(chunks, (slice(0, _LARGE_ROWS),))
        budget.assert_not_called()

    def test_breakable_graph_replay_still_chunks(self):
        """Breakable-graph replay sets get_is_capture_mode(), but its eager breaks
        run this path for real and must stay budgeted."""
        with patch.object(
            indexer_module.capture_mode, "is_in_breakable_cuda_graph", return_value=True
        ):
            self.assertTrue(get_is_capture_mode())
            chunks, _ = _plan_chunks(
                num_rows=_LARGE_ROWS, num_cols=_LARGE_COLS, budget_bytes=6 << 30
            )
        self.assertGreater(len(chunks), 1)


def _row_chunks(*, rows_per_chunk, num_rows, num_cols, device):
    if rows_per_chunk is None:
        return (slice(0, num_rows),)
    return tuple(
        slice(start, min(start + rows_per_chunk, num_rows))
        for start in range(0, num_rows, rows_per_chunk)
    )


def _fake_deep_gemm():
    # Logits carry each row's weight, so a mis-offset q/weights slice shows up.
    return SimpleNamespace(
        fp8_mqa_logits=lambda q, kv, w, ks, ke, clean_logits: (
            w[:, :1].expand(q.shape[0], kv[0].shape[0]).clone()
        )
    )


def _encoding_topk(*, width, calls):
    def topk(
        *,
        logits,
        pool_lens,
        seq_lens,
        page_table,
        topk_offsets,
        row_starts,
        out_rows,
        page_table_row_index,
    ):
        calls.append(
            {"page_table": page_table, "page_table_row_index": page_table_row_index}
        )
        # Each row encodes the weight it was scored with, its pooled length and
        # its request or offset index.
        if page_table_row_index is not None:
            index = page_table_row_index
        elif topk_offsets is not None:
            index = topk_offsets
        else:
            index = torch.zeros_like(pool_lens)
        row_ids = (
            logits[:, 0].to(torch.int32) * 1_000_000
            + pool_lens.to(torch.int32) * 1_000
            + index.to(torch.int32)
        )
        result = row_ids.unsqueeze(1).expand(-1, width).contiguous()
        # topk_from_pooled_history_logits pads to out_rows with -1.
        if out_rows is not None and out_rows > result.shape[0]:
            padded = torch.full((out_rows, width), -1, dtype=torch.int32)
            padded[: result.shape[0]] = result
            result = padded
        return result

    return topk


def _expected_rows(*, pool_lens, index, width, pad_rows=0):
    # Row r scored with its own inputs: weight r + 1 (see _fake_deep_gemm).
    weight_ids = torch.arange(1, pool_lens.shape[0] + 1, dtype=torch.int32)
    row_ids = weight_ids * 1_000_000 + pool_lens * 1_000 + index
    rows = row_ids.to(torch.int32).unsqueeze(1).expand(-1, width)
    padding = torch.full((pad_rows, width), -1, dtype=torch.int32)
    return torch.cat([rows, padding])


def _chunk_backend(*, topk, index_topk=4, index_kpool=4):
    backend = SimpleNamespace(
        index_topk=index_topk,
        index_kpool=index_kpool,
        alt_stream=None,
        _topk_from_kpool_logits=topk,
        _get_index_k_read_buffer=lambda pool, layer_id: None,
        _fp8_mqa_logits=IndexerKPool._fp8_mqa_logits,
    )
    backend._kpool_topk_by_row_chunks = MethodType(
        IndexerKPool._kpool_topk_by_row_chunks, backend
    )
    return backend


def _fp8_zeros(*shape):
    return torch.zeros(shape, dtype=torch.uint8).view(torch.float8_e4m3fn)


def _fake_aiter_modules():
    # _fp8_mqa_logits imports AITER lazily on ROCm; same row encoding as DeepGEMM.
    kernel = ModuleType("aiter.ops.triton.fp8_mqa_logits")
    kernel.fp8_mqa_logits = lambda q, k, scale, w, starts, ends, clean_logits: (
        w[:, :1].expand(q.shape[0], k.shape[0]).clone()
    )
    return {
        "aiter": ModuleType("aiter"),
        "aiter.ops": ModuleType("aiter.ops"),
        "aiter.ops.triton": ModuleType("aiter.ops.triton"),
        "aiter.ops.triton.fp8_mqa_logits": kernel,
    }


def _refuse_deep_gemm(*args, **kwargs):
    raise AssertionError("ROCm reached DeepGEMM instead of AITER")


@contextmanager
def _chunking_patches(*, rows_per_chunk, kv_pool, rocm=False):
    deep_gemm = (
        SimpleNamespace(fp8_mqa_logits=_refuse_deep_gemm) if rocm else _fake_deep_gemm()
    )
    with ExitStack() as stack:
        # deep_gemm is bound only under `if is_cuda()`, so create it on CPU.
        stack.enter_context(
            patch.object(indexer_module, "deep_gemm", deep_gemm, create=True)
        )
        stack.enter_context(
            patch.object(
                indexer_module,
                "_mqa_logits_row_chunks",
                partial(_row_chunks, rows_per_chunk=rows_per_chunk),
            )
        )
        stack.enter_context(
            patch.object(indexer_module, "get_token_to_kv_pool", return_value=kv_pool)
        )
        stack.enter_context(
            patch.object(indexer_module, "_should_fuse_kpool_topk", return_value=True)
        )
        stack.enter_context(patch.object(indexer_module, "is_hip", return_value=rocm))
        if rocm:
            stack.enter_context(patch.dict(sys.modules, _fake_aiter_modules()))
        yield


class TestKPoolPlanChunking(CustomTestCase):
    NUM_REQ_SLOTS = 64
    TOPK = 4
    PAD_ROWS = 5

    def _drive(self, *, req_ids, rows_per_chunk, topk_method, rocm=False):
        n_real = len(req_ids)
        total_k_rows = 16
        page_table = torch.arange(self.NUM_REQ_SLOTS * 8, dtype=torch.int32).reshape(
            self.NUM_REQ_SLOTS, 8
        )
        plan = SimpleNamespace(
            seq_lens_expanded=torch.full((n_real,), 8, dtype=torch.int32),
            pooled_seq_lens_expanded=torch.arange(1, n_real + 1, dtype=torch.int32),
            ragged_q_ks=torch.zeros(n_real, dtype=torch.int32),
            ragged_q_ke=torch.full((n_real,), total_k_rows, dtype=torch.int32),
            ragged_total_k_rows=total_k_rows,
            ragged_k_u8=torch.zeros((total_k_rows, 4), dtype=torch.uint8),
            ragged_k_scale=torch.zeros((total_k_rows, 1), dtype=torch.float32),
            ragged_concat_page_table=torch.zeros(total_k_rows, dtype=torch.int32),
            ragged_paged_page_table=page_table,
            ragged_paged_page_table_row_index=torch.tensor(req_ids, dtype=torch.int32),
        )
        # Padded batch: q_fp8 has rows past the plan's real ones.
        total_q = n_real + self.PAD_ROWS
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                kpool_extend_plan=plan,
                topk_indices_offset=torch.arange(total_q, dtype=torch.int32) + 100,
            ),
            topk_transform_method=topk_method,
        )
        weights = torch.arange(1, total_q + 1, dtype=torch.float32).reshape(
            total_q, 1, 1
        )
        calls = []
        backend = _chunk_backend(topk=_encoding_topk(width=self.TOPK, calls=calls))
        with (
            _chunking_patches(
                rows_per_chunk=rows_per_chunk, kv_pool=object(), rocm=rocm
            ),
            patch(
                "sglang.srt.layers.attention.dsa.kpool_fp8_index."
                "gather_index_k_scale_prefix_into",
                lambda **kwargs: None,
            ),
        ):
            result = IndexerKPool._get_topk_ragged_kpool_plan(
                backend,
                forward_batch=None,
                layer_id=0,
                q_fp8=_fp8_zeros(total_q, 4),
                weights=weights,
                metadata=metadata,
            )
        return result, calls, page_table

    def test_every_chunk_gets_the_whole_request_indexed_table(self):
        """Every chunk must see req_to_token whole: a sliced table resolves
        request 1 to row 17 once a 16-row chunk shifts its base."""
        req_ids = [1, 9, 2, 40, 7, 3, 63, 0] * 4
        _, calls, page_table = self._drive(
            req_ids=req_ids, rows_per_chunk=16, topk_method=TopkTransformMethod.PAGED
        )
        self.assertEqual(len(calls), 2)
        for chunk_idx, call in enumerate(calls):
            self.assertIs(call["page_table"], page_table)
            start = chunk_idx * 16
            torch.testing.assert_close(
                call["page_table_row_index"],
                torch.tensor(req_ids[start : start + 16], dtype=torch.int32),
            )

    def _reference(self, *, req_ids, topk_method):
        # Every real row scored with its own inputs, then -1 padding.
        n_real = len(req_ids)
        index = (
            torch.tensor(req_ids, dtype=torch.int32)
            if topk_method == TopkTransformMethod.PAGED
            else torch.arange(n_real, dtype=torch.int32) + 100
        )
        return _expected_rows(
            pool_lens=torch.arange(1, n_real + 1, dtype=torch.int32),
            index=index,
            width=self.TOPK,
            pad_rows=self.PAD_ROWS,
        )

    def test_chunked_rows_match_the_single_call_including_padding(self):
        req_ids = [5, 11, 2, 60, 9, 33, 1] * 4
        for method in (TopkTransformMethod.PAGED, TopkTransformMethod.RAGGED):
            with self.subTest(method=method):
                expected, one_call, _ = self._drive(
                    req_ids=req_ids, rows_per_chunk=None, topk_method=method
                )
                actual, chunked, _ = self._drive(
                    req_ids=req_ids, rows_per_chunk=5, topk_method=method
                )
                self.assertEqual(len(one_call), 1)
                self.assertGreater(len(chunked), 1)
                reference = self._reference(req_ids=req_ids, topk_method=method)
                torch.testing.assert_close(expected, reference, rtol=0, atol=0)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_chunked_rows_score_through_aiter_on_rocm(self):
        """On ROCm every row chunk must score through _fp8_mqa_logits, which
        routes to AITER; reaching DeepGEMM there fails the forward."""
        req_ids = [5, 11, 2, 60, 9, 33, 1] * 4
        actual, chunked, _ = self._drive(
            req_ids=req_ids,
            rows_per_chunk=5,
            topk_method=TopkTransformMethod.PAGED,
            rocm=True,
        )
        self.assertGreater(len(chunked), 1)
        torch.testing.assert_close(
            actual,
            self._reference(req_ids=req_ids, topk_method=TopkTransformMethod.PAGED),
            rtol=0,
            atol=0,
        )


def _causal_seq_lens(*, q_lens, seq_lens):
    # Extend row j of a request attends to its prefix plus rows 0..j.
    return torch.cat(
        [
            torch.arange(seq_len - q_len + 1, seq_len + 1, dtype=torch.int32)
            for q_len, seq_len in zip(q_lens, seq_lens)
        ]
    )


class TestKPoolPerRequestChunking(CustomTestCase):
    TOPK = 4
    POOL = 4

    def _drive(self, *, q_lens, seq_lens, rows_per_chunk):
        token_nums = sum(q_lens)
        forward_batch = SimpleNamespace(
            batch_size=len(q_lens),
            extend_seq_lens_cpu=list(q_lens),
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
            req_pool_indices=torch.tensor([7, 3][: len(q_lens)], dtype=torch.int64),
        )
        seqlens_expanded = _causal_seq_lens(q_lens=q_lens, seq_lens=seq_lens)
        metadata = SimpleNamespace(
            get_page_table_64=lambda: torch.zeros((len(q_lens), 1), dtype=torch.int32),
            get_seqlens_expanded=lambda: seqlens_expanded,
            topk_transform_method=TopkTransformMethod.RAGGED,
            attn_metadata=SimpleNamespace(
                topk_indices_offset=torch.arange(token_nums, dtype=torch.int32) + 100,
            ),
        )
        # Fully cached current K: the path scores it without a pool gather.
        cache = [
            (0, _fp8_zeros(seq_len // self.POOL, 4), torch.zeros(seq_len // self.POOL))
            for seq_len in seq_lens
        ]
        weights = torch.arange(1, token_nums + 1, dtype=torch.float32).reshape(
            token_nums, 1
        )
        calls = []
        backend = _chunk_backend(
            topk=_encoding_topk(width=self.TOPK + self.POOL - 1, calls=calls),
            index_topk=self.TOPK,
            index_kpool=self.POOL,
        )
        with _chunking_patches(
            rows_per_chunk=rows_per_chunk, kv_pool=SimpleNamespace(page_size=64)
        ):
            result = IndexerKPool._get_topk_ragged_kpool(
                backend,
                forward_batch=forward_batch,
                layer_id=0,
                q_fp8=_fp8_zeros(token_nums, 4),
                weights=weights,
                metadata=metadata,
                extend_pooled_cache=cache,
            )
        return result, calls

    def test_a_long_request_after_another_matches_the_single_call(self):
        """A request's row chunks slice relative to that request, not to the
        batch; the second request here starts at q offset 5."""
        q_lens, seq_lens = [5, 19], [40, 96]
        expected, one_call = self._drive(
            q_lens=q_lens, seq_lens=seq_lens, rows_per_chunk=None
        )
        actual, chunked = self._drive(
            q_lens=q_lens, seq_lens=seq_lens, rows_per_chunk=4
        )
        self.assertEqual(len(one_call), 2)
        self.assertGreater(len(chunked), len(one_call))
        pool_lens = torch.div(
            _causal_seq_lens(q_lens=q_lens, seq_lens=seq_lens),
            self.POOL,
            rounding_mode="floor",
        )
        reference = _expected_rows(
            pool_lens=pool_lens,
            index=torch.arange(pool_lens.shape[0], dtype=torch.int32) + 100,
            width=self.TOPK + self.POOL - 1,
        )
        torch.testing.assert_close(expected, reference, rtol=0, atol=0)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
