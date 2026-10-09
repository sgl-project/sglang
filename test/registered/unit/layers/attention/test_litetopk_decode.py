"""SGLANG_OPT_LITETOPK_DECODE in the DSA and DeepSeek-V4.1 indexers: off by default,
the default top-k for anything LiteTopK cannot serve, and selector buffers that
captured CUDA graphs can rely on."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention import litetopk_decode as litetopk_ops
from sglang.kernels.ops.attention.litetopk_decode import BF16_TOP512, FP32_TOP2048
from sglang.srt.environ import envs
from sglang.srt.layers.attention import litetopk_decode
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)
from sglang.srt.layers.attention.dsv4.v41_indexer import litetopk as v41_litetopk
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, enter_scope

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakePlan:
    def __init__(self, rows, storage):
        self.rows = rows
        self.storage = storage
        self.histogram = torch.zeros((rows, 1024), dtype=torch.int32)
        self.selected = []

    def select(self, scores, lengths, table, *, out, rows_per_table_row=1):
        self.selected.append((scores, lengths, table, rows_per_table_row))
        return out.fill_(7)


class _FakeStorage:
    allocated: list = []

    def __init__(self, config, max_rows, device, candidate_capacity=8192):
        _FakeStorage.allocated.append(self)
        self.max_rows = max_rows
        self.nbytes = max_rows * 72 * 1024
        self.histogram, self.workspace = Mock(), Mock()

    def plan(self, rows):
        assert 1 <= rows <= self.max_rows
        return _FakePlan(rows, self)


class TestSwitchAndCapabilities(CustomTestCase):
    def setUp(self):
        litetopk_decode.get_litetopk_decode.cache_clear()
        self.addCleanup(litetopk_decode.get_litetopk_decode.cache_clear)

    def test_off_by_default_without_probing(self):
        probe = AssertionError("the support probe ran with the flag off")
        with patch.object(litetopk_decode, "unsupported_reason", side_effect=probe):
            for config in (FP32_TOP2048, BF16_TOP512):
                self.assertIsNone(litetopk_decode.get_litetopk_decode(config))

    def test_deep_gemm_capability_detection(self):
        """SGLang's pinned DeepGEMM has no histogram argument: the FP32 path must
        detect that rather than fail inside the first decode."""

        def without(q, kv, w, lens, table, schedule, max_len, clean_logits=False):
            pass

        def with_histogram(q, kv, w, lens, table, schedule, max_len, histogram=None):
            pass

        cases = [
            (FP32_TOP2048, dict(fp8_paged_mqa_logits=without), "takes no histogram"),
            (FP32_TOP2048, dict(fp8_paged_mqa_logits=with_histogram), None),
            (BF16_TOP512, dict(fp4_paged_mqa_logits_bf16=print), "has no get_paged"),
        ]
        for config, api, expected in cases:
            producer = types.ModuleType("deep_gemm")
            for name, fn in api.items():
                setattr(producer, name, fn)
            with (
                patch.dict(sys.modules, {"deep_gemm": producer}),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.version, "cuda", "13.0"),
                patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            ):
                reason = litetopk_ops.unsupported_reason(config)
            if expected is None:
                self.assertIsNone(reason, api)
            else:
                self.assertIn(expected, reason)


class TestPlans(CustomTestCase):
    """Plans are views of one storage. A captured graph replays the buffers it saw,
    so its storage must outlive every later eager batch, and nothing may be
    allocated while a graph is captured."""

    def setUp(self):
        _FakeStorage.allocated = []
        self.capture_mode = self.capturing = False
        for patcher in (
            patch.object(litetopk_decode, "LiteTopKStorage", _FakeStorage),
            patch.object(
                litetopk_decode, "get_is_capture_mode", lambda: self.capture_mode
            ),
            patch.object(
                torch.cuda, "is_current_stream_capturing", lambda: self.capturing
            ),
            patch.object(torch.cuda, "current_stream", lambda device=None: "stream"),
        ):
            enter_scope(self, patcher)
        self.buffers = litetopk_decode.LiteTopKDecode(
            config=FP32_TOP2048, device=torch.device("cpu")
        )

    def _allocated_rows(self):
        return [storage.max_rows for storage in _FakeStorage.allocated]

    def test_row_counts_share_one_storage_that_grows_geometrically(self):
        for rows in (8, 3, 5, 8, 1):
            self.buffers.plan(rows)
        self.assertEqual(self._allocated_rows(), [8])
        for rows in range(1, 101):
            self.buffers.plan(rows)
        self.assertEqual(self._allocated_rows(), [8, 16, 32, 64, 128])
        # A replaced storage stays allocated until its eager calls have run.
        for old in _FakeStorage.allocated[:-1]:
            old.workspace.record_stream.assert_called_once_with("stream")

    def test_captured_plans_outlive_eager_growth(self):
        self.capture_mode = True  # warmup before capture
        captured = self.buffers.plan(3)
        self.capturing = True
        self.assertIs(self.buffers.plan(3), captured)
        self.capture_mode = self.capturing = False
        for rows in range(10, 100):
            self.buffers.plan(rows)
        self.assertIs(self.buffers.plan(3), captured)
        eager = self.buffers.plan(50)
        self.assertIsNot(eager.storage, captured.storage)

    def test_nothing_is_allocated_during_capture(self):
        self.buffers.plan(4)
        self.capture_mode = self.capturing = True
        self.assertIsNotNone(self.buffers.plan(2))  # a view of the storage
        self.assertIsNone(self.buffers.plan(9))
        self.assertEqual(self._allocated_rows(), [4])


# ---- DSA indexer


def _fake_deep_gemm(calls):
    producer = types.ModuleType("deep_gemm")

    def logits(q, kv, w, lens, table, schedule, max_len, clean_logits=False, **kw):
        rows = q.shape[0] * q.shape[1]
        calls.append(dict(rows=rows, q_shape=tuple(q.shape), kwargs=kw))
        return torch.arange(rows * max_len, dtype=torch.float32).view(rows, max_len)

    producer.fp8_paged_mqa_logits = logits
    producer.get_paged_mqa_logits_metadata = lambda lens, block, sms: torch.zeros(
        (sms + 1, 2), dtype=torch.int32
    )
    return producer


class TestDSAIndexer(CustomTestCase):
    def setUp(self):
        self.calls = []
        self.plans = []
        for patcher in (
            patch.object(dsa_indexer, "_is_cuda", True),
            patch.object(dsa_indexer, "_is_hip", False),
            patch.object(dsa_indexer, "_is_xpu", False),
            patch.object(
                dsa_indexer, "deep_gemm", _fake_deep_gemm(self.calls), create=True
            ),
            patch.object(
                dsa_indexer,
                "get_token_to_kv_pool",
                lambda: SimpleNamespace(page_size=64),
            ),
            patch.object(
                DSAIndexerMetadata,
                "topk_transform",
                lambda md, logits, topk, **kw: ("default", logits),
            ),
        ):
            enter_scope(self, patcher)

    def _indexer(self, litetopk=True, sm_count=148):
        def plan(rows):
            p = _FakePlan(rows, None)
            self.plans.append(p)
            return p

        indexer = SimpleNamespace(
            paged_mqa_logits_backend=DSAPagedMQALogitsBackend.DEEPGEMM,
            sm_count=sm_count,
            n_heads=32,
            index_topk=2048,
            num_init_tokens=0,
            num_local_tokens=0,
            litetopk=SimpleNamespace(plan=plan) if litetopk else None,
            _get_index_k_read_buffer=lambda pool, layer_id: torch.zeros(
                (40, 64 * 132), dtype=torch.uint8
            ),
        )
        for name in ("_litetopk_plan", "_mask_init_and_local_tokens"):
            setattr(
                indexer,
                name,
                types.MethodType(getattr(dsa_indexer.Indexer, name), indexer),
            )
        return indexer

    def _call(self, indexer, batch, next_n=1, mode=ForwardMode.DECODE, **overrides):
        rows = batch * next_n
        lens = torch.arange(rows, dtype=torch.int32) + 3000
        attn = SimpleNamespace(
            real_page_table=torch.arange(rows * 8, dtype=torch.int32).view(rows, 8),
            cache_seqlens_int32=lens[next_n - 1 :: next_n].contiguous(),
            dsa_seqlens_expanded=lens,
            dsa_extend_seq_lens_list=[next_n] * batch,
        )
        fields = dict(
            attn_metadata=attn,
            topk_transform_method=TopkTransformMethod.PAGED,
            paged_mqa_schedule_metadata=torch.zeros((149, 2), dtype=torch.int32),
            paged_mqa_ctx_lens_2d=lens.view(batch, next_n),
        )
        fields.update(overrides)
        metadata = DSAIndexerMetadata(**fields)
        result = dsa_indexer.Indexer._get_topk_paged(
            indexer,
            SimpleNamespace(forward_mode=mode),
            3,
            torch.zeros((rows, 32, 128)),
            torch.zeros((rows, 32, 1)),
            metadata,
        )
        return result, metadata

    def test_decode_and_verify_select_from_the_histogram(self):
        for next_n, mode in ((1, ForwardMode.DECODE), (2, ForwardMode.TARGET_VERIFY)):
            self.calls.clear()
            self.plans.clear()
            result, metadata = self._call(self._indexer(), 3, next_n, mode)
            (plan,) = self.plans
            (call,) = self.calls
            self.assertEqual(call["q_shape"][:2], (3, next_n))
            self.assertIs(call["kwargs"]["histogram"], plan.histogram)
            ((scores, lengths, table, per_table),) = plan.selected
            self.assertEqual(tuple(scores.shape), (3 * next_n, 8 * 64))
            self.assertIs(lengths, metadata.get_seqlens_expanded())
            self.assertIs(table, metadata.get_page_table_64())
            self.assertEqual(per_table, 1)
            self.assertEqual(tuple(result.shape), (3 * next_n, 2048))
            self.assertTrue(bool((result == 7).all()))

    def test_batches_beyond_the_sms_slice_the_histogram(self):
        self._call(self._indexer(sm_count=2), batch=5)
        (plan,) = self.plans
        offsets = [
            (c["kwargs"]["histogram"].data_ptr() - plan.histogram.data_ptr()) // 4096
            for c in self.calls
        ]
        self.assertEqual(offsets, [0, 2, 4])
        self.assertEqual(
            [c["kwargs"]["histogram"].shape[0] for c in self.calls], [2, 2, 1]
        )

    def test_calls_it_cannot_serve_keep_the_default_top_k(self):
        """DeepGEMM gets no histogram argument (the pinned DeepGEMM rejects it) and
        the default top-k transform runs."""
        ragged = dict(topk_transform_method=TopkTransformMethod.RAGGED)
        for litetopk, mode, next_n, overrides in (
            (False, ForwardMode.DECODE, 1, {}),  # flag off
            (True, ForwardMode.DRAFT_EXTEND_V2, 1, {}),
            (True, ForwardMode.DECODE, 1, dict(force_unfused_topk=True)),
            (True, ForwardMode.DECODE, 1, ragged),
            # DeepGEMM's histogram serves at most 4 verify tokens per request.
            (True, ForwardMode.TARGET_VERIFY, 5, {}),
        ):
            self.calls.clear()
            indexer = self._indexer(litetopk=litetopk)
            result, _ = self._call(indexer, 2, next_n, mode, **overrides)
            self.assertEqual(result[0], "default", (mode, overrides))
            self.assertEqual([c["kwargs"] for c in self.calls], [{}])
        self.assertEqual(self.plans, [])
        # Only 32 x 128 heads, top-2048, no forced tokens, DeepGEMM producer.
        glm = dict(
            n_heads=32,
            head_dim=128,
            index_topk=2048,
            forces_tokens=False,
            paged_mqa_logits_backend=DSAPagedMQALogitsBackend.DEEPGEMM,
        )
        marker = object()
        with patch.object(dsa_indexer, "get_litetopk_decode", return_value=marker):
            self.assertIs(dsa_indexer._get_dsa_litetopk(**glm), marker)
            for change in (
                dict(n_heads=64),
                dict(head_dim=64),
                dict(index_topk=1024),
                dict(forces_tokens=True),
                dict(paged_mqa_logits_backend=DSAPagedMQALogitsBackend.CUTEDSL),
            ):
                self.assertIsNone(dsa_indexer._get_dsa_litetopk(**{**glm, **change}))

    def test_causal_verify_lengths_only_where_litetopk_serves_verify(self):
        """The histogram must count the keys each verify token sees; more draft
        tokens than LiteTopK serves, or the flag off, keep the final length."""
        from sglang.srt.layers.attention import dsa_backend

        for litetopk, draft_tokens, want in (
            (object(), 4, True),
            (object(), 6, False),
            (None, 3, False),
        ):
            with patch.object(
                dsa_backend, "get_litetopk_decode", return_value=litetopk
            ):
                got = dsa_backend._litetopk_causal_verify_ctx_lens(draft_tokens)
            self.assertEqual(got, want, draft_tokens)
        cache = torch.tensor([10, 20], dtype=torch.int32)
        expanded = torch.tensor([8, 9, 10, 18, 19, 20], dtype=torch.int32)
        build = (
            dsa_backend.DeepseekSparseAttnBackend._build_paged_mqa_schedule_2d_ctx_lens
        )
        with patch.object(
            dsa_backend, "get_platform", return_value=SimpleNamespace(is_sm100=True)
        ):
            for causal, want in (
                (False, [[10, 10, 10], [20, 20, 20]]),
                (True, [[8, 9, 10], [18, 19, 20]]),
            ):
                backend = SimpleNamespace(
                    speculative_num_draft_tokens=3, paged_mqa_causal_ctx_lens=causal
                )
                got = build(backend, ForwardMode.TARGET_VERIFY, cache, expanded, 2)
                self.assertEqual(got.tolist(), want)


# ---- DeepSeek-V4.1 indexer


def _v41_call(hook, rows, *, heads=32, head_bytes=64, schedule=True, out_cols=512):
    data = SimpleNamespace(
        q_fp4=torch.zeros((rows, 1, heads, head_bytes), dtype=torch.int8),
        q_sf=torch.zeros((rows, 1, heads), dtype=torch.int32),
        k_cache=None,
        weights=torch.zeros((rows, heads)),
    )
    metadata = SimpleNamespace(
        compressed_seq_lens=torch.full((rows,), 1000, dtype=torch.int32),
        page_table=torch.zeros((rows, 8), dtype=torch.int32),
        litetopk_schedule=torch.zeros((149, 2), dtype=torch.int32)
        if schedule
        else None,
        litetopk_request_ids=torch.arange(rows, dtype=torch.int32),
        litetopk_tokens_per_request=1,
        max_compressed_seq_len=1024,
    )
    return hook.scores(
        data=data,
        metadata=metadata,
        out=torch.empty((rows, out_cols), dtype=torch.int32),
    )


class TestDsv41(CustomTestCase):
    def setUp(self):
        self.buffers = SimpleNamespace(plan=Mock())
        self.hook = v41_litetopk.Dsv41LiteTopK(self.buffers)

    def test_unsupported_calls_keep_the_default_top_k(self):
        for kwargs in (
            dict(schedule=False),  # no schedule for this forward (unsupported hint)
            dict(heads=64),  # the BF16 producer takes 32 index heads
            dict(head_bytes=32),  # of 128 dimensions (64 bytes of packed e2m1)
            dict(out_cols=1024),  # another top-k
        ):
            self.assertIsNone(_v41_call(self.hook, 2, **kwargs), kwargs)
        self.buffers.plan.assert_not_called()

    def test_no_schedule_for_forwards_it_cannot_serve(self):
        producer = types.ModuleType("deep_gemm")
        producer.get_paged_mqa_logits_bf16_metadata = Mock()
        base = dict(
            compressed_page_size=128,
            compressed_seq_lens=torch.full((12,), 1000, dtype=torch.int32),
            row_chunk=0,
            deep_gemm_metadata=torch.empty((149, 2), dtype=torch.int32),
        )
        ids = torch.arange(12)
        with patch.dict(sys.modules, {"deep_gemm": producer}):
            for change, tokens, rows in (
                ({}, 7, ids),  # more tokens per request than the producer tiles
                (dict(compressed_page_size=64), 6, ids),
                (dict(row_chunk=4), 6, ids),
                (dict(deep_gemm_metadata=[None]), 6, ids),  # chunked rows
                ({}, 6, ids[:6]),  # request IDs that do not match the rows
            ):
                metadata = SimpleNamespace(**{**base, **change})
                self.hook.prepare_metadata(metadata, rows, tokens)
                self.assertIsNone(metadata.litetopk_schedule, change)
                self.assertIsNone(metadata.litetopk_request_ids, change)
        producer.get_paged_mqa_logits_bf16_metadata.assert_not_called()

    def test_metadata_copy_keeps_captured_schedule_buffers(self):
        from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

        def make(lens):
            md = PagedIndexerMetadata(
                page_size=256,
                compressed_page_size=128,
                page_table=torch.zeros((4, 8), dtype=torch.int32),
                compressed_seq_lens=torch.full((4,), lens, dtype=torch.int32),
                use_topk_v2=False,
            )
            md.litetopk_schedule = torch.full((149, 2), lens, dtype=torch.int32)
            md.litetopk_request_ids = torch.full((4,), lens, dtype=torch.int32)
            return md

        with envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True):
            captured, live = make(1), make(2)
            buffers = (captured.litetopk_schedule, captured.litetopk_request_ids)
            captured.copy_(live)
            self.assertIs(captured.litetopk_schedule, buffers[0])
            self.assertIs(captured.litetopk_request_ids, buffers[1])
            self.assertTrue(bool((captured.litetopk_schedule == 2).all()))
            # A forward that has no schedule must not keep a stale one.
            live.litetopk_schedule = None
            captured.copy_(live)
            self.assertIsNone(captured.litetopk_schedule)

    def test_backend_prepares_every_forward_with_live_ids(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as backend
        from sglang.srt.model_executor.runner_utils import capture_mode

        # Unequal adjacent runs must reach the schedule unchanged.
        ids = torch.tensor([9, 9, 3, 3, 3, 3, 3, 8], dtype=torch.int64)
        for n, mode in ((1, ForwardMode.DECODE), (6, ForwardMode.TARGET_VERIFY)):
            hook = Mock()
            c1, c2 = object(), object()
            metadata = backend.DSV4Metadata(
                core_attn_metadata=SimpleNamespace(low_ratios=(1, 2)),
                indexer_metadata=None,
                c1_indexer_metadata=c1,
                c2_indexer_metadata=c2,
                low_ratio_req_indices=ids,
            )
            obj = SimpleNamespace(
                forward_metadata=metadata,
                full_topk_indexer=SimpleNamespace(litetopk=hook),
            )
            batch = SimpleNamespace(
                forward_mode=mode,
                out_cache_loc=None,
                spec_info=SimpleNamespace(draft_token_num=n),
            )
            with (
                patch.object(
                    capture_mode, "skip_low_ratio_indexer", return_value=False
                ),
                patch.object(backend, "token_req_indices", return_value=ids) as live,
            ):
                backend.DeepseekV4AttnBackend.init_forward_metadata_in_graph(obj, batch)
                live.assert_not_called()
                self.assertEqual(
                    [c.args for c in hook.prepare_metadata.call_args_list],
                    [(c1, ids, n), (c2, ids, n)],
                )
                metadata.low_ratio_req_indices = None
                backend.DeepseekV4AttnBackend.init_forward_metadata_in_graph(obj, batch)
                live.assert_called_once_with(batch)


if __name__ == "__main__":
    unittest.main()
