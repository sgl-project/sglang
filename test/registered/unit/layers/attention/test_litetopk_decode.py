"""SGLANG_OPT_LITETOPK_DECODE in the DSA and DeepSeek-V4.1 indexers: off by default,
the default top-k for anything LiteTopK cannot serve, and selector buffers that
captured CUDA graphs can rely on."""

import inspect
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention import litetopk_decode as litetopk_ops
from sglang.kernels.ops.attention.litetopk_decode import BF16_TOP512, BF16_TOP2048
from sglang.srt.environ import envs
from sglang.srt.layers.attention import litetopk_decode
from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
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
        self.storage = storage
        self.histogram = torch.zeros((rows, 1024), dtype=torch.int32)
        self.selected = []

    def select(self, scores, lengths, table, *, out, rows_per_table_row=1):
        self.selected.append((scores, lengths, table, rows_per_table_row))
        return out.fill_(7)


class _FakeStorage:
    allocated: list = []

    def __init__(self, config, max_rows, device):
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
            for config in (BF16_TOP2048, BF16_TOP512):
                self.assertIsNone(litetopk_decode.get_litetopk_decode(config))

    def test_deep_gemm_capability_detection(self):
        """SGLang's pinned DeepGEMM counts no histogram, and DeepGEMM builds from before
        its SM100 BF16-only API (``logits_dtype``) return other logits: LiteTopK must
        detect every keyword it relies on rather than fail inside the first decode."""

        def taking(*names):
            fn = Mock()
            fn.__signature__ = inspect.Signature(
                [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY) for n in names]
            )
            return fn

        logits, schedule = "fp8_fp4_paged_mqa_logits", "get_paged_mqa_logits_metadata"
        dev = {schedule: taking("indices"), logits: taking("indices", "histogram")}
        for change, expected in (
            ({schedule: None}, f"has no {schedule}"),
            ({schedule: print}, "takes no indices"),  # no signature: nothing declared
            ({logits: taking("logits_dtype", "indices")}, "takes no histogram"),
            ({logits: taking("logits_dtype", "indices", "histogram")}, "predates"),
            ({}, None),
        ):
            api = {k: v for k, v in {**dev, **change}.items() if v is not None}
            with (
                patch.dict(sys.modules, {"deep_gemm": SimpleNamespace(**api)}),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.version, "cuda", "13.0"),
                patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
            ):
                reason = litetopk_ops.unsupported_reason()
            if expected is None:
                self.assertIsNone(reason, change)
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
            config=BF16_TOP2048, device=torch.device("cpu")
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


class TestDSAIndexer(CustomTestCase):
    def setUp(self):
        self.calls = []
        self.plans = []

        def logits(q, kv, w, lens, table, schedule, max_len, clean_logits=False, **kw):
            rows = q.shape[0] * q.shape[1]
            self.calls.append(dict(rows=rows, q_shape=tuple(q.shape), kwargs=kw))
            return torch.arange(rows * max_len, dtype=torch.float32).view(rows, max_len)

        def schedule(lens, block, sms):
            return torch.zeros((sms + 1, 2), dtype=torch.int32)

        for patcher in (
            patch.multiple(
                dsa_indexer,
                _is_cuda=True,
                _is_hip=False,
                _is_xpu=False,
                resolve_fp8_paged_mqa_logits_fn=lambda: logits,
                resolve_paged_mqa_logits_metadata_fn=lambda: schedule,
                get_token_to_kv_pool=lambda: SimpleNamespace(index_page_size=64),
            ),
            patch.object(
                DSAIndexerMetadata,
                "topk_transform",
                lambda md, logits, topk, **kw: ("default", logits),
            ),
        ):
            enter_scope(self, patcher)

    def _indexer(self, litetopk=True, sm_count=8):
        def plan(rows):
            p = _FakePlan(rows, None)
            self.plans.append(p)
            return p

        indexer = SimpleNamespace(
            paged_mqa_logits_backend=DSAPagedMQALogitsBackend.DEEPGEMM,
            mx_mqa_logits=True,
            ue8m0_index_cache=True,
            sm_count=sm_count,
            index_topk=2048,
            num_init_tokens=0,
            num_local_tokens=0,
            litetopk=SimpleNamespace(plan=plan) if litetopk else None,
            _get_index_k_read_buffer=lambda pool, layer_id: torch.zeros(
                (40, 64 * 132), dtype=torch.uint8
            ),
        )
        for name in ("_litetopk_plan", "_mask_init_and_local_tokens"):
            method = getattr(dsa_indexer.Indexer, name)
            setattr(indexer, name, types.MethodType(method, indexer))
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
            paged_mqa_schedule_metadata=torch.zeros((9, 2), dtype=torch.int32),
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
        # SM100 DeepGEMM takes one row per verify token, so any draft count is served.
        for next_n, mode in ((1, ForwardMode.DECODE), (5, ForwardMode.TARGET_VERIFY)):
            self.calls.clear()
            self.plans.clear()
            result, metadata = self._call(self._indexer(sm_count=16), 3, next_n, mode)
            (plan,) = self.plans
            (call,) = self.calls
            self.assertEqual(call["q_shape"][:2], (3 * next_n, 1))
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
            (True, ForwardMode.DECODE, 1, dict(topk_backend=DSATopKBackend.FLASHINFER)),
        ):
            self.calls.clear()
            indexer = self._indexer(litetopk=litetopk)
            result, _ = self._call(indexer, 2, next_n, mode, **overrides)
            self.assertEqual(result[0], "default", (mode, overrides))
            self.assertEqual([c["kwargs"] for c in self.calls], [{}])
        self.assertEqual(self.plans, [])
        # Only 32 x 128 heads, top-2048, no forced tokens, and the UE8M0 index cache,
        # which only DeepGEMM's SM100 MXFP8 logits read (not the torch fallback).
        glm = dict(n_heads=32, head_dim=128, index_topk=2048, forces_tokens=False)
        glm.update(ue8m0_index_cache=True)
        refused = dict(n_heads=64, head_dim=64, index_topk=1024, forces_tokens=True)
        refused.update(ue8m0_index_cache=False)
        marker = object()
        with patch.object(dsa_indexer, "get_litetopk_decode", return_value=marker):
            self.assertIs(dsa_indexer._get_dsa_litetopk(**glm), marker)
            for key, value in refused.items():
                self.assertIsNone(dsa_indexer._get_dsa_litetopk(**{**glm, key: value}))

    def test_adapter_passes_a_histogram_only_when_given(self):
        """DeepGEMM builds without a histogram reject the keyword; with one, the logits
        stay BF16 for LiteTopK instead of widening for the default top-k."""
        from sglang.srt.layers.attention import deep_gemm_mqa_logits as mqa

        logits = torch.ones((2, 64), dtype=torch.bfloat16)
        producer = SimpleNamespace(fp8_fp4_paged_mqa_logits=Mock(return_value=logits))
        args = (torch.zeros((2, 1, 32, 128)), torch.zeros(1), torch.zeros((2, 32)))
        args += (None,) * 3
        histogram = torch.zeros((2, 1024), dtype=torch.int32)
        with patch.dict(sys.modules, {"deep_gemm": producer}):
            widened = mqa.fp8_paged_mqa_logits(*args, 64)
            raw = mqa.fp8_paged_mqa_logits(*args, 64, histogram=histogram)
        default, served = producer.fp8_fp4_paged_mqa_logits.call_args_list
        self.assertNotIn("histogram", default.kwargs)
        self.assertEqual(widened.dtype, torch.float32)
        self.assertIs(served.kwargs["histogram"], histogram)
        self.assertIs(raw, logits)

    def test_causal_verify_lengths_only_where_litetopk_serves_verify(self):
        """LiteTopK's histogram counts the keys each verify token sees; with another
        top-k backend or the flag off, verify keeps the final length."""
        from sglang.srt.layers.attention import dsa_backend

        sgl_kernel, flashinfer = DSATopKBackend.SGL_KERNEL, DSATopKBackend.FLASHINFER
        for buffers, topk_backend, want in (
            (object(), sgl_kernel, True),
            (object(), flashinfer, False),
            (None, sgl_kernel, False),
        ):
            with patch.object(dsa_backend, "get_litetopk_decode", return_value=buffers):
                got = dsa_backend._litetopk_causal_verify_ctx_lens(topk_backend)
            self.assertEqual(got, want, topk_backend)
        cache = torch.tensor([10, 20], dtype=torch.int32)
        expanded = torch.tensor([8, 9, 10, 18, 19, 20], dtype=torch.int32)
        with patch.object(
            dsa_backend, "get_platform", return_value=SimpleNamespace(is_sm100=True)
        ):
            for attrs, want in (
                ({}, [[10, 10, 10], [20, 20, 20]]),  # the class default
                (dict(paged_mqa_causal_ctx_lens=True), [[8, 9, 10], [18, 19, 20]]),
            ):
                backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
                vars(backend).update(speculative_num_draft_tokens=3, **attrs)
                got = backend._build_paged_mqa_schedule_2d_ctx_lens(
                    ForwardMode.TARGET_VERIFY, cache, expanded, 2
                )
                self.assertEqual(got.tolist(), want)
                # Graph replays overwrite these lengths: they must not alias `expanded`.
                self.assertNotEqual(got.data_ptr(), expanded.data_ptr())


# ---- DeepSeek-V4.1 indexer


def _v41_call(hook, rows, *, heads=32, head_bytes=64, schedule=True, out_cols=512):
    data = SimpleNamespace(
        q_fp4=torch.zeros((rows, 1, heads, head_bytes), dtype=torch.int8),
        q_sf=torch.zeros((rows, 1, heads), dtype=torch.int32),
        weights=torch.zeros((rows, heads)),
    )
    metadata = SimpleNamespace(
        litetopk_schedule=torch.zeros((9, 2), dtype=torch.int32) if schedule else None,
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
            dict(schedule=False),  # no schedule for this forward
            dict(heads=64),  # DeepSeek-V4.1 has 32 index heads
            dict(head_bytes=32),  # of 128 dimensions (64 bytes of packed e2m1)
            dict(out_cols=1024),  # another top-k
        ):
            self.assertIsNone(_v41_call(self.hook, 2, **kwargs), kwargs)
        self.buffers.plan.assert_not_called()

    def test_per_forward_schedule(self):
        """Decode reuses the forward's own DeepGEMM schedule (one request per row)
        unless TileLang's JIT builder made it; verify rows of a request share one ID;
        forwards LiteTopK cannot serve get no schedule."""
        build = Mock(return_value="own")
        producer = SimpleNamespace(
            get_paged_mqa_logits_metadata=build, get_num_sms=lambda: 8
        )
        main = torch.empty((9, 2), dtype=torch.int32)
        base = dict(
            compressed_page_size=128,
            compressed_seq_lens=torch.full((12,), 1000, dtype=torch.int32),
            row_chunk=0,
            deep_gemm_metadata=main,
            force_deep_gemm_metadata=False,
        )
        for change, tokens, tilelang, want in (
            (dict(compressed_page_size=64), 1, False, None),
            (dict(row_chunk=4), 1, False, None),
            (dict(deep_gemm_metadata=[main]), 1, False, None),  # chunked rows
            ({}, 5, False, None),  # rows that are not whole requests
            ({}, 1, False, main),
            ({}, 1, True, "own"),
            ({}, 4, False, "own"),
        ):
            metadata = SimpleNamespace(**{**base, **change})
            with (
                patch.dict(sys.modules, {"deep_gemm": producer}),
                envs.SGLANG_OPT_USE_TILELANG_INDEXER.override(tilelang),
            ):
                self.hook.prepare_metadata(metadata, tokens)
            self.assertIs(metadata.litetopk_schedule, want, change)
            ids = metadata.litetopk_request_ids
            want_ids = None if want is None else [r // tokens for r in range(12)]
            self.assertEqual(None if ids is None else ids.tolist(), want_ids)
        self.assertEqual(
            [c.kwargs["indices"].tolist() for c in build.call_args_list],
            [list(range(12)), [r // 4 for r in range(12)]],
        )

    def test_metadata_copy_takes_the_live_schedule(self):
        """Graph replays copy the live metadata (built without a LiteTopK schedule)
        into the captured one; the captured graph rebuilds its own schedule."""
        from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

        def make():
            return PagedIndexerMetadata(
                page_size=256,
                compressed_page_size=128,
                page_table=torch.zeros((4, 8), dtype=torch.int32),
                compressed_seq_lens=torch.full((4,), 1000, dtype=torch.int32),
                use_topk_v2=False,
            )

        with envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True):
            captured, live = make(), make()
            captured.litetopk_schedule = torch.zeros((9, 2), dtype=torch.int32)
            captured.litetopk_request_ids = torch.zeros((4,), dtype=torch.int32)
            captured.copy_(live)
        self.assertIsNone(captured.litetopk_schedule)
        self.assertIsNone(captured.litetopk_request_ids)

    def test_backend_prepares_every_forward(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as backend
        from sglang.srt.model_executor.runner_utils import capture_mode

        for n, mode in ((1, ForwardMode.DECODE), (6, ForwardMode.TARGET_VERIFY)):
            hook = Mock()
            c1, c2 = object(), object()
            metadata = backend.DSV4Metadata(
                core_attn_metadata=SimpleNamespace(low_ratios=(1, 2)),
                indexer_metadata=None,
                c1_indexer_metadata=c1,
                c2_indexer_metadata=c2,
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
            with patch.object(
                capture_mode, "skip_low_ratio_indexer", return_value=False
            ):
                backend.DeepseekV4AttnBackend.init_forward_metadata_in_graph(obj, batch)
            self.assertEqual(
                [c.args for c in hook.prepare_metadata.call_args_list],
                [(c1, n), (c2, n)],
            )


if __name__ == "__main__":
    unittest.main()
