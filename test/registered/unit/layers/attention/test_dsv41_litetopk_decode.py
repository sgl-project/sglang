"""The opt-in LiteTopK hook of the DeepSeek-V4.1 decode index top-k: off by default, a
fallback when unsupported, and selector buffers that captured CUDA graphs can rely
on."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.kernels.experimental.litetopk_decode as litetopk_decode_package
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.v41_indexer import litetopk
from sglang.srt.layers.attention.dsv4.v41_indexer.types import Selection
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FUSED = "sglang.kernels.experimental.litetopk_decode.bf16"


def _fake_fused(storages: list) -> types.ModuleType:
    """Stands in for the experimental module (DeepGEMM + JIT); records the
    storages it allocates."""

    class FakePlan:
        def __init__(self, rows, storage):
            self.rows = rows
            self.storage = storage

        def scores(self, **kwargs):
            return torch.zeros((self.rows, 256))

        def select(self, *, scores, context_lens, block_table, out):
            return out.fill_(7)

    class FakeStorage:
        def __init__(self, max_rows, device, candidate_capacity=8192):
            storages.append(max_rows)
            self.max_rows = max_rows
            self.nbytes = max_rows * 72 * 1024
            self.histogram = self.workspace = self.output = torch.zeros(1)

        def plan(self, rows):
            assert 1 <= rows <= self.max_rows
            return FakePlan(rows, self)

    module = types.ModuleType(_FUSED)
    module.Bf16Dsv41DecodeStorage = FakeStorage
    module.TOPK = 512
    module.PAGE_SIZE = 128
    return module


def _call(
    hook,
    rows: int,
    page_size: int = 128,
    heads: int = 32,
    head_bytes: int = 64,
    weights_dtype: torch.dtype = torch.float32,
    device="cpu",
):
    data = SimpleNamespace(
        q_fp4=torch.zeros((rows, 1, heads, head_bytes), dtype=torch.int8),
        q_sf=torch.zeros((rows, 1, heads), dtype=torch.int32),
        k_cache=None,
        weights=torch.zeros((rows, heads), dtype=weights_dtype),
    )
    metadata = SimpleNamespace(
        compressed_page_size=page_size,
        compressed_seq_lens=torch.full((rows,), 1000, dtype=torch.int32),
        page_table=torch.zeros((rows, 8), dtype=torch.int32),
        deep_gemm_metadata=None,
        bf16_schedule=torch.empty((1, 2), dtype=torch.int32),
        bf16_indices=torch.arange(rows, dtype=torch.int32),
        bf16_tokens_per_request=1,
        max_compressed_seq_len=1024,
    )
    out = Selection(
        page_indices=torch.full((rows, 512), -1, dtype=torch.int32, device=device),
        raw_indices=None,
    )
    logits = hook.scores(data=data, metadata=metadata, out=out)
    if logits is not None:
        hook.select(logits=logits, metadata=metadata, out=out)
    return logits, out


class TestLiteTopKDecodeSwitch(CustomTestCase):
    def setUp(self):
        litetopk.get_litetopk_decode.cache_clear()
        self.addCleanup(litetopk.get_litetopk_decode.cache_clear)

    def test_off_by_default(self):
        """Unset, the hook is None and its startup probe (GPU kernels) never runs."""
        probe_ran = AssertionError("the support probe ran with the hook off")
        with (
            envs.SGLANG_OPT_DSV41_LITETOPK_DECODE.override(True),
            patch.object(litetopk, "_unsupported_reason", side_effect=probe_ran),
        ):
            envs.SGLANG_OPT_DSV41_LITETOPK_DECODE.clear()
            self.assertIsNone(litetopk.get_litetopk_decode())

    def test_unsupported_keeps_default_top_k(self):
        with (
            envs.SGLANG_OPT_DSV41_LITETOPK_DECODE.override(True),
            patch.object(
                litetopk, "_unsupported_reason", return_value="needs an SM100 GPU"
            ),
            self.assertLogs(litetopk.logger, level="WARNING") as logs,
        ):
            self.assertIsNone(litetopk.get_litetopk_decode())
        self.assertIn("needs an SM100 GPU", "\n".join(logs.output))


class TestLiteTopKDecodePlans(CustomTestCase):
    """Plans are views of a shared storage. A captured graph replays the buffers it
    saw, so its plan and storage must outlive every eager batch, and nothing may
    be allocated while a graph captures."""

    def setUp(self):
        self.storages = []
        fake = _fake_fused(self.storages)
        self.capture_mode = False
        self.capturing = False
        for patcher in (
            patch.dict(sys.modules, {_FUSED: fake}),
            patch.object(litetopk_decode_package, "bf16", fake, create=True),
            patch.object(litetopk, "get_is_capture_mode", lambda: self.capture_mode),
            patch.object(
                torch.cuda, "is_current_stream_capturing", lambda: self.capturing
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.hook = litetopk.LiteTopKDecode(
            device=torch.device("cpu"), check=False, check_file=None
        )

    def test_selects_into_the_selection(self):
        logits, out = _call(self.hook, rows=3)
        self.assertIsNotNone(logits)
        self.assertTrue(bool((out.page_indices == 7).all()))

    def test_row_counts_share_one_storage(self):
        for rows in (8, 3, 5, 8, 1):
            _call(self.hook, rows=rows)
        self.assertEqual(self.storages, [8])
        self.assertEqual(len({id(p.storage) for p in self.hook._plans.values()}), 1)

    def test_storage_grows_geometrically(self):
        for rows in range(1, 101):
            _call(self.hook, rows=rows)
        self.assertEqual(self.storages, [1, 2, 4, 8, 16, 32, 64, 128])

    def test_eager_plans_move_to_a_grown_storage(self):
        _call(self.hook, rows=2)
        first = self.hook._plans[2]
        _call(self.hook, rows=5)
        _call(self.hook, rows=2)
        self.assertEqual(self.storages, [2, 5])
        self.assertIsNot(self.hook._plans[2], first)
        # Nothing refers to the first storage any more: it is freed
        self.assertTrue(
            all(p.storage is self.hook._storage for p in self.hook._plans.values())
        )

    def test_capture_session_plans_outlive_eager_batches(self):
        self.capture_mode = True
        _call(self.hook, rows=3)
        captured = self.hook._plans[3]
        self.capture_mode = False
        for rows in range(10, 100):
            _call(self.hook, rows=rows)
        _call(self.hook, rows=3)
        self.assertIs(self.hook._plans[3], captured)
        self.assertEqual(self.storages[0], 3)

    def test_eager_plan_reused_by_a_capture_is_kept(self):
        _call(self.hook, rows=4)
        plan = self.hook._plans[4]
        self.capture_mode = True
        _call(self.hook, rows=4)
        self.capture_mode = False
        for rows in range(10, 100):
            _call(self.hook, rows=rows)
        _call(self.hook, rows=4)
        self.assertIs(self.hook._plans[4], plan)

    def test_no_storage_is_allocated_during_capture(self):
        self.capture_mode = self.capturing = True
        logits, out = _call(self.hook, rows=5)
        self.assertIsNone(logits)
        self.assertEqual(self.storages, [])
        self.assertTrue(bool((out.page_indices == -1).all()))

    def test_capture_takes_a_view_of_the_storage(self):
        _call(self.hook, rows=8)
        self.capture_mode = self.capturing = True
        logits, _ = _call(self.hook, rows=5)
        self.assertIsNotNone(logits)
        self.assertEqual(self.storages, [8])

    def test_capture_beyond_the_storage_keeps_default_top_k(self):
        _call(self.hook, rows=4)
        self.capture_mode = self.capturing = True
        logits, _ = _call(self.hook, rows=9)
        self.assertIsNone(logits)
        self.assertEqual(self.storages, [4])

    def test_plan_from_warmup_serves_the_capture(self):
        self.capture_mode = True
        _call(self.hook, rows=5)
        self.capturing = True
        logits, _ = _call(self.hook, rows=5)
        self.assertIsNotNone(logits)
        self.assertEqual(self.storages, [5])

    def test_unsupported_calls_keep_default_top_k(self):
        for kwargs in (
            dict(page_size=64),  # another index-K page
            dict(heads=64),  # DeepGEMM's histogram needs 32 index heads
            dict(head_bytes=32),  # of 128 dimensions (64 bytes of packed e2m1)
            dict(weights_dtype=torch.float16),
            dict(device="meta"),  # not the selector's device
        ):
            logits, _ = _call(self.hook, rows=2, **kwargs)
            self.assertIsNone(logits, kwargs)
        self.assertEqual(self.storages, [])

    def test_metadata_rebuilt_for_every_forward(self):
        calls = []
        producer = types.ModuleType("deep_gemm")
        producer.get_num_sms = lambda: 148

        def schedule(lens, page, sms, **kwargs):
            calls.append((lens.clone(), page, sms, kwargs))
            return torch.full((149, 2), len(calls), dtype=torch.int32)

        producer.get_paged_mqa_logits_bf16_metadata = schedule
        metadata = SimpleNamespace(
            compressed_page_size=128,
            compressed_seq_lens=torch.full((12,), 1000, dtype=torch.int32),
            row_chunk=0,
            deep_gemm_metadata=torch.empty((149, 2), dtype=torch.int32),
        )
        with patch.dict(sys.modules, {"deep_gemm": producer}):
            for n in range(1, 7):
                ids = torch.arange(12, dtype=torch.int64) // n
                self.hook.prepare_metadata(metadata, ids, n)
                self.assertEqual(metadata.bf16_tokens_per_request, n)
                self.assertEqual(calls[-1][3]["tokens_per_request"], n)
                self.assertEqual(metadata.bf16_indices.dtype, torch.int32)
                self.assertTrue(torch.equal(calls[-1][3]["indices"], ids.int()))
            before = metadata.bf16_schedule
            metadata.compressed_seq_lens.add_(17)
            self.hook.prepare_metadata(metadata, ids, 6)
            self.assertIsNot(metadata.bf16_schedule, before)
            self.assertTrue(bool((calls[-1][0] == 1017).all()))
            self.hook.prepare_metadata(metadata, ids, 7)
            self.assertIsNone(metadata.bf16_schedule)
            self.assertIsNone(metadata.bf16_indices)
            self.assertEqual(len(calls), 7)


class TestLiteTopKBackendMetadata(CustomTestCase):
    def test_backend_forwards_cpu_hint_and_live_ragged_ids(self):
        from sglang.srt.layers.attention import deepseek_v4_backend as backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner_utils import capture_mode

        # Unequal adjacent runs must reach the schedule unchanged.
        ids = torch.tensor([9, 9, 3, 3, 3, 3, 3, 8], dtype=torch.int64)
        for n, mode in (
            (1, ForwardMode.DECODE),
            (5, ForwardMode.TARGET_VERIFY),
            (6, ForwardMode.TARGET_VERIFY),
        ):
            hook = Mock()
            c1, c2 = object(), object()
            metadata = backend.DSV4Metadata(
                core_attn_metadata=SimpleNamespace(low_ratios=[1, 2]),
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
                patch.object(
                    backend, "token_req_indices", return_value=ids
                ) as fallback,
            ):
                backend.DeepseekV4AttnBackend.init_forward_metadata_in_graph(obj, batch)
                fallback.assert_not_called()
                self.assertEqual(hook.prepare_metadata.call_count, 2)
                for call, paged in zip(hook.prepare_metadata.call_args_list, (c1, c2)):
                    self.assertIs(call.args[0], paged)
                    self.assertIs(call.args[1], ids)
                    self.assertEqual(call.args[2], n)
                metadata.low_ratio_req_indices = None
                backend.DeepseekV4AttnBackend.init_forward_metadata_in_graph(obj, batch)
                fallback.assert_called_once_with(batch)


if __name__ == "__main__":
    unittest.main()
