"""PLE side states share physical working and tracking rows with recurrent state."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from test_declared_parts import _build
from torch.overrides import TorchFunctionMode

from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART, NGramPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpPLELayer,
    _commit_ple_batch,
    _PLEBatch,
    _prepare_ple_batch,
)
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestPLEStateCoordinates(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_working_and_tracking_rows_after_state_compaction(self):
        for moved in (False, True):
            for replay in (False, True):
                for fast in (False, True):
                    with self.subTest(moved=moved, replay=replay, fast=fast):
                        self._check_rows(moved, replay, fast)

    def _check_rows(self, moved, replay, fast):
        storage, _, _ = _build(state=True, ple=True)
        alloc = storage.allocator
        pool = alloc.get_kvcache()
        alloc.state.bind_routes(pool.layer_router)
        slots = alloc.state_slots.alloc(8)
        if moved:
            alloc.state_slots.free(slots[:4])
            alloc.mamba_allocator.flush_for_allocation()
        working, tracking = slots[-1:], slots[-2:-1]
        physical = alloc.mamba_allocator.ids.translate(working)
        track_physical = alloc.mamba_allocator.ids.translate(tracking)
        self.assertEqual(torch.equal(working, physical), not moved)
        req = storage.requests
        req.req_index_to_mamba_index_mapping[1] = working[0]
        reader = RunnerIds(
            req_to_token=req.req_to_token,
            token_to_kv_pool_allocator=alloc,
            token_to_kv_pool=pool,
            page_size=4,
            device="cpu",
        )
        backend = object.__new__(MambaAttnBackendBase)
        backend.device = "cpu"
        backend.topk = 1
        backend.req_to_token_pool = req
        backend.state_pool = alloc.state.pool
        backend._state_ids = reader.state_ids(req.state_sub_pool)
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([99]),
            req_pool_indices=torch.tensor([1]),
            seq_lens=torch.tensor([8]),
            seq_lens_sum=8,
            seq_lens_cpu=torch.tensor([8]),
            out_cache_loc=torch.tensor([1]),
            mamba_track_indices=tracking.clone(),
            mamba_track_mask=torch.tensor([True]),
        )
        if replay:
            backend._fused_state_indices_ok = False
            backend.replayssm_write_pos_list = None
            backend.replayssm_force_flush_list = None
            backend.replayssm_ring_indices_list = None
            backend.state_indices_list = [torch.zeros(1, dtype=torch.int32)]
            backend.query_start_loc_list = [torch.zeros(2, dtype=torch.int32)]
            backend.cached_cuda_graph_decode_query_start_loc = torch.arange(2)
            backend.mamba_track_indices_buf = torch.zeros(1, dtype=torch.int64)
            backend.forward_metadata = backend._capture_metadata(
                1, fb.req_pool_indices, fb.forward_mode, None
            )
            captured_tracks = backend.state_tracking_indices
            backend.forward_metadata = backend._replay_metadata(
                1,
                fb.req_pool_indices,
                fb.forward_mode,
                None,
                fb.seq_lens_cpu,
                num_padding=0,
                mamba_track_indices=fb.mamba_track_indices,
            )
            self.assertEqual(
                captured_tracks.data_ptr(), backend.state_tracking_indices.data_ptr()
            )
            torch.testing.assert_close(fb.mamba_track_indices, tracking)
        else:
            backend.init_forward_metadata(fb)
            torch.testing.assert_close(
                fb.mamba_track_indices, track_physical, check_dtype=False
            )
        ngram = alloc.state.ngram_pool
        ngram.context.copy_(
            torch.arange(ngram.context.shape[0])[:, None] * 100
            + torch.tensor([11, 12, 13])
        )
        history = ngram.get_context(physical).clone()
        before_ngram = ngram.context.clone()
        conv = alloc.state.short_conv_pool.layer_cache(1)
        conv.copy_(torch.arange(conv.numel()).reshape_as(conv))
        before_conv = conv.clone()
        x = torch.tensor([[1.0, 2.0, 3.0]])
        expected_conv = torch.cat((conv[physical].float(), x.unsqueeze(-1)), -1)
        ple = SimpleNamespace(
            layer_id=1,
            conv_channels=3,
            short_conv_dilation=1,
            short_conv_state_len=4,
            conv1d=SimpleNamespace(weight=torch.ones(3, 1, 5)),
        )
        with (
            forward_context(
                ForwardContext(
                    attn_backend=SimpleNamespace(
                        req_to_token_pool=req,
                        token_to_kv_pool=pool,
                        runner_ids=reader,
                        linear_attn_backend=backend,
                        forward_state_slots=backend.forward_state_slots,
                    )
                )
            ),
            envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.override(fast),
        ):
            batch = _prepare_ple_batch(
                fb.input_ids, fb, ngram_size=4, ngram_eos_token_id=17
            )
            torch.testing.assert_close(
                batch.ngram_context, torch.cat((history, fb.input_ids[:, None]), 1)
            )
            actual = Qwen4ExpPLELayer._short_conv(ple, x, fb, batch)
            torch.testing.assert_close(
                actual, torch.nn.functional.silu(expected_conv.sum(-1))
            )
            _commit_ple_batch(batch, fb)
        next_history = torch.cat((history[:, 1:], fb.input_ids[:, None]), 1)
        for dst in (physical, track_physical):
            before_ngram[dst] = next_history
            before_conv[dst] = expected_conv[:, :, 1:].to(conv.dtype)
        torch.testing.assert_close(ngram.context, before_ngram)
        torch.testing.assert_close(conv, before_conv)
        self.assertEqual(
            req.get_mamba_indices(fb.req_pool_indices).tolist(), working.tolist()
        )
        self.assertEqual(storage.accounting_violations(), [])

    def test_ascend_graph_metadata_checkpoints_static_ple_state(self):
        """Capture and replay save both PLE parts to the recurrent checkpoint rows."""
        from sglang.srt.mem_cache.kv.decl import STATE_PART
        from sglang.srt.mem_cache.kv.router import KVLayerRouter
        from sglang.srt.mem_cache.ple_state_pool import (
            NGRAM_PART,
            SHORT_CONV_PART,
            NGramPool,
            ShortConvPool,
        )
        from sglang.test.unified_allocator_fixtures import setup_allocator_context

        setup_allocator_context()
        # Import-only NPU kernel dependencies; this test executes the production
        # Ascend metadata methods and PLE tensor writes on CPU.
        npu_ops = ModuleType("sgl_kernel_npu.mamba.mamba_state_update_triton")
        npu_ops.conv_state_rollback = None
        npu_ops.move_intermediate_cache = None
        with patch.dict("sys.modules", {npu_ops.__name__: npu_ops}):
            from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
                AscendMambaAttnBackendBase,
            )

        for fast in (False, True):
            with self.subTest(fast=fast):
                conv_pool = ShortConvPool(
                    size=4,
                    state_shape=(1, 2),
                    layer_ids=[0],
                    dtype=torch.float32,
                    device="cpu",
                )
                ngram = NGramPool(size=4, context_len=2, eos_token_id=17, device="cpu")
                pool = SimpleNamespace(layer_transfer_counter=None)
                pool.layer_router = KVLayerRouter(
                    pool, sub_pools=(), views={}, layers={}
                )
                for part, view in (
                    (STATE_PART, SimpleNamespace()),
                    (SHORT_CONV_PART, conv_pool),
                    (NGRAM_PART, ngram),
                ):
                    pool.layer_router.bind_part(
                        sub_pool="mamba", part=part, view=view, indices={0: 0}
                    )
                mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
                req = SimpleNamespace(
                    size=2,
                    state_sub_pool="mamba",
                    holds_state_slots=True,
                    get_mamba_indices=lambda rows: mapping[rows],
                )
                reader = SimpleNamespace(
                    state_ids=lambda _: SimpleNamespace(ids=lambda indices: indices),
                    request_ids=lambda: SimpleNamespace(ids=lambda rows: rows),
                )
                backend = AscendMambaAttnBackendBase(
                    SimpleNamespace(
                        server_args=SimpleNamespace(enable_mis=False),
                        device="cpu",
                        is_draft_worker=False,
                        req_to_token_pool=req,
                        token_to_kv_pool=pool,
                        runner_ids=reader,
                    )
                )
                backend.init_cuda_graph_state(2, 2)
                tracking = torch.tensor([3, 4])
                fb = ForwardBatch(
                    forward_mode=ForwardMode.DECODE,
                    batch_size=2,
                    input_ids=torch.tensor([99, 77]),
                    req_pool_indices=torch.tensor([1, 2]),
                    seq_lens=torch.tensor([8, 8]),
                    seq_lens_sum=16,
                    seq_lens_cpu=torch.tensor([8, 8]),
                    out_cache_loc=torch.tensor([1, 2]),
                    mamba_track_indices=tracking,
                    mamba_track_mask=torch.tensor([True, False]),
                )
                conv = conv_pool.layer_cache(0)
                conv.copy_(torch.arange(10).reshape_as(conv))
                ngram.context.copy_(torch.arange(10).reshape_as(ngram.context))
                ple = SimpleNamespace(
                    layer_id=0,
                    conv_channels=1,
                    short_conv_dilation=1,
                    short_conv_state_len=2,
                    conv1d=SimpleNamespace(weight=torch.ones(1, 1, 3)),
                )
                with (
                    forward_context(
                        ForwardContext(
                            attn_backend=SimpleNamespace(
                                req_to_token_pool=req,
                                token_to_kv_pool=pool,
                                runner_ids=reader,
                                linear_attn_backend=backend,
                                forward_state_slots=backend.forward_state_slots,
                            )
                        )
                    ),
                    envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.override(fast),
                ):
                    for phase in ("capture", "replay", "untracked"):
                        if phase == "replay":
                            tracking.copy_(torch.tensor([4, 3]))
                        elif phase == "untracked":
                            fb.mamba_track_indices = None
                            fb.mamba_track_mask = None
                        backend.init_forward_metadata_out_graph(
                            fb, in_capture=phase == "capture"
                        )
                        expected_conv = conv.clone()
                        expected_ngram = ngram.context.clone()
                        x = torch.tensor([[4.0], [6.0]])
                        next_conv = torch.cat((conv[1:3, :, 1:], x.unsqueeze(-1)), -1)
                        next_ngram = torch.cat(
                            (ngram.context[1:3, 1:], fb.input_ids[:, None]), -1
                        )
                        expected_conv[1:3] = next_conv
                        expected_ngram[1:3] = next_ngram
                        if phase != "untracked":
                            # Masked rows use the reserved sink without a dynamic
                            # selection; the disabled checkpoint itself is untouched.
                            destinations = torch.tensor([tracking[0], 0])
                            expected_conv[destinations] = next_conv
                            expected_ngram[destinations] = next_ngram
                            self.assertEqual(
                                backend.forward_metadata.mamba_track_indices.data_ptr(),
                                tracking.data_ptr(),
                            )
                        batch = _prepare_ple_batch(
                            fb.input_ids, fb, ngram_size=3, ngram_eos_token_id=17
                        )
                        Qwen4ExpPLELayer._short_conv(ple, x, fb, batch)
                        _commit_ple_batch(batch, fb)
                        torch.testing.assert_close(conv, expected_conv)
                        torch.testing.assert_close(ngram.context, expected_ngram)
                    fb.forward_mode = ForwardMode.IDLE
                    backend.init_forward_metadata_out_graph(fb)
                    batch = _prepare_ple_batch(
                        fb.input_ids, fb, ngram_size=3, ngram_eos_token_id=17
                    )
                    self.assertIsNone(batch)
                    _commit_ple_batch(batch, fb)
                    torch.testing.assert_close(conv, expected_conv)
                    torch.testing.assert_close(ngram.context, expected_ngram)

    def test_verify_ngram_commit_builds_the_row_mask_only_for_stable_rows(self):
        """Without PP spec stable rows the verify commit writes positional rows and
        computes no row mask; with them, request rows and the padding row."""

        class CountNe(TorchFunctionMode):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                if func is torch.Tensor.ne:
                    self.calls += 1
                return func(*args, **(kwargs or {}))

        # Request 0 verifies two tokens; request 1 is a padding row.
        batch = _PLEBatch(
            mode=ForwardMode.TARGET_VERIFY,
            use_decode_fast_path=False,
            physical_tokens=4,
            processed_tokens=2,
            lengths=torch.tensor([2, 0]),
            row_width=2,
            req_indices=torch.tensor([0, 0]),
            token_offsets=torch.tensor([0, 1]),
            valid_tokens=torch.tensor([True, True, False, False]),
            state_indices=torch.tensor([0, 0]),
            query_start_loc=None,
            ngram_context=torch.tensor([[5, 6, 7, 8], [17, 17, 17, 17]]),
            ngram_eos_token_id=17,
        )
        fb = SimpleNamespace(req_pool_indices=torch.tensor([3, 1]))
        step_contexts = torch.tensor([[[6, 7], [7, 8]], [[17, 17], [17, 17]]])
        for stable_rows, rows in ((False, [0, 1]), (True, [3, 4])):
            with self.subTest(stable_rows=stable_rows):
                ngram = NGramPool(
                    size=4,
                    context_len=2,
                    eos_token_id=17,
                    device="cpu",
                    spec_state_size=4,
                    speculative_num_draft_tokens=2,
                )
                ngram.intermediate_context.fill_(-1)
                expected = ngram.intermediate_context.clone()
                expected[rows] = step_contexts
                count_ne = CountNe()
                with (
                    forward_context(
                        ForwardContext(
                            attn_backend=SimpleNamespace(
                                req_to_token_pool=SimpleNamespace(size=4),
                                token_to_kv_pool=SimpleNamespace(
                                    layer_router=SimpleNamespace(
                                        part_view={NGRAM_PART: ngram}.get
                                    )
                                ),
                            )
                        )
                    ),
                    envs.SGLANG_ENABLE_PP_SPEC.override(stable_rows),
                    count_ne,
                ):
                    _commit_ple_batch(batch, fb)
                torch.testing.assert_close(ngram.intermediate_context, expected)
                self.assertEqual(count_ne.calls, int(stable_rows))


if __name__ == "__main__":
    unittest.main()
