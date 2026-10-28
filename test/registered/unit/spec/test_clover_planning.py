# SPDX-License-Identifier: Apache-2.0
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferIndicesUpdaterPrefill,
    fast_prefill_plan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.clover_draft import CloverPreviewFlashInferBackend
from sglang.srt.speculative.clover_info import CloverPreviewInput
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCloverPlanning(CustomTestCase):
    def test_verify_preplan_marker_and_shape_change(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.speculative.clover_info import CloverVerifyInput

        verify = CloverVerifyInput.__new__(CloverVerifyInput)
        for mode, ready in [
            (ForwardMode.TARGET_VERIFY, True),
            (ForwardMode.IDLE, False),
        ]:
            fb = ForwardBatch.__new__(ForwardBatch)
            fb.batch_size = 1
            fb.input_ids = torch.empty(16, dtype=torch.int64)
            with patch(
                "sglang.srt.speculative.clover_info.DFlashVerifyInput.prepare_for_verify",
                return_value=(fb, True),
            ):
                result, graph = verify.prepare_for_verify(
                    SimpleNamespace(forward_mode=mode), None
                )
            self.assertIs(result, fb)
            self.assertTrue(graph)
            self.assertEqual(fb.needs_forward_metadata_init(), not ready)
            fb.input_ids = torch.empty(32, dtype=torch.int64)
            self.assertTrue(fb.needs_forward_metadata_init())

    def test_capture_planner_preserves_existing_routes(self):
        # Install only after capture's ordinary plan initialized the wrapper.
        cases = [
            (
                ForwardMode.TARGET_VERIFY,
                SpecInputType.CLOVER_VERIFY,
                True,
                "fa2",
                None,
                False,
                True,
            ),
            (
                ForwardMode.TARGET_VERIFY,
                SpecInputType.DFLASH_VERIFY,
                True,
                "fa2",
                None,
                False,
                True,
            ),
            (
                ForwardMode.TARGET_VERIFY,
                SpecInputType.EAGLE_VERIFY,
                True,
                "fa2",
                None,
                False,
                False,
            ),
            (
                ForwardMode.TARGET_VERIFY,
                SpecInputType.CLOVER_VERIFY,
                True,
                "fa2",
                None,
                True,
                False,
            ),
            (
                ForwardMode.DRAFT_EXTEND_V2,
                None,
                True,
                "fa2",
                None,
                False,
                True,
            ),
            (
                ForwardMode.DLLM_EXTEND,
                SpecInputType.CLOVER_DRAFT,
                True,
                "fa2",
                None,
                False,
                True,
            ),
            (
                ForwardMode.DLLM_EXTEND,
                SpecInputType.CLOVER_DRAFT,
                False,
                "fa2",
                None,
                False,
                False,
            ),
            (ForwardMode.DLLM_EXTEND, None, True, "fa2", None, False, False),
            (
                ForwardMode.DLLM_EXTEND,
                SpecInputType.CLOVER_DRAFT,
                True,
                "fa3",
                None,
                False,
                False,
            ),
            (
                ForwardMode.DLLM_EXTEND,
                SpecInputType.CLOVER_DRAFT,
                True,
                "fa2",
                "swa",
                False,
                False,
            ),
        ]
        for backend_type in (FlashInferAttnBackend, CloverPreviewFlashInferBackend):
            for mode, kind, paged, impl, dispatch, masked, fast in cases:
                with self.subTest(
                    backend=backend_type.__name__,
                    mode=mode,
                    kind=kind,
                    paged=paged,
                    impl=impl,
                    dispatch=dispatch,
                    masked=masked,
                ):
                    backend = backend_type.__new__(backend_type)
                    regular = Mock()
                    wrapper = SimpleNamespace(begin_forward=regular)
                    backend.prefill_cuda_graph_metadata = {1: [wrapper]}
                    backend.draft_extend_cuda_graph_metadata = {1: [wrapper]}
                    backend.prefill_backend = impl
                    backend.dispatch_reason = dispatch
                    backend.use_paged = paged
                    backend.dllm_config = SimpleNamespace(block_size=16)
                    backend.use_sliding_window_kv_pool = False
                    backend._prepare_cuda_graph_metadata = Mock()

                    def update(*args, **kwargs):
                        # Dropping preview metadata would silently select normal
                        # extend planning while capture still installs fast-plan.
                        if mode.is_dllm_extend():
                            self.assertIs(
                                kwargs["spec_info"],
                                (
                                    spec
                                    if backend_type is CloverPreviewFlashInferBackend
                                    else None
                                ),
                            )
                        wrapper.begin_forward()

                    backend.indices_updater_prefill = SimpleNamespace(update=update)
                    spec = (
                        None
                        if kind is None
                        else SimpleNamespace(
                            spec_input_type=kind,
                            custom_mask=torch.ones(1) if masked else None,
                            ragged_verify_layout=None,
                        )
                    )
                    if kind == SpecInputType.CLOVER_DRAFT:
                        spec = CloverPreviewInput(16)
                        spec.custom_mask = torch.ones(1) if masked else None
                    batch = SimpleNamespace(
                        batch_size=1,
                        forward_mode=mode,
                        spec_info=spec,
                        req_pool_indices=torch.tensor([0]),
                        seq_lens=torch.tensor([101]),
                        seq_lens_cpu=torch.tensor([101]),
                        seq_lens_sum=101,
                        encoder_lens=None,
                        positions=torch.arange(16),
                        kv_loc_plan=object(),
                        out_cache_loc=None,
                    )
                    if (
                        backend_type is CloverPreviewFlashInferBackend
                        and mode.is_dllm_extend()
                        and spec is None
                    ):
                        with self.assertRaisesRegex(TypeError, "CloverPreviewInput"):
                            backend.init_forward_metadata_out_graph(
                                batch, in_capture=True
                            )
                        continue
                    backend.init_forward_metadata_out_graph(batch, in_capture=True)
                    self.assertEqual(regular.call_count, 1)
                    if fast and (
                        not mode.is_dllm_extend()
                        or backend_type is CloverPreviewFlashInferBackend
                    ):
                        self.assertIs(wrapper.begin_forward.func, fast_prefill_plan)
                    else:
                        self.assertIs(wrapper.begin_forward, regular)

    def test_preview_worker_and_capture_share_width_contract(self):
        from sglang.srt.speculative.clover_draft import CloverDraftCudaGraphRunner
        from sglang.srt.speculative.clover_worker import CloverWorker

        # Worker batches and capture batches must agree even when the capture
        # bucket pads a live batch. A missing spec silently drops the fast path.
        for width in (2, 4, 8, 16, 32):
            for bs in (1, 2, 3, 4):
                with self.subTest(width=width, bs=bs):
                    worker = CloverWorker.__new__(CloverWorker)
                    worker.block_size = width
                    worker.device = "cpu"
                    worker.model_runner = SimpleNamespace(kv_index_translator=None)
                    prefixes = torch.arange(bs) * 7 + 7
                    batch = SimpleNamespace(
                        batch_size=lambda: bs,
                        seq_lens=prefixes,
                        seq_lens_cpu=prefixes,
                        req_pool_indices=torch.arange(bs),
                        sampling_info=None,
                        spec_algorithm=None,
                    )
                    fb = worker._preview_batch(
                        batch,
                        torch.zeros(bs, width, dtype=torch.int64),
                        (prefixes[:, None] + torch.arange(width)).reshape(-1),
                        torch.arange(bs * width),
                        SimpleNamespace(bind=lambda *args: None),
                    )
                    runner = CloverDraftCudaGraphRunner.__new__(
                        CloverDraftCudaGraphRunner
                    )
                    runner.captured_req_width = width
                    padded_bs = 4 if bs == 3 else bs
                    for spec in (fb.spec_info, runner.get_spec_info(padded_bs * width)):
                        self.assertIsInstance(spec, CloverPreviewInput)
                        self.assertTrue(spec.is_draft_input())
                        self.assertEqual(spec.num_tokens_per_req, width)
                        self.assertEqual(spec.draft_token_num, width)
                        self.assertEqual(spec.num_tokens_for_logprob_per_req, width)
                        self.assertIsNone(spec.hidden_states)
                    self.assertTrue(fb.forward_mode.is_dllm_extend())
                    self.assertEqual(
                        fb.seq_lens_cpu.tolist(), (prefixes + width).tolist()
                    )
                    self.assertEqual(fb.seq_lens_sum, int(prefixes.sum()) + bs * width)
                    self.assertEqual(fb.extend_seq_lens_cpu, [width] * bs)
                    self.assertEqual(
                        fb.extend_start_loc.tolist(), list(range(0, bs * width, width))
                    )

    def test_preview_host_layout_uses_batch_speculative_width(self):
        # Preview KV lengths include the block already. Host CSR must match query layout
        # for both graph buckets and changing KV lengths, without stale values.
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.attn_backend = SimpleNamespace(
            kv_index_translator=SimpleNamespace(
                reads_are_translated=False,
                pack_read_stream=lambda plan, **kwargs: kwargs["out"].zero_(),
            ),
            dq_paged_kernel_lens=None,
        )
        updater.num_qo_heads = 16
        updater.num_kv_heads = 2
        updater.head_dim = 128
        updater.q_data_type = updater.data_type = torch.bfloat16
        updater.kv_last_page_len = torch.ones(2, dtype=torch.int32)
        snapshots = []

        def capture(wrapper, qo, kv, *args, **kwargs):
            torch.testing.assert_close(qo, kwargs["qo_indptr_host"])
            torch.testing.assert_close(kv, kwargs["kv_indptr_host"])
            snapshots.append((qo.tolist(), kv.tolist(), kwargs["max_q_len"]))

        with patch(
            "sglang.srt.layers.attention.flashinfer_backend.fast_prefill_plan",
            side_effect=capture,
        ) as fast:
            wrapper = SimpleNamespace(
                begin_forward=partial(fast, None),
                is_cuda_graph_enabled=True,
                _custom_mask_buf=None,
            )
            for width, lengths in [(16, [31]), (16, [105]), (32, [43, 59])]:
                seq_lens = torch.tensor(lengths, dtype=torch.int32)
                bs = len(lengths)
                updater.call_begin_forward(
                    None,
                    wrapper,
                    torch.arange(bs),
                    seq_lens,
                    sum(lengths),
                    seq_lens,
                    seq_lens - width,
                    None,
                    torch.zeros(bs + 1, dtype=torch.int32),
                    torch.zeros(bs + 1, dtype=torch.int32),
                    False,
                    CloverPreviewInput(width),
                    seq_lens_cpu=seq_lens,
                    custom_kv_indices=torch.zeros(sum(lengths), dtype=torch.int32),
                    plan=object(),
                )
        self.assertEqual(
            snapshots,
            [
                ([0, 16], [0, 31], 16),
                ([0, 16], [0, 105], 16),
                ([0, 32, 64], [0, 43, 102], 32),
            ],
        )


if __name__ == "__main__":
    unittest.main()
