"""MTP must seed drafts once and preserve in-flight PDMux prefill state."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
)
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_prefill_graph,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4_nextn import DeepseekV4ModelNextN
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_common import build_eagle_verify_input
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    get_plan_stream,
    pdmux_prefill_handoff,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEaglePDMux(unittest.TestCase):
    def make_worker(self, idle=False):
        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        target = SimpleNamespace(
            set_hicache_consumer=Mock(),
            _maybe_finalize_elastic_cuda_graph_scale=Mock(),
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(num_hidden_layers=3)
            ),
        )
        target.forward_batch_split_prefill = lambda batch, **kwargs: (
            TpModelWorker.forward_batch_split_prefill(target, batch, **kwargs)
        )
        worker._target_worker = target
        worker._draft_worker = SimpleNamespace(
            draft_runner=SimpleNamespace(tp_group=object()),
            draft_owns_attention=False,
        )
        ids = torch.empty(0, dtype=torch.long) if idle else torch.tensor([1, 2, 3, 4])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE if idle else ForwardMode.SPLIT_PREFILL,
            input_ids=ids,
            split_index=0,
            split_forward_count=1,
            split_forward_batch=None,
            extend_num_tokens=ids.numel(),
            seq_lens=torch.tensor([] if idle else [2, 2], dtype=torch.int64),
            spec_info=None,
            hicache_consumer_index=9,
            sampling_info=Mock(),
        )
        persistent = SimpleNamespace(
            input_ids=torch.cat((ids, torch.tensor([99]))), split_index=0
        )
        hidden = torch.empty((0, 6)) if idle else torch.arange(24).reshape(4, 6).float()
        logits = SimpleNamespace(hidden_states=hidden, mm_input_embeds=None)
        sampled = torch.empty(0, dtype=torch.long) if idle else torch.tensor([5, 6])

        def forward(fb, split_forward_count):
            fb.split_index += split_forward_count
            return ModelRunnerOutput(
                logits_output=logits if fb.split_index == 3 else None,
                can_run_graph=False,
            )

        target.model_runner.forward = Mock(side_effect=forward)
        target.model_runner.sample = Mock(return_value=sampled)
        next_draft = object()

        def draft(local_batch, target_hidden, next_ids, mm):
            self.assertIs(target_hidden, hidden)
            self.assertIs(next_ids, sampled)
            self.assertEqual(
                local_batch.forward_mode,
                ForwardMode.IDLE if idle else ForwardMode.EXTEND,
            )
            torch.testing.assert_close(local_batch.input_ids, ids)
            self.assertIsNot(local_batch, batch)
            local_batch.input_ids = torch.tensor([-1])
            local_batch.spec_info = object()
            return next_draft

        worker._draft_worker._draft_extend_for_prefill = Mock(side_effect=draft)
        return worker, batch, persistent, next_draft

    def run_split_prefill(self, idle=False):
        worker, batch, persistent, next_draft = self.make_worker(idle)
        prefill_stream, decode_stream = Mock(), Mock()
        event = prefill_stream.record_event.return_value
        prefill_group = object()
        with (
            patch(
                "sglang.srt.managers.tp_worker.ForwardBatch.init_new",
                return_value=persistent,
            ) as init,
            patch(
                "sglang.srt.multiplex.pdmux_context.get_current_stream_idx",
                return_value=0,
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_stream_groups",
                return_value=[(prefill_stream, decode_stream)],
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_parallel",
                return_value=SimpleNamespace(tp_group=prefill_group),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.draft_tp_context",
                return_value=nullcontext(),
            ) as draft_context,
        ):
            for index in range(3):
                batch.split_index = index
                result = worker.forward_batch_split_prefill(batch)
                # Match scheduler: input_ids is discarded after every slice.
                batch.input_ids = None
                if index < 2:
                    self.assertIsNone(result.next_draft_input)
                    prefill_stream.wait_stream.assert_not_called()
                    worker._draft_worker._draft_extend_for_prefill.assert_not_called()
                    prefill_stream.record_event.assert_not_called()
                    decode_stream.wait_event.assert_not_called()
            self.assertEqual(init.call_count, 1)
            self.assertEqual(
                init.call_args.kwargs["capture_hidden_mode"], CaptureHiddenMode.FULL
            )
        self.assertIs(result.next_draft_input, next_draft)
        self.assertIs(result.new_seq_lens, batch.seq_lens)
        self.assertIsNone(batch.spec_info)
        self.assertEqual(
            batch.forward_mode, ForwardMode.IDLE if idle else ForwardMode.SPLIT_PREFILL
        )
        worker._draft_worker._draft_extend_for_prefill.assert_called_once()
        draft_context.assert_called_once_with(False)
        prefill_stream.wait_stream.assert_called_once_with(decode_stream)
        prefill_stream.record_event.assert_called_once_with()
        decode_stream.wait_event.assert_called_once_with(event)
        self.assertEqual(worker._target_worker.set_hicache_consumer.call_count, 3)

    def test_seeds_mtp_only_after_final_slice_without_mutating_target(self):
        self.run_split_prefill()

    def test_idle_dp_rank_participates_in_final_draft_extend(self):
        self.run_split_prefill(idle=True)

    def test_ordinary_prefill_keeps_full_capture_and_publishes_seq_lens(self):
        worker, batch, _, _ = self.make_worker()
        worker.speculative_algorithm = SimpleNamespace(is_standalone=lambda: False)
        output = SimpleNamespace(new_seq_lens=None)
        worker._target_worker.forward_batch_generation = Mock(return_value=output)
        worker._draft_worker = None
        publish = Mock()
        self.assertIs(worker._forward_prefill_batch(batch, on_publish=publish), output)
        worker._target_worker.forward_batch_generation.assert_called_once_with(
            batch, pp_proxy_tensors=None, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        publish.assert_called_once_with(batch.seq_lens)

    def make_eager(self, draft=False):
        runner = EagerRunner.__new__(EagerRunner)
        runner.enable_pdmux = True
        prefill, decode = Mock(), Mock()
        decode.clear_forward_metadata_for_idle.side_effect = lambda: setattr(
            decode, "forward_metadata", None
        )
        runner.model_runner = SimpleNamespace(
            is_draft_worker=draft,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            get_decode_attn_backend=lambda: decode,
            attn_backend=prefill,
            decode_attn_backend=decode,
            attn_dcp_size=1,
            device="cpu",
            device_timer=None,
            prefill_cuda_graph_runner=None,
            _pp_kwargs=lambda proxy: {},
            _extend_forward_kwargs=lambda batch, proxy: {},
            model=SimpleNamespace(
                forward=Mock(side_effect=lambda *args, **kwargs: get_attn_backend())
            ),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            batch_size=1,
            needs_forward_metadata_init=lambda: True,
        )
        return runner, batch, prefill, decode

    def test_eager_verify_preserves_prefill_metadata(self):
        runner, batch, prefill, decode = self.make_eager()
        with (
            patch(
                "sglang.srt.model_executor.runner.eager_runner.get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=1),
            ),
            patch(
                "sglang.srt.model_executor.runner.eager_runner.is_cp_active",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.runner.eager_runner.device_timer_ctx",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.model_executor.runner.eager_runner.maybe_publish_prefill_shared_read_done"
            ),
        ):
            self.assertIs(runner._execute_extend(batch), decode)
        decode.init_forward_metadata.assert_called_once_with(batch)
        prefill.init_forward_metadata.assert_not_called()
        prefill.prepare_prefill_shared_read_snapshot.assert_not_called()

    def test_idle_verify_clears_only_decode_metadata(self):
        runner, batch, prefill, decode = self.make_eager()
        sentinel = object()
        prefill.forward_metadata = sentinel
        batch.forward_mode = ForwardMode.IDLE
        batch.batch_size = 0
        with patch(
            "sglang.srt.model_executor.runner.eager_runner.device_timer_ctx",
            return_value=nullcontext(),
        ):
            self.assertIs(runner._execute_idle(batch), decode)
        self.assertIsNone(decode.forward_metadata)
        self.assertIs(prefill.forward_metadata, sentinel)

    def test_empty_idle_clears_real_dsv_state_without_planning(self):
        runner, batch, prefill, _ = self.make_eager()
        decode = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        decode.online_c128_mtp = Mock()
        decode.forward_metadata = object()
        decode.tail_forward_metadata = object()
        decode.encoder_replay = True
        decode.init_forward_metadata = Mock(
            side_effect=AssertionError("zero-token planning")
        )
        runner.model_runner.get_decode_attn_backend = lambda: decode
        prefill.forward_metadata = object()
        batch.forward_mode, batch.batch_size = ForwardMode.IDLE, 0
        with patch(
            "sglang.srt.model_executor.runner.eager_runner.device_timer_ctx",
            return_value=nullcontext(),
        ):
            self.assertIs(runner._execute_idle(batch), decode)
        decode.online_c128_mtp.clear.assert_called_once_with()
        decode.init_forward_metadata.assert_not_called()
        self.assertIsNone(decode.forward_metadata)
        self.assertIsNone(decode.tail_forward_metadata)
        self.assertFalse(decode.encoder_replay)
        self.assertIsNotNone(prefill.forward_metadata)

    def test_default_idle_reset_does_not_plan_metadata(self):
        backend = AttentionBackend()
        backend.forward_metadata = object()
        backend.init_forward_metadata = Mock(
            side_effect=AssertionError("zero-token planning")
        )
        backend.clear_forward_metadata_for_idle()
        backend.init_forward_metadata.assert_not_called()
        self.assertIsNone(backend.forward_metadata)

    def test_draft_backend_is_not_replaced_by_target_decode_lane(self):
        runner, _, prefill, _ = self.make_eager(draft=True)
        backend, _ = runner._resolve_decode_pdmux()
        self.assertIs(backend, prefill)

    def test_split_idle_and_drafts_bypass_graph_replay(self):
        for draft, mode, split_count in (
            (False, ForwardMode.IDLE, 1),
            (True, ForwardMode.IDLE, None),
            (True, ForwardMode.EXTEND, None),
        ):
            with self.subTest(draft=draft, mode=mode):
                runner = ModelRunner.__new__(ModelRunner)
                runner.is_draft_worker = draft
                runner.spec_algorithm = SimpleNamespace(is_eagle=lambda: True)
                runner.device = "cuda"
                runner.attn_backend = Mock()
                runner.decode_cuda_graph_runner = Mock()
                runner.prefill_cuda_graph_runner = Mock()
                runner.eager_runner = Mock()
                runner.eager_runner.execute.return_value = "eager"
                runner.forward_split_prefill = Mock(return_value="split")
                runner._prepare_eager_forward_batch = Mock()
                runner._maybe_execute_deferred_mamba_cow_and_clear = Mock()
                runner.pp_group = SimpleNamespace(is_last_rank=True)
                batch = SimpleNamespace(forward_mode=mode, global_num_tokens_cpu=None)
                with (
                    patch(
                        "sglang.srt.model_executor.model_runner.get_disagg",
                        return_value=SimpleNamespace(enable_pdmux=True),
                    ),
                    patch(
                        "sglang.srt.model_executor.model_runner.get_global_dwdp_manager",
                        return_value=None,
                    ),
                ):
                    result = runner._forward_raw(
                        batch, None, split_forward_count=split_count
                    )
                self.assertFalse(result.can_run_graph)
                self.assertEqual(
                    result.logits_output, "split" if split_count else "eager"
                )
                runner.decode_cuda_graph_runner.execute.assert_not_called()
                runner.prefill_cuda_graph_runner.execute.assert_not_called()

    def test_target_backend_selection_preserves_draft_backend(self):
        runner = ModelRunner.__new__(ModelRunner)
        runner.attn_backend, runner.decode_attn_backend = object(), object()
        for enabled, draft, expected in (
            (True, False, runner.decode_attn_backend),
            (True, True, runner.attn_backend),
            (False, False, runner.attn_backend),
        ):
            runner.is_draft_worker = draft
            runner.spec_algorithm = SimpleNamespace(is_eagle=lambda: True)
            with patch(
                "sglang.srt.model_executor.model_runner.get_disagg",
                return_value=SimpleNamespace(enable_pdmux=enabled),
            ):
                self.assertIs(runner.get_decode_attn_backend(), expected)

    def test_chunked_mtp_prefill_rotates_tokens_and_multimodal_embeddings(self):
        chunked, other = object(), object()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.tensor([1, 2, 3, 4, 5]),
            extend_lens=[3, 2],
            reqs=[chunked, other],
            chunked_req=chunked,
            chunked_req_next_prompt_token=7,
            sampling_info=object(),
        )
        target_hidden = torch.arange(10).reshape(5, 2).float()
        mm = target_hidden + 20
        bonus = torch.tensor([8, 9])
        output = SimpleNamespace(
            next_token_logits=torch.ones(2, 4), hidden_states=torch.ones(2, 2)
        )
        runner = SimpleNamespace(
            canary_manager=None,
            forward=Mock(return_value=SimpleNamespace(logits_output=output)),
        )
        draft = SimpleNamespace(
            speculative_algorithm=SimpleNamespace(is_standalone=lambda: False),
            draft_runner=runner,
            seed_dsa_topk_from_draft_extend=False,
            topk=1,
        )
        fb = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.ForwardBatch.init_new",
                return_value=fb,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_spec",
                return_value=SimpleNamespace(speculative_use_rejection_sampling=False),
            ),
            patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_nan"),
            patch("sglang.srt.speculative.eagle_worker_v2.maybe_detect_inf"),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.renorm_draft_probs",
                side_effect=lambda logits, *args: torch.softmax(logits, dim=-1),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.fast_topk",
                side_effect=torch.topk,
            ),
        ):
            result = EagleDraftWorker._draft_extend_for_prefill(
                draft, batch, target_hidden, bonus, mm
            )
        torch.testing.assert_close(batch.input_ids, torch.tensor([2, 3, 7, 5, 9]))
        torch.testing.assert_close(fb.mm_input_embeds[[0, 1, 3]], mm[[1, 2, 4]])
        self.assertIs(batch.spec_info.hidden_states, target_hidden)
        self.assertIs(result.hidden_states, output.hidden_states)
        self.assertIs(result.bonus_tokens, bonus)
        self.assertFalse(fb.return_logprob)

    def test_verify_tree_uses_decode_mask_buffer(self):
        prefill, decode = Mock(), Mock()
        mask = torch.empty(16, dtype=torch.bool)
        decode.verify_mask = SimpleNamespace(
            mode=TreeMaskMode.QLEN_ONLY, is_read=True, buffer=mask, fits=lambda bs: True
        )
        target = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=prefill, get_decode_attn_backend=lambda: decode
            )
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens=torch.tensor([3]),
            seq_lens_sum=None,
        )
        kernel_result = (
            mask,
            torch.tensor([3]),
            torch.tensor([[0]]),
            torch.tensor([[-1]]),
            torch.tensor([[-1]]),
            torch.tensor([1]),
        )
        with patch(
            "sglang.srt.speculative.eagle_worker_common.build_tree_kernel_efficient",
            return_value=kernel_result,
        ) as kernel:
            result = build_eagle_verify_input(
                batch,
                SimpleNamespace(bonus_tokens=torch.tensor([1])),
                None,
                None,
                None,
                None,
                target_worker=target,
                topk=1,
                num_steps=1,
                num_draft_tokens=2,
                tree_mask_mode=TreeMaskMode.QLEN_ONLY,
                device="cpu",
            )
        self.assertIs(result.custom_mask, mask)
        self.assertIs(kernel.call_args.args[10], mask)

    def test_accepted_mamba_states_commit_through_decode_backend(self):
        prefill, decode = Mock(), Mock()
        target = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=object(),
                req_to_token_pool=SimpleNamespace(mamba_pool=None),
                attn_backend=prefill,
                get_decode_attn_backend=lambda: decode,
                model=object(),
            )
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            mamba_track_indices=None,
            req_pool_indices=torch.tensor([0]),
        )
        with (
            patch(
                "sglang.srt.speculative.spec_utils.mambaish_config",
                return_value=object(),
            ),
            patch(
                "sglang.srt.speculative.spec_utils._verify_commit_step_indices",
                return_value=(torch.tensor([1]), None),
            ),
        ):
            commit_mamba_states_after_verify(
                target, batch, torch.tensor([2]), torch.tensor([[0, 1]]), 2
            )
        decode.update_mamba_state_after_mtp_verify.assert_called_once()
        prefill.update_mamba_state_after_mtp_verify.assert_not_called()

    def test_draft_graphs_are_not_captured_on_ordinary_streams(self):
        worker = SimpleNamespace(
            cuda_graph_runner=object(), cuda_graph_runner_for_draft_extend=object()
        )
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.get_disagg",
            return_value=SimpleNamespace(enable_pdmux=True),
        ):
            EagleDraftWorker._capture_cuda_graphs(worker)
        self.assertIsNone(worker.cuda_graph_runner)
        self.assertIsNone(worker.cuda_graph_runner_for_draft_extend)

    def test_draft_preserves_caller_per_step_backend(self):
        runner, _, _, _ = self.make_eager(draft=True)
        per_step = object()
        with forward_context(ForwardContext(attn_backend=per_step)):
            backend, scope = runner._resolve_decode_pdmux()
            self.assertIs(backend, per_step)
            with scope:
                self.assertIs(get_attn_backend(), per_step)

    def test_final_idle_without_logits_builds_empty_mhc_draft_seed(self):
        worker, batch, persistent, next_draft = self.make_worker(idle=True)
        worker.device = "cpu"
        batch.split_index = 2
        batch.split_forward_batch = persistent

        def target(fb, split_forward_count):
            fb.split_index = 3
            return ModelRunnerOutput(logits_output=None, can_run_graph=False)

        worker.target_worker.model_runner.forward.side_effect = target
        worker._draft_worker.draft_runner.model_config = SimpleNamespace(
            spec_hidden_size=6, dtype=torch.float32
        )
        worker._draft_worker.draft_runner.spec_algorithm = SimpleNamespace(
            is_standalone=lambda: False
        )
        worker._draft_worker._draft_extend_for_prefill = Mock(return_value=next_draft)
        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.pdmux_prefill_handoff",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
                return_value=nullcontext(),
            ),
        ):
            result = worker.forward_batch_split_prefill(batch)
        call = worker._draft_worker._draft_extend_for_prefill.call_args.args
        self.assertEqual(tuple(call[1].shape), (0, 6))
        self.assertEqual(call[1].dtype, torch.float32)
        self.assertEqual(call[2].numel(), 0)
        self.assertEqual(call[2].dtype, torch.int64)
        self.assertIs(result.next_draft_input, next_draft)
        self.assertIsNone(batch.spec_info)

    def test_dspark_draft_keeps_per_stream_graph_and_backend(self):
        eager, _, _, decode = self.make_eager(draft=True)
        eager.model_runner.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        self.assertIs(eager._resolve_decode_pdmux()[0], decode)
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = True
        runner.spec_algorithm = SimpleNamespace(is_eagle=lambda: False)
        runner.device = "cuda"
        runner.attn_backend = Mock()
        runner.hisparse_coordinator = None
        runner.decode_cuda_graph_runner = Mock()
        runner.decode_cuda_graph_runner.can_run_graph.return_value = True
        runner.decode_cuda_graph_runner.execute.return_value = "stream-graph"
        batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        with patch(
            "sglang.srt.model_executor.model_runner.get_disagg",
            return_value=SimpleNamespace(enable_pdmux=True),
        ):
            result = runner._forward_raw(batch, None)
        self.assertTrue(result.can_run_graph)
        self.assertEqual(result.logits_output, "stream-graph")

    def test_forced_eagle_prefill_graph_capture_stays_eager(self):
        runner = SimpleNamespace(
            is_draft_worker=True,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
        )
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.cuda_graph_setup."
                "check_cuda_graph_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.cuda_graph_setup."
                "get_disagg",
                return_value=SimpleNamespace(enable_pdmux=True),
            ),
        ):
            result = capture_prefill_graph(
                model_runner=runner, eager_runner=object(), force_for_draft_worker=True
            )
        self.assertIsNone(result.runner)

    def test_planner_stays_on_selected_pdmux_stream(self):
        with (
            patch(
                "sglang.srt.speculative.spec_utils.envs."
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.speculative.spec_utils.get_disagg",
                return_value=SimpleNamespace(enable_pdmux=True),
            ),
            patch(
                "sglang.srt.speculative.spec_utils.torch.get_device_module"
            ) as device,
        ):
            stream, scope = get_plan_stream("cuda")
            self.assertIsNone(stream)
            with scope:
                pass
        device.assert_not_called()

    def test_final_handoff_fences_previous_and_next_decode_in_order(self):
        calls = []
        event = object()
        prefill = SimpleNamespace(
            wait_stream=lambda decode: calls.append("previous-decode"),
            record_event=lambda: calls.append("handoff-event") or event,
        )
        decode = SimpleNamespace(
            wait_event=lambda recorded: calls.append(("next-decode", recorded))
        )
        with (
            patch(
                "sglang.srt.multiplex.pdmux_context.get_current_stream_idx",
                return_value=0,
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_stream_groups",
                return_value=[(prefill, decode)],
            ),
            pdmux_prefill_handoff(),
        ):
            calls.append("draft-forward")
        self.assertEqual(
            calls,
            [
                "previous-decode",
                "draft-forward",
                "handoff-event",
                ("next-decode", event),
            ],
        )

    def test_nextn_consumes_and_returns_flattened_mhc_stream(self):
        class Projection(nn.Module):
            def forward(self, x):
                return x, None

        model = DeepseekV4ModelNextN.__new__(DeepseekV4ModelNextN)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=2)
        model.hc_mult, model.hc_eps, model.rms_norm_eps = 3, 0.0, 1e-6
        model.embed_tokens = nn.Embedding(8, 2)
        model.enorm, model.hnorm, model.shared_head = (
            nn.Identity(),
            nn.Identity(),
            nn.Module(),
        )
        model.shared_head.norm = nn.Identity()
        model.e_proj = model.h_proj = Projection()
        model.hc_head_fn = nn.Parameter(torch.zeros(3, 6))
        model.hc_head_scale = nn.Parameter(torch.ones(1))
        model.hc_head_base = nn.Parameter(torch.zeros(3))
        model.decoder = Mock(
            side_effect=lambda **kw: (kw["hidden_states"], None, None, None)
        )
        with (
            patch(
                "sglang.srt.models.deepseek_v4_nextn.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch(
                "sglang.srt.models.deepseek_v4_nextn.dsa_use_prefill_cp",
                return_value=False,
            ),
        ):
            for idle in (False, True):
                with self.subTest(idle=idle):
                    n = 0 if idle else 2
                    ids = torch.tensor([] if idle else [1, 2], dtype=torch.int64)
                    hc = torch.arange(n * 6).reshape(n, 6).float()
                    batch = ForwardBatch(
                        forward_mode=ForwardMode.IDLE if idle else ForwardMode.EXTEND,
                        batch_size=0 if idle else 1,
                        input_ids=ids,
                        req_pool_indices=torch.arange(0 if idle else 1),
                        seq_lens=torch.tensor([] if idle else [2], dtype=torch.int64),
                        out_cache_loc=torch.arange(n),
                        seq_lens_sum=n,
                        positions=torch.arange(n),
                        spec_info=SimpleNamespace(hidden_states=hc),
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                    )
                    hidden, captured = model(ids, batch.positions, batch)
                    expected = model.embed_tokens(ids)[:, None, :] + hc.reshape(n, 3, 2)
                    torch.testing.assert_close(captured, expected.flatten(1))
                    torch.testing.assert_close(hidden, expected.sum(1) * 0.5)
                    self.assertEqual(tuple(captured.shape), (n, 6))


if __name__ == "__main__":
    unittest.main()
