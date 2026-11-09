"""Lifecycle tests for dLLMs with an explicit context-encoding phase."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocation import _alloc_extend_loc_with_kv_reuse
from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq, release_only_retirer

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Req(FakeReq):
    def __init__(self, context_len=8, block_size=4, *, prefill=False):
        self.origin_input_ids = array("q", range(context_len))
        self.output_ids = array("q")
        self.full_untruncated_fill_ids = self.origin_input_ids + array(
            "q", [0] * block_size
        )
        self.prefix_len = context_len
        Req.init_diffusion_llm(self, None)
        self.dllm_block_offset = context_len
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.dllm_phase_prefill = prefill
        self.extend_end = context_len + block_size
        self.kv = ReqKvInfo(
            req_pool_idx=1,
            kv_allocated_len=context_len + block_size,
            kv_committed_len=context_len + block_size,
        )
        self.finished_reason = None
        self.finish_on_update = False
        self.retracted_stain = False
        self.sampling_params = SimpleNamespace(max_new_tokens=1024)
        self.time_stats = SimpleNamespace(set_completion_time=Mock())

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    extend_len = Req.extend_len
    get_dllm_forward_phase = Req.get_dllm_forward_phase
    finish_outside_step = Req.finish_outside_step

    def is_dllm_prefill(self):
        return self.dllm_phase_prefill

    def update_finish_state(self, new_accepted_len=1):
        if self.finish_on_update:
            self.finished_reason = object()

    def finished(self):
        return self.finished_reason is not None


class _Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.dllm_forward_metadata = tuple(
            (req.get_dllm_forward_phase(), req.extend_end) for req in reqs
        )
        self.forward_mode = (
            ForwardMode.EXTEND
            if all(req.is_dllm_prefill() for req in reqs)
            else ForwardMode.DLLM_EXTEND
        )
        self.return_logprob = False
        self.prefill_stats = object()
        self.dp_cooperation_info = None

    def batch_size(self):
        return len(self.reqs)


class _Scheduler(SchedulerDllmMixin):
    def __init__(self, *, fdfo, block_size=4, context_len=128):
        self.dllm_config = DllmConfig(
            algorithm="Gemma4Renoise",
            algorithm_config={},
            block_size=block_size,
            mask_id=-1,
            max_running_requests=8,
            first_done_first_out_mode=fdfo,
            requires_separate_context_encoding=True,
        )
        self.metrics_reporter = SimpleNamespace(
            num_generated_tokens=0,
            report_prefill_stats=Mock(),
        )
        self.token_to_kv_pool_allocator = SimpleNamespace(
            free_group_begin=Mock(),
            free_group_end=Mock(),
        )
        self.tree_cache = object()
        self.request_retirer = release_only_retirer(tree_cache=self.tree_cache)
        self.output_streamer = SimpleNamespace(stream_output=Mock())
        self.model_config = SimpleNamespace(context_len=context_len)


def _result(next_token_ids, *, accept_lengths=None, algo_states=None):
    return SimpleNamespace(
        copy_done=None,
        dllm_forward_count=1,
        next_token_ids=next_token_ids,
        accept_length_per_req_cpu=accept_lengths,
        dllm_algo_state=algo_states,
        can_run_cuda_graph=False,
    )


class TestGemma4ContextLifecycle(unittest.TestCase):
    def test_scheduler_uses_standard_extend_for_context_only(self):
        scheduler = SimpleNamespace(
            dllm_config=SimpleNamespace(requires_separate_context_encoding=True),
            dllm_manager=SimpleNamespace(
                get_prefill_requests=Mock(return_value=[object()]),
                get_decode_requests=Mock(return_value=[]),
            ),
            _process_batch_by_phase=Mock(),
        )

        mode = SchedulerDllmMixin._process_dllm_batches(
            scheduler, Mock(), running_batch=Mock()
        )
        self.assertEqual(mode, ForwardMode.EXTEND)

        scheduler.dllm_config.requires_separate_context_encoding = False
        mode = SchedulerDllmMixin._process_dllm_batches(
            scheduler, Mock(), running_batch=Mock()
        )
        self.assertEqual(mode, ForwardMode.DLLM_EXTEND)

        scheduler.dllm_config.requires_separate_context_encoding = True
        scheduler.dllm_manager.get_prefill_requests.return_value = []
        scheduler.dllm_manager.get_decode_requests.return_value = [object()]
        mode = SchedulerDllmMixin._process_dllm_batches(
            scheduler, Mock(), running_batch=Mock()
        )
        self.assertEqual(mode, ForwardMode.DLLM_EXTEND)

    def test_context_result_skips_fdfo_token_processing(self):
        scheduler = _Scheduler(fdfo=True)
        req = _Req(prefill=True)
        result = _result(None)

        SchedulerDllmMixin.process_batch_result_dllm(scheduler, _Batch([req]), result)

        scheduler.token_to_kv_pool_allocator.free_group_begin.assert_not_called()
        scheduler.output_streamer.stream_output.assert_not_called()
        scheduler.metrics_reporter.report_prefill_stats.assert_called_once()
        self.assertEqual(req.dllm_forward_count, 1)
        self.assertEqual(list(req.output_ids), [])

    def test_scheduler_counts_rounds_without_output(self):
        for fdfo in (False, True):
            with self.subTest(fdfo=fdfo):
                scheduler = _Scheduler(fdfo=fdfo)
                scheduler.dllm_config.requires_separate_context_encoding = False
                req = _Req(prefill=not fdfo)
                result = (
                    _result([[1, 2, 3, 4]], accept_lengths=[0], algo_states=[{}])
                    if fdfo
                    else _result([])
                )
                for expected in (1, 2):
                    scheduler.process_batch_result_dllm(_Batch([req]), result)
                    self.assertEqual(req.dllm_forward_count, expected)
                    self.assertEqual(list(req.output_ids), [])

    def test_phase_and_extent_are_snapshotted_before_next_schedule(self):
        scheduler = _Scheduler(fdfo=True)
        req = _Req(prefill=True)
        req.prefix_len, req.extend_end = 0, 8
        batch = _Batch([req])
        # Overlap can prepare the next decode round before this result arrives.
        req.dllm_phase_prefill = False
        req.extend_end = 12
        scheduler.process_batch_result_dllm(batch, _result(None))
        self.assertEqual(req.dllm_forward_count, 1)
        self.assertEqual(req.dllm_prompt_forward_count, 1)
        self.assertEqual(req.dllm_generation_forward_count, 0)
        self.assertEqual(req.dllm_computed_end, 8)

    def test_invalid_forward_metadata_leaves_all_request_counts_unchanged(self):
        for metadata in (None, (("prompt", 8),), (("prompt", 8),) * 3):
            with self.subTest(metadata=metadata):
                scheduler = _Scheduler(fdfo=True)
                reqs = [_Req(prefill=True), _Req(prefill=True)]
                batch = _Batch(reqs)
                batch.dllm_forward_metadata = metadata
                before = [vars(req).copy() for req in reqs]
                with self.assertRaisesRegex(ValueError, "one snapshot per request"):
                    scheduler.process_batch_result_dllm(batch, _result(None))
                self.assertEqual([vars(req) for req in reqs], before)

    def test_empty_forward_metadata_is_valid_for_empty_batch(self):
        scheduler = _Scheduler(fdfo=True)
        scheduler.process_batch_result_dllm(_Batch([]), _result(None))
        scheduler.metrics_reporter.report_prefill_stats.assert_called_once()

    def test_context_forward_classification_and_zero_work(self):
        scheduler = _Scheduler(fdfo=True)
        req = _Req(prefill=True)
        # Initial prompt, later block context, replay, then resumed context.
        for prefix, end, replay_until, calls, expected, computed_end in (
            (0, 8, 0, 1, (1, 0, 0), 8),
            (8, 12, 0, 1, (1, 1, 0), 12),
            (0, 8, 12, 1, (1, 1, 1), 12),
            (12, 16, 12, 1, (1, 2, 1), 16),
            (16, 20, 12, 0, (1, 2, 1), 16),
        ):
            with self.subTest(prefix=prefix, end=end, replay_until=replay_until):
                req.prefix_len, req.extend_end = prefix, end
                req.dllm_recompute_until = replay_until
                result = _result(None)
                result.dllm_forward_count = calls
                scheduler.process_batch_result_dllm(_Batch([req]), result)
                self.assertEqual(
                    (
                        req.dllm_prompt_forward_count,
                        req.dllm_generation_forward_count,
                        req.dllm_recompute_forward_count,
                    ),
                    expected,
                )
                self.assertEqual(req.dllm_forward_count, sum(expected))
                self.assertEqual(req.dllm_computed_end, computed_end)

    def test_repeated_retraction_retains_work_and_replay_boundary(self):
        req = Req(
            "replay",
            "",
            array("q", range(8)),
            SamplingParams(),
            dllm_config=_Scheduler(fdfo=True).dllm_config,
        )
        req.output_ids.extend([9, 10, 11, 12])
        req.dllm_forward_count = 7
        req.dllm_prompt_forward_count = 2
        req.dllm_generation_forward_count = 4
        req.dllm_recompute_forward_count = 1
        # The computed canvas can extend beyond committed output.
        req.dllm_computed_end = 16
        req.reset_for_retract()
        self.assertEqual(req.dllm_recompute_until, 12)
        req.dllm_phase = DllmReqPhase.STAGING_PREFILL
        req.prefix_len, req.extend_end = 0, 8
        self.assertEqual(req.get_dllm_forward_phase(), "recompute")
        # Retract again before replay reaches its previous high-water mark.
        req.reset_for_retract()
        self.assertEqual(req.dllm_recompute_until, 12)
        self.assertEqual(
            (
                req.dllm_forward_count,
                req.dllm_prompt_forward_count,
                req.dllm_generation_forward_count,
                req.dllm_recompute_forward_count,
            ),
            (7, 2, 4, 1),
        )
        req.prefix_len, req.extend_end = 12, 16
        self.assertEqual(req.get_dllm_forward_phase(), "generation")
        req.dllm_phase = DllmReqPhase.STAGING_DECODE
        self.assertEqual(req.get_dllm_forward_phase(), "generation")

        # Embedding-input retraction discards output and restarts generation.
        # Its old generated positions must not classify new context as replay.
        req.input_embeds = [[0.0]] * len(req.origin_input_ids)
        req.reset_for_retract()
        self.assertEqual(len(req.output_ids), 0)
        self.assertEqual(req.dllm_recompute_until, 8)
        req.dllm_phase = DllmReqPhase.STAGING_PREFILL
        req.prefix_len, req.extend_end = 0, 8
        self.assertEqual(req.get_dllm_forward_phase(), "recompute")
        req.output_ids.extend([13, 14, 15, 16])
        req.prefix_len, req.extend_end = 8, 12
        self.assertEqual(req.get_dllm_forward_phase(), "generation")
        fresh = Req(req.rid, "", req.origin_input_ids, SamplingParams())
        self.assertEqual(fresh.dllm_forward_count, 0)

    def test_context_admission_obeys_live_swa_budget(self):
        allocator = SimpleNamespace(
            page_size=4,
            full_available_size=lambda: 128,
            swa_available_size=lambda: 32,
        )
        cache = SimpleNamespace(
            full_evictable_size=lambda: 0,
            window_evictable_size=lambda name: 0,
        )
        budget = SWAPrefillBudget(allocator, cache)
        adder = object.__new__(PrefillAdder)
        adder.memory_budget = budget
        adder.page_size = 4
        adder.dllm_config = SimpleNamespace(requires_separate_context_encoding=True)
        adder.dllm_block_size = 8
        adder.rem_dllm_tokens = 64
        adder.is_hybrid_swa = True
        req = _Req(context_len=40, block_size=8, prefill=True)

        for reserved, expected in ((0, 28), (20, 8), (28, 0), (32, -4)):
            with self.subTest(reserved=reserved):
                budget.swa_offset = reserved
                self.assertEqual(adder._get_dllm_remain_tokens(req), expected)
        req.dllm_phase_prefill = False
        budget.swa_offset = 0
        self.assertEqual(adder._get_dllm_remain_tokens(req), 8)

    def test_context_prefill_is_not_capped_to_canvas_length(self):
        adder = object.__new__(PrefillAdder)
        adder.dllm_config = SimpleNamespace(requires_separate_context_encoding=True)
        adder.dllm_block_size = 256
        adder.rem_dllm_tokens = 1024
        adder.can_run_list = []
        adder._state_slots_for_req = lambda req: 0
        adder._replay_len = lambda req, prefix_len: 0
        adder._update_prefill_budget = Mock()
        adder._account_prefill_cache_admission = Mock()

        adder.page_size = 1
        adder.exact_chunk_fill = False
        adder.rem_chunk_tokens = None
        adder.rem_input_tokens = 1024
        adder._check_prefill_budget = Mock(return_value=(True, None))
        adder._check_prefill_tile_budget = Mock(return_value=None)
        req = _Req(context_len=300, block_size=256, prefill=True)
        req.host_hit_length = req.storage_hit_length = 0
        req.prefix_len = 0
        admission = adder._select_prefill_admission(
            req,
            total_tokens=1024,
            host_hit_length=0,
            window_host_hit_lengths={},
            truncation_align_size=None,
        )
        self.assertEqual((admission.prefix_len, admission.extend_len), (0, 300))
        PrefillAdder._add_dllm_req(adder, req)
        self.assertEqual((req.prefix_len, req.extend_end), (0, 300))

        req.dllm_phase_prefill = False
        req.prefix_len = 300
        PrefillAdder._add_dllm_req(adder, req)
        self.assertEqual((req.prefix_len, req.extend_end), (300, 556))

    def test_chunked_context_prefill_stops_at_context_boundary(self):
        adder = object.__new__(PrefillAdder)
        adder.dllm_config = SimpleNamespace(requires_separate_context_encoding=True)
        adder.can_run_list = []
        adder._get_dllm_remain_tokens = lambda req=None: 64
        adder._state_slots_for_req = lambda req: 0
        adder._update_prefill_budget = Mock()

        req = _Req(context_len=10, block_size=4, prefill=True)
        req.prefix_len = 7

        result = PrefillAdder.add_dllm_staging_req(adder, req)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual((req.prefix_len, req.extend_end), (7, 10))
        self.assertEqual(adder.can_run_list, [req])

    def test_completed_canvas_frees_only_decoder_pages_and_keeps_slot(self):
        context_len, block_size, page_size = 10, 256, 256
        req = Req(
            "completed-canvas", "", array("q", range(context_len)), SamplingParams()
        )
        req.full_untruncated_fill_ids.extend([0] * block_size)
        req.dllm_block_offset = context_len
        req.extend_end = context_len + block_size
        req.kv.req_pool_idx = 3
        req.kv.kv_allocated_len = context_len + block_size
        req.kv.kv_committed_len = context_len + block_size

        req_to_token = torch.arange(4 * 600, dtype=torch.int64).view(4, 600)
        allocator = SimpleNamespace(page_size=page_size, free_segment=Mock())
        scheduler = _Scheduler(fdfo=True, block_size=block_size)
        scheduler.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        scheduler.token_to_kv_pool_allocator = allocator
        scheduler.stash_chunked_request = Mock()

        SchedulerDllmMixin._stash_dllm_context(scheduler, req)

        expected = req_to_token[3, page_size : context_len + block_size]
        freed = allocator.free_segment.call_args.args[0]
        torch.testing.assert_close(freed, expected)
        self.assertEqual(allocator.free_segment.call_args.kwargs, {"start_pos": 256})
        self.assertEqual(req.kv.req_pool_idx, 3)
        self.assertEqual(req.kv.kv_committed_len, context_len)
        self.assertEqual(req.kv.kv_allocated_len, context_len)
        self.assertEqual(req.extend_end, context_len)
        scheduler.stash_chunked_request.assert_called_once_with(req)

    def test_unresolved_fdfo_preserves_state_and_reuses_exact_slots(self):
        context_len, block_size, slot = 5, 4, 2
        scheduler = _Scheduler(fdfo=True, block_size=block_size)
        req = _Req(context_len=context_len, block_size=block_size)
        req.kv.req_pool_idx = slot
        state = {"step": 7}
        next_canvas = torch.tensor([9, 8, 7, 6])

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        ) as release:
            SchedulerDllmMixin.process_batch_result_dllm(
                scheduler,
                _Batch([req]),
                _result(
                    [next_canvas],
                    accept_lengths=[0],
                    algo_states=[state],
                ),
            )
        release.assert_not_called()
        self.assertEqual(req.kv.req_pool_idx, slot)
        self.assertEqual(req.kv.kv_allocated_len, context_len + block_size)
        self.assertEqual(req.dllm_incomplete_ids.tolist(), next_canvas.tolist())
        self.assertIs(req.dllm_algo_state, state)

        req_to_token = torch.arange(3 * 32, dtype=torch.int64).view(3, 32)
        alloc_batch = SimpleNamespace(
            device=torch.device("cpu"),
            reqs=[req],
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
        reused = _alloc_extend_loc_with_kv_reuse(
            alloc_batch,
            [True],
            [req_to_token[slot, :context_len]],
            torch.tensor([slot]),
            torch.tensor([context_len]),
            torch.tensor([block_size]),
            torch.tensor([slot]),
            block_size,
        )
        torch.testing.assert_close(
            reused, req_to_token[slot, context_len : context_len + block_size]
        )

    def test_finished_context_model_never_inserts_canvas_kv(self):
        scheduler = _Scheduler(fdfo=False)
        req = _Req()
        req.finish_on_update = True
        canvas = torch.tensor([3, 2, 1, 0])

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        ) as release:
            SchedulerDllmMixin.process_batch_result_dllm(
                scheduler, _Batch([req]), _result([canvas])
            )

        release.assert_called_once_with(req, scheduler.tree_cache, checkpoint=False)

    def test_context_boundary_stops_sync_and_fdfo(self):
        block_size = 4
        for fdfo in (False, True):
            with self.subTest(fdfo=fdfo):
                scheduler = _Scheduler(fdfo=fdfo, block_size=block_size, context_len=14)
                req = _Req(context_len=7, block_size=block_size)
                canvas = torch.tensor([3, 2, 1, 0])
                result = _result(
                    [canvas],
                    accept_lengths=[block_size] if fdfo else None,
                    algo_states=[None] if fdfo else None,
                )

                with patch(
                    "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
                ) as release:
                    SchedulerDllmMixin.process_batch_result_dllm(
                        scheduler, _Batch([req]), result
                    )

                self.assertTrue(req.finished())
                release.assert_called_once_with(
                    req, scheduler.tree_cache, checkpoint=False
                )


class TestGemma4RequestValidation(unittest.TestCase):
    def _validate(
        self,
        sampling_params=None,
        *,
        algorithm="Gemma4Renoise",
        **req_kwargs,
    ):
        scheduler = _Scheduler(fdfo=False)
        scheduler.dllm_config.algorithm = algorithm
        req = Req(
            "request",
            "",
            array("q", [1]),
            sampling_params if sampling_params is not None else SamplingParams(),
            dllm_config=scheduler.dllm_config,
            **req_kwargs,
        )
        return SchedulerDllmMixin.validate_dllm_request(scheduler, req)

    def test_logprob_rejection_never_enters_the_generation_queue(self):
        from sglang.srt.disaggregation.mode import DisaggregationMode
        from sglang.srt.managers.io_struct import AbortReq
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.runtime_context import get_context, get_parallel

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_session_radix_cache = False
        scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.tokenizer = None
        scheduler.dllm_config = _Scheduler(fdfo=False).dllm_config
        scheduler._maybe_namespace_elastic_radix_cache = Mock()
        scheduler._record_kv_event_lora_name = Mock()
        scheduler._add_request_to_queue = Mock()
        scheduler.init_req_max_new_tokens = Mock()
        scheduler.output_streamer = Mock()
        sender = Mock()
        scheduler.ipc_channels = SimpleNamespace(send_to_tokenizer=sender)
        scheduler.rust_server = None
        scheduler.tree_cache = Mock()
        scheduler.beam_coordinator = Mock()
        scheduler.hisparse_coordinator = None
        scheduler.uocr_runtime = None
        scheduler.init_request_retirer()
        recv_req = MagicMock(
            session_params=None,
            session_id=None,
            input_embeds=None,
            bootstrap_port=1,
        )
        req = Req(
            "unsupported-logprobs",
            "Hello",
            array("q", [1]),
            SamplingParams(),
            return_logprob=True,
            top_logprobs_num=1,
            dllm_config=scheduler.dllm_config,
        )
        with (
            get_context().override_server_args(weight_version="test"),
            get_parallel().override(tp_rank=0),
            patch("sglang.srt.managers.scheduler.Req", return_value=req),
            patch(
                "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                return_value=1,
            ),
        ):
            scheduler.handle_generate_request(recv_req)
        sender.send_output.assert_called_once()
        abort, original = sender.send_output.call_args.args
        self.assertIsInstance(abort, AbortReq)
        self.assertIs(original, req)
        self.assertEqual(abort.finished_reason["status_code"], 400)
        self.assertIn("return_logprob", abort.finished_reason["message"])
        scheduler._add_request_to_queue.assert_not_called()
        scheduler.output_streamer.stream_output.assert_not_called()

    def test_core_and_posthoc_controls_are_supported(self):
        params = SamplingParams(
            max_new_tokens=17,
            stop="END",
            stop_token_ids={2},
            stop_regex=r"DONE",
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            n=2,
            ignore_eos=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            no_stop_trim=True,
            stream_interval=2,
            custom_params={"tag": "allowed"},
        )
        self.assertIsNone(self._validate(params, stream=True))

    def test_unsupported_sampling_parameters(self):
        cases = [
            ("frequency_penalty", {"frequency_penalty": 0.1}),
            ("presence_penalty", {"presence_penalty": 0.1}),
            ("repetition_penalty", {"repetition_penalty": 1.1}),
            ("min_new_tokens", {"min_new_tokens": 1}),
            ("min_p", {"min_p": 0.1}),
            ("sampling_seed", {"sampling_seed": 7}),
            ("logit_bias", {"logit_bias": {"7": 1.0}}),
            ("json_schema", {"json_schema": "{}"}),
            ("regex", {"regex": r"\d+"}),
            ("ebnf", {"ebnf": 'root ::= "x"'}),
            ("structural_tag", {"structural_tag": "<tag>"}),
        ]
        for field, values in cases:
            with self.subTest(field=field):
                error = self._validate(SamplingParams(**values))
                self.assertIsNotNone(error)
                self.assertIn(field, error)

    def test_unsupported_request_flags(self):
        cases = [
            ("return_logprob", {"return_logprob": True}),
            ("top_logprobs_num", {"top_logprobs_num": 2}),
            ("token_ids_logprob", {"token_ids_logprob": [3]}),
            ("return_sampling_mask", {"return_sampling_mask": True}),
            (
                "return_flat_raw_top_logprobs",
                {"return_flat_raw_top_logprobs": True},
            ),
            ("custom_logit_processor", {"custom_logit_processor": "processor"}),
        ]
        for field, values in cases:
            with self.subTest(field=field):
                error = self._validate(**values)
                self.assertIsNotNone(error)
                self.assertIn(field, error)

    def test_reports_every_unsupported_field(self):
        error = self._validate(
            SamplingParams(frequency_penalty=0.1, presence_penalty=0.2),
            return_logprob=True,
        )
        for field in ("frequency_penalty", "presence_penalty", "return_logprob"):
            self.assertIn(field, error)
        self.assertIn("sampling is governed by the renoise schedule", error)

    def test_validation_is_gemma4_renoise_specific(self):
        self.assertIsNone(
            self._validate(SamplingParams(top_p=0.9), algorithm="LowConfidence")
        )
        scheduler = SimpleNamespace(dllm_config=None)
        self.assertIsNone(SchedulerDllmMixin.validate_dllm_request(scheduler, object()))

    def test_greedy_temperature_is_accepted_after_normalization(self):
        params = SamplingParams(temperature=0)
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.top_k, 1)
        self.assertIsNone(self._validate(params))


if __name__ == "__main__":
    unittest.main()
