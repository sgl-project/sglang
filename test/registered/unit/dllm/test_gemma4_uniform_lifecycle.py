import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.allocation import _alloc_extend_loc_with_kv_reuse
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, context_len=8, block_size=4, *, prefill=False):
        self.origin_input_ids = array("q", range(context_len))
        self.output_ids = array("q")
        self.full_untruncated_fill_ids = self.origin_input_ids + array(
            "q", [0] * block_size
        )
        self.prefix_indices = torch.arange(context_len, dtype=torch.int64)
        self.dllm_block_offset = context_len
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.dllm_phase_prefill = prefill
        self.extend_range = SimpleNamespace(
            start=context_len, end=context_len + block_size, length=block_size
        )
        self.req_pool_idx = 1
        self.kv = SimpleNamespace(
            kv_allocated_len=context_len + block_size,
            swa_evicted_seqlen=0,
        )
        self.kv_committed_len = context_len + block_size
        self.finished_reason = None
        self.finish_on_update = False
        self.retracted_stain = False
        self.sampling_params = SimpleNamespace(max_new_tokens=1024)
        self.time_stats = SimpleNamespace(set_completion_time=Mock())

    @property
    def seqlen(self):
        return len(self.origin_input_ids) + len(self.output_ids)

    def is_dllm_prefill(self):
        return self.dllm_phase_prefill

    def set_extend_range(self, start, end):
        self.extend_range = SimpleNamespace(start=start, end=end, length=end - start)

    def update_finish_state(self, new_accepted_len=1):
        if self.finish_on_update:
            self.finished_reason = object()

    def finished(self):
        return self.finished_reason is not None


class _Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.return_logprob = False
        self.prefill_stats = object()
        self.dp_cooperation_info = None

    def batch_size(self):
        return len(self.reqs)


class _Scheduler:
    def __init__(self, *, fdfo, block_size=4, context_len=128):
        self.dllm_config = SimpleNamespace(
            algorithm="Gemma4Renoise",
            first_done_first_out_mode=fdfo,
            is_uniform=True,
            block_size=block_size,
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
        self.output_streamer = SimpleNamespace(stream_output=Mock())
        self.model_config = SimpleNamespace(context_len=context_len)


def _result(next_token_ids, *, accept_lengths=None, algo_states=None):
    return SimpleNamespace(
        copy_done=None,
        next_token_ids=next_token_ids,
        accept_length_per_req_cpu=accept_lengths,
        dllm_algo_state=algo_states,
        can_run_cuda_graph=False,
    )


class TestGemma4UniformLifecycle(unittest.TestCase):
    def test_encoder_result_skips_fdfo_token_processing(self):
        scheduler = _Scheduler(fdfo=True)
        req = _Req(prefill=True)
        result = _result(None)

        SchedulerDllmMixin.process_batch_result_dllm(scheduler, _Batch([req]), result)

        scheduler.token_to_kv_pool_allocator.free_group_begin.assert_not_called()
        scheduler.output_streamer.stream_output.assert_not_called()
        scheduler.metrics_reporter.report_prefill_stats.assert_called_once()

    def test_uniform_encoder_is_not_capped_to_canvas_length(self):
        adder = object.__new__(PrefillAdder)
        adder.dllm_config = SimpleNamespace(is_uniform=True)
        adder.dllm_block_size = 256
        adder.rem_dllm_tokens = 1024
        adder.can_run_list = []
        adder._mamba_gap_budget_for_req = lambda req: 0
        adder._update_prefill_budget = Mock()

        req = _Req(context_len=300, block_size=256, prefill=True)
        req.host_hit_length = req.storage_hit_length = 0
        PrefillAdder._add_dllm_req(adder, req, 0)
        self.assertEqual((req.extend_range.start, req.extend_range.end), (0, 300))

        req.dllm_phase_prefill = False
        PrefillAdder._add_dllm_req(adder, req, 300)
        self.assertEqual((req.extend_range.start, req.extend_range.end), (300, 556))

    def test_chunked_uniform_prefill_stops_at_context_boundary(self):
        adder = object.__new__(PrefillAdder)
        adder.dllm_config = SimpleNamespace(is_uniform=True)
        adder.can_run_list = []
        adder._get_dllm_remain_tokens = lambda req=None: 64
        adder._mamba_gap_budget_for_req = lambda req: 0
        adder._update_prefill_budget = Mock()

        req = _Req(context_len=10, block_size=4, prefill=True)
        req.prefix_indices = torch.arange(7)

        result = PrefillAdder.add_dllm_staging_req(adder, req)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual((req.extend_range.start, req.extend_range.end), (7, 10))
        self.assertEqual(adder.can_run_list, [req])

    def test_completed_canvas_frees_only_decoder_pages_and_keeps_slot(self):
        context_len, block_size, page_size = 10, 256, 256
        req = _Req(context_len=context_len, block_size=block_size)
        req.req_pool_idx = 3
        req.kv.kv_allocated_len = context_len + block_size
        req.kv_committed_len = context_len + block_size

        req_to_token = torch.arange(4 * 600, dtype=torch.int64).view(4, 600)
        allocator = SimpleNamespace(page_size=page_size, free_segment=Mock())
        scheduler = _Scheduler(fdfo=True, block_size=block_size)
        scheduler.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        scheduler.token_to_kv_pool_allocator = allocator
        scheduler.stash_chunked_request = Mock()

        SchedulerDllmMixin._stash_completed_uniform_canvas(scheduler, req)

        expected = req_to_token[3, page_size : context_len + block_size]
        freed = allocator.free_segment.call_args.args[0]
        torch.testing.assert_close(freed, expected)
        self.assertEqual(allocator.free_segment.call_args.kwargs, {"start_pos": 256})
        self.assertEqual(req.req_pool_idx, 3)
        self.assertEqual(req.kv_committed_len, context_len)
        self.assertEqual(req.kv.kv_allocated_len, context_len)
        self.assertEqual(
            (req.extend_range.start, req.extend_range.end),
            (context_len, context_len),
        )
        scheduler.stash_chunked_request.assert_called_once_with(req)

    def test_unresolved_fdfo_preserves_state_and_reuses_exact_slots(self):
        context_len, block_size, slot = 5, 4, 2
        scheduler = _Scheduler(fdfo=True, block_size=block_size)
        req = _Req(context_len=context_len, block_size=block_size)
        req.req_pool_idx = slot
        state = {"step": 7}
        next_canvas = torch.tensor([9, 8, 7, 6])

        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release:
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
        self.assertEqual(req.req_pool_idx, slot)
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
            torch.tensor([slot]),
            torch.tensor([context_len]),
            torch.tensor([block_size]),
            torch.tensor([slot]),
            block_size,
        )
        torch.testing.assert_close(
            reused, req_to_token[slot, context_len : context_len + block_size]
        )

    def test_finished_uniform_request_never_inserts_canvas_kv(self):
        scheduler = _Scheduler(fdfo=False)
        req = _Req()
        req.finish_on_update = True
        canvas = torch.tensor([3, 2, 1, 0])

        with patch("sglang.srt.dllm.mixin.scheduler.release_kv_cache") as release:
            SchedulerDllmMixin.process_batch_result_dllm(
                scheduler, _Batch([req]), _result([canvas])
            )

        release.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)

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
                    "sglang.srt.dllm.mixin.scheduler.release_kv_cache"
                ) as release:
                    SchedulerDllmMixin.process_batch_result_dllm(
                        scheduler, _Batch([req]), result
                    )

                self.assertTrue(req.finished())
                release.assert_called_once_with(
                    req, scheduler.tree_cache, is_insert=False
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
