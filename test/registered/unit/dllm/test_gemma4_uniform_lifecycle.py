"""Lifecycle tests for dLLMs with an explicit context-encoding phase."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.manager import DllmManager
from sglang.srt.dllm.prefill_adder import DllmPrefillAdder
from sglang.srt.dllm.req_state import DllmReqPhase, DllmReqState
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.mem_cache.allocation import _alloc_extend_loc_with_kv_reuse
from sglang.srt.mem_cache.prefill_budget import SWAPrefillBudget
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_request_lists import make_request_retirer
from sglang.test.test_utils import enter_scope, published_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req:
    def __init__(self, context_len=8, block_size=4, *, prefill=False):
        self.origin_input_ids = array("q", range(context_len))
        self.output_ids = array("q")
        self.full_untruncated_fill_ids = self.origin_input_ids + array(
            "q", [0] * block_size
        )
        self.prefix_len = context_len
        self.dllm = DllmReqState(
            config=None,
            phase=(
                DllmReqPhase.STAGING_PREFILL if prefill else DllmReqPhase.STAGING_DECODE
            ),
            block_offset=context_len,
        )
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

    def update_finish_state(self, new_accepted_len=1):
        if self.finish_on_update:
            self.finished_reason = object()

    def finished(self):
        return self.finished_reason is not None


class _Batch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.forward_mode = (
            ForwardMode.EXTEND
            if all(req.dllm.is_prefill() for req in reqs)
            else ForwardMode.DLLM_EXTEND
        )
        self.return_logprob = False
        self.prefill_stats = object()
        self.dp_cooperation_info = None

    def batch_size(self):
        return len(self.reqs)


def _manager(*, fdfo, block_size=4, context_len=128):
    kv = SimpleNamespace(
        tree_cache=object(),
        req_to_token_pool=None,
        token_to_kv_pool_allocator=SimpleNamespace(
            free_group_begin=Mock(),
            free_group_end=Mock(),
        ),
    )
    return DllmManager(
        dllm_config=DllmConfig(
            algorithm="Gemma4Renoise",
            algorithm_config={},
            block_size=block_size,
            mask_id=-1,
            max_running_requests=8,
            first_done_first_out_mode=fdfo,
            requires_separate_context_encoding=True,
        ),
        kv=kv,
        services=SimpleNamespace(
            metrics_sink=SimpleNamespace(
                num_generated_tokens=0,
                report_prefill_stats=Mock(),
            ),
            output_streamer=SimpleNamespace(stream_output=Mock()),
        ),
        policy=None,
        new_token_ratio_tracker=None,
        limits=SimpleNamespace(
            max_prefill_tokens=0, chunked_prefill_size=None, truncation_align_size=None
        ),
        context_len=context_len,
        beam_coordinator=None,
        step_builder=None,
        request_retirer=make_request_retirer(kv=kv),
    )


def _result(next_token_ids, *, accept_lengths=None, algo_states=None):
    return SimpleNamespace(
        copy_done=None,
        next_token_ids=next_token_ids,
        accept_length_per_req_cpu=accept_lengths,
        dllm_algo_state=algo_states,
        can_run_cuda_graph=False,
    )


class TestGemma4ContextLifecycle(unittest.TestCase):
    def setUp(self):
        enter_scope(
            self,
            published_topology(page_size=1, priority_scheduling_preemption_threshold=0),
        )

    def test_scheduler_uses_standard_extend_for_context_only(self):
        manager = SimpleNamespace(
            dllm_config=SimpleNamespace(requires_separate_context_encoding=True),
            get_prefill_requests=Mock(return_value=[object()]),
            get_decode_requests=Mock(return_value=[]),
            _process_batch_by_phase=Mock(),
        )

        mode = DllmManager._process_dllm_batches(manager, Mock())
        self.assertEqual(mode, ForwardMode.EXTEND)

        manager.dllm_config.requires_separate_context_encoding = False
        mode = DllmManager._process_dllm_batches(manager, Mock())
        self.assertEqual(mode, ForwardMode.DLLM_EXTEND)

        manager.dllm_config.requires_separate_context_encoding = True
        manager.get_prefill_requests.return_value = []
        manager.get_decode_requests.return_value = [object()]
        mode = DllmManager._process_dllm_batches(manager, Mock())
        self.assertEqual(mode, ForwardMode.DLLM_EXTEND)

    def test_context_result_skips_fdfo_token_processing(self):
        manager = _manager(fdfo=True)
        req = _Req(prefill=True)
        result = _result(None)

        manager._write_blocks(_Batch([req]), result)

        manager.token_to_kv_pool_allocator.free_group_begin.assert_not_called()
        manager.output_streamer.stream_output.assert_not_called()

    def test_context_admission_obeys_live_swa_budget(self):
        allocator = SimpleNamespace(
            page_size=4,
            full_available_size=lambda: 128,
            swa_available_size=lambda: 32,
        )
        cache = SimpleNamespace(
            full_evictable_size=lambda: 0,
            swa_evictable_size=lambda: 0,
        )
        budget = SWAPrefillBudget(allocator, cache)
        adder = object.__new__(DllmPrefillAdder)
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
        req.dllm.phase = DllmReqPhase.STAGING_DECODE
        budget.swa_offset = 0
        self.assertEqual(adder._get_dllm_remain_tokens(req), 8)

    def test_context_prefill_is_not_capped_to_canvas_length(self):
        adder = object.__new__(DllmPrefillAdder)
        adder.dllm_config = SimpleNamespace(requires_separate_context_encoding=True)
        adder.dllm_block_size = 256
        adder.rem_dllm_tokens = 1024
        adder.can_run_list = []
        adder._mamba_gap_budget_for_req = lambda req: 0
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
            swa_host_hit_length=0,
            truncation_align_size=None,
        )
        self.assertEqual((admission.prefix_len, admission.extend_len), (0, 300))
        DllmPrefillAdder._add_dllm_req(adder, req)
        self.assertEqual((req.prefix_len, req.extend_end), (0, 300))

        req.dllm.phase = DllmReqPhase.STAGING_DECODE
        req.prefix_len = 300
        DllmPrefillAdder._add_dllm_req(adder, req)
        self.assertEqual((req.prefix_len, req.extend_end), (300, 556))

    def test_chunked_context_prefill_stops_at_context_boundary(self):
        adder = object.__new__(DllmPrefillAdder)
        adder.dllm_config = SimpleNamespace(requires_separate_context_encoding=True)
        adder.can_run_list = []
        adder._get_dllm_remain_tokens = lambda req=None: 64
        adder._mamba_gap_budget_for_req = lambda req: 0
        adder._update_prefill_budget = Mock()

        req = _Req(context_len=10, block_size=4, prefill=True)
        req.prefix_len = 7

        result = DllmPrefillAdder.add_dllm_staging_req(adder, req)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual((req.prefix_len, req.extend_end), (7, 10))
        self.assertEqual(adder.can_run_list, [req])

    def test_completed_canvas_frees_only_decoder_pages_and_keeps_slot(self):
        context_len, block_size, page_size = 10, 256, 256
        req = Req(
            "completed-canvas", "", array("q", range(context_len)), SamplingParams()
        )
        req.full_untruncated_fill_ids.extend([0] * block_size)
        req.dllm = DllmReqState(config=None, block_offset=context_len)
        req.extend_end = context_len + block_size
        req.kv.req_pool_idx = 3
        req.kv.kv_allocated_len = context_len + block_size
        req.kv.kv_committed_len = context_len + block_size

        req_to_token = torch.arange(4 * 600, dtype=torch.int64).view(4, 600)
        allocator = SimpleNamespace(page_size=page_size, free_segment=Mock())
        manager = _manager(fdfo=True, block_size=block_size)
        manager.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        manager.token_to_kv_pool_allocator = allocator

        with patch("sglang.srt.dllm.manager.checkpoint_kv_cache") as checkpoint:
            manager._stash_dllm_context(req)

        expected = req_to_token[3, page_size : context_len + block_size]
        freed = allocator.free_segment.call_args.args[0]
        torch.testing.assert_close(freed, expected)
        self.assertEqual(allocator.free_segment.call_args.kwargs, {"start_pos": 256})
        self.assertEqual(req.kv.req_pool_idx, 3)
        self.assertEqual(req.kv.kv_committed_len, context_len)
        self.assertEqual(req.kv.kv_allocated_len, context_len)
        self.assertEqual(req.extend_end, context_len)
        checkpoint.assert_called_once_with(req, manager.tree_cache)

    def test_unresolved_fdfo_preserves_state_and_reuses_exact_slots(self):
        context_len, block_size, slot = 5, 4, 2
        manager = _manager(fdfo=True, block_size=block_size)
        req = _Req(context_len=context_len, block_size=block_size)
        req.kv.req_pool_idx = slot
        state = {"step": 7}
        next_canvas = torch.tensor([9, 8, 7, 6])

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        ) as release:
            manager._write_blocks(
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
        self.assertEqual(req.dllm.incomplete_ids.tolist(), next_canvas.tolist())
        self.assertIs(req.dllm.algo_state, state)

        req_to_token = torch.arange(3 * 32, dtype=torch.int64).view(3, 32)
        alloc_batch = SimpleNamespace(
            device=torch.device("cpu"),
            reqs=[req],
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
        reused = _alloc_extend_loc_with_kv_reuse(
            alloc_batch,
            [block_size],
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
        manager = _manager(fdfo=False)
        req = _Req()
        req.finish_on_update = True
        canvas = torch.tensor([3, 2, 1, 0])

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
        ) as release:
            manager._write_blocks(_Batch([req]), _result([canvas]))

        release.assert_called_once_with(req, manager.tree_cache, checkpoint=False)

    def test_context_boundary_stops_sync_and_fdfo(self):
        block_size = 4
        for fdfo in (False, True):
            with self.subTest(fdfo=fdfo):
                manager = _manager(fdfo=fdfo, block_size=block_size, context_len=14)
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
                    manager._write_blocks(_Batch([req]), result)

                self.assertTrue(req.finished())
                release.assert_called_once_with(
                    req, manager.tree_cache, checkpoint=False
                )


class TestGemma4RequestValidation(unittest.TestCase):
    def setUp(self):
        enter_scope(
            self,
            published_topology(page_size=1, priority_scheduling_preemption_threshold=0),
        )

    def _validate(
        self,
        sampling_params=None,
        *,
        algorithm="Gemma4Renoise",
        **req_kwargs,
    ):
        manager = _manager(fdfo=False)
        manager.dllm_config.algorithm = algorithm
        req = Req(
            "request",
            "",
            array("q", [1]),
            sampling_params if sampling_params is not None else SamplingParams(),
            **req_kwargs,
        )
        return manager.validate_request(req)

    def test_logprob_rejection_never_enters_the_generation_queue(self):
        from sglang.srt.managers.io_struct import AbortReq
        from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
        from sglang.srt.runtime_context import get_context, get_parallel
        from sglang.test.scheduler_request_intake import make_request_intake

        sender = Mock()
        intake = make_request_intake(
            model_config=SimpleNamespace(hf_eos_token_id={1}, vocab_size=128),
            mode_owners=ModeOwners(dllm=_manager(fdfo=False)),
            output_streamer=Mock(),
            ipc_channels=SimpleNamespace(send_to_tokenizer=sender),
        )
        intake._maybe_namespace_elastic_radix_cache = Mock()
        intake.add_request_to_queue = Mock()
        intake.init_req_max_new_tokens = Mock()
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
        )
        with (
            get_context().override_server_args(weight_version="test"),
            get_parallel().override(tp_rank=0),
            patch(
                "sglang.srt.managers.scheduler_components.request_intake.Req",
                return_value=req,
            ),
            patch(
                "sglang.srt.managers.scheduler_components.request_intake."
                "BeamCoordinator.request_beam_width",
                return_value=1,
            ),
        ):
            intake.handle_generate_request(recv_req)
        sender.send_output.assert_called_once()
        abort, original = sender.send_output.call_args.args
        self.assertIsInstance(abort, AbortReq)
        self.assertIs(original, req)
        self.assertEqual(abort.finished_reason["status_code"], 400)
        self.assertIn("return_logprob", abort.finished_reason["message"])
        intake.add_request_to_queue.assert_not_called()
        intake.output_streamer.stream_output.assert_not_called()

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

    def test_greedy_temperature_is_accepted_after_normalization(self):
        params = SamplingParams(temperature=0)
        self.assertEqual(params.temperature, 1.0)
        self.assertEqual(params.top_k, 1)
        self.assertIsNone(self._validate(params))


if __name__ == "__main__":
    unittest.main()
