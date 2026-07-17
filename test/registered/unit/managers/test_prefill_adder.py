import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.dllm.config import _validate_multi_block_prefill_backend
from sglang.srt.dllm.mixin.req import DllmReqPhase, ReqDllmMixin
from sglang.srt.dllm.mixin.scheduler import DllmManager, SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class TestPrefillAdder(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.mock_tree_cache = self.create_tree_cache()
        self.mock_token_allocator = self.create_token_allocator()

    def create_tree_cache(
        self,
        *,
        full_evictable_size: int = 0,
        swa_evictable_size: int = 0,
        evictable_size: int = 0,
    ) -> MagicMock:
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = full_evictable_size
        tree_cache.swa_evictable_size.return_value = swa_evictable_size
        tree_cache.evictable_size.return_value = evictable_size
        tree_cache.disable = False
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        return tree_cache

    def create_token_allocator(
        self,
        *,
        full_available_size: int = 0,
        swa_available_size: int = 0,
        available_size: int = 0,
    ) -> MagicMock:
        allocator = MagicMock()
        allocator.full_available_size.return_value = full_available_size
        allocator.swa_available_size.return_value = swa_available_size
        allocator.available_size.return_value = available_size
        return allocator

    def create_running_batch(self, reqs=None) -> MagicMock:
        batch = MagicMock()
        batch.reqs = list(reqs or [])
        batch.release_req.return_value = None
        batch.filter_batch.return_value = None
        return batch

    def create_server_args(
        self, *, schedule_low_priority_values_first: bool
    ) -> MagicMock:
        server_args = MagicMock()
        server_args.schedule_low_priority_values_first = (
            schedule_low_priority_values_first
        )
        return server_args

    def create_mock_req(self, rid, priority, max_new_tokens, output_len=0, wait_time=0):
        req = MagicMock(spec=Req)
        req.rid = str(rid)
        req.priority = priority
        req.prefix_indices = []
        req.full_untruncated_fill_ids = []
        req.output_ids = [0] * output_len
        req.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
        req.time_stats = SimpleNamespace(wait_queue_entry_time=wait_time)
        req.retracted_stain = False
        req.finished.return_value = False
        req.needs_host_load_back.return_value = False
        return req

    def create_adder(self, running_batch, **kwargs):
        defaults = dict(
            page_size=1,
            tree_cache=self.mock_tree_cache,
            token_to_kv_pool_allocator=self.mock_token_allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=None,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        defaults.update(kwargs)
        return PrefillAdder(**defaults)

    def create_dllm_req(
        self,
        *,
        origin_len: int,
        prefix_len: int,
        is_prefill: bool,
        output_len: int = 0,
        block_size: int = 32,
    ):
        req = self.create_mock_req(
            "dllm", priority=0, max_new_tokens=128, output_len=output_len
        )
        req.origin_input_ids = [1] * origin_len
        req.prefix_indices = [0] * prefix_len
        req.is_dllm_prefill.return_value = is_prefill
        req.full_untruncated_fill_ids = [1] * (origin_len + output_len + block_size)
        req.set_extend_range.side_effect = lambda start, end: setattr(
            req, "extend_range", Range(start, end)
        )
        return req

    def create_dllm_adder(self, *, is_prefill: bool, rem_input_tokens: int = 10000):
        self.mock_token_allocator.available_size.return_value = 10000
        dllm_config = SimpleNamespace(
            block_size=32,
            prefill_block_size=128,
            max_running_requests=2,
        )
        return self.create_adder(
            self.create_running_batch(),
            page_size=32,
            rem_input_tokens=rem_input_tokens,
            dllm_config=dllm_config,
            dllm_is_prefill=is_prefill,
        )

    def test_dllm_multi_block_prefill_requires_flashinfer(self):
        _validate_multi_block_prefill_backend(
            block_size=32,
            prefill_block_size=1024,
            prefill_attention_backend="flashinfer",
        )
        # Existing single-block behavior remains backend-independent.
        _validate_multi_block_prefill_backend(
            block_size=32,
            prefill_block_size=32,
            prefill_attention_backend="triton",
        )
        with self.assertRaisesRegex(ValueError, "requires the FlashInfer"):
            _validate_multi_block_prefill_backend(
                block_size=32,
                prefill_block_size=1024,
                prefill_attention_backend="triton",
            )

    def test_dllm_prefill_uses_phase_budget_and_block_aligned_context(self):
        adder = self.create_dllm_adder(is_prefill=True)
        req = self.create_dllm_req(origin_len=300, prefix_len=0, is_prefill=True)

        self.assertEqual(adder.rem_dllm_tokens, 256)
        self.assertTrue(adder._add_dllm_req(req, 0))
        self.assertEqual(req.extend_range, Range(0, 128))
        self.assertEqual(adder.rem_dllm_tokens, 128)

        # 300 is not block aligned. The final 12 prompt tokens must be decoded
        # with masks rather than being committed as a non-aligned prefill tail.
        tail_req = self.create_dllm_req(origin_len=300, prefix_len=288, is_prefill=True)
        self.assertFalse(adder._add_dllm_req(tail_req, 288))

    def test_dllm_prefill_respects_max_prefill_tokens(self):
        adder = self.create_dllm_adder(is_prefill=True, rem_input_tokens=64)
        req = self.create_dllm_req(origin_len=256, prefix_len=0, is_prefill=True)

        self.assertTrue(adder._add_dllm_req(req, 0))
        self.assertEqual(req.extend_range, Range(0, 64))

    def test_dllm_large_prefill_advances_decode_position_offset(self):
        req = SimpleNamespace(
            dllm_initialized=True,
            dllm_incomplete_ids=array("q"),
            dllm_block_offset=0,
            extend_range=Range(0, 128),
            dllm_config=SimpleNamespace(block_size=32, mask_id=0),
            origin_input_ids=array("q", [1] * 128),
            output_ids=array("q"),
        )

        ReqDllmMixin._init_fill_ids_for_dllm(req)

        self.assertEqual(req.dllm_block_offset, 128)

    def test_dllm_unadmitted_req_does_not_advance_position_offset(self):
        req = SimpleNamespace(
            dllm_initialized=True,
            dllm_incomplete_ids=array("q"),
            dllm_block_offset=0,
            extend_range=None,
            dllm_config=SimpleNamespace(block_size=32, mask_id=0),
            origin_input_ids=array("q", [1] * 128),
            output_ids=array("q"),
        )

        ReqDllmMixin._init_fill_ids_for_dllm(req)

        self.assertEqual(req.dllm_block_offset, 0)

    def test_dllm_scheduler_selects_decode_after_preparing_incoming_reqs(self):
        req = MagicMock()
        manager = MagicMock()
        manager.get_prefill_requests.return_value = []
        manager.get_decode_requests.return_value = [req]
        running_batch = SimpleNamespace(batch_is_full=False, reqs=[])
        scheduler = SimpleNamespace(
            enable_priority_preemption=False,
            policy=MagicMock(),
            waiting_queue=[],
            dllm_manager=manager,
            tree_cache=MagicMock(),
            _should_skip_prefill=lambda *, running_batch: False,
            _fetch_waiting_reqs=lambda: None,
            _create_dllm_prefill_adder=MagicMock(),
            _process_dllm_batches=MagicMock(return_value=ForwardMode.DLLM_EXTEND),
        )
        adder = SimpleNamespace(can_run_list=[req])
        scheduler._create_dllm_prefill_adder.return_value = adder
        scheduler._update_state_for_batch = MagicMock()
        scheduler._create_dllm_batch = MagicMock(return_value=MagicMock())

        batch = SchedulerDllmMixin.get_new_batch_dllm(scheduler, running_batch)

        self.assertIsNotNone(batch)
        manager.init_next_round.assert_called_once_with(scheduler.tree_cache)
        scheduler._process_dllm_batches.assert_called_once_with(
            adder, running_batch=running_batch, is_prefill=False
        )

    def test_dllm_manager_prepares_incoming_req_before_phase_selection(self):
        req = MagicMock()
        req.dllm_phase = DllmReqPhase.INCOMING_PREFILL
        req.init_next_round_input.side_effect = lambda _: setattr(
            req, "dllm_phase", DllmReqPhase.STAGING_DECODE
        )
        req.is_dllm_prefill.return_value = False
        manager = DllmManager(dllm_config=SimpleNamespace(max_running_requests=1))
        manager.waiting_queue = [req]

        manager.init_next_round(MagicMock())

        self.assertEqual(req.dllm_phase, DllmReqPhase.INCOMING_DECODE)

    def test_dllm_admitted_incoming_req_becomes_staging(self):
        req = MagicMock()
        req.dllm_phase = DllmReqPhase.INCOMING_DECODE
        adder = MagicMock()
        adder.can_run_list = [req]
        adder.add_one_req.return_value = AddReqResult.CONTINUE
        scheduler = SimpleNamespace(
            get_num_allocatable_reqs=lambda _: 2,
            enable_priority_preemption=False,
            truncation_align_size=None,
        )
        running_batch = SimpleNamespace(batch_is_full=False, reqs=[])

        result = SchedulerDllmMixin.process_dllm_incoming_reqs(
            scheduler, adder, [req], running_batch=running_batch
        )

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(req.dllm_phase, DllmReqPhase.STAGING_DECODE)

    def test_dllm_decode_stays_at_one_fixed_block(self):
        adder = self.create_dllm_adder(is_prefill=False)
        req = self.create_dllm_req(origin_len=20, prefix_len=0, is_prefill=False)

        self.assertEqual(adder.rem_dllm_tokens, 64)
        self.assertTrue(adder._add_dllm_req(req, 0))
        self.assertEqual(req.extend_range, Range(0, 32))

    def test_dllm_scheduler_uses_normal_extend_only_for_prefill(self):
        scheduler = SimpleNamespace(
            dllm_manager=MagicMock(), _process_batch_by_phase=MagicMock()
        )
        running_batch = MagicMock()
        scheduler.dllm_manager.get_prefill_requests.return_value = [MagicMock()]

        self.assertEqual(
            SchedulerDllmMixin._process_dllm_batches(
                scheduler,
                MagicMock(),
                running_batch=running_batch,
                is_prefill=True,
            ),
            ForwardMode.EXTEND,
        )

        self.assertEqual(
            SchedulerDllmMixin._process_dllm_batches(
                scheduler,
                MagicMock(),
                running_batch=running_batch,
                is_prefill=False,
            ),
            ForwardMode.DLLM_EXTEND,
        )

    def test_dllm_scheduler_propagates_explicit_prefill_phase(self):
        scheduler = SimpleNamespace(
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
            tree_cache=object(),
            model_config=object(),
            enable_overlap=False,
            spec_algorithm=object(),
            dllm_config=object(),
            adder=MagicMock(),
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
        )
        module = "sglang.srt.dllm.mixin.scheduler"

        for forward_mode, expected in (
            (ForwardMode.EXTEND, True),
            (ForwardMode.DLLM_EXTEND, False),
        ):
            batch = MagicMock()
            with (
                patch(f"{module}.ScheduleBatch.init_new", return_value=batch) as init,
                patch(
                    "sglang.srt.managers.scheduler_components.metrics_reporter."
                    "PrefillStats.from_adder",
                    return_value=object(),
                ),
            ):
                result = SchedulerDllmMixin._create_dllm_batch(
                    scheduler,
                    [MagicMock()],
                    forward_mode,
                    scheduler.adder,
                    scheduler.running_batch,
                )

            self.assertIs(result, batch)
            self.assertEqual(init.call_args.kwargs["is_dllm_prefill"], expected)
            self.assertEqual(batch.forward_mode, forward_mode)

    def test_dllm_prefill_cuda_graph_capability_gate(self):
        class FakeBreakableBackend:
            pass

        runner = SimpleNamespace(
            backend=FakeBreakableBackend(),
            capture_num_tokens=[32, 128],
            capture_hidden_mode=None,
            device="cuda",
            max_num_tokens=128,
            model_runner=SimpleNamespace(attn_backend=object()),
            _is_full_backend=False,
            _has_unsupported_mha_prefix=lambda _batch: False,
            _has_inactive_dp_rank=lambda _batch: False,
            _pad_to_bucket=lambda raw_size, buckets: next(
                bucket for bucket in buckets if bucket >= raw_size
            ),
        )
        forward_batch = SimpleNamespace(
            dllm_config=SimpleNamespace(),
            is_dllm_prefill=True,
            forward_mode=ForwardMode.EXTEND,
            input_ids=[1] * 32,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            capture_hidden_mode=None,
            global_num_tokens_cpu=None,
            return_logprob=False,
        )
        module = "sglang.srt.model_executor.runner.prefill_cuda_graph_runner"
        with (
            patch(f"{module}.BreakableCudaGraphBackend", FakeBreakableBackend),
            patch(
                f"{module}._is_flashinfer_attention_backend", return_value=True
            ) as is_flashinfer,
            patch(f"{module}._is_hip", False),
            patch(f"{module}.is_npu", return_value=False),
        ):
            self.assertTrue(PrefillCudaGraphRunner.can_run_graph(runner, forward_batch))

            forward_batch.forward_mode = ForwardMode.DLLM_EXTEND
            self.assertFalse(
                PrefillCudaGraphRunner.can_run_graph(runner, forward_batch)
            )
            forward_batch.forward_mode = ForwardMode.EXTEND

            forward_batch.is_dllm_prefill = False
            self.assertFalse(
                PrefillCudaGraphRunner.can_run_graph(runner, forward_batch)
            )
            forward_batch.is_dllm_prefill = True

            forward_batch.input_ids = [1] * 31
            self.assertFalse(
                PrefillCudaGraphRunner.can_run_graph(runner, forward_batch)
            )
            forward_batch.input_ids = [1] * 32

            runner.backend = object()
            self.assertFalse(
                PrefillCudaGraphRunner.can_run_graph(runner, forward_batch)
            )
            runner.backend = FakeBreakableBackend()

            runner.device = "cpu"
            self.assertFalse(
                PrefillCudaGraphRunner.can_run_graph(runner, forward_batch)
            )
            runner.device = "cuda"

            is_flashinfer.return_value = False
            self.assertFalse(
                PrefillCudaGraphRunner.can_run_graph(runner, forward_batch)
            )
            is_flashinfer.return_value = True

            forward_batch.input_embeds = object()
            self.assertFalse(
                PrefillCudaGraphRunner.can_run_graph(runner, forward_batch)
            )
            forward_batch.input_embeds = None

            # Ordinary prefill retains the existing upward-bucket behavior.
            forward_batch.dllm_config = None
            forward_batch.input_ids = [1] * 31
            self.assertTrue(PrefillCudaGraphRunner.can_run_graph(runner, forward_batch))

    def test_preempt_success_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req, mock_server_args)

        self.assertTrue(success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)  # 50 + 75 + 100 - 50 = 175
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req, mock_server_args)

        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 125)  # 50 + 75 + 100 - 100 = 125
        running_batch.release_req.assert_called_once()

    def test_preempt_fail_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=2, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_fail_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=0, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=-1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_skip_already_preempted_request(self):
        params = [
            ("req_prio_0", 0, 50),
            ("req_prio_1", 1, 75),
            ("req_prio_2", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = 225
        self.mock_token_allocator.available_size.return_value = 225

        # New request preempts req_prio_0
        first_req = self.create_mock_req(
            "new_req_prio_1", priority=1, max_new_tokens=49
        )
        first_success = adder.preempt_to_schedule(first_req, mock_server_args)
        self.assertTrue(first_success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)
        running_batch.release_req.assert_called_once()

        # Second call needs more tokens than currently free, so it would need to
        # preempt req_prio_0 again if already-preempted requests were not filtered out.
        second_req = self.create_mock_req(
            "second_new_req_prio_1", priority=1, max_new_tokens=76
        )
        second_success = adder.preempt_to_schedule(second_req, mock_server_args)

        self.assertFalse(second_success)
        self.assertEqual(adder.rem_total_token_offset, 175)
        self.assertEqual(adder.preempt_list.count(running_reqs[0]), 1)
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_once(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=75)

        success = adder.preempt_to_schedule(new_req, mock_server_args)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 375
        )  # 50 + 75 + 100 + 125 + 125 - 100 = 375
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_twice(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=200)

        success = adder.preempt_to_schedule(new_req, mock_server_args)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertIn(running_reqs[3], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 250
        )  # 50 + 75 + 100 + 125 + 125 - 100 - 125 = 250
        self.assertEqual(running_batch.release_req.call_count, 2)

    def test_mixed_chunk_prefill_budgets(self):
        self.mock_token_allocator.available_size.return_value = 1000

        decode_reqs = [
            self.create_mock_req(f"decode_{i}", priority=0, max_new_tokens=50)
            for i in range(8)
        ]
        running_batch = self.create_running_batch(decode_reqs)

        adder = self.create_adder(
            running_batch,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(decode_reqs),
        )

        self.assertEqual(adder.rem_input_tokens, 192)  # 200 - 8
        self.assertEqual(adder.rem_chunk_tokens, 56)  # 64 - 8
        self.assertEqual(adder.rem_total_token_offset, 408)  # 8 + 8 * 50
        self.assertEqual(adder.cur_rem_token_offset, 8)
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)

        # Add a prefill that exactly consumes the chunk budget
        req1 = self.create_mock_req("req1", priority=0, max_new_tokens=64)
        req1.host_hit_length = 0
        req1.prefix_indices = []
        req1.full_untruncated_fill_ids = list(range(56))
        req1.last_node = MagicMock()
        req1.sampling_params.ignore_eos = False

        result1 = adder.add_one_req(
            req1, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder.can_run_list), 1)
        self.assertEqual(adder.rem_chunk_tokens, 0)  # 56 - 56
        self.assertEqual(adder.rem_input_tokens, 136)  # 192 - 56
        self.assertEqual(result1, AddReqResult.OTHER)

        # 3 decode requests finished
        remaining_decode_reqs = decode_reqs[3:]
        running_batch2 = self.create_running_batch(remaining_decode_reqs)

        adder2 = self.create_adder(
            running_batch2,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(remaining_decode_reqs),
        )

        self.assertEqual(adder2.rem_input_tokens, 195)  # 200 - 5
        self.assertEqual(adder2.rem_chunk_tokens, 59)  # 64 - 5
        self.assertEqual(adder2.rem_total_token_offset, 255)  # 5 + 5 * 50
        self.assertEqual(adder2.budget_state(), AddReqResult.CONTINUE)

        # Same prefill no longer exhausts the chunk budget
        req2 = self.create_mock_req("req2", priority=0, max_new_tokens=64)
        req2.host_hit_length = 0
        req2.prefix_indices = []
        req2.full_untruncated_fill_ids = list(range(56))
        req2.last_node = MagicMock()
        req2.sampling_params.ignore_eos = False

        result2 = adder2.add_one_req(
            req2, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 1)
        self.assertEqual(adder2.rem_chunk_tokens, 3)  # 59 - 56 = 3 remaining
        self.assertEqual(result2, AddReqResult.CONTINUE)

        # Fit last small prefill request
        req3 = self.create_mock_req("req3", priority=0, max_new_tokens=16)
        req3.host_hit_length = 0
        req3.prefix_indices = []
        req3.full_untruncated_fill_ids = list(range(3))
        req3.last_node = MagicMock()
        req3.sampling_params.ignore_eos = False

        result3 = adder2.add_one_req(
            req3, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 2)
        self.assertEqual(adder2.rem_chunk_tokens, 0)  # 3 - 3 = 0
        self.assertEqual(result3, AddReqResult.OTHER)

    def _build_hybrid_swa_chunked_req(
        self,
        *,
        page_size,
        rem_swa,
        rem_chunk=2048,
        extend_input_len=500,
        is_hybrid_swa=True,
        full_available=100_000,
    ):
        self.mock_token_allocator.swa_available_size.return_value = rem_swa
        self.mock_token_allocator.full_available_size.return_value = full_available
        self.mock_token_allocator.available_size.return_value = full_available
        self.mock_tree_cache.sliding_window_size = 128
        adder = self.create_adder(
            self.create_running_batch(),
            page_size=page_size,
            rem_chunk_tokens=rem_chunk,
        )
        adder.is_hybrid_swa = is_hybrid_swa

        req = self.create_mock_req("chunked", priority=0, max_new_tokens=128)
        req.prefix_indices = []
        req.full_untruncated_fill_ids = list(range(extend_input_len))
        # set_extend_range is the only writer of extend_range; the production
        # path reads req.extend_range.length right after calling it, so the mock
        # must actually set the attribute (a spec=Req mock has the method but
        # not the instance attribute).
        req.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(
                req, "extend_range", Range(start, end)
            )
        )
        return adder, req

    def test_add_chunked_req_hybrid_swa_reserves_page_for_alloc_extend(self):
        # alloc_extend needs extend_num_tokens + page_size per request. If the
        # scheduler hands out all of rem_swa_tokens, alloc_extend cannot get its
        # extra page and OOMs. With the fix, extend_input_len must cap at
        # rem_swa_tokens - page_size so the page is reserved.
        PAGE_SIZE = 64
        REM_SWA = 100
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=REM_SWA
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)  # truncated → chunked prefill continues
        req.set_extend_range.assert_called_once()
        start, end = req.set_extend_range.call_args.args
        new_len = end - start
        self.assertLessEqual(new_len + PAGE_SIZE, REM_SWA)
        self.assertEqual(new_len, REM_SWA - PAGE_SIZE)

    def test_add_chunked_req_hybrid_swa_defers_when_swa_below_page(self):
        # When rem_swa_tokens <= page_size there is no room to serve even the
        # reservation, so the chunked req must be deferred (returned unchanged)
        # instead of falling back to rem_chunk_tokens and bypassing SWA budget.
        PAGE_SIZE = 64
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=PAGE_SIZE
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)
        req.set_extend_range.assert_not_called()
        self.assertEqual(len(adder.can_run_list), 0)

    def test_swa_budget_for_req(self):
        cases = [
            # (extend, rem_chunk, window, page, expected, label)
            (64, None, 128, 16, 128 + 16, "no_cap_floor_active"),
            (200, None, 256, 32, 256 + 32, "no_cap_floor_active_other_dims"),
            (300, None, 128, 16, 300 + 16, "no_cap_floor_inactive"),
            (200, 50, 64, 8, 64 + 8, "cap_binds_then_floor"),
            (300, 500, 64, 64, 300 + 64, "cap_does_not_bind"),
            (0, None, 128, 16, 128 + 16, "extend_zero_floor_only"),
        ]
        for extend, rem_chunk, window, page, expected, label in cases:
            with self.subTest(label=label):
                self.mock_tree_cache.sliding_window_size = window
                adder = self.create_adder(
                    self.create_running_batch(),
                    page_size=page,
                    rem_chunk_tokens=rem_chunk,
                )
                self.assertEqual(adder._swa_budget_for_req(extend), expected)

    def test_add_chunked_req_non_hybrid_no_swa_reservation(self):
        # Non-hybrid path: the SWA-pool reservation must NOT apply, otherwise
        # the fix would regress non-SWA models.
        PAGE_SIZE = 16
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE,
            rem_swa=10,
            rem_chunk=500,
            extend_input_len=200,
            is_hybrid_swa=False,
            full_available=300,
        )

        result = adder.add_chunked_req(req)
        self.assertIsNone(result)
        req.set_extend_range.assert_called_once_with(0, 200)
        self.assertIn(req, adder.can_run_list)


if __name__ == "__main__":
    unittest.main()
