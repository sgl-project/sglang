import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.layernorm import RMSNorm  # noqa: E402
from sglang.srt.managers.scheduler_components.pp_dspark_draft import (  # noqa: E402
    PPDSparkDraftCoordinator,
)
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    SchedulerPPMixin,
    _pp_snapshot_forward_batch,
    _pp_use_batched_result_relay,
)
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.runner.base_runner import (  # noqa: E402
    _allocate_decode_buffers,
)
from sglang.srt.model_executor.runner_utils.buffers import (  # noqa: E402
    DecodeInputBuffers,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM  # noqa: E402
from sglang.srt.models.deepseek_v4_dspark import (  # noqa: E402
    DeepseekV4ForCausalLMDSpark,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_pp import (  # noqa: E402
    draft_owner,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (  # noqa: E402
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (  # noqa: E402
    DSparkWorkerV2,
    PPDSparkCommitState,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSparkPPContext(CustomTestCase):
    def test_batched_result_relay_gate_is_cuda_pp2_replicated_dspark_only(self):
        cases = [
            (True, 2, True, True, True),
            (True, 2, True, False, False),
            (True, 2, False, True, False),
            (True, 4, True, True, False),
            (False, 2, True, True, False),
        ]
        for cuda_available, pp_size, replicated, enabled, expected in cases:
            with (
                self.subTest(
                    cuda_available=cuda_available,
                    pp_size=pp_size,
                    replicated=replicated,
                    enabled=enabled,
                ),
                patch(
                    "sglang.srt.managers.scheduler_pp_mixin._is_npu",
                    False,
                ),
                patch(
                    "sglang.srt.managers.scheduler_pp_mixin.torch.cuda.is_available",
                    return_value=cuda_available,
                ),
                patch(
                    "sglang.srt.managers.scheduler_pp_mixin.get_parallel",
                    return_value=SimpleNamespace(pp_size=pp_size),
                ),
                patch(
                    "sglang.srt.managers.scheduler_pp_mixin.get_spec",
                    return_value=SimpleNamespace(
                        speculative_dspark_pp_replicated_draft=replicated
                    ),
                ),
                patch(
                    "sglang.srt.managers.scheduler_pp_mixin.envs."
                    "SGLANG_PP_DSPARK_BATCHED_RESULT_RELAY.get",
                    return_value=enabled,
                ),
            ):
                self.assertEqual(_pp_use_batched_result_relay(), expected)

    def test_batched_result_relay_waits_for_send_and_recv_events(self):
        send_ready_event = Mock()
        recv_event = Mock()
        d2h_event = Mock()
        current_stream = Mock()
        batch_result = SimpleNamespace(logits_output=None)
        target = SimpleNamespace(
            forward_mode=SimpleNamespace(is_prebuilt=Mock(return_value=False)),
            return_logprob=False,
        )
        pp_group = SimpleNamespace(
            is_last_rank=True,
            send_recv_tensor_dict=Mock(return_value={"next_token_ids": torch.ones(1)}),
        )
        scheduler = SimpleNamespace(
            pp_group=pp_group,
            attn_tp_group=object(),
            pp_comm_stream_ctx=nullcontext(),
            copy_stream_ctx=nullcontext(),
            copy_stream=Mock(),
            schedule_stream=Mock(),
            device_module=SimpleNamespace(
                Event=Mock(return_value=d2h_event),
                current_stream=Mock(return_value=current_stream),
            ),
            _pp_record_comm_event=Mock(return_value=recv_event),
            _pp_prep_batch_result=Mock(return_value=batch_result),
        )
        outputs = PPProxyTensors({"accept_lens": torch.ones(1)})
        queue = deque([(send_ready_event, outputs)])

        next_outputs, result, event, send_work = (
            SchedulerPPMixin._pp2_send_recv_output_tensors_batched(
                scheduler,
                next_first_rank_mb_id=0,
                next_mb_id=0,
                mbs=[target],
                mb_metadata=[object()],
                last_rank_comm_queue=queue,
                pp_outputs=None,
            )
        )

        current_stream.wait_event.assert_called_once_with(send_ready_event)
        pp_group.send_recv_tensor_dict.assert_called_once()
        scheduler.copy_stream.wait_event.assert_called_once_with(recv_event)
        self.assertIs(result, batch_result)
        self.assertIs(event, d2h_event)
        self.assertIn("next_token_ids", next_outputs.tensors)
        self.assertEqual(send_work, [])

    def test_forward_snapshot_copies_draft_counts_only_for_replicated_dspark(self):
        snapshot = SimpleNamespace()
        req = object()
        batch = SimpleNamespace(
            reqs=[req],
            spec_algorithm=SimpleNamespace(is_none=lambda: False),
            req_pool_indices=torch.tensor([3]),
            orig_seq_lens=torch.tensor([7]),
            draft_global_num_tokens=[1, 2],
            draft_global_num_tokens_for_logprob=[1, 2],
            copy=Mock(return_value=snapshot),
        )

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_spec",
            return_value=SimpleNamespace(speculative_dspark_pp_replicated_draft=False),
        ):
            result = _pp_snapshot_forward_batch(batch)
            self.assertFalse(hasattr(result, "draft_global_num_tokens"))

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_spec",
            return_value=SimpleNamespace(speculative_dspark_pp_replicated_draft=True),
        ):
            result = _pp_snapshot_forward_batch(batch)
            self.assertEqual(result.draft_global_num_tokens, [1, 2])
            self.assertEqual(result.draft_global_num_tokens_for_logprob, [1, 2])
            self.assertTrue(torch.equal(result.orig_seq_lens, torch.tensor([7])))
            self.assertIsNot(result.reqs, batch.reqs)
            self.assertIs(result.reqs[0], req)
            self.assertIsNot(result.req_pool_indices, batch.req_pool_indices)

    def test_idle_verify_without_ragged_layout_uses_target_verify_mode(self):
        target_worker = Mock()
        target_worker.forward_batch_generation.return_value = object()
        executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
        executor.model_runner = SimpleNamespace(device="cpu")
        executor.verify_num_draft_tokens = 5
        executor.verify_epilogue = None
        executor.target_worker = target_worker
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            seq_lens=torch.empty((0,), dtype=torch.int64),
            req_pool_indices=torch.empty((0,), dtype=torch.int64),
            global_num_tokens=[0, 1, 1, 1],
            global_num_tokens_for_logprob=[0, 1, 1, 1],
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "DFlashVerifyInput.prepare_for_verify",
            return_value=(
                SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY),
                False,
            ),
        ):
            executor.run_idle_participation(batch=batch, idle_layout=None)

        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        forward_batch = target_worker.forward_batch_generation.call_args.kwargs[
            "forward_batch"
        ]
        self.assertEqual(forward_batch.forward_mode, ForwardMode.TARGET_VERIFY)

    def test_replicated_commit_writes_only_locally_owned_rows(self):
        rids = []
        for owner in range(2):
            suffix = 0
            while True:
                rid = f"owner-{owner}-{suffix}"
                if draft_owner(rid, 2) == owner:
                    rids.append(rid)
                    break
                suffix += 1

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.device = "cpu"
        worker.ps = SimpleNamespace(pp_rank=0, pp_size=2)
        worker._replicate_pp_prefill_context = False
        worker._kv_injector = Mock()
        state = PPDSparkCommitState(
            rids=tuple(rids),
            cache_loc=torch.tensor([10, 11, 20, 21, 22]),
            cache_loc_2d=None,
            positions=torch.tensor([0, 1, 7, 8, 9]),
            state_slot=None,
            token_counts=(2, 3),
        )
        projected = torch.arange(20).view(5, 4)

        worker.commit_pp_draft_context(
            state=state,
            rids=tuple(rids),
            projected_context=projected,
            commit_lens=None,
        )

        kwargs = worker._kv_injector.inject_projected_context.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["projected_context"], projected[:2]))
        self.assertTrue(torch.equal(kwargs["cache_loc"], torch.tensor([10, 11])))
        self.assertTrue(torch.equal(kwargs["positions"], torch.tensor([0, 1])))

    def test_radix_prefill_commit_writes_all_rows_on_each_replica(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.device = "cpu"
        worker.ps = SimpleNamespace(pp_rank=0, pp_size=2)
        worker._replicate_pp_prefill_context = True
        worker._kv_injector = Mock()
        state = PPDSparkCommitState(
            rids=("request-a", "request-b"),
            cache_loc=torch.tensor([10, 11, 20, 21, 22]),
            cache_loc_2d=None,
            positions=torch.tensor([0, 1, 7, 8, 9]),
            state_slot=None,
            token_counts=(2, 3),
        )
        projected = torch.arange(20).view(5, 4)

        worker.commit_pp_draft_context(
            state=state,
            rids=state.rids,
            projected_context=projected,
            commit_lens=None,
        )

        kwargs = worker._kv_injector.inject_projected_context.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["projected_context"], projected))
        self.assertTrue(torch.equal(kwargs["cache_loc"], state.cache_loc))
        self.assertTrue(torch.equal(kwargs["positions"], state.positions))

    @patch(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.alloc_verify_window"
    )
    def test_empty_owner_partition_runs_idle_draft_with_gathered_counts(
        self, alloc_verify_window
    ):
        alloc_verify_window.return_value = SimpleNamespace()
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.device = "cpu"
        worker.ps = SimpleNamespace(pp_rank=0, pp_size=2, attn_dp_rank=0)
        worker._pp_draft_dp_enabled = True
        worker.verify_num_draft_tokens = 5
        worker._block_pos_offsets = torch.empty(0)
        worker.model_runner = Mock()
        worker._proposer = Mock()
        worker._draft_context = Mock(return_value=nullcontext())
        worker._observers = Mock()
        worker._observers.segment.return_value = nullcontext()
        reqs = []
        suffix = 0
        while len(reqs) < 2:
            rid = f"owner-1-{suffix}"
            if draft_owner(rid, 2) == 1:
                reqs.append(
                    SimpleNamespace(
                        rid=rid,
                        bootstrap_room=1,
                        retraction_count=0,
                        spec_verify_ct=2,
                    )
                )
            suffix += 1
        batch = SimpleNamespace(
            reqs=reqs,
            seq_lens=torch.tensor([7, 9]),
            global_num_tokens=[3, 2, 5, 2],
            global_num_tokens_for_logprob=[3, 2, 5, 2],
            draft_global_num_tokens=[0, 2, 1, 0],
        )
        draft_input = SimpleNamespace(new_seq_lens=torch.tensor([8, 10]))

        payload = worker.prepare_pp_draft(batch, draft_input)

        self.assertEqual(payload, {"identities": []})
        idle_batch = worker._proposer.run_idle_participation.call_args.args[0]
        self.assertEqual(idle_batch.global_num_tokens, [0, 2, 1, 0])
        self.assertEqual(idle_batch.global_num_tokens_for_logprob, [0, 2, 1, 0])

    @patch(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
        return_value=SimpleNamespace(attn_dp_enabled=True),
    )
    def test_replicated_final_idle_lane_defers_draft_to_scheduler(self, _):
        calls = []
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.device = "cpu"
        worker.ps = SimpleNamespace(pp_rank=1, pp_size=2, attn_dp_rank=0)
        worker._pp_draft_dp_enabled = True
        worker._replicated_pp_decode = True
        worker._draft_is_moe = True
        worker._observers = Mock()
        worker._verify_executor = Mock()
        verify_result = object()
        worker._verify_executor.run_idle_participation.side_effect = lambda **kwargs: (
            calls.append(("verify", kwargs["batch"])),
            verify_result,
        )[1]
        worker._proposer = Mock()
        worker._proposer.run_idle_participation.side_effect = lambda batch: (
            calls.append(("draft", batch))
        )
        worker._idle_verify_ragged_layout = Mock(return_value=None)
        idle_result = object()
        worker._decode_idle_result = Mock(return_value=idle_result)
        pp_proxy_tensors = object()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True),
            spec_info=DFlashDraftInputV2.create_idle_input(device="cpu"),
            global_num_tokens=[0, 2, 1, 0],
            global_num_tokens_for_logprob=[0, 2, 1, 0],
            draft_global_num_tokens=[0, 2, 1, 0],
        )

        result = worker._forward_decode(
            batch, on_publish=None, pp_proxy_tensors=pp_proxy_tensors
        )

        self.assertIs(result, idle_result)
        worker._decode_idle_result.assert_called_once_with(
            on_publish=None, draft_idle=True
        )
        self.assertEqual([name for name, _ in calls], ["verify"])
        worker._proposer.run_idle_participation.assert_not_called()
        self.assertIs(calls[0][1], batch)
        self.assertIs(
            worker._verify_executor.run_idle_participation.call_args.kwargs[
                "pp_proxy_tensors"
            ],
            pp_proxy_tensors,
        )

    @patch(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
        return_value=SimpleNamespace(attn_dp_enabled=True),
    )
    def test_replicated_nonfinal_idle_lane_only_forwards_verify_proxy(self, _):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.device = "cpu"
        worker.ps = SimpleNamespace(pp_rank=0, pp_size=2, attn_dp_rank=0)
        worker._replicated_pp_decode = True
        worker._draft_is_moe = True
        worker._observers = Mock()
        worker._verify_executor = Mock()
        verify_result = object()
        worker._verify_executor.run_idle_participation.return_value = verify_result
        worker._proposer = Mock()
        worker._idle_verify_ragged_layout = Mock(return_value=None)
        worker._decode_idle_result = Mock()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True),
            spec_info=DFlashDraftInputV2.create_idle_input(device="cpu"),
            global_num_tokens=[0, 2, 1, 0],
        )

        result = worker._forward_decode(
            batch, on_publish=None, pp_proxy_tensors=object()
        )

        self.assertIs(result, verify_result)
        worker._decode_idle_result.assert_not_called()
        worker._proposer.run_idle_participation.assert_not_called()

    def test_prepare_pp_idle_draft_uses_gathered_owner_counts(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.ps = SimpleNamespace(attn_dp_rank=0)
        worker._pp_draft_dp_enabled = True
        worker._draft_context = Mock(return_value=nullcontext())
        worker._observers = Mock()
        worker._observers.segment.return_value = nullcontext()
        worker._proposer = Mock()
        batch = SimpleNamespace(draft_global_num_tokens=[0, 2, 1, 0])

        worker.prepare_pp_idle_draft(batch)

        idle_batch = worker._proposer.run_idle_participation.call_args.args[0]
        self.assertEqual(idle_batch.global_num_tokens, [0, 2, 1, 0])
        self.assertEqual(idle_batch.global_num_tokens_for_logprob, [0, 2, 1, 0])

    def test_pp_launch_schedules_idle_draft_on_last_stage(self):
        result = GenerationBatchResult(pp_dspark_draft_idle=True)
        draft_coordinator = Mock()
        scheduler = SimpleNamespace(
            forward_stream_ctx=nullcontext(),
            forward_stream=Mock(),
            schedule_stream=Mock(),
            run_batch=Mock(return_value=result),
            _pp_wait_forward_dependencies=Mock(),
            pp_dspark_draft=draft_coordinator,
            _pp_prepare_tensor_dict=Mock(return_value={}),
            device_module=SimpleNamespace(
                Event=Mock(return_value=Mock()), current_stream=Mock()
            ),
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        batch = SimpleNamespace(
            reqs=[],
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        metadata = [None]

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_spec",
            return_value=SimpleNamespace(speculative_dspark_pp_replicated_draft=False),
        ):
            SchedulerPPMixin._pp_launch_batch(
                scheduler, 0, batch, None, metadata, deque()
            )

        draft_coordinator.on_batch_launched.assert_called_once_with(batch, result)

    def test_coordinator_runs_idle_draft_only_on_last_stage(self):
        result = GenerationBatchResult(pp_dspark_draft_idle=True)
        model_worker = SimpleNamespace(prepare_pp_idle_draft=Mock())
        scheduler = SimpleNamespace(
            model_worker=model_worker,
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        coordinator = PPDSparkDraftCoordinator(scheduler)
        batch = object()

        coordinator.on_batch_launched(batch, result)
        model_worker.prepare_pp_idle_draft.assert_called_once_with(batch)

        scheduler.pp_group.is_last_rank = False
        coordinator.on_batch_launched(batch, result)
        self.assertEqual(model_worker.prepare_pp_idle_draft.call_count, 1)

    def test_coordinator_enqueues_bubble_draft_without_running_it(self):
        next_draft_input = object()
        result = GenerationBatchResult(
            accept_lens=torch.ones(1, dtype=torch.int32),
            next_draft_input=next_draft_input,
            pp_dspark_projected_context=torch.ones(1),
        )
        scheduler = SimpleNamespace(
            model_worker=SimpleNamespace(prepare_pp_draft=Mock()),
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        coordinator = PPDSparkDraftCoordinator(scheduler)
        batch = SimpleNamespace(
            reqs=[],
            req_pool_indices=torch.tensor([3]),
            orig_seq_lens=torch.tensor([7]),
        )

        with patch(
            "sglang.srt.managers.scheduler_components.pp_dspark_draft.get_spec",
            return_value=SimpleNamespace(speculative_draft_scheduling_policy="bubble"),
        ):
            coordinator.on_batch_launched(batch, result)

        scheduler.model_worker.prepare_pp_draft.assert_not_called()
        self.assertEqual(len(coordinator._pending), 1)
        work = coordinator._pending[0]
        self.assertIsNot(work.batch, batch)
        self.assertTrue(torch.equal(work.batch.orig_seq_lens, torch.tensor([7])))
        self.assertIs(work.draft_input, next_draft_input)

        batch.orig_seq_lens = None
        self.assertTrue(torch.equal(work.batch.orig_seq_lens, torch.tensor([7])))

    def test_first_rank_defers_relayed_bubble_proposals(self):
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True),
        )
        coordinator = PPDSparkDraftCoordinator(scheduler)
        batch = SimpleNamespace(
            reqs=[],
            req_pool_indices=torch.tensor([3]),
            orig_seq_lens=torch.tensor([7]),
        )
        draft_input = object()
        outputs = PPProxyTensors({})

        with patch(
            "sglang.srt.managers.scheduler_components.pp_dspark_draft.get_spec",
            return_value=SimpleNamespace(speculative_draft_scheduling_policy="bubble"),
        ):
            coordinator.on_relayed_proposals(batch, draft_input, outputs)

        self.assertEqual(len(coordinator._pending), 1)
        work = coordinator._pending[0]
        self.assertIsNot(work.batch, batch)
        self.assertIs(work.draft_input, draft_input)
        self.assertIs(work.pp_outputs, outputs)

    def test_last_rank_drains_bubble_draft_as_typed_message(self):
        batch = SimpleNamespace(reqs=[], req_pool_indices=torch.tensor([3]))
        draft_input = object()
        proposal = {"identities": [(1, 2, 3)]}
        event = Mock()
        scheduler = SimpleNamespace(
            _pp_commit_comm_work=Mock(),
            forward_stream_ctx=nullcontext(),
            _pp_send_dict_to_next_stage=Mock(return_value=["send-work"]),
            model_worker=SimpleNamespace(prepare_pp_draft=Mock(return_value=proposal)),
            device_module=SimpleNamespace(
                Event=Mock(return_value=event), current_stream=Mock()
            ),
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        coordinator = PPDSparkDraftCoordinator(scheduler)
        coordinator.enqueue(batch, draft_input)

        with (
            patch(
                "sglang.srt.managers.scheduler_components.pp_dspark_draft.get_parallel",
                return_value=SimpleNamespace(pp_rank=1),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.pp_dspark_draft.get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_scheduling_policy="bubble"
                ),
            ),
        ):
            coordinator.drain()

        draft_batch = scheduler.model_worker.prepare_pp_draft.call_args.args[0]
        self.assertIsNot(draft_batch, batch)
        self.assertIs(
            scheduler.model_worker.prepare_pp_draft.call_args.args[1],
            draft_input,
        )
        args, kwargs = scheduler._pp_send_dict_to_next_stage.call_args
        self.assertEqual(kwargs["msg_type"], "dspark_draft")
        self.assertEqual(args[0]["dspark_next_1_identities"], [(1, 2, 3)])
        self.assertEqual(coordinator._send_work, ["send-work"])

    def test_first_rank_drains_and_installs_both_owner_proposals(self):
        batch = SimpleNamespace(reqs=[], req_pool_indices=torch.tensor([3]))
        draft_input = object()
        outputs = PPProxyTensors({})
        local = {"identities": [(0, 0, 1)]}
        remote = {"dspark_next_1_identities": [(1, 0, 1)]}
        scheduler = SimpleNamespace(
            _pp_commit_comm_work=Mock(),
            forward_stream_ctx=nullcontext(),
            forward_stream=Mock(),
            _pp_recv_typed_dict=Mock(return_value=(remote, None)),
            model_worker=SimpleNamespace(
                prepare_pp_draft=Mock(return_value=local),
                install_pp_draft=Mock(),
            ),
            attn_tp_group=object(),
            device_module=SimpleNamespace(
                Event=Mock(return_value=Mock()), current_stream=Mock()
            ),
            pp_group=SimpleNamespace(is_last_rank=False, is_first_rank=True),
        )
        coordinator = PPDSparkDraftCoordinator(scheduler)
        coordinator.enqueue(batch, draft_input, outputs)

        with (
            patch(
                "sglang.srt.managers.scheduler_components.pp_dspark_draft.get_parallel",
                return_value=SimpleNamespace(pp_size=2),
            ),
            patch(
                "sglang.srt.managers.scheduler_components.pp_dspark_draft.get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_scheduling_policy="bubble"
                ),
            ),
        ):
            coordinator.drain()

        scheduler._pp_recv_typed_dict.assert_called_once_with(
            expected_kind="dspark_draft",
            all_gather_group=scheduler.attn_tp_group,
        )
        self.assertEqual(
            scheduler.model_worker.install_pp_draft.call_count,
            2,
        )
        self.assertIn("dspark_next_0_identities", outputs.tensors)
        self.assertIn("dspark_next_1_identities", outputs.tensors)

    def test_pp_relay_schedules_idle_draft_on_first_stage(self):
        model_worker = SimpleNamespace(prepare_pp_idle_draft=Mock())
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=True),
            forward_stream=Mock(),
            copy_stream=Mock(),
            forward_stream_ctx=nullcontext(),
            model_worker=model_worker,
        )
        coordinator = PPDSparkDraftCoordinator(scheduler)
        batch = object()
        outputs = PPProxyTensors({"dspark_draft_idle": torch.ones(1)})

        coordinator.on_relayed_idle(batch, outputs)

        scheduler.forward_stream.wait_stream.assert_called_once_with(
            scheduler.copy_stream
        )
        model_worker.prepare_pp_idle_draft.assert_called_once_with(batch)
        scheduler.copy_stream.wait_stream.assert_called_once_with(
            scheduler.forward_stream
        )

    def test_pp_spec_verify_buffers_use_token_axis(self):
        max_bs = 64
        num_tokens_per_req = 6
        max_num_token = max_bs * num_tokens_per_req
        common_kwargs = dict(
            device=torch.device("cpu"),
            max_bs=max_bs,
            max_num_token=max_num_token,
            hidden_size=4,
            dtype=torch.float32,
            num_dp_ranks=1,
            pp_size=8,
            is_encoder_decoder=False,
            require_mlp_tp_gather=False,
            seq_len_fill_value=1,
            encoder_len_fill_value=0,
            num_tokens_per_req=num_tokens_per_req,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            hc_hidden_size=16,
        )
        eager_kwargs = {
            key: value
            for key, value in common_kwargs.items()
            if key not in ("num_dp_ranks", "pp_size")
        }
        with (
            patch(
                "sglang.srt.model_executor.runner.base_runner.get_parallel",
                return_value=SimpleNamespace(num_dp_ranks=1, pp_size=8),
            ),
            patch(
                "sglang.srt.model_executor.runner.base_runner.enable_num_token_non_padded",
                return_value=False,
            ),
            patch(
                "sglang.srt.model_executor.runner_utils.buffers.enable_num_token_non_padded",
                return_value=False,
            ),
        ):
            eager_buffers = _allocate_decode_buffers(vocab_size=8, **eager_kwargs)
            graph_buffers = DecodeInputBuffers.create(
                next_token_logits_buffer=torch.zeros((max_num_token, 8)),
                **common_kwargs,
            )

        self.assertEqual(
            eager_buffers.pp_proxy_tensors["hidden_states"].shape,
            (max_num_token, 16),
        )
        self.assertEqual(
            graph_buffers.pp_proxy_tensors["hidden_states"].shape,
            (max_num_token, 16),
        )

    def test_deepseek_v4_projected_context_is_normalized_before_kv_write(self):
        model = DeepseekV4ForCausalLMDSpark.__new__(DeepseekV4ForCausalLMDSpark)
        torch.nn.Module.__init__(model)
        stage = torch.nn.Module()
        stage.main_norm = RMSNorm(4, eps=1e-6)
        model.stages = torch.nn.ModuleList([stage])
        projected_context = torch.randn(5, 4)
        expected = stage.main_norm(projected_context)
        write_context_hidden_kv = Mock()
        model._write_context_hidden_kv = write_context_hidden_kv

        model.write_projected_context_kv(
            projected_context=projected_context,
            swa_loc=torch.arange(5),
            positions=torch.arange(5),
            pool=object(),
        )

        torch.testing.assert_close(
            write_context_hidden_kv.call_args.kwargs["main_x"],
            expected,
        )

    def test_deepseek_v4_capture_is_local_to_each_pp_rank(self):
        model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_last_rank=False)
        model.model = SimpleNamespace(
            start_layer=10,
            end_layer=20,
            dspark_layers_to_capture=None,
        )
        model.capture_aux_hidden_states = False

        with patch(
            "sglang.srt.models.deepseek_v4.get_spec",
            return_value=SimpleNamespace(speculative_dspark_pp_replicated_draft=True),
        ):
            model.set_dspark_layers_to_capture([5, 12, 18, 25])

        self.assertTrue(model.capture_aux_hidden_states)
        self.assertEqual(model.model.dspark_layers_to_capture, [12, 18])

    def test_deepseek_v4_non_replicated_capture_stays_on_last_pp_rank(self):
        model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_last_rank=False)
        model.model = SimpleNamespace(dspark_layers_to_capture=None)
        model.capture_aux_hidden_states = False

        with patch(
            "sglang.srt.models.deepseek_v4.get_spec",
            return_value=SimpleNamespace(speculative_dspark_pp_replicated_draft=False),
        ):
            model.set_dspark_layers_to_capture([5, 12, 18, 25])

        self.assertFalse(model.capture_aux_hidden_states)
        self.assertIsNone(model.model.dspark_layers_to_capture)


if __name__ == "__main__":
    unittest.main()
