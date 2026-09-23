"""CPU contract tests for AFD configuration and eager execution."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.afd.afd_config import AFDRole, check_afd_server_args
from sglang.srt.afd.afd_sync import AFDSyncMsg
from sglang.srt.afd.comm.base import AFDCommConfig
from sglang.srt.afd.dispatcher import AFDDispatcher
from sglang.srt.afd.ffn.compute import AFDFFNCompute
from sglang.srt.afd.ffn.executor import AFDFFNExecutor
from sglang.srt.afd.layers import AFDMoELayer
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_args(**overrides):
    values = dict(
        model_path="dummy",
        afd_role="attn",
        afd_ffn_addr="127.0.0.1:30001",
        afd_backend="nccl",
        afd_ffn_moe_backend="none",
        cuda_graph_config=default_cuda_graph_config(),
    )
    values.update(overrides)
    return ServerArgs(**values)


class TestAFDServerArgsCheck(unittest.TestCase):
    def test_attention_role_resolves_eager_execution_and_routing(self):
        for disaggregation_mode in ("null", "decode"):
            with self.subTest(disaggregation_mode=disaggregation_mode):
                args = _make_args(disaggregation_mode=disaggregation_mode)
                original_graph_config = args.cuda_graph_config
                check_afd_server_args(args)
                resolved = resolving_view(args)
                self.assertEqual(resolved.moe_a2a_backend, "afd")
                self.assertEqual(resolved.afd_ffn_moe_backend, "none")
                self.assertTrue(resolved.disable_shared_experts_fusion)
                for phase in Phase.ALL:
                    self.assertEqual(
                        getattr(resolved.cuda_graph_config, phase).backend,
                        Backend.DISABLED,
                    )
                self.assertEqual(args.moe_a2a_backend, "none")
                self.assertIs(args.cuda_graph_config, original_graph_config)

    def test_ffn_role_disables_http_warmup_and_resolves_standard_topk_runner(self):
        args = _make_args(
            afd_role="ffn",
            afd_ffn_addr=None,
            moe_runner_backend="flashinfer_trtllm",
        )
        check_afd_server_args(args)
        resolved = resolving_view(args)
        self.assertTrue(resolved.skip_server_warmup)
        self.assertEqual(resolved.moe_a2a_backend, "none")
        self.assertEqual(resolved.moe_runner_backend, "triton")

    def test_disabled_afd_preserves_existing_configuration(self):
        args = _make_args(afd_role=None, afd_ffn_addr=None)
        original_graph_config = args.cuda_graph_config
        check_afd_server_args(args)
        resolved = resolving_view(args)
        self.assertEqual(resolved.moe_a2a_backend, "none")
        self.assertIs(resolved.cuda_graph_config, original_graph_config)
        self.assertFalse(resolved.disable_shared_experts_fusion)
        self.assertFalse(resolved.skip_server_warmup)

    def test_rejects_unsupported_execution_and_role_combinations(self):
        invalid = (
            ({"afd_ffn_addr": None}, "afd-ffn-addr"),
            ({"disaggregation_mode": "prefill"}, "disaggregation_mode"),
            ({"speculative_algorithm": "EAGLE"}, "speculative_algorithm"),
            ({"pp_size": 2}, "pp_size"),
            ({"ep_size": 2}, "ep-size"),
            ({"tp_size": 2}, "attn_tp_size"),
            ({"afd_role": "ffn"}, "attn-side"),
            (
                {
                    "afd_role": "ffn",
                    "afd_ffn_addr": None,
                    "disaggregation_mode": "decode",
                },
                "disaggregation_mode",
            ),
            (
                {"afd_role": None, "moe_a2a_backend": "afd"},
                "requires --afd-role attn",
            ),
        )
        for overrides, error in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, error):
                    check_afd_server_args(_make_args(**overrides))


class _CPUExperts(torch.nn.Module):
    def __init__(self, *, fuse_topk=False, runner_scale=1.0, post_scale=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.should_fuse_routed_scaling_factor_in_topk = fuse_topk
        self.runner_scale = runner_scale
        self.post_scale = post_scale

    def forward(self, hidden_states, topk):
        weights = ((topk.topk_ids + 1) * topk.topk_weights).sum(dim=1, keepdim=True)
        return hidden_states * weights * self.runner_scale


class _CPUComm:
    def __init__(self, role, incoming, outgoing):
        self.config = AFDCommConfig(role, 1, 1, 0, "localhost", 30001, "cpu")
        self.incoming = incoming
        self.outgoing = outgoing
        self.transfers = []

    def send_to(self, tensors, peer_global_rank):
        self.transfers.append(
            ("send", peer_global_rank, tuple(t.shape for t in tensors))
        )
        self.outgoing.put(tuple(tensor.clone() for tensor in tensors))

    def recv_from(self, tensors, peer_global_rank):
        received = self.incoming.get(timeout=5)
        if len(tensors) != len(received):
            raise AssertionError("AFD tensor count mismatch")
        for target, source in zip(tensors, received, strict=True):
            if target.shape != source.shape or target.dtype != source.dtype:
                raise AssertionError("AFD tensor shape or dtype mismatch")
            target.copy_(source)
        self.transfers.append(
            ("recv", peer_global_rank, tuple(t.shape for t in tensors))
        )


class TestAFDEagerRoundtrip(unittest.TestCase):
    def _pair(self):
        layers = [
            AFDMoELayer(0, _CPUExperts(fuse_topk=True), 2, 4, 2, 4, 0, 2.0),
            AFDMoELayer(1, _CPUExperts(runner_scale=3.0), 2, 4, 2, 4, 0, 3.0),
            AFDMoELayer(2, _CPUExperts(post_scale=True), 2, 4, 2, 4, 0, 4.0),
        ]
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(dtype=torch.float32, model=None)
            ),
            tp_group=SimpleNamespace(world_size=1, rank_in_group=0),
        )
        with (
            patch(
                "sglang.srt.afd.ffn.compute.collect_afd_moe_layers",
                return_value=layers,
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe."
                "requires_post_expert_routed_scaling",
                side_effect=lambda experts: experts.post_scale,
            ),
        ):
            compute = AFDFFNCompute(scheduler)
        requests, responses = Queue(), Queue()
        attn_comm = _CPUComm(AFDRole.ATTN, responses, requests)
        ffn_comm = _CPUComm(AFDRole.FFN, requests, responses)
        dispatcher = AFDDispatcher(
            attn_comm, layers, torch.float32, torch.device("cpu")
        )
        executor = AFDFFNExecutor(scheduler, ffn_comm, compute)
        return dispatcher, executor, attn_comm, ffn_comm

    def test_prefill_decode_roundtrip_applies_each_scaling_factor_once(self):
        dispatcher, executor, attn_comm, ffn_comm = self._pair()
        ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
        weights = torch.full((3, 2), 0.25)
        with (
            patch("sglang.srt.afd.ffn.executor.set_is_extend_in_batch") as set_extend,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 1
                expected = hidden.clone()
                step_id = dispatcher.begin_batch(mode, 3)
                result = pool.submit(
                    executor.run_step, AFDSyncMsg(step_id, [3], int(mode), 3)
                )
                for layer_id, scale in enumerate((2.0, 3.0, 4.0)):
                    handle = dispatcher.dispatch(layer_id, hidden, ids, weights)
                    hidden = dispatcher.combine(handle)
                    expected = (
                        expected * ((ids + 1) * weights).sum(1, keepdim=True) * scale
                    )
                    torch.testing.assert_close(hidden, expected)
                result.result(timeout=10)
                torch.testing.assert_close(weights, torch.full((3, 2), 0.25))
            self.assertEqual(
                [c.args[0] for c in set_extend.call_args_list], [True, False]
            )
        self.assertEqual(len(attn_comm.transfers), 12)
        self.assertEqual(len(ffn_comm.transfers), 12)
        self.assertEqual({peer for _, peer, _ in attn_comm.transfers}, {0})
        self.assertEqual({peer for _, peer, _ in ffn_comm.transfers}, {1})

    def test_zero_token_steps_do_not_touch_transport_or_experts(self):
        dispatcher, executor, attn_comm, ffn_comm = self._pair()
        with patch.object(executor._compute, "run") as run:
            for mode in (ForwardMode.IDLE, ForwardMode.DECODE):
                step_id = dispatcher.begin_batch(mode, 0)
                executor.run_step(AFDSyncMsg(step_id, [0], int(mode), 0))
        run.assert_not_called()
        self.assertEqual(attn_comm.transfers, [])
        self.assertEqual(ffn_comm.transfers, [])

    def test_invalid_control_and_layer_order_fail_before_transport(self):
        dispatcher, executor, attn_comm, ffn_comm = self._pair()
        invalid_messages = (
            AFDSyncMsg(0, [1, 1], int(ForwardMode.DECODE), 2),
            AFDSyncMsg(0, [2], int(ForwardMode.DECODE), 3),
            AFDSyncMsg(0, [1], int(ForwardMode.IDLE), 1),
        )
        for message in invalid_messages:
            with self.subTest(message=message), self.assertRaises(RuntimeError):
                executor.run_step(message)
        dispatcher.begin_batch(ForwardMode.DECODE, 1)
        with self.assertRaisesRegex(RuntimeError, "layer order mismatch"):
            dispatcher.dispatch(
                1,
                torch.zeros(1, 2),
                torch.zeros(1, 2, dtype=torch.int32),
                torch.zeros(1, 2),
            )
        with self.assertRaisesRegex(RuntimeError, "previous batch"):
            dispatcher.begin_batch(ForwardMode.DECODE, 1)
        self.assertEqual(attn_comm.transfers, [])
        self.assertEqual(ffn_comm.transfers, [])


class TestAFDSchedulerRouting(unittest.TestCase):
    def test_pd_prebuilt_placeholder_does_not_send_an_ffn_step(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.metrics_reporter = Mock()
        scheduler.forward_ct = 0
        scheduler._sched_idled = True
        scheduler.scripted_scheduler_hook = None
        scheduler.profiler_manager = Mock()
        scheduler.forward_sleep_time = None
        scheduler._afd_pre_run_batch = Mock()
        result = object()
        scheduler._run_batch_prebuilt = Mock(return_value=result)
        batch = SimpleNamespace(forward_mode=ForwardMode.PREBUILT)

        actual = Scheduler.run_batch.__wrapped__(scheduler, batch)

        self.assertIs(actual, result)
        scheduler._run_batch_prebuilt.assert_called_once_with(batch)
        scheduler._afd_pre_run_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
