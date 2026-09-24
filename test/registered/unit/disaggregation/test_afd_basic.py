"""CPU contract tests for AFD configuration, fixed buffers, and execution."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from queue import Queue
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.afd.afd_config import AFDRole, check_afd_server_args
from sglang.srt.afd.afd_handshake import (
    AFDGraphCaptureVariant,
    AFDHandshakeInfo,
    AFDHandshakeResult,
    build_afd_ffn_capacity_plan,
)
from sglang.srt.afd.afd_scheduler import (
    AFDSyncSender,
    _decide_afd_attn_graph,
    prepare_afd_attn_batch,
)
from sglang.srt.afd.afd_sync import AFDSyncMsg
from sglang.srt.afd.comm.base import AFDCommConfig
from sglang.srt.afd.dispatcher import AFDDispatcher
from sglang.srt.afd.ffn.compute import AFDFFNCompute
from sglang.srt.afd.ffn.executor import AFDFFNExecutor
from sglang.srt.afd.layers import AFDMoELayer
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
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
    def test_attention_role_preserves_decode_graph_and_resolves_routing(self):
        for disaggregation_mode, backend in product(
            ("null", "decode"), ("none", "deepep")
        ):
            with self.subTest(disaggregation_mode=disaggregation_mode, backend=backend):
                args = _make_args(
                    disaggregation_mode=disaggregation_mode, moe_a2a_backend=backend
                )
                original_graph_config = args.cuda_graph_config
                check_afd_server_args(args)
                resolved = resolving_view(args)
                self.assertEqual(resolved.moe_a2a_backend, "afd")
                self.assertEqual(resolved.afd_ffn_moe_backend, backend)
                self.assertTrue(resolved.disable_shared_experts_fusion)
                self.assertEqual(
                    resolved.cuda_graph_config.decode.backend, Backend.FULL
                )
                self.assertEqual(
                    resolved.cuda_graph_config.prefill.backend, Backend.DISABLED
                )
                self.assertEqual(args.moe_a2a_backend, backend)
                self.assertIs(args.cuda_graph_config, original_graph_config)

    def test_ffn_role_disables_http_warmup_and_resolves_standard_topk_runner(self):
        for backend in ("none", "deepep"):
            with self.subTest(backend=backend):
                args = _make_args(
                    afd_role="ffn",
                    afd_ffn_addr=None,
                    moe_a2a_backend=backend,
                    moe_runner_backend="flashinfer_trtllm",
                )
                check_afd_server_args(args)
                resolved = resolving_view(args)
                self.assertTrue(resolved.skip_server_warmup)
                self.assertEqual(resolved.moe_a2a_backend, backend)
                self.assertEqual(resolved.afd_ffn_moe_backend, backend)
                self.assertEqual(
                    resolved.moe_runner_backend,
                    "triton" if backend == "none" else "flashinfer_trtllm",
                )

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
    def __init__(
        self, *, fuse_topk=False, runner_scale=1.0, post_scale=False, barrier=None
    ):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.should_fuse_routed_scaling_factor_in_topk = fuse_topk
        self.runner_scale = runner_scale
        self.post_scale = post_scale
        self.barrier = barrier
        self.inputs = []
        self.routing_ids = []
        self.output_ptrs = []

    def forward(self, hidden_states, topk):
        self.inputs.append(hidden_states.clone())
        self.routing_ids.append(topk.topk_ids.clone())
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        weights = ((topk.topk_ids + 1) * topk.topk_weights).sum(dim=1, keepdim=True)
        output = hidden_states * weights * self.runner_scale
        self.output_ptrs.append(output.data_ptr())
        return output


class _CPUComm:
    def __init__(self, config, channels):
        self.config = config
        self.channels = channels
        self.transfers = []
        self.sent_tensor_ptrs = []

    def send_to(self, tensors, peer_global_rank):
        self.sent_tensor_ptrs.append(tuple(t.data_ptr() for t in tensors))
        self.transfers.append(
            ("send", peer_global_rank, tuple(t.shape for t in tensors))
        )
        self.channels[self.config.global_rank, peer_global_rank].put(
            tuple(tensor.clone() for tensor in tensors)
        )

    def recv_from(self, tensors, peer_global_rank):
        received = self.channels[peer_global_rank, self.config.global_rank].get(
            timeout=5
        )
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
    def _executor(self, comm, layers, *, scattered=False, deepep_mode="normal"):
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    dtype=torch.float32, model=None, device="cpu"
                )
            ),
            tp_group=SimpleNamespace(
                world_size=comm.config.num_ffn_ranks,
                rank_in_group=comm.config.instance_rank,
            ),
            ps=SimpleNamespace(gpu_id=0),
            device_module=SimpleNamespace(),
        )
        backend = "deepep" if scattered else "none"
        attn = AFDHandshakeInfo(
            "attn",
            "model",
            comm.config.num_attn_ranks,
            "nccl",
            backend,
            "float32",
            [],
            list(range(1, 17)),
        )
        ffn = AFDHandshakeInfo(
            "ffn", "model", comm.config.num_ffn_ranks, "nccl", backend, "float32", []
        )
        handshake = AFDHandshakeResult(ffn, attn, "localhost", 30002, 30003)
        disagg = SimpleNamespace(afd_ffn_moe_backend="deepep" if scattered else "none")
        with (
            patch("sglang.srt.afd.ffn.compute.get_disagg", return_value=disagg),
            patch("sglang.srt.afd.ffn.executor.get_disagg", return_value=disagg),
            patch(
                "sglang.srt.afd.ffn.compute.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(deepep_mode=deepep_mode),
                    graph=SimpleNamespace(
                        disable_cuda_graph_padding=False, enable_cudagraph_gc=False
                    ),
                    features=SimpleNamespace(enable_memory_saver=False),
                ),
            ),
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
            return AFDFFNExecutor(scheduler, comm, compute, handshake)

    def _comm(self, role, rank, num_attn, num_ffn, channels):
        return _CPUComm(
            AFDCommConfig(role, num_attn, num_ffn, rank, "localhost", 30001, "cpu"),
            channels,
        )

    def _pair(self, *, scattered=False):
        layers = [
            AFDMoELayer(0, _CPUExperts(fuse_topk=True), 2, 4, 2, 4, 0, 2.0),
            AFDMoELayer(1, _CPUExperts(runner_scale=3.0), 2, 4, 2, 4, 0, 3.0),
            AFDMoELayer(2, _CPUExperts(post_scale=True), 2, 4, 2, 4, 0, 4.0),
        ]
        channels = {(0, 1): Queue(), (1, 0): Queue()}
        attn_comm = self._comm(AFDRole.ATTN, 0, 1, 1, channels)
        ffn_comm = self._comm(AFDRole.FFN, 0, 1, 1, channels)
        dispatcher = AFDDispatcher(
            comm=attn_comm,
            layers=layers,
            token_buckets=list(range(1, 17)),
            dtype=torch.float32,
            device=torch.device("cpu"),
            scattered=scattered,
        )
        executor = self._executor(ffn_comm, layers, scattered=scattered)
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
                step_id = dispatcher.begin_batch(mode, 3, 3)
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
        for scattered in (False, True):
            with self.subTest(scattered=scattered):
                dispatcher, executor, attn_comm, ffn_comm = self._pair(
                    scattered=scattered
                )
                with patch.object(executor._compute, "run") as run:
                    for mode in (ForwardMode.IDLE, ForwardMode.DECODE):
                        step_id = dispatcher.begin_batch(mode, 0, 0)
                        executor.run_step(AFDSyncMsg(step_id, [0], int(mode), 0))
                run.assert_not_called()
                self.assertEqual(attn_comm.transfers, [])
                self.assertEqual(ffn_comm.transfers, [])

    def test_invalid_control_and_layer_order_fail_before_transport(self):
        dispatcher, executor, attn_comm, ffn_comm = self._pair()
        invalid_messages = (
            AFDSyncMsg(0, [1, 1], int(ForwardMode.DECODE), 2),
            AFDSyncMsg(0, [2], int(ForwardMode.DECODE), 1),
            AFDSyncMsg(0, [1], int(ForwardMode.IDLE), 1),
        )
        for message in invalid_messages:
            with self.subTest(message=message), self.assertRaises(RuntimeError):
                executor.run_step(message)
        dispatcher.begin_batch(ForwardMode.DECODE, 1, 1)
        with self.assertRaisesRegex(RuntimeError, "layer order mismatch"):
            dispatcher.dispatch(
                1,
                torch.zeros(1, 2),
                torch.zeros(1, 2, dtype=torch.int32),
                torch.zeros(1, 2),
            )
        with self.assertRaisesRegex(RuntimeError, "previous batch"):
            dispatcher.begin_batch(ForwardMode.DECODE, 1, 1)
        self.assertEqual(attn_comm.transfers, [])
        self.assertEqual(ffn_comm.transfers, [])

    def _asymmetric_roundtrip(self, tokens, num_ffn, scattered, mode):
        num_attn = len(tokens)
        world_size = num_attn + num_ffn
        channels = {
            (src, dst): Queue()
            for src in range(world_size)
            for dst in range(world_size)
            if src != dst
        }
        barrier = Barrier(num_ffn) if scattered else None
        ffn_comms, executors = [], []
        for rank in range(num_ffn):
            comm = self._comm(AFDRole.FFN, rank, num_attn, num_ffn, channels)
            experts = _CPUExperts(
                runner_scale=1.0 if scattered else rank + 1, barrier=barrier
            )
            layer = AFDMoELayer(0, experts, 2, 4, 2, 4, 0, 1.0)
            ffn_comms.append(comm)
            executors.append(self._executor(comm, [layer], scattered=scattered))
        local_hidden = [
            torch.arange(count * 2, dtype=torch.float32).reshape(count, 2)
            + 10 * (rank + 1)
            for rank, count in enumerate(tokens)
        ]
        global_hidden = torch.cat(local_hidden)
        inputs = local_hidden if scattered else [global_hidden.clone() for _ in tokens]
        dispatchers, handles = [], []
        wire_capacity = max(tokens) if scattered else sum(tokens)
        msg = AFDSyncMsg(0, tokens, int(mode), wire_capacity)
        with (
            patch("sglang.srt.afd.ffn.executor.set_is_extend_in_batch"),
            ThreadPoolExecutor(max_workers=num_ffn) as pool,
        ):
            futures = [pool.submit(executor.run_step, msg) for executor in executors]
            for rank, hidden in enumerate(inputs):
                comm = self._comm(AFDRole.ATTN, rank, num_attn, num_ffn, channels)
                dispatcher = AFDDispatcher(
                    comm=comm,
                    layers=executors[0]._layers,
                    token_buckets=list(range(1, 17)),
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    scattered=scattered,
                )
                local_wire = wire_capacity if mode.is_decode() else len(hidden)
                dispatcher.begin_batch(mode, len(hidden), local_wire)
                handle = None
                if len(hidden) or (mode.is_decode() and scattered):
                    ids = torch.tensor([[0, 1]], dtype=torch.int32).expand(
                        len(hidden), 2
                    )
                    weights = torch.full((len(hidden), 2), 0.25)
                    handle = dispatcher.dispatch(0, hidden, ids, weights)
                dispatchers.append(dispatcher)
                handles.append(handle)
            outputs = [
                dispatcher.combine(handle) if handle is not None else hidden.clone()
                for dispatcher, handle, hidden in zip(
                    dispatchers, handles, inputs, strict=True
                )
            ]
            for future in futures:
                future.result(timeout=10)
        for actual, original in zip(
            inputs,
            local_hidden if scattered else [global_hidden] * num_attn,
            strict=True,
        ):
            torch.testing.assert_close(actual, original)
        self.assertTrue(all(channel.empty() for channel in channels.values()))
        return local_hidden, outputs, executors, ffn_comms

    def test_full_asymmetric_fanout_sums_partials_without_overwriting_inputs(self):
        for tokens, num_ffn, partial_scales in (
            ([2, 0, 1], 2, [1, 2, 0]),
            ([2, 1], 3, [4, 2]),
        ):
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(tokens=tokens, num_ffn=num_ffn, mode=mode):
                    inputs, outputs, executors, _ = self._asymmetric_roundtrip(
                        tokens, num_ffn, False, mode
                    )
                    gathered = torch.cat(inputs)
                    for output, scale in zip(outputs, partial_scales, strict=True):
                        torch.testing.assert_close(output, gathered * 0.75 * scale)
                    torch.testing.assert_close(
                        sum(outputs), gathered * 0.75 * sum(range(1, num_ffn + 1))
                    )
                    for executor in executors:
                        torch.testing.assert_close(
                            executor._layers[0].experts.inputs[0], gathered
                        )

    def test_scattered_fanin_splits_responses_and_keeps_empty_ffn_in_collectives(self):
        for tokens, num_ffn, source_ranks in (
            ([2, 0, 1], 2, [(0, 2), (1,)]),
            ([2, 1], 3, [(0,), (1,), ()]),
        ):
            for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
                with self.subTest(tokens=tokens, num_ffn=num_ffn, mode=mode):
                    inputs, outputs, executors, ffn_comms = self._asymmetric_roundtrip(
                        tokens, num_ffn, True, mode
                    )
                    for output, hidden in zip(outputs, inputs, strict=True):
                        torch.testing.assert_close(output, hidden * 0.75)
                    for executor, comm, sources in zip(
                        executors, ffn_comms, source_ranks, strict=True
                    ):
                        source_inputs = [inputs[rank] for rank in sources]
                        if mode.is_decode():
                            source_inputs = [
                                torch.nn.functional.pad(
                                    hidden, (0, 0, 0, max(tokens) - len(hidden))
                                )
                                for hidden in source_inputs
                            ]
                        expected = (
                            torch.cat(source_inputs) if sources else torch.empty(0, 2)
                        )
                        self.assertEqual(len(executor._layers[0].experts.inputs), 1)
                        torch.testing.assert_close(
                            executor._layers[0].experts.inputs[0], expected
                        )
                        if not len(expected):
                            self.assertEqual(comm.transfers, [])

    def test_low_latency_fanin_capacity_updates_every_native_dispatcher(self):
        attn = AFDHandshakeInfo(
            "attn", "model", 3, "nccl", "deepep", "float32", [], [32, 64, 96]
        )
        ffn = AFDHandshakeInfo("ffn", "model", 2, "nccl", "deepep", "float32", [])
        attn.decode_graph_token_buckets = [32, 64]
        ffn.decode_graph_token_buckets = [1]
        plan = build_afd_ffn_capacity_plan(ffn, attn, use_deepep_low_latency=True)
        self.assertEqual(plan.graph_token_buckets, (64, 128))
        self.assertEqual(plan.deepep_dispatch_limit, 192)
        inners = [
            SimpleNamespace(
                deepep_mode=DeepEPMode.AUTO,
                _normal_dispatcher=SimpleNamespace(num_max_dispatch_tokens_per_rank=1),
                _low_latency_dispatcher=SimpleNamespace(
                    num_max_dispatch_tokens_per_rank=1
                ),
            )
            for _ in range(2)
        ]
        experts = _CPUExperts()
        experts.dispatcher = SimpleNamespace(_inners=inners)
        layer = AFDMoELayer(0, experts, 2, 4, 2, 4, 0, 1.0)
        comm = self._comm(AFDRole.FFN, 0, 3, 2, {})
        compute = self._executor(
            comm, [layer], scattered=True, deepep_mode="auto"
        )._compute
        compute.configure_native_deepep_dispatch_limit(plan.deepep_dispatch_limit)
        for inner in inners:
            self.assertEqual(
                inner._normal_dispatcher.num_max_dispatch_tokens_per_rank, 192
            )
            self.assertEqual(
                inner._low_latency_dispatcher.num_max_dispatch_tokens_per_rank, 192
            )
        attn.decode_token_buckets = [32, 64, 512]
        self.assertEqual(
            build_afd_ffn_capacity_plan(ffn, attn, True).deepep_dispatch_limit, 1024
        )
        attn.decode_token_buckets = [32, 64, 513]
        with self.assertRaisesRegex(RuntimeError, "required=1026"):
            build_afd_ffn_capacity_plan(ffn, attn, True)
        self.assertIsNone(
            build_afd_ffn_capacity_plan(ffn, attn, False).deepep_dispatch_limit
        )
        attn.decode_graph_token_buckets = []
        self.assertEqual(
            build_afd_ffn_capacity_plan(ffn, attn, False).graph_token_buckets, ()
        )

    def test_staged_fixed_buffer_padding_reuse_and_prefill_exact_size(self):
        layer = AFDMoELayer(3, torch.nn.Identity(), 2, 4, 2, 4, 0, None)
        channels = {(0, 1): Queue(), (1, 0): Queue()}
        comm = self._comm(AFDRole.ATTN, 0, 1, 1, channels)
        dispatcher = AFDDispatcher(
            comm=comm,
            layers=[layer],
            token_buckets=[4, 8],
            dtype=torch.float32,
            device=torch.device("cpu"),
            scattered=True,
        )
        first_slot_ptr = None
        for index, (mode, size, capacity) in enumerate(
            (
                (ForwardMode.DECODE, 3, 4),
                (ForwardMode.DECODE, 5, 8),
                (ForwardMode.DECODE, 0, 4),
                (ForwardMode.EXTEND, 5, 5),
            ),
            start=1,
        ):
            with self.subTest(mode=mode, size=size):
                hidden = torch.arange(size * 2, dtype=torch.float32).reshape(size, 2)
                ids = torch.zeros(size, 2, dtype=torch.int32)
                weights = torch.ones(size, 2)
                dispatcher.begin_batch(mode, size, capacity)
                handle = dispatcher.dispatch(3, hidden, ids, weights)
                if mode.is_decode():
                    self.assertTrue(dispatcher.needs_empty_decode_dispatch)
                    if first_slot_ptr is None:
                        first_slot_ptr = handle.hidden_states.data_ptr()
                    self.assertEqual(handle.hidden_states.data_ptr(), first_slot_ptr)
                    self.assertTrue(torch.all(handle.hidden_states[size:] == 0))
                    self.assertTrue(torch.all(handle.topk_ids[size:] == -1))
                    self.assertTrue(torch.all(handle.topk_weights[size:] == 0))
                else:
                    self.assertIsNone(handle.slot_id)
                sent = channels[1, 0].get_nowait()
                self.assertEqual(tuple(sent[0].shape), (capacity, 2))
                channels[0, 1].put((torch.full((capacity, 2), float(index)),))
                output = dispatcher.combine(handle)
                self.assertEqual(tuple(output.shape), (size, 2))
                self.assertTrue(torch.all(output == index))

    def test_integrated_nccl_reuses_inputs_and_masks_compute_padding(self):
        channels = {(0, 1): Queue(), (1, 0): Queue()}
        attn_comm = self._comm(AFDRole.ATTN, 0, 1, 1, channels)
        ffn_comm = self._comm(AFDRole.FFN, 0, 1, 1, channels)
        layer = AFDMoELayer(0, _CPUExperts(), 2, 4, 2, 4, 0, 1.0)
        executor = self._executor(ffn_comm, [layer], scattered=True)
        executor._compute._allocate_graph_buffers(4)
        dispatcher = AFDDispatcher(
            comm=attn_comm,
            layers=[layer],
            token_buckets=[2, 4],
            dtype=torch.float32,
            device=torch.device("cpu"),
            scattered=True,
        )
        hidden = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 1
        ids = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
        weights = torch.full((2, 2), 0.25)
        with (
            patch.object(
                executor._compute,
                "run_integrated_layer",
                wraps=executor._compute.run_integrated_layer,
            ) as run,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            result = pool.submit(executor._run_integrated_graph_step, (2,), (4,))
            output = dispatcher.decode_graph_wave(0, hidden, ids, weights)
            result.result(timeout=10)
        torch.testing.assert_close(
            output, hidden * ((ids + 1) * weights).sum(1, keepdim=True)
        )
        self.assertEqual(
            attn_comm.sent_tensor_ptrs[0],
            tuple(t.data_ptr() for t in (hidden, ids, weights)),
        )
        self.assertFalse(run.call_args.kwargs["stage_inputs"])
        self.assertEqual(tuple(layer.experts.inputs[0].shape), (4, 2))
        self.assertTrue(torch.all(layer.experts.inputs[0][2:] == 0))
        self.assertTrue(torch.all(layer.experts.routing_ids[0][2:] == -1))
        self.assertEqual(ffn_comm.sent_tensor_ptrs[0], (layer.experts.output_ptrs[0],))
        self.assertTrue(all(channel.empty() for channel in channels.values()))


class TestAFDSchedulerRouting(unittest.TestCase):
    def test_scattered_control_preserves_cross_dp_counts_and_local_empty_mode(self):
        from sglang.srt.managers.scheduler_components.dp_attn import (
            _update_gather_batch,
        )

        for tokens, gather, expected in (
            ([2, 5, 1], False, [2, 5, 1]),
            ([0, 5, 1], False, [0, 5, 1]),
            ([0, 3], True, [3, 3]),
        ):
            with self.subTest(tokens=tokens, gather=gather):
                scheduler = SimpleNamespace(
                    afd_handshake=SimpleNamespace(
                        local=SimpleNamespace(
                            num_ranks=len(tokens), decode_token_buckets=[4, 8]
                        )
                    ),
                    ps=SimpleNamespace(tp_rank=0),
                    _afd_sync_sender=Mock(),
                    tp_worker=SimpleNamespace(model_runner=SimpleNamespace()),
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.DECODE if tokens[0] else ForwardMode.IDLE,
                )
                sync_info = SimpleNamespace(
                    num_tokens=tokens[0],
                    num_tokens_for_logprob=tokens[0],
                    global_num_tokens=tokens,
                    global_num_tokens_for_logprob=tokens,
                    global_forward_mode=int(ForwardMode.DECODE),
                    is_extend_in_batch=False,
                    tbo_split_seq_index=None,
                    can_run_decode_cuda_graph=False,
                    can_run_prefill_cuda_graph=False,
                    prefill_cuda_graph_max_prefix_len=None,
                )
                _update_gather_batch(batch, sync_info, require_mlp_tp_gather=gather)
                dispatcher = Mock()
                dispatcher.begin_batch.return_value = 7
                with (
                    patch(
                        "sglang.srt.afd.afd_scheduler.is_afd_attn_moe_scattered",
                        return_value=True,
                    ),
                    patch(
                        "sglang.srt.afd.afd_scheduler.get_afd_dispatcher",
                        return_value=dispatcher,
                    ),
                    patch(
                        "sglang.srt.layers.dp_attention.get_attention_dp_size",
                        return_value=len(tokens),
                    ),
                    patch(
                        "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                        return_value=MoeA2ABackend.AFD_ATTN,
                    ),
                ):
                    prepare_afd_attn_batch(scheduler, batch)
                dispatcher.begin_batch.assert_called_once_with(
                    forward_mode=ForwardMode.DECODE,
                    dp_capacity=expected[0],
                    wire_capacity=8 if max(expected) > 4 else 4,
                )
                self.assertEqual(
                    scheduler.tp_worker.model_runner.afd_cuda_graph_batch_size,
                    max(expected),
                )
                scheduler._afd_sync_sender.prepare_for_batch.assert_called_once_with(
                    step_id=7,
                    tokens_per_rank=expected,
                    forward_mode=ForwardMode.DECODE,
                    wire_capacity=8 if max(expected) > 4 else 4,
                )

    def test_control_publishes_final_graph_decision_and_eager_fallback(self):
        layer = AFDMoELayer(0, torch.nn.Identity(), 2, 4, 2, 4, 0, None)
        channels = {(0, 1): Queue(), (1, 0): Queue()}
        comm = _CPUComm(
            AFDCommConfig(AFDRole.ATTN, 1, 1, 0, "localhost", 1, "cpu"), channels
        )
        dispatcher = AFDDispatcher(
            comm=comm,
            layers=[layer],
            token_buckets=[4, 8],
            dtype=torch.float32,
            device=torch.device("cpu"),
            scattered=False,
        )
        dispatcher.enable_communication_graph_variants(
            (AFDGraphCaptureVariant(4, 4, (4,), (4,)),)
        )
        with patch("sglang.srt.afd.afd_scheduler.AFDSyncMsgSender") as transport:
            sender = AFDSyncSender(SimpleNamespace(dist_host="localhost", sync_port=1))
        for capacity, can_run_graph, expected in (
            (4, True, True),
            (4, False, False),
            (8, True, False),
        ):
            with self.subTest(capacity=capacity, can_run_graph=can_run_graph):
                step_id = dispatcher.begin_batch(ForwardMode.DECODE, capacity, capacity)
                sender.prepare_for_batch(
                    step_id, [capacity], ForwardMode.DECODE, capacity
                )
                self.assertEqual(transport.return_value.send.call_count, step_id)
                decision = _decide_afd_attn_graph(can_run_graph, dispatcher)
                self.assertEqual(decision.use_local_model_graph, expected)
                sender.send_prepared(decision.use_paired_communication_graph)
                self.assertEqual(
                    transport.return_value.send.call_args.args[0].communication_graph,
                    expected,
                )
                with self.assertRaisesRegex(RuntimeError, "no prepared"):
                    sender.send_prepared(expected)
                if not expected:
                    work = dispatcher.dispatch(
                        0,
                        torch.ones(capacity, 2),
                        torch.zeros(capacity, 2, dtype=torch.int32),
                        torch.ones(capacity, 2),
                    )
                    channels[1, 0].get_nowait()
                    channels[0, 1].put((torch.ones(capacity, 2),))
                    dispatcher.combine(work)

    def test_breakable_logits_output_buffer_roundtrip(self):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            BreakableCudaGraphBackend,
        )

        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        output = LogitsProcessorOutput(
            next_token_logits=torch.arange(20, dtype=torch.float32).reshape(4, 5),
            hidden_states=torch.arange(8, dtype=torch.float32).reshape(4, 2),
            full_logits=None,
            customized_info={"marker": 7},
        )
        buffer = backend._alloc_full_buffer(output, 4)
        backend._copy_output_to_buffer(output, buffer, backend._output_rows(output, 4))
        sliced = backend._slice_output(buffer, 3)
        torch.testing.assert_close(
            sliced.next_token_logits, output.next_token_logits[:3]
        )
        torch.testing.assert_close(sliced.hidden_states, output.hidden_states[:3])
        self.assertIsNone(sliced.full_logits)
        self.assertEqual(sliced.customized_info, {"marker": 7})

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
