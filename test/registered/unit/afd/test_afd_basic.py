"""CPU contract tests for AFD configuration and eager execution."""

import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from queue import Queue
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.afd.afd_config import AFDRole, check_afd_server_args
from sglang.srt.afd.afd_handshake import AFDHandshakeInfo, build_afd_ffn_capacity_plan
from sglang.srt.afd.afd_scheduler import prepare_afd_attn_batch
from sglang.srt.afd.afd_sync import AFDSyncMsg
from sglang.srt.afd.comm.base import AFDCommConfig
from sglang.srt.afd.dispatcher import AFDDispatcher
from sglang.srt.afd.ffn.compute import AFDFFNCompute
from sglang.srt.afd.ffn.executor import AFDFFNExecutor
from sglang.srt.afd.layers import AFDMoELayer
from sglang.srt.arg_groups.attention_hook import handle_deterministic_inference
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_context,
    get_flags,
    get_forward,
    get_parallel,
    get_resources,
    override_platform,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

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


class TestAFDServerArgsCheck(CustomTestCase):
    def test_deterministic_afd_tp1_uses_the_same_nccl_settings_as_tp2(self):
        """Asymmetric AFD peers must agree on communicator channel settings."""
        nccl_env = {
            "NCCL_ALGO": "allreduce:tree",
            "NCCL_MIN_NCHANNELS": "8",
            "NCCL_MAX_NCHANNELS": "8",
        }
        for role, tp_size, requires_nccl in (
            (None, 1, False),
            (None, 2, True),
            ("attn", 1, True),
            ("ffn", 1, True),
        ):
            with (
                self.subTest(role=role, tp_size=tp_size),
                patch.dict(os.environ, {"SGLANG_DETERMINISTIC_NCCL_NCHANNELS": "8"}),
                override_platform(is_hip=False),
            ):
                for name in nccl_env:
                    os.environ.pop(name, None)
                args = _make_args(
                    afd_role=role,
                    tp_size=tp_size,
                    enable_deterministic_inference=True,
                    attention_backend="triton",
                    disable_custom_all_reduce=False,
                    enable_torch_symm_mem=True,
                )
                args._model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
                )
                handle_deterministic_inference(args)
                resolved = resolving_view(args)
                self.assertEqual(resolved.disable_custom_all_reduce, requires_nccl)
                self.assertEqual(resolved.enable_torch_symm_mem, not requires_nccl)
                for name, expected in nccl_env.items():
                    if requires_nccl:
                        self.assertEqual(os.environ[name], expected)
                    else:
                        self.assertNotIn(name, os.environ)

    def test_ffn_rejects_low_latency_only_deepep_prefill(self):
        """The decode admission bound does not bound prefill token counts."""
        for backend, mode in product(
            ("none", "deepep"), ("auto", "normal", "low_latency")
        ):
            with self.subTest(backend=backend, mode=mode):
                args = _make_args(
                    afd_role="ffn",
                    afd_ffn_addr=None,
                    moe_a2a_backend=backend,
                    moe_runner_backend="deep_gemm",
                    deepep_mode=mode,
                )
                if backend == "deepep" and mode == "low_latency":
                    with self.assertRaisesRegex(ValueError, "prefill capacity"):
                        check_afd_server_args(args)
                else:
                    check_afd_server_args(args)
                    self.assertEqual(resolving_view(args).deepep_mode, mode)

    def test_attention_role_resolves_eager_execution_and_routing(self):
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
                for phase in Phase.ALL:
                    self.assertEqual(
                        getattr(resolved.cuda_graph_config, phase).backend,
                        Backend.DISABLED,
                    )
                self.assertEqual(args.moe_a2a_backend, backend)
                self.assertIs(args.cuda_graph_config, original_graph_config)

    def test_ffn_role_disables_http_warmup_and_resolves_standard_topk_runner(self):
        for backend in ("none", "deepep"):
            with (
                self.subTest(backend=backend),
                patch(
                    "sglang.srt.afd.afd_config.model_config_of",
                    return_value=SimpleNamespace(quantization=None, hf_quant_config={}),
                ),
            ):
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
                self.assertEqual(resolved.moe_runner_backend, "flashinfer_trtllm")

    def test_full_ffn_runner_preserves_materialized_routing(self):
        cases = (
            (None, None, {}, "flashinfer_trtllm"),
            ("fp8", "fp8", {"weight_block_size": [128, 128]}, "flashinfer_trtllm"),
            (None, "fp8", {}, "triton"),
            ("fp8", "fp8", {}, "triton"),
            ("modelopt_fp8", "modelopt_fp8", {}, "triton"),
            ("mxfp8", "mxfp8", {}, "flashinfer_trtllm"),
            ("modelopt_fp4", "modelopt_fp4", {}, "flashinfer_trtllm"),
            ("nvfp4_online", "nvfp4_online", {}, "flashinfer_trtllm"),
        )
        for runner, (quant, model_quant, metadata, expected) in product(
            ("flashinfer_trtllm", "flashinfer_trtllm_routed"), cases
        ):
            with (
                self.subTest(runner=runner, quantization=quant, metadata=metadata),
                patch(
                    "sglang.srt.afd.afd_config.model_config_of",
                    return_value=SimpleNamespace(
                        quantization=model_quant, hf_quant_config=metadata
                    ),
                ),
            ):
                args = _make_args(
                    afd_role="ffn",
                    afd_ffn_addr=None,
                    moe_runner_backend=runner,
                    quantization=quant,
                )
                check_afd_server_args(args)
                self.assertEqual(resolving_view(args).moe_runner_backend, expected)

        for quant in ("mxfp8", "modelopt_fp4", "nvfp4_online", "modelopt_mixed"):
            with self.subTest(incompatible_quantization=quant):
                args = _make_args(
                    afd_role="ffn",
                    afd_ffn_addr=None,
                    moe_runner_backend="experimental_sgl_trtllm",
                    quantization=quant,
                )
                with self.assertRaisesRegex(ValueError, "cannot fall back to triton"):
                    check_afd_server_args(args)

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
            (
                {"disaggregation_mode": "decode", "enable_pd_role_switch": True},
                "enable_pd_role_switch",
            ),
            ({"decoupled_spec_role": "drafter"}, "decoupled_spec_role"),
            ({"decoupled_spec_role": "verifier"}, "decoupled_spec_role"),
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
        self.extend_modes = []

    def forward(self, hidden_states, topk):
        self.inputs.append(hidden_states.clone())
        self.extend_modes.append(get_forward().is_extend_in_batch)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        weights = ((topk.topk_ids + 1) * topk.topk_weights).sum(dim=1, keepdim=True)
        return hidden_states * weights * self.runner_scale


class _CPUComm:
    def __init__(self, config, channels):
        self.config = config
        self.channels = channels
        self.transfers = []

    def send_to(self, tensors, peer_global_rank):
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


class TestAFDEagerRoundtrip(CustomTestCase):
    def _executor(self, comm, layers, *, scattered=False, deepep_mode="normal"):
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(dtype=torch.float32, model=None)
            ),
            tp_group=SimpleNamespace(
                world_size=comm.config.num_ffn_ranks,
                rank_in_group=comm.config.instance_rank,
            ),
        )
        with (
            get_context().override_server_args(
                afd_ffn_moe_backend="deepep" if scattered else "none",
                deepep_mode=deepep_mode,
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
            return AFDFFNExecutor(scheduler, comm, compute)

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
            attn_comm, layers, torch.float32, torch.device("cpu"), scattered=scattered
        )
        executor = self._executor(ffn_comm, layers, scattered=scattered)
        return dispatcher, executor, attn_comm, ffn_comm

    def test_prefill_decode_roundtrip_applies_each_scaling_factor_once(self):
        dispatcher, executor, attn_comm, ffn_comm = self._pair()
        ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
        weights = torch.full((3, 2), 0.25)
        with (
            get_forward().scoped(is_extend_in_batch=False),
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
                self.assertFalse(get_forward().is_extend_in_batch)
            for layer in executor._layers:
                self.assertEqual(layer.experts.extend_modes, [True, False])
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
            AFDSyncMsg(0, [2], int(ForwardMode.DECODE), 3),
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
            get_forward().scoped(is_extend_in_batch=False),
            ThreadPoolExecutor(max_workers=num_ffn) as pool,
        ):
            futures = [pool.submit(executor.run_step, msg) for executor in executors]
            for rank, hidden in enumerate(inputs):
                comm = self._comm(AFDRole.ATTN, rank, num_attn, num_ffn, channels)
                dispatcher = AFDDispatcher(
                    comm,
                    executors[0]._layers,
                    torch.float32,
                    torch.device("cpu"),
                    scattered,
                )
                dispatcher.begin_batch(mode, len(hidden), len(hidden))
                handle = None
                if len(hidden):
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
                        expected = (
                            torch.cat([inputs[rank] for rank in sources])
                            if sources
                            else torch.empty(0, 2)
                        )
                        self.assertEqual(len(executor._layers[0].experts.inputs), 1)
                        torch.testing.assert_close(
                            executor._layers[0].experts.inputs[0], expected
                        )
                        if not len(expected):
                            self.assertEqual(comm.transfers, [])

    def test_low_latency_fanin_capacity_updates_every_native_dispatcher(self):
        attn = AFDHandshakeInfo(
            "attn", "model", 3, "nccl", "deepep", "float32", [], [96]
        )
        ffn = AFDHandshakeInfo("ffn", "model", 2, "nccl", "deepep", "float32", [])
        plan = build_afd_ffn_capacity_plan(ffn, attn, use_deepep_low_latency=True)
        self.assertEqual(plan.deepep_dispatch_limit, 192)
        small_attn = AFDHandshakeInfo(
            "attn", "model", 1, "nccl", "deepep", "float32", [], [16]
        )
        small_plan = build_afd_ffn_capacity_plan(
            ffn, small_attn, use_deepep_low_latency=True
        )
        self.assertEqual(
            small_plan.deepep_dispatch_limit,
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.default,
        )
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
        attn.decode_token_buckets = [512]
        self.assertEqual(
            build_afd_ffn_capacity_plan(ffn, attn, True).deepep_dispatch_limit, 1024
        )
        attn.decode_token_buckets = [513]
        with self.assertRaisesRegex(RuntimeError, "required=1026"):
            build_afd_ffn_capacity_plan(ffn, attn, True)
        self.assertIsNone(
            build_afd_ffn_capacity_plan(ffn, attn, False).deepep_dispatch_limit
        )


class TestAFDSchedulerRouting(CustomTestCase):
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
                messages = []
                scheduler = SimpleNamespace(
                    afd_handshake=SimpleNamespace(
                        local=SimpleNamespace(num_ranks=len(tokens))
                    ),
                    _afd_sync_sender=SimpleNamespace(send=messages.append),
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
                    can_run_draft_cuda_graph=False,
                    can_run_prefill_cuda_graph=False,
                    prefill_cuda_graph_max_prefix_len=None,
                )
                _update_gather_batch(batch, sync_info, require_mlp_tp_gather=gather)
                dispatcher = AFDDispatcher(
                    _CPUComm(
                        AFDCommConfig(
                            AFDRole.ATTN, len(tokens), 1, 0, "localhost", 30001, "cpu"
                        ),
                        {},
                    ),
                    [],
                    torch.float32,
                    torch.device("cpu"),
                    scattered=True,
                )
                with (
                    get_context().override_server_args(
                        afd_role="attn",
                        afd_ffn_moe_backend="deepep",
                        tp_size=len(tokens),
                        attn_dp_size=len(tokens),
                    ),
                    get_parallel().override(tp_rank=0),
                    get_flags().moe.override(a2a_backend=MoeA2ABackend.AFD_ATTN),
                    get_resources().override(afd_dispatcher=dispatcher),
                ):
                    prepare_afd_attn_batch(scheduler, batch)
                self.assertEqual(
                    messages,
                    [AFDSyncMsg(0, expected, int(ForwardMode.DECODE), max(expected))],
                )

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
        batch = SimpleNamespace(forward_mode=ForwardMode.PREBUILT, extend_num_tokens=0)

        actual = Scheduler.run_batch.__wrapped__(scheduler, batch)

        self.assertIs(actual, result)
        scheduler._run_batch_prebuilt.assert_called_once_with(batch)
        scheduler._afd_pre_run_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
