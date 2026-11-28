# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract checks: python test/manual/ascend/test_kt_npu_offload.py.

Load the standalone helper directly so these checks do not require SGLang's
serving dependencies, CANN, or torch_npu. They do not validate NPU execution.
"""

import importlib.util
import os
import sys
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

HELPER = Path(__file__).resolve().parents[3] / "python/sglang/srt/layers/moe/kt_npu.py"
spec = importlib.util.spec_from_file_location("kt_npu_under_test", HELPER)
kt_npu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kt_npu)


class TestKTNPUOffload(unittest.TestCase):
    def test_configuration_matrix(self):
        for switch in ("", "0", "1"):
            with patch.dict(os.environ, {"KT_NPU_EXPERIMENTAL_TP2": switch}):
                for tp in (1, 2, 4):
                    for ep in (1, 2):
                        for experts in (None, 0, 32):
                            for backend in ("none", "deepep"):
                                allowed = (
                                    ep == 1
                                    and (tp == 1 or (tp == 2 and switch == "1"))
                                    and experts == 32
                                    and backend == "none"
                                )
                                with self.subTest(
                                    switch=switch,
                                    tp=tp,
                                    ep=ep,
                                    experts=experts,
                                    backend=backend,
                                ):
                                    if allowed:
                                        kt_npu.validate_kt_npu_config(
                                            tp, ep, experts, backend
                                        )
                                        self.assertEqual(
                                            os.environ[
                                                "KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"
                                            ],
                                            "1",
                                        )
                                    else:
                                        with self.assertRaises(ValueError):
                                            kt_npu.validate_kt_npu_config(
                                                tp, ep, experts, backend
                                            )

    def test_routing_preserves_inputs_and_resident_contribution(self):
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor([[0, 31, 32, 255]], dtype=dtype)
            weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
            ids_before, weights_before = ids.clone(), weights.clone()
            safe_ids, safe_weights = kt_npu.mask_cpu_expert_routing_npu(
                ids, weights, 32
            )
            torch.testing.assert_close(
                safe_ids, torch.tensor([[0, 31, 0, 0]], dtype=dtype)
            )
            torch.testing.assert_close(
                safe_weights, torch.tensor([[0.1, 0.2, 0.0, 0.0]])
            )
            torch.testing.assert_close(ids, ids_before)
            torch.testing.assert_close(weights, weights_before)

    def test_all_cpu_and_empty_routes(self):
        ids = torch.tensor([[32, 255]])
        safe_ids, weights = kt_npu.mask_cpu_expert_routing_npu(
            ids, torch.ones(1, 2), 32
        )
        self.assertEqual(safe_ids.sum().item(), 0)
        self.assertEqual(weights.sum().item(), 0)
        ids = torch.empty((0, 6), dtype=torch.int64)
        result = kt_npu.mask_cpu_expert_routing_npu(ids, torch.empty((0, 6)), 32)
        self.assertEqual(result[0].shape, (0, 6))

    def test_per_layer_paths(self):
        for template in ("/weights/layer{}.gguf", "/weights/layer{layer_idx}.gguf"):
            self.assertEqual(
                kt_npu.resolve_kt_weight_path(template, 7), "/weights/layer7.gguf"
            )
        self.assertEqual(
            kt_npu.resolve_kt_weight_path("/weights/all", 7), "/weights/all"
        )

    def test_graph_callback_copies_then_runs_cpu(self):
        events = []
        stream = SimpleNamespace(npu_stream=42)
        wrapper = Mock()
        wrapper.copy_inputs_to_cpu_buffers.side_effect = lambda *args: events.append(
            "copy"
        )
        wrapper.run_pinned_forward_sync.side_effect = lambda *args: events.append("cpu")

        def launch(actual_stream, callback, args):
            self.assertIs(actual_stream, stream)
            events.append("launch")
            callback(args)

        fake_npu = SimpleNamespace(npu=SimpleNamespace(_launch_host_func=launch))
        x = SimpleNamespace(device="npu:0")
        with (
            patch.dict(sys.modules, {"torch_npu": fake_npu}),
            patch.object(
                torch,
                "npu",
                SimpleNamespace(current_stream=lambda device: stream),
                create=True,
            ),
        ):
            kt_npu.submit_npu_graph(wrapper, x, "ids", "weights")
        self.assertEqual(events, ["copy", "launch", "cpu"])
        wrapper.run_pinned_forward_sync.assert_called_once_with(x, 42)

    def test_subscription_only_suppresses_already_subscribed(self):
        subscribe = Mock(side_effect=RuntimeError("ACL 107011"))
        fake_npu = SimpleNamespace(npu=SimpleNamespace(_subscribe_report=subscribe))
        with patch.dict(sys.modules, {"torch_npu": fake_npu}):
            kt_npu.subscribe_npu_report("stream")
            subscribe.side_effect = RuntimeError("ACL 107030")
            with self.assertRaisesRegex(RuntimeError, "107030"):
                kt_npu.subscribe_npu_report("stream")

    def test_wrapper_cpu_owner_and_merge(self):
        """Execute the real wrapper with mocked serving/native dependencies."""
        topk_type = namedtuple("TopK", "topk_weights topk_ids topk_logits")
        dispatch_type = namedtuple("Dispatch", "hidden_states topk_output")
        combine_type = namedtuple("Combine", "hidden_states")
        parallel = SimpleNamespace(tp_rank=0, tp_size=2)
        stubs = {
            "sglang.srt.arg_groups.model_override_base": SimpleNamespace(
                model_config_of=Mock()
            ),
            "sglang.srt.layers.moe.kt_npu": kt_npu,
            "sglang.srt.layers.quantization.base_config": SimpleNamespace(
                FusedMoEMethodBase=object
            ),
            "sglang.srt.runtime_context": SimpleNamespace(
                get_device=lambda: SimpleNamespace(device="npu"),
                get_parallel=lambda: parallel,
                get_exec=Mock(),
                get_schedule=Mock(),
            ),
            "sglang.srt.utils": SimpleNamespace(get_compiler_backend=lambda: "eager"),
            "sglang.srt.layers.moe.token_dispatcher": SimpleNamespace(
                StandardCombineInput=combine_type
            ),
            "sglang.srt.layers.moe.topk": SimpleNamespace(
                TopKOutputChecker=SimpleNamespace(format_is_bypassed=lambda topk: False)
            ),
            "kt_kernel": SimpleNamespace(KTMoEWrapper=Mock()),
        }
        wrapper_spec = importlib.util.spec_from_file_location(
            "kt_wrapper_under_test", HELPER.with_name("kt_ep_wrapper.py")
        )
        module = importlib.util.module_from_spec(wrapper_spec)
        stubs[wrapper_spec.name] = module
        with (
            patch.dict(sys.modules, stubs),
            patch.object(torch, "compile", lambda **kwargs: lambda fn: fn),
        ):
            wrapper_spec.loader.exec_module(module)
            for graph in (False, True):
                for rank in (0, 1):
                    parallel.tp_rank = rank
                    config = SimpleNamespace(num_gpu_experts=1)
                    gpu = Mock()
                    gpu.apply.return_value = combine_type(torch.ones(1, 2))
                    method = module.KTEPWrapperMethod(gpu, config)
                    method.wrapper = Mock() if rank == 0 else None
                    method.moe_runner_config = SimpleNamespace(activation="silu")
                    if rank == 0:
                        method.wrapper.sync_forward.return_value = torch.full(
                            (1, 2), 2.0
                        )
                        method.wrapper.copy_forward_output_to_device.return_value = (
                            torch.full((1, 2), 2.0)
                        )
                    x = SimpleNamespace(device=SimpleNamespace(type="npu"))
                    ids = torch.tensor([[0, 1]])
                    weights = torch.tensor([[0.25, 0.75]])
                    dispatch = dispatch_type(x, topk_type(weights, ids, None))
                    with (
                        patch.object(module, "npu_graph_capturing", return_value=graph),
                        patch.object(module, "_kt_stream_handle", return_value=42),
                        patch.object(module, "submit_npu_graph") as graph_submit,
                    ):
                        result = method.apply(None, dispatch)
                    torch.testing.assert_close(
                        result.hidden_states,
                        torch.full((1, 2), 3.0 if rank == 0 else 1.0),
                    )
                    torch.testing.assert_close(ids, torch.tensor([[0, 1]]))
                    masked = gpu.apply.call_args.args[1].topk_output
                    torch.testing.assert_close(
                        masked.topk_weights, torch.tensor([[0.25, 0.0]])
                    )
                    if rank == 0:
                        if graph:
                            graph_submit.assert_called_once_with(
                                method.wrapper, x, ids, weights
                            )
                            method.wrapper.submit_forward.assert_not_called()
                            method.wrapper.copy_forward_output_to_device.assert_called_once_with(
                                x
                            )
                        else:
                            method.wrapper.submit_forward.assert_called_once_with(
                                x, ids, weights, 42
                            )
                            method.wrapper.sync_forward.assert_called_once_with(x, 42)
                    else:
                        graph_submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
