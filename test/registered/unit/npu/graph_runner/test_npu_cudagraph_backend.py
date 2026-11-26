"""CPU unit tests for the NPU CUDA-graph backend's Python orchestration."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.npu.graph_runner import npu_cudagraph_backend as mod
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=10, suite="extra-a-test-npu")


def _device_module_stub():
    return SimpleNamespace(
        current_device=mock.Mock(return_value=3),
        synchronize=mock.Mock(),
        set_device=mock.Mock(),
        Event=mock.Mock(),
    )


def _make_backend(runner=None):
    # The backend holds its runner from construction (no bind hook): pass
    # a white-box runner to exercise a specific replay() branch.
    if runner is None:
        runner = SimpleNamespace(
            device_module=_device_module_stub(), enable_torch_compile=False
        )
    with (
        get_parallel().override(
            tp_group=SimpleNamespace(world_size=1, barrier=mock.Mock())
        ),
        mock.patch.object(mod.TorchMemorySaverAdapter, "create", return_value=None),
    ):
        return mod.NPUCudaGraphBackend(runner), runner


def _make_eagle_draft_runner(
    architectures, speculative_num_steps=3, raw_bs=2, bs=4, index_topk=None
):
    # White-box instance: __new__ bypasses the heavy __init__; replay() and
    # _build_eagle_draft_update_payload read only the fields set below.
    # index_topk: is_deepseek_dsa requires arch match AND index_topk present
    # in the HF config, so a DSA mock must carry both.
    runner = EAGLEDraftCudaGraphRunner.__new__(EAGLEDraftCudaGraphRunner)
    runner.device_module = _device_module_stub()
    runner.speculative_num_steps = speculative_num_steps
    runner.raw_bs = raw_bs
    runner.bs = bs
    runner.model_runner = SimpleNamespace(
        kv_cache_configurator=SimpleNamespace(modelslim_cache_layout=None),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=architectures, index_topk=index_topk
            )
        ),
    )
    return runner


def _make_decode_runner(raw_bs=2, bs=2):
    # get_update_attr_name() dispatches on isinstance, so the decode branch
    # needs a real DecodeCudaGraphRunner (white-box, same __new__ trick).
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.device_module = _device_module_stub()
    runner.raw_bs = raw_bs
    runner.bs = bs
    runner.captured_req_width = 1
    runner.model_runner = SimpleNamespace(
        kv_cache_configurator=SimpleNamespace(modelslim_cache_layout=None),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
        ),
        spec_algorithm=SimpleNamespace(
            is_dspark=mock.Mock(return_value=False),
            is_dflash=mock.Mock(return_value=False),
        ),
    )
    runner.capture_forward_mode = SimpleNamespace(
        is_target_verify=mock.Mock(return_value=False)
    )
    return runner


class TestNPUCudaGraphBackend(unittest.TestCase):
    def test_replay_update_converts_legacy_sequence_lengths_to_int32_tensor(self):
        backend, runner = _make_backend()
        graph = mock.Mock()
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay_with_input_update(
            shape_key,
            seq_lens=[7, 9],
            attr_name="seq_lens",
            attr_type=torch.empty(0),
        )

        runner.device_module.set_device.assert_called_once_with(3)
        graph.replay.assert_called_once_with()
        update_input = graph.update.call_args.kwargs["cpu_update_input"]
        self.assertEqual(len(update_input), 1)
        self.assertEqual(set(update_input[0]), {"seq_lens"})
        self.assertEqual(update_input[0]["seq_lens"].dtype, torch.int32)
        self.assertTrue(torch.equal(update_input[0]["seq_lens"], torch.tensor([7, 9])))
        self.assertEqual(output, "output")

    def test_replay_eagle_draft_runner_builds_per_step_update_lists(self):
        # Derived property: one {attr: lens} dict per in-graph draft forward.
        # The kth forward reads KV lengths seq_len + k + 1 (its query's path
        # depth in the draft tree); padding rows carry 0 so attention skips
        # them. num_steps=3 means 2 in-graph forwards (the last loop round
        # only selects tokens), hence num_steps - 1 dicts.
        backend, _ = _make_backend(
            _make_eagle_draft_runner(
                architectures=["Qwen3MoeForCausalLM"],
                speculative_num_steps=3,
                raw_bs=2,
                bs=4,
            )
        )
        graph = mock.Mock()
        shape_key = ShapeKey(size=4)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay(
            shape_key, SimpleNamespace(seq_lens_cpu=torch.tensor([5, 3]))
        )

        self.assertEqual(output, "output")
        update_input = graph.update.call_args.kwargs["cpu_update_input"]
        self.assertEqual(
            update_input,
            [
                {"actual_seq_lengths_kv": [6, 4, 0, 0]},
                {"actual_seq_lengths_kv": [7, 5, 0, 0]},
            ],
        )
        graph.replay.assert_called_once_with()

    def test_replay_deepseek_dsa_bypasses_input_update(self):
        # DSA/DSV4 run a sync-free path: replay() must not rebind host
        # inputs (graph.update) and go straight to graph.replay().
        backend, _ = _make_backend(
            _make_eagle_draft_runner(
                architectures=["DeepseekV3ForCausalLM"], index_topk=8
            )
        )
        graph = mock.Mock()
        shape_key = ShapeKey(size=4)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay(shape_key, SimpleNamespace())

        self.assertEqual(output, "output")
        graph.update.assert_not_called()
        graph.replay.assert_called_once_with()

    def test_replay_decode_runner_uses_single_legacy_payload(self):
        # The merged call site must build the same payload the legacy
        # seq_lens/attr_name/attr_type=[] convention produced: one dict with
        # attr actual_seq_lengths_kv and the live seq_lens list.
        backend, _ = _make_backend(_make_decode_runner(raw_bs=2, bs=2))
        graph = mock.Mock()
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = graph
        backend._outputs[shape_key] = "output"

        output = backend.replay(
            shape_key,
            SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_target_verify=mock.Mock(return_value=False)
                ),
                seq_lens_cpu=torch.tensor([5, 3]),
            ),
        )

        self.assertEqual(output, "output")
        update_input = graph.update.call_args.kwargs["cpu_update_input"]
        self.assertEqual(update_input, [{"actual_seq_lengths_kv": [5, 3]}])
        graph.replay.assert_called_once_with()

    def test_pingpong_disabled_by_default_and_async_requires_pingpong(self):
        with (
            mock.patch.object(
                mod.envs.SGLANG_NPU_GRAPH_PINGPONG, "get", return_value=False
            ),
            mock.patch.object(
                mod.envs.SGLANG_NPU_GRAPH_ASYNC_UPDATE, "get", return_value=True
            ),
        ):
            backend, _ = _make_backend()
        self.assertFalse(backend._pingpong_enabled)
        self.assertFalse(backend._async_update_enabled)
        self.assertEqual(backend._num_slots, 1)
        backend.cleanup()

    def test_pingpong_captures_two_slots_sharing_same_pool(self):
        import sys

        captured_pools = []
        created_graphs = []

        class _FakeGraph:
            pass

        def _fake_graph_ctx(
            graph, *, pool=None, stream=None, auto_dispatch_capture=False
        ):
            captured_pools.append(pool)
            return mod.empty_context()

        def _make_fake_graph():
            g = _FakeGraph()
            created_graphs.append(g)
            return g

        fake_npu = SimpleNamespace(NPUGraph=_make_fake_graph, graph=_fake_graph_ctx)
        with (
            mock.patch.object(
                mod.envs.SGLANG_NPU_GRAPH_PINGPONG, "get", return_value=True
            ),
            mock.patch.object(
                mod.envs.SGLANG_NPU_GRAPH_ASYNC_UPDATE, "get", return_value=False
            ),
        ):
            backend, _ = _make_backend()

        backend._pool = ("shared-pool", 0)
        shape_key = ShapeKey(size=4)
        call_idx = 0

        def forward_fn():
            nonlocal call_idx
            call_idx += 1
            return f"out-{call_idx}"

        post_warmup = mock.Mock()
        with (
            mock.patch.dict(sys.modules, {"torch_npu": SimpleNamespace()}),
            mock.patch.object(torch, "npu", fake_npu, create=True),
        ):
            backend._capture_one(shape_key, forward_fn, post_warmup_hook=post_warmup)

        self.assertEqual(len(created_graphs), 2)
        self.assertEqual(captured_pools, [("shared-pool", 0), ("shared-pool", 0)])
        self.assertEqual(backend._graphs[shape_key], created_graphs)
        # 2 warmup calls + 2 capture slot calls -> outputs are out-3, out-4.
        self.assertEqual(backend._outputs[shape_key], ["out-3", "out-4"])
        # post_warmup_hook called after each warmup (2) and between slot 0 & 1 (1).
        self.assertEqual(post_warmup.call_count, 3)
        backend.cleanup()

    def test_pingpong_alternates_slots_and_waits_per_slot_fence(self):
        with (
            mock.patch.object(
                mod.envs.SGLANG_NPU_GRAPH_PINGPONG, "get", return_value=True
            ),
            mock.patch.object(
                mod.envs.SGLANG_NPU_GRAPH_ASYNC_UPDATE, "get", return_value=False
            ),
        ):
            backend, runner = _make_backend()

        events = [mock.Mock(), mock.Mock(), mock.Mock()]
        runner.device_module.Event.side_effect = events
        g0, g1 = mock.Mock(), mock.Mock()
        shape_key = ShapeKey(size=4)
        backend._graphs[shape_key] = [g0, g1]
        backend._outputs[shape_key] = ["out0", "out1"]

        # Step 0 -> slot 0: no prior fence synchronized.
        self.assertEqual(
            backend.replay_with_input_update(
                shape_key, [10], attr_name="k", attr_type=[]
            ),
            "out0",
        )
        g0.replay.assert_called_once()
        g1.replay.assert_not_called()
        events[0].record.assert_called_once()
        events[0].synchronize.assert_not_called()

        # Step 1 -> slot 1: must NOT wait on slot 0's fence (events[0]).
        self.assertEqual(
            backend.replay_with_input_update(
                shape_key, [11], attr_name="k", attr_type=[]
            ),
            "out1",
        )
        g1.replay.assert_called_once()
        events[0].synchronize.assert_not_called()
        events[1].record.assert_called_once()

        # Step 2 -> slot 0 again: now waits on slot 0's fence (events[0]), not events[1].
        self.assertEqual(
            backend.replay_with_input_update(
                shape_key, [12], attr_name="k", attr_type=[]
            ),
            "out0",
        )
        events[0].synchronize.assert_called_once()
        events[1].synchronize.assert_not_called()
        self.assertEqual(g0.replay.call_count, 2)
        backend.cleanup()

    def test_pingpong_async_update_defers_future_until_slot_reuse_and_cleanup(self):
        import threading

        with (
            mock.patch.object(
                mod.envs.SGLANG_NPU_GRAPH_PINGPONG, "get", return_value=True
            ),
            mock.patch.object(
                mod.envs.SGLANG_NPU_GRAPH_ASYNC_UPDATE, "get", return_value=True
            ),
        ):
            backend, _ = _make_backend()

        gate = threading.Event()
        completed = []

        def slow_update(cpu_update_input):
            gate.wait(timeout=2.0)
            completed.append(cpu_update_input)

        g0 = SimpleNamespace(update=slow_update, replay=mock.Mock())
        g1 = SimpleNamespace(update=mock.Mock(), replay=mock.Mock())
        shape_key = ShapeKey(size=2)
        backend._graphs[shape_key] = [g0, g1]
        backend._outputs[shape_key] = ["out0", "out1"]

        # Step 0 must return immediately without waiting for gate to be set.
        out0 = backend.replay_with_input_update(
            shape_key, [1], attr_name="k", attr_type=[]
        )
        self.assertEqual(out0, "out0")
        self.assertIn((shape_key, 0), backend._pending_update_futures)
        self.assertEqual(completed, [])

        # Release gate; cleanup must drain pending futures cleanly.
        gate.set()
        backend.cleanup()
        self.assertEqual(completed, [[{"k": [1]}]])
        self.assertEqual(backend._pending_update_futures, {})


if __name__ == "__main__":
    unittest.main()
