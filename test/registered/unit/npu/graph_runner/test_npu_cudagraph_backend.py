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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        get_parallel().override(tp_group=SimpleNamespace(barrier=mock.Mock())),
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
    runner = EAGLEDraftCudaGraphRunner.__new__(EAGLDraftCudaGraphRunner)
    runner.device_module = _device_module_stub()
    runner.speculative_num_steps = speculative_num_steps
    runner.raw_bs = raw_bs
    runner.bs = bs
    runner.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=architectures, index_topk=index_topk
            )
        )
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


if __name__ == "__main__":
    unittest.main()
