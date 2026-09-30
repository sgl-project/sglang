"""Regression for concurrent dynamic NPU graph updates and synchronizing ops."""

import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.utils.common import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")


def exercise_graph_update():
    import torch
    import torch_npu
    import torch_npu.npu.graphs as graph_api

    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )

    torch.npu.set_device(0)
    torch.manual_seed(1234)
    q = torch.randn(1, 4, 1, 64, device="npu", dtype=torch.float16)
    k = torch.randn(1, 4, 128, 64, device="npu", dtype=torch.float16)
    v = torch.randn_like(k)

    def attention(length):
        return torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=4,
            num_key_value_heads=4,
            input_layout="BNSD",
            scale=0.125,
            actual_seq_lengths_kv=[length],
        )[0]

    runner = SimpleNamespace(
        device_module=torch.npu,
        model_runner=SimpleNamespace(tp_group=SimpleNamespace(barrier=lambda: None)),
    )
    backend = NPUCudaGraphBackend(runner)
    backend.capture_one("test", lambda: attention(128))
    graph = backend._graphs["test"]
    original_end = graph_api.graph_task_update_end
    original_replay = graph.replay
    for length in (64, 96, 128):
        reached = threading.Event()
        release = threading.Event()
        submitted = threading.Event()
        errors = []
        outputs = []

        def delayed_end(stream):
            reached.set()
            if not release.wait(10):
                raise TimeoutError("Test did not release the graph update")
            original_end(stream)

        def replay():
            original_replay()
            submitted.set()

        def run_graph():
            try:
                torch.npu.set_device(0)
                outputs.append(
                    backend.replay_with_input_update(
                        "test", [length], attr_name="actual_seq_lengths_kv"
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        def synchronizing_op():
            try:
                torch.npu.set_device(0)
                # Nonzero synchronizes inside the launch-queue consumer.
                indices = q.nonzero()
                assert indices.shape[0] == q.numel()
            except BaseException as exc:
                errors.append(exc)

        graph_api.graph_task_update_end = delayed_end
        graph.replay = replay
        replay_thread = threading.Thread(target=run_graph, daemon=True)
        replay_thread.start()
        assert reached.wait(10), "Graph update did not start"
        assert submitted.wait(10), "Graph replay was not submitted"
        copy_thread = threading.Thread(target=synchronizing_op, daemon=True)
        copy_thread.start()
        time.sleep(0.1)
        release.set()
        replay_thread.join(10)
        copy_thread.join(10)
        assert not replay_thread.is_alive() and not copy_thread.is_alive()
        assert not errors, errors
        graph_api.graph_task_update_end = original_end
        graph.replay = original_replay
        torch.npu.synchronize()
        torch.testing.assert_close(outputs[0], attention(length), rtol=1e-3, atol=1e-3)
    backend.cleanup()


@unittest.skipUnless(is_npu(), "requires Ascend NPU")
class TestNPUGraphUpdate(unittest.TestCase):
    def test_synchronizing_op_does_not_deadlock_update(self):
        # A deadlocked native thread can hold the GIL; isolate it so the test
        # times out and fails instead of hanging the complete CI worker.
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--exercise"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--exercise" in sys.argv:
        exercise_graph_update()
    else:
        unittest.main()
