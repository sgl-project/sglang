"""CPU source-isolated probe tests; no SGLang/CANN qualification implied."""

import argparse
import ast
import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

OMNI_ROOT = None

ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "python/sglang/srt/hardware_backend/npu/graph_runner"
SPEC = importlib.util.spec_from_file_location(
    "probe_submission", BACKEND_DIR / "npu_graph_submission.py"
)
submission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submission)


def load_method(path, class_name, method_name):
    import torch

    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    namespace = {
        "Any": object,
        "ShapeKey": object,
        "torch": torch,
        "npu_graph_submission": submission.npu_graph_submission,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[method_name]


class SubmissionTests(unittest.TestCase):
    def test_encoder_actual_replay_path(self):
        if OMNI_ROOT is None:
            self.skipTest("pass --omni-root to exercise the encoder source")
        import torch

        run = load_method(
            OMNI_ROOT / "sglang_omni/models/qwen3_asr/encoder_cuda_graph.py",
            "Qwen3ASREncoderLayerStackGraphRunner",
            "run",
        )
        calls = []
        graph = object()
        entry = SimpleNamespace(
            hidden_states=torch.zeros(8, 2), graph=graph, output=torch.ones(8, 2)
        )
        runner = SimpleNamespace(
            _is_npu=True,
            _capture_failed=False,
            _max_seqlen=8,
            _plan=lambda total, windows: (8, [8 - total]),
            _graphs={8: entry},
            _make_cu_seqlens=lambda lengths, device: torch.tensor([0, lengths[0], 8]),
            _device_module=SimpleNamespace(current_stream=lambda: "encoder"),
            _npu_update_stream=SimpleNamespace(
                wait_stream=lambda stream: calls.append("wait")
            ),
            _graph_backend=SimpleNamespace(replay=lambda graph: calls.append("replay")),
            _update_npu_attention_tasks=lambda entry, lengths: calls.append(
                ("update", lengths)
            ),
        )
        name = "sglang.srt.hardware_backend.npu.graph_runner.npu_graph_submission"
        with patch.dict(sys.modules, {name: submission}):
            output = run(runner, torch.ones(3, 2), [3])
        self.assertEqual(calls, ["wait", "replay", ("update", [3, 8])])
        self.assertEqual(tuple(output.shape), (3, 2))
        entry.output.zero_()
        self.assertTrue(torch.all(output == 1))

    def test_decoder_order_payload_thread_and_errors(self):
        run = load_method(
            BACKEND_DIR / "npu_cudagraph_backend.py",
            "NPUCudaGraphBackend",
            "replay_with_input_update",
        )
        caller = threading.get_ident()
        for failure in (None, "wait", "replay", "update"):
            with self.subTest(failure=failure):
                calls = []
                error = RuntimeError("injected")

                def record(name):
                    calls.append((name, threading.get_ident()))
                    if name == failure:
                        raise error

                stream = SimpleNamespace(wait_stream=lambda compute: record("wait"))
                payload = [{"seq_lens": [3, 7]}]

                def update(*, cpu_update_input):
                    self.assertIs(cpu_update_input, payload)
                    record("update")

                output = object()
                backend = SimpleNamespace(
                    _graphs={
                        1: SimpleNamespace(
                            replay=lambda: record("replay"), update=update
                        )
                    },
                    _outputs={1: output},
                    _device_id=0,
                    _device_module=SimpleNamespace(
                        set_device=lambda device: record("device"),
                        current_stream=lambda: "compute",
                    ),
                )
                graphs = SimpleNamespace(
                    _GraphDispatchMode=SimpleNamespace(update_stream=stream)
                )
                with patch.dict(sys.modules, {"torch_npu.npu.graphs": graphs}):
                    if failure:
                        with self.assertRaises(RuntimeError) as caught:
                            run(backend, 1, None, cpu_update_input=payload)
                        self.assertIs(caught.exception, error)
                    else:
                        self.assertIs(
                            run(backend, 1, None, cpu_update_input=payload), output
                        )
                expected = ["device", "wait", "replay", "update"]
                if failure:
                    expected = expected[: expected.index(failure) + 1]
                self.assertEqual(calls, [(name, caller) for name in expected])

    def test_worker_transactions_do_not_interleave(self):
        entered = threading.Event()
        competitor = threading.Event()
        release = threading.Event()
        calls = []
        errors = []

        def worker(name):
            try:
                if name == "decoder":
                    competitor.set()
                stream = SimpleNamespace(
                    wait_stream=lambda compute: calls.append((name, "wait"))
                )
                device = SimpleNamespace(current_stream=lambda: name)
                with submission.npu_graph_submission(device, stream, source=name):
                    calls.append((name, "replay"))
                    if name == "encoder":
                        entered.set()
                        if not release.wait(3):
                            raise RuntimeError("test release timeout")
                    calls.append((name, "update"))
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=worker, args=("encoder",), daemon=True)
        second = threading.Thread(target=worker, args=("decoder",), daemon=True)
        first.start()
        self.assertTrue(entered.wait(3))
        second.start()
        try:
            self.assertTrue(competitor.wait(3))
            self.assertEqual(calls, [("encoder", "wait"), ("encoder", "replay")])
        finally:
            release.set()
            first.join(3)
            second.join(3)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(
            calls,
            [
                (name, action)
                for name in ("encoder", "decoder")
                for action in ("wait", "replay", "update")
            ],
        )

    def test_exception_releases_lock_for_another_worker(self):
        device = SimpleNamespace(current_stream=lambda: "compute")
        stream = SimpleNamespace(wait_stream=lambda compute: None)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with submission.npu_graph_submission(device, stream, source="failed"):
                raise RuntimeError("injected")
        done = threading.Event()

        def worker():
            with submission.npu_graph_submission(device, stream, source="next"):
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(3)
        self.assertTrue(done.is_set())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--omni-root", type=Path)
    args, remaining = parser.parse_known_args()
    OMNI_ROOT = args.omni_root
    unittest.main(argv=[sys.argv[0]] + remaining)
