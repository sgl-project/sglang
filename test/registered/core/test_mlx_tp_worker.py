import dataclasses
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _install_stub(fullname: str, **attrs):
    module = types.ModuleType(fullname)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[fullname] = module
    return module


def _load_tp_worker_module():
    @dataclasses.dataclass
    class _StubGenerationBatchResult:
        logits_output: object
        next_token_ids: object = None
        can_run_cuda_graph: bool = False

    @dataclasses.dataclass
    class _StubLogitsProcessorOutput:
        next_token_logits: object

    class _StubTpModelWorker:
        def forward_batch_generation(self, *args, **kwargs):
            raise NotImplementedError

    _install_stub("sglang")
    _install_stub("sglang.srt")
    _install_stub("sglang.srt.managers")
    _install_stub("sglang.srt.model_executor")
    _install_stub("sglang.srt.layers")
    _install_stub(
        "sglang.srt.managers.schedule_batch",
        ModelWorkerBatch=object,
    )
    _install_stub(
        "sglang.srt.managers.tp_worker",
        TpModelWorker=_StubTpModelWorker,
    )
    _install_stub(
        "sglang.srt.managers.utils",
        GenerationBatchResult=_StubGenerationBatchResult,
    )
    _install_stub(
        "sglang.srt.model_executor.forward_batch_info",
        ForwardBatch=object,
        PPProxyTensors=object,
    )
    _install_stub(
        "sglang.srt.layers.logits_processor",
        LogitsProcessorOutput=_StubLogitsProcessorOutput,
    )

    module_path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/hardware_backend/mlx/tp_worker.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_mlx_tp_worker_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeForwardMode:
    def __init__(self, mode: str):
        self.mode = mode

    def is_idle(self):
        return self.mode == "idle"

    def is_extend(self):
        return self.mode == "extend"

    def is_decode(self):
        return self.mode == "decode"


class _FakeMlxRunner:
    def __init__(self):
        self._request_states = {}
        self.removed_request_ids = []

    def prefill(self, req_id: str, token_ids: list[int]) -> int:
        next_token = token_ids[-1] + 1
        self._request_states[req_id] = {
            "token_ids": list(token_ids) + [next_token],
        }
        return next_token

    def decode_batch(self, req_ids: list[str]) -> list[int]:
        missing = [req_id for req_id in req_ids if req_id not in self._request_states]
        if missing:
            raise KeyError(missing[0])

        next_tokens = []
        for req_id in req_ids:
            state = self._request_states[req_id]
            next_token = state["token_ids"][-1] + 1
            state["token_ids"].append(next_token)
            next_tokens.append(next_token)
        return next_tokens

    def remove_request(self, req_id: str):
        self.removed_request_ids.append(req_id)
        self._request_states.pop(req_id, None)

    def clear(self):
        self._request_states.clear()


class TestMlxTpModelWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tp_worker_module = _load_tp_worker_module()

    def _make_worker(self):
        worker = object.__new__(self.tp_worker_module.MlxTpModelWorker)
        worker._mlx_runner = _FakeMlxRunner()
        return worker

    @staticmethod
    def _make_extend_batch(reqs_with_tokens: list[tuple[str, list[int]]]):
        flat_input_ids = [
            token_id for _, token_ids in reqs_with_tokens for token_id in token_ids
        ]
        return SimpleNamespace(
            forward_mode=_FakeForwardMode("extend"),
            reqs=[SimpleNamespace(rid=req_id) for req_id, _ in reqs_with_tokens],
            input_ids=torch.tensor(flat_input_ids, dtype=torch.long),
            extend_seq_lens=[len(token_ids) for _, token_ids in reqs_with_tokens],
        )

    @staticmethod
    def _make_decode_batch(req_ids: list[str]):
        return SimpleNamespace(
            forward_mode=_FakeForwardMode("decode"),
            reqs=[SimpleNamespace(rid=req_id) for req_id in req_ids],
        )

    def test_request_state_survives_batches_where_request_is_temporarily_absent(self):
        worker = self._make_worker()

        worker._forward_batch_generation_mlx(
            self._make_extend_batch([("req-a", [10, 11, 12])])
        )
        self.assertIn("req-a", worker._mlx_runner._request_states)

        worker._forward_batch_generation_mlx(
            self._make_extend_batch([("req-b", [20, 21, 22])])
        )
        self.assertIn("req-a", worker._mlx_runner._request_states)
        self.assertEqual(worker._mlx_runner.removed_request_ids, [])

        result = worker._forward_batch_generation_mlx(
            self._make_decode_batch(["req-a", "req-b"])
        )

        self.assertEqual(result.next_token_ids.tolist(), [14, 24])

    def test_cleanup_hooks_remove_request_state_when_requested(self):
        worker = self._make_worker()

        worker._forward_batch_generation_mlx(
            self._make_extend_batch([("req-a", [10, 11, 12]), ("req-b", [20, 21, 22])])
        )
        self.assertEqual(set(worker._mlx_runner._request_states), {"req-a", "req-b"})

        worker.cleanup_requests(["req-a"])
        self.assertEqual(set(worker._mlx_runner._request_states), {"req-b"})
        self.assertEqual(worker._mlx_runner.removed_request_ids, ["req-a"])

        worker.clear_runtime_state()
        self.assertEqual(worker._mlx_runner._request_states, {})


if __name__ == "__main__":
    unittest.main()
