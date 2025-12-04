import importlib
import importlib.machinery
import sys
import types
import unittest
from types import SimpleNamespace

# Provide lightweight stubs for optional native deps so we can import the mixin in a CPU-only env.
triton_stub = types.ModuleType("triton")
triton_language_stub = types.ModuleType("triton.language")
triton_stub.language = triton_language_stub
triton_stub.__spec__ = importlib.machinery.ModuleSpec("triton", None)
triton_language_stub.__spec__ = importlib.machinery.ModuleSpec("triton.language", None)
sys.modules.setdefault("triton", triton_stub)
sys.modules.setdefault("triton.language", triton_language_stub)
triton_testing_stub = types.ModuleType("triton.testing")
triton_testing_stub.__spec__ = importlib.machinery.ModuleSpec("triton.testing", None)
sys.modules.setdefault("triton.testing", triton_testing_stub)
sys.modules.setdefault("zmq", types.ModuleType("zmq"))

import torch

# Torch on macOS may expose torch.mps without a Stream type; make a stub to satisfy type hints.
if hasattr(torch, "mps") and not hasattr(torch.mps, "Stream"):

    class _MpsStream:  # pragma: no cover - just a placeholder for import-time checks
        pass

    torch.mps.Stream = _MpsStream


def _install_stub_module(name: str, attrs: dict) -> None:
    try:
        importlib.import_module(name)
        return
    except Exception:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules.setdefault(name, module)


_install_stub_module(
    "sglang.srt.disaggregation.utils",
    {"DisaggregationMode": SimpleNamespace(DECODE="decode")},
)
_install_stub_module("sglang.srt.environ", {"envs": SimpleNamespace()})
_install_stub_module(
    "sglang.srt.layers.logits_processor", {"LogitsProcessorOutput": object}
)
_install_stub_module(
    "sglang.srt.managers.io_struct",
    {
        "AbortReq": object,
        "BatchEmbeddingOutput": object,
        "BatchTokenIDOutput": object,
    },
)
_install_stub_module(
    "sglang.srt.managers.schedule_batch",
    {
        "BaseFinishReason": object,
        "Req": object,
        "RequestStage": SimpleNamespace(),
        "ScheduleBatch": object,
    },
)
_install_stub_module(
    "sglang.srt.mem_cache.common", {"release_kv_cache": lambda *_, **__: None}
)
_install_stub_module(
    "sglang.srt.tracing.trace",
    {
        "trace_slice": lambda *_, **__: None,
        "trace_slice_batch": lambda *_, **__: None,
        "trace_slice_end": lambda *_, **__: None,
    },
)

from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)


class _DummyScheduler(SchedulerOutputProcessorMixin):
    def __init__(self, reasoning_parser: str | None):
        self.server_args = SimpleNamespace(reasoning_parser=reasoning_parser)


class _DummyTokenizer:
    def __init__(self, think_start_id, think_end_id):
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id


class _DummyReq:
    def __init__(self, tokenizer: _DummyTokenizer):
        self.tokenizer = tokenizer
        self.in_reasoning_phase = False
        self.num_reasoning_tokens = 0


class TestUpdateReasoningTokens(unittest.TestCase):
    def test_no_reasoning_parser(self):
        scheduler = _DummyScheduler(reasoning_parser=None)
        req = _DummyReq(_DummyTokenizer(1, 2))

        scheduler._update_reasoning_tokens_for_req(req, 1)

        self.assertEqual(req.num_reasoning_tokens, 0)
        self.assertFalse(req.in_reasoning_phase)

    def test_counts_between_markers(self):
        scheduler = _DummyScheduler(reasoning_parser="deepseek-r1")
        req = _DummyReq(_DummyTokenizer(11, 12))

        scheduler._update_reasoning_tokens_for_req(req, 11)  # <think>
        self.assertTrue(req.in_reasoning_phase)
        scheduler._update_reasoning_tokens_for_req(req, 99)  # reasoning content
        self.assertEqual(req.num_reasoning_tokens, 1)
        scheduler._update_reasoning_tokens_for_req(req, 12)  # </think>

        self.assertFalse(req.in_reasoning_phase)
        scheduler._update_reasoning_tokens_for_req(req, 100)  # normal text
        self.assertEqual(req.num_reasoning_tokens, 1)

    def test_forced_reasoning_without_start_token(self):
        scheduler = _DummyScheduler(reasoning_parser="deepseek-r1")
        req = _DummyReq(_DummyTokenizer(None, 7))

        scheduler._update_reasoning_tokens_for_req(req, 99)
        scheduler._update_reasoning_tokens_for_req(req, 100)
        self.assertTrue(req.in_reasoning_phase)
        self.assertEqual(req.num_reasoning_tokens, 2)

        scheduler._update_reasoning_tokens_for_req(req, 7)  # </think>
        self.assertFalse(req.in_reasoning_phase)
        self.assertEqual(req.num_reasoning_tokens, 2)


if __name__ == "__main__":
    unittest.main()
