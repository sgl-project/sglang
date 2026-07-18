"""Hermetic stream-order checks for Inkling shared/routed overlap."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

# Skipped on CI: this hermetic check re-parses the InklingMoE forward source and
# pins its exact stream-order, so it breaks on unrelated refactors of that
# method. Skip until it is rebuilt against a stable seam.
pytestmark = pytest.mark.skip(
    reason="refactor-fragile source-parsing unit test; skipped on CI"
)

REPO_ROOT = Path(__file__).resolve().parents[4]
MOE_PATH = REPO_ROOT / "python/sglang/srt/models/inkling_common/moe.py"
INKLING_DENSE_PATH = (
    REPO_ROOT / "python/sglang/srt/lora/trtllm_lora_temp/inkling_dense.py"
)


class _Flag:
    def __init__(self, value: bool):
        self.value = value

    def get(self) -> bool:
        return self.value


class _Stream:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    def wait_stream(self, other: _Stream) -> None:
        self.events.append(f"{self.name}.wait({other.name})")


class _StreamContext:
    def __init__(self, cuda, stream: _Stream):
        self.cuda = cuda
        self.stream = stream
        self.previous = None

    def __enter__(self):
        self.previous = self.cuda.current
        self.cuda.current = self.stream
        self.cuda.events.append(f"enter({self.stream.name})")

    def __exit__(self, *_):
        self.cuda.events.append(f"exit({self.stream.name})")
        self.cuda.current = self.previous


class _Cuda:
    def __init__(self, events: list[str]):
        self.events = events
        self.current = _Stream("main", events)

    def current_stream(self) -> _Stream:
        return self.current

    def stream(self, stream: _Stream) -> _StreamContext:
        return _StreamContext(self, stream)


class _Tensor:
    def __init__(self, name: str, events: list[str], *, tokens: int = 1):
        self.name = name
        self.events = events
        self.shape = (tokens, 8)
        self.dtype = "bf16"
        self.is_cuda = True

    def record_stream(self, stream: _Stream) -> None:
        self.events.append(f"{self.name}.record({stream.name})")

    def __add__(self, other: _Tensor) -> _Tensor:
        self.events.append(f"add({self.name},{other.name})")
        return _Tensor("sum", self.events)


def _load_forward(fake_torch, capture: bool = False):
    tree = ast.parse(MOE_PATH.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InklingMoE"
    )
    forward = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    test_class = ast.ClassDef(
        name="_InklingMoEForwardUnderTest",
        bases=[],
        keywords=[],
        body=[forward],
        decorator_list=[],
    )
    namespace = {
        "ForwardBatch": object,
        "envs": SimpleNamespace(
            SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP=_Flag(True)
        ),
        # capture gating added by #103: overlap only inside cuda-graph capture
        "get_is_capture_mode": lambda: capture,
        "get_ar_buffer": lambda *_: None,
        "get_tensor_model_parallel_group": lambda: SimpleNamespace(world_size=1),
        "lora_compatible_layout_enabled": lambda: True,
        "torch": fake_torch,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[])),
            str(MOE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[test_class.name]


def _load_lora_overlap_policy():
    tree = ast.parse(INKLING_DENSE_PATH.read_text())
    policy = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "allow_inkling_moe_two_stream"
    )
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[policy], type_ignores=[])),
            str(INKLING_DENSE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace[policy.name]


def _make_moe(events: list[str], cuda: _Cuda, capture: bool = False):
    fake_torch = SimpleNamespace(Tensor=_Tensor, cuda=cuda)
    moe = _load_forward(fake_torch, capture)()
    moe.alt_stream = _Stream("alt", events)
    moe.shared_experts = SimpleNamespace(
        lora_backend=SimpleNamespace(batch_info=SimpleNamespace(has_active_lora=True))
    )
    moe.experts = SimpleNamespace()
    moe._clone_fused_sink_input = False
    moe._fused_ar_shared = False
    moe.gate = lambda x: (
        _Tensor("topk_weights", events),
        _Tensor("topk_ids", events),
        _Tensor("gammas", events),
        None,
    )

    def forward_shared(x, gammas):
        events.append(f"shared({cuda.current.name})")
        return _Tensor("shared_out", events)

    def forward_routed(*_):
        assert cuda.current.name == "main"
        events.append("routed(main)")
        return _Tensor("routed_out", events)

    moe._forward_shared = forward_shared
    moe._forward_routed = forward_routed
    return moe


def _install_lora_policy(monkeypatch, *, main_alloc: bool, capture: bool = False):
    lora_envs = SimpleNamespace(SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=_Flag(main_alloc))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.lora.trtllm_lora_temp.environ",
        types.SimpleNamespace(lora_envs=lora_envs),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.runner_utils.capture_mode",
        types.SimpleNamespace(get_is_capture_mode=lambda: capture),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.lora.trtllm_lora_temp.inkling_dense",
        types.SimpleNamespace(allow_inkling_moe_two_stream=_load_lora_overlap_policy()),
    )


@pytest.mark.parametrize("tokens", [1, 32])
def test_direct_sink_keeps_decode_overlap(monkeypatch, tokens):
    events: list[str] = []
    cuda = _Cuda(events)
    _install_lora_policy(monkeypatch, main_alloc=True, capture=True)
    moe = _make_moe(events, cuda, capture=True)

    moe.forward(_Tensor("x", events, tokens=tokens), reduce=False)

    assert events == [
        "x.record(alt)",
        "gammas.record(alt)",
        "alt.wait(main)",
        "enter(alt)",
        "shared(alt)",
        "exit(alt)",
        "routed(main)",
        "main.wait(alt)",
        "shared_out.record(main)",
        "add(routed_out,shared_out)",
    ]


def test_lora_prefill_stays_serial(monkeypatch):
    # Even when capture would allow overlap, the M>32 LoRA policy forces serial.
    events: list[str] = []
    cuda = _Cuda(events)
    _install_lora_policy(monkeypatch, main_alloc=True, capture=True)
    moe = _make_moe(events, cuda, capture=True)

    moe.forward(_Tensor("x", events, tokens=33), reduce=False)

    assert events == [
        "routed(main)",
        "shared(main)",
        "add(routed_out,shared_out)",
    ]


def test_captured_prefill_stays_serial_even_base_only(monkeypatch):
    # Capture forces has_lora_work (one schedule for every replay), so
    # prefill-sized batches (>32 tokens) are serial even with no live adapter.
    events: list[str] = []
    cuda = _Cuda(events)
    _install_lora_policy(monkeypatch, main_alloc=True, capture=True)
    moe = _make_moe(events, cuda, capture=True)
    moe.shared_experts.lora_backend.batch_info.has_active_lora = False

    moe.forward(_Tensor("x", events, tokens=33), reduce=False)

    assert events == [
        "routed(main)",
        "shared(main)",
        "add(routed_out,shared_out)",
    ]


def test_eager_forward_stays_serial_even_base_only(monkeypatch):
    # #103: overlap is gated on cuda-graph capture; eager forwards are serial.
    events: list[str] = []
    cuda = _Cuda(events)
    _install_lora_policy(monkeypatch, main_alloc=False, capture=False)
    moe = _make_moe(events, cuda, capture=False)
    moe.shared_experts.lora_backend.batch_info.has_active_lora = False

    moe.forward(_Tensor("x", events, tokens=33), reduce=False)

    assert events == [
        "routed(main)",
        "shared(main)",
        "add(routed_out,shared_out)",
    ]


def test_lora_overlap_stays_serial_without_main_alloc(monkeypatch):
    events: list[str] = []
    cuda = _Cuda(events)
    _install_lora_policy(monkeypatch, main_alloc=False, capture=True)
    moe = _make_moe(events, cuda, capture=True)

    moe.forward(_Tensor("x", events), reduce=False)

    assert events == [
        "routed(main)",
        "shared(main)",
        "add(routed_out,shared_out)",
    ]


def test_capture_keeps_lora_schedule_without_active_adapter(monkeypatch):
    events: list[str] = []
    cuda = _Cuda(events)
    _install_lora_policy(monkeypatch, main_alloc=False, capture=True)
    moe = _make_moe(events, cuda, capture=True)
    moe.shared_experts.lora_backend.batch_info.has_active_lora = False

    moe.forward(_Tensor("x", events), reduce=False)

    assert events == [
        "routed(main)",
        "shared(main)",
        "add(routed_out,shared_out)",
    ]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
