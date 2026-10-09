"""Inkling's v2 overlap dispatch must not inherit the experimental gate."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.lora.trtllm_lora_temp.inkling_dense import allow_inkling_moe_two_stream
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _backend(name, active):
    return SimpleNamespace(
        name=name, batch_info=SimpleNamespace(has_active_lora=active)
    )


@pytest.mark.parametrize("backend", ["triton_v2", "triton"])
@pytest.mark.parametrize("sink_bound", [None, False, True])
@pytest.mark.parametrize("overlap_enabled", [False, True])
def test_outer_overlap_dispatch(monkeypatch, backend, sink_bound, overlap_enabled):
    """Missing expert wrappers must not send v2 through the legacy overlap gate."""
    from sglang.srt.lora.trtllm_lora_temp import inkling_dense
    from sglang.srt.models.inkling_common import moe

    sink = SimpleNamespace()
    routed = SimpleNamespace()
    if sink_bound is not None:
        sink.lora_backend = routed.lora_backend = _backend(backend, True)
        sink.set_lora = sink_bound
    calls = []
    alt_stream = Mock()
    current_stream = Mock()
    layer = SimpleNamespace(
        shared_experts=sink,
        experts=routed,
        alt_stream=alt_stream,
        gate=lambda x: (None, None, None, None),
        _clone_fused_sink_input=False,
        _forward_routed=lambda *args: calls.append("routed"),
        _forward_shared=lambda *args: calls.append("sink"),
    )
    legacy_gate = Mock(return_value=False)
    monkeypatch.setattr(inkling_dense, "allow_inkling_moe_two_stream", legacy_gate)
    monkeypatch.setattr(moe, "lora_compatible_layout_enabled", lambda: True)
    monkeypatch.setattr(moe, "get_lora", lambda: SimpleNamespace(lora_backend=backend))
    monkeypatch.setattr(moe, "get_is_capture_mode", lambda: True)
    monkeypatch.setattr(
        moe.envs.SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP,
        "get",
        lambda: overlap_enabled,
    )
    monkeypatch.setattr(moe.torch.cuda, "current_stream", lambda: current_stream)
    monkeypatch.setattr(moe.torch.cuda, "stream", lambda stream: nullcontext())
    x = SimpleNamespace(is_cuda=True, shape=(64, 6144))
    moe.InklingMoE.forward(layer, x, reduce=False)
    if backend == "triton_v2":
        legacy_gate.assert_not_called()
    else:
        legacy_gate.assert_called_once_with(sink, routed, 64)
    if backend == "triton_v2" and overlap_enabled:
        assert calls == ["sink", "routed"]
        alt_stream.wait_stream.assert_called_once_with(current_stream)
        current_stream.wait_stream.assert_called_once_with(alt_stream)
    else:
        assert calls == ["routed", "sink"]
        current_stream.wait_stream.assert_not_called()


def test_no_lora_work_always_overlaps():
    idle = SimpleNamespace(lora_backend=_backend("triton", False))
    assert allow_inkling_moe_two_stream(idle, idle, 8)


def test_experimental_path_keeps_its_env_gate(monkeypatch):
    from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

    busy = SimpleNamespace(lora_backend=_backend("triton", True), set_lora=False)
    monkeypatch.setattr(
        lora_envs.SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC, "get", lambda: False
    )
    assert not allow_inkling_moe_two_stream(busy, busy, 8)
    monkeypatch.setattr(
        lora_envs.SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC, "get", lambda: True
    )
    assert allow_inkling_moe_two_stream(busy, busy, 8)
    assert not allow_inkling_moe_two_stream(busy, busy, 64)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
