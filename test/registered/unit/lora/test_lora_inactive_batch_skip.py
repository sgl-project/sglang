"""Skip dense LoRA for adapter-free batches, except when capturing adapter graphs.
Decode uses the separate nolora graph variant for adapter-free batches.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    _lora_backend_skips_inactive,
)
from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Layer(SimpleNamespace):
    """A LoRA layer stand-in with the real ``lora_active`` rule."""

    @property
    def lora_active(self):
        return BaseLayerWithLoRA.lora_active.fget(self)


class _Backend(SimpleNamespace):
    def get_batch_info(self):
        return self.batch_info


def _layer(
    active,
    *,
    skip_dense,
    skip_batches=False,
    set_lora=True,
    batch=True,
    backend_name="triton_v2",
):
    backend = _Backend(
        name=backend_name,
        batch_info=SimpleNamespace(has_active_lora=active) if batch else None,
        skip_inactive_lora_batches=skip_batches,
        skip_inactive_dense_lora=skip_dense,
    )
    return _Layer(set_lora=set_lora, lora_backend=backend)


def _active(layer):
    return BaseLayerWithLoRA.lora_active.fget(layer)


@pytest.fixture
def capture(monkeypatch):
    def set_state(capturing, variant=None):
        monkeypatch.setattr(capture_mode, "is_capture_mode", capturing)
        monkeypatch.setattr(capture_mode, "_capture_lora_variant", variant)

    return set_state


def test_active_adapter_or_legacy_backend_keeps_lora(capture):
    capture(False)
    assert _active(_layer(True, skip_dense=True))
    assert _active(_layer(False, skip_dense=False))
    assert not _active(_layer(True, skip_dense=True, set_lora=False))
    assert not _active(_layer(True, skip_dense=True, batch=False))


def test_inactive_batch_skips_dense_lora_outside_capture(capture):
    capture(False)
    assert not _active(_layer(False, skip_dense=True))
    # the whole-batch skip (UNO) still wins regardless of the dense flag
    assert not _active(_layer(False, skip_dense=False, skip_batches=True))


@pytest.mark.parametrize(
    "variant,expected", [(None, True), ("lora", True), ("nolora", False)]
)
def test_capture_keeps_lora_except_for_the_nolora_variant(capture, variant, expected):
    capture(True, variant)
    assert _active(_layer(False, skip_dense=True)) is expected


def test_backend_flags_and_decode_runner_default():
    assert BaseLoRABackend.skip_inactive_dense_lora is False
    assert TritonV2LoRABackend.skip_inactive_dense_lora is True
    assert not _lora_backend_skips_inactive(SimpleNamespace(lora_manager=None))
    manager = SimpleNamespace(
        lora_backend=SimpleNamespace(skip_inactive_dense_lora=True)
    )
    assert _lora_backend_skips_inactive(SimpleNamespace(lora_manager=manager))
    manager = SimpleNamespace(
        lora_backend=SimpleNamespace(skip_inactive_dense_lora=False)
    )
    assert not _lora_backend_skips_inactive(SimpleNamespace(lora_manager=manager))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
