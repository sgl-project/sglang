"""Skip dense LoRA for adapter-free batches, except when capturing adapter graphs.
Decode uses the separate nolora graph variant for adapter-free batches.
"""

from types import SimpleNamespace

import pytest

from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.utils import kv_b_lora_correction
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


def _layer(active, *, skip_dense, set_lora=True, batch=True, backend_name="triton_v2"):
    backend = _Backend(
        name=backend_name,
        batch_info=SimpleNamespace(has_active_lora=active) if batch else None,
        skip_inactive_lora_batches=False,
        skip_inactive_dense_lora=skip_dense,
    )
    return _Layer(set_lora=set_lora, lora_backend=backend)


def _attn(layer):
    return SimpleNamespace(kv_b_proj=layer)


@pytest.fixture
def capture(monkeypatch):
    def set_state(capturing, variant=None):
        monkeypatch.setattr(capture_mode, "is_capture_mode", capturing)
        monkeypatch.setattr(capture_mode, "_capture_lora_variant", variant)

    return set_state


def test_outside_capture_only_an_inactive_v2_batch_skips_lora(capture):
    capture(False)
    assert _layer(True, skip_dense=True).lora_active
    assert _layer(False, skip_dense=False).lora_active
    assert not _layer(False, skip_dense=True).lora_active
    assert not _layer(True, skip_dense=True, set_lora=False).lora_active
    assert not _layer(True, skip_dense=True, batch=False).lora_active


@pytest.mark.parametrize(
    "variant,expected", [(None, True), ("lora", True), ("nolora", False)]
)
def test_capture_keeps_lora_except_for_the_nolora_variant(capture, variant, expected):
    capture(True, variant)
    layer = _layer(False, skip_dense=True)
    assert layer.lora_active is expected
    # The kv_b correction shares the capture boundary.
    assert kv_b_lora_correction(_attn(layer)) == ("v2" if expected else None)


def test_kv_b_correction_runs_v2_legacy_or_nothing(capture):
    capture(False)
    assert kv_b_lora_correction(_attn(_layer(True, skip_dense=True))) == "v2"
    # An inactive triton_v2 batch runs neither correction: the legacy one
    # cannot read the v2 batch info.
    assert kv_b_lora_correction(_attn(_layer(False, skip_dense=True))) is None
    legacy = dict(skip_dense=False, backend_name="triton")
    assert kv_b_lora_correction(_attn(_layer(True, **legacy))) == "legacy"
    assert kv_b_lora_correction(_attn(_layer(False, **legacy))) == "legacy"
    assert (
        kv_b_lora_correction(_attn(_layer(True, skip_dense=True, set_lora=False)))
        is None
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
