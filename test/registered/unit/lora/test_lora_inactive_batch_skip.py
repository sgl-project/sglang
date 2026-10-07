"""Skip dense LoRA for adapter-free batches, except when capturing adapter graphs.
Decode uses the separate nolora graph variant for adapter-free batches.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest

from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.utils import capturing_lora_graph
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


def test_capture_lora_variant_restores_the_previous_label(capture):
    capture(True, None)
    with capture_mode.capture_lora_variant("lora"):
        assert capture_mode.get_capture_lora_variant() == "lora"
        with capture_mode.capture_lora_variant("nolora"):
            assert capture_mode.get_capture_lora_variant() == "nolora"
            assert not capturing_lora_graph()
        assert capture_mode.get_capture_lora_variant() == "lora"
    assert capture_mode.get_capture_lora_variant() is None
    with pytest.raises(RuntimeError):
        with capture_mode.capture_lora_variant("nolora"):
            raise RuntimeError("capture failed")
    assert capture_mode.get_capture_lora_variant() is None
    # A later single-graph capture (prefill, a draft runner) keeps its LoRA work.
    assert capturing_lora_graph()


BACKENDS = [
    ("sglang.srt.lora.backend.triton_backend", "TritonLoRABackend"),
    ("sglang.srt.lora.backend.triton_v2_backend", "TritonV2LoRABackend"),
    ("sglang.srt.lora.backend.chunked_backend", "ChunkedSgmvLoRABackend"),
    ("sglang.srt.lora.backend.torch_backend", "TorchNativeLoRABackend"),
    ("sglang.srt.lora.backend.ascend_backend", "AscendLoRABackend"),
]


@pytest.mark.parametrize("module,name", BACKENDS)
def test_graph_init_accepts_the_runner_keywords(module, name):
    try:
        cls = getattr(importlib.import_module(module), name)
    except ImportError as exc:  # a platform backend whose imports are absent here
        pytest.skip(f"{module}: {exc}")
    inspect.signature(cls.init_decode_cuda_graph_batch_info).bind(
        None, max_bs_in_cuda_graph=8, num_tokens_per_req=1
    )
    inspect.signature(cls.init_prefill_cuda_graph_batch_info).bind(
        None, max_num_tokens=64, max_num_requests=8
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
