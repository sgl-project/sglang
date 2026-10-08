"""Capture-variant labels and the LoRA graph-init keyword interface."""

import importlib
import inspect

import pytest

from sglang.srt.model_executor.runner_utils import capture_mode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def capture(monkeypatch):
    def set_state(capturing, variant=None):
        monkeypatch.setattr(capture_mode, "is_capture_mode", capturing)
        monkeypatch.setattr(capture_mode, "_capture_lora_variant", variant)

    return set_state


def test_capture_lora_variant_restores_the_previous_label(capture):
    capture(True, None)
    with capture_mode.capture_lora_variant("lora"):
        assert capture_mode.get_capture_lora_variant() == "lora"
        with capture_mode.capture_lora_variant("nolora"):
            assert capture_mode.get_capture_lora_variant() == "nolora"
        assert capture_mode.get_capture_lora_variant() == "lora"
    assert capture_mode.get_capture_lora_variant() is None
    with pytest.raises(RuntimeError):
        with capture_mode.capture_lora_variant("nolora"):
            raise RuntimeError("capture failed")
    assert capture_mode.get_capture_lora_variant() is None


BACKENDS = [
    ("sglang.srt.lora.backend.triton_backend", "TritonLoRABackend"),
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
