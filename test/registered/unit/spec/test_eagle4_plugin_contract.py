"""Check the installed EAGLE-4 plugin against the actual host, without a model.

Run in an environment with ``pip install -e 'python[eagle4]'``. The plugin is
optional; the generic host hook tests do not require it.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("eagle_harness")

from eagle_harness.sglang import hooks, plugin  # noqa: E402

from sglang.srt.models.registry import ModelRegistry  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.srt.speculative.spec_registry import _REGISTRY  # noqa: E402


@pytest.fixture
def algorithm():
    snapshot = _REGISTRY.copy()
    _REGISTRY.pop("EAGLE_HARNESS", None)
    try:
        assert hooks.hook_problems() == []
        assert plugin.register()
        yield SpeculativeAlgorithm.from_string("EAGLE_HARNESS")
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(snapshot)


def test_registration_preserves_the_eagle_runtime_contract(algorithm):
    assert plugin.register()  # idempotent in every worker process
    assert algorithm.is_eagle()
    assert algorithm.uses_aux_hidden_states()
    assert algorithm.need_topk()
    assert algorithm.carries_draft_hidden_states()
    assert algorithm.supports_overlap
    assert "LlamaForCausalLMEagle4" in ModelRegistry.get_supported_archs()


def test_multimodal_requests_are_rejected_before_scheduling(algorithm):
    assert (
        algorithm.validate_request(None, has_mm_inputs=True, enable_overlap=True)
        is not None
    )
    assert (
        algorithm.validate_request(None, has_mm_inputs=False, enable_overlap=True)
        is None
    )


@pytest.mark.parametrize(
    "req, message",
    [
        (SimpleNamespace(custom_logit_processor="processor"), "custom logit"),
        (
            SimpleNamespace(
                sampling_params=SimpleNamespace(watermark={"enabled": True})
            ),
            "watermark",
        ),
    ],
)
def test_request_features_cannot_bypass_sampling(algorithm, req, message):
    assert message in algorithm.validate_request(
        req, has_mm_inputs=False, enable_overlap=True
    )


@pytest.mark.parametrize("device", ["npu", "cpu", "xpu", "mps"])
def test_unsupported_device_cannot_launch_the_cuda_drafter(algorithm, device):
    with pytest.raises(NotImplementedError, match="requires CUDA or ROCm"):
        algorithm.create_worker(SimpleNamespace(device=device))
