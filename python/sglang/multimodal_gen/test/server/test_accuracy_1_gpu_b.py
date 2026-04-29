import pytest

from sglang.multimodal_gen.test.server.accuracy_config import (
    ComponentType,
    get_skip_reason,
    should_skip_component,
)
from sglang.multimodal_gen.test.server.accuracy_utils import (
    run_native_component_accuracy_case,
    run_text_encoder_accuracy_case,
)
from sglang.multimodal_gen.test.server.component_accuracy import AccuracyEngine
from sglang.multimodal_gen.test.server.testcase_configs import ACCURACY_ONE_GPU_CASES_B


@pytest.mark.parametrize("case", ACCURACY_ONE_GPU_CASES_B, ids=lambda x: x.id)
class TestAccuracy1GPU_B:
    """1-GPU Component Accuracy Suite (Set B)."""

    def test_vae_accuracy(self, case):
        if should_skip_component(case, ComponentType.VAE):
            pytest.skip(get_skip_reason(case, ComponentType.VAE))
        run_native_component_accuracy_case(
            AccuracyEngine, case, ComponentType.VAE, "diffusers", 1
        )

    def test_transformer_accuracy(self, case):
        if should_skip_component(case, ComponentType.TRANSFORMER):
            pytest.skip(get_skip_reason(case, ComponentType.TRANSFORMER))
        run_native_component_accuracy_case(
            AccuracyEngine, case, ComponentType.TRANSFORMER, "diffusers", 1
        )

    def test_encoder_accuracy(self, case):
        if should_skip_component(case, ComponentType.TEXT_ENCODER):
            pytest.skip(get_skip_reason(case, ComponentType.TEXT_ENCODER))
        run_text_encoder_accuracy_case(AccuracyEngine, case, 1)
