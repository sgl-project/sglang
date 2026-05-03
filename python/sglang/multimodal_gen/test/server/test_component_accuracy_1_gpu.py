import pytest

from sglang.multimodal_gen.test.server.accuracy_config import (
    ComponentType,
    should_skip_component,
)
from sglang.multimodal_gen.test.server.accuracy_testcase_configs import (
    ACCURACY_ONE_GPU_CASES,
)
from sglang.multimodal_gen.test.server.accuracy_utils import (
    run_native_component_accuracy_case,
    run_text_encoder_accuracy_case,
)
from sglang.multimodal_gen.test.server.component_accuracy import AccuracyEngine


def _case_id(case):
    return case.id


VAE_CASES_1GPU = [
    case
    for case in ACCURACY_ONE_GPU_CASES
    if not should_skip_component(case, ComponentType.VAE)
]
TRANSFORMER_CASES_1GPU = [
    case
    for case in ACCURACY_ONE_GPU_CASES
    if not should_skip_component(case, ComponentType.TRANSFORMER)
]
TEXT_ENCODER_CASES_1GPU = [
    case
    for case in ACCURACY_ONE_GPU_CASES
    if not should_skip_component(case, ComponentType.TEXT_ENCODER)
]


class TestComponentAccuracy1GPU:
    """1-GPU component accuracy suite."""

    @pytest.mark.parametrize("case", VAE_CASES_1GPU, ids=_case_id)
    def test_vae_accuracy(self, case):
        run_native_component_accuracy_case(
            AccuracyEngine,
            case,
            ComponentType.VAE,
            "diffusers",
            case.server_args.num_gpus,
        )

    @pytest.mark.parametrize("case", TRANSFORMER_CASES_1GPU, ids=_case_id)
    def test_transformer_accuracy(self, case):
        run_native_component_accuracy_case(
            AccuracyEngine,
            case,
            ComponentType.TRANSFORMER,
            "diffusers",
            case.server_args.num_gpus,
        )

    @pytest.mark.parametrize("case", TEXT_ENCODER_CASES_1GPU, ids=_case_id)
    def test_encoder_accuracy(self, case):
        run_text_encoder_accuracy_case(
            AccuracyEngine,
            case,
            case.server_args.num_gpus,
        )
