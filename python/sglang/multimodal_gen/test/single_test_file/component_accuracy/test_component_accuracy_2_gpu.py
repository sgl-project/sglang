import pytest

from sglang.multimodal_gen.test.single_test_file.component_accuracy.config import (
    ComponentType,
    get_skip_reason,
    should_skip_component,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.engine import (
    AccuracyEngine,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.testcase_configs import (
    ACCURACY_TWO_GPU_CASES,
    get_component_duplicate_skip_reason,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    run_native_component_accuracy_case,
    run_text_encoder_accuracy_case,
)

VAE_CHANNELS_LAST_3D_PARITY_CASE_IDS = {
    "wan2_2_i2v_a14b_2gpu",
}
VAE_CHANNELS_LAST_3D_PARITY_CASES = [
    case
    for case in ACCURACY_TWO_GPU_CASES
    if case.id in VAE_CHANNELS_LAST_3D_PARITY_CASE_IDS
]


@pytest.mark.parametrize("case", ACCURACY_TWO_GPU_CASES, ids=lambda case: case.id)
class TestComponentAccuracy2GPU:
    """2-GPU component accuracy suite."""

    def test_vae_accuracy(self, case):
        if should_skip_component(case, ComponentType.VAE):
            pytest.skip(get_skip_reason(case, ComponentType.VAE))
        duplicate_reason = get_component_duplicate_skip_reason(case, ComponentType.VAE)
        if duplicate_reason:
            pytest.skip(duplicate_reason)
        run_native_component_accuracy_case(
            AccuracyEngine,
            case,
            ComponentType.VAE,
            "diffusers",
            case.server_args.num_gpus,
        )

    def test_transformer_accuracy(self, case):
        if should_skip_component(case, ComponentType.TRANSFORMER):
            pytest.skip(get_skip_reason(case, ComponentType.TRANSFORMER))
        duplicate_reason = get_component_duplicate_skip_reason(
            case, ComponentType.TRANSFORMER
        )
        if duplicate_reason:
            pytest.skip(duplicate_reason)
        run_native_component_accuracy_case(
            AccuracyEngine,
            case,
            ComponentType.TRANSFORMER,
            "diffusers",
            case.server_args.num_gpus,
        )

    def test_encoder_accuracy(self, case):
        if should_skip_component(case, ComponentType.TEXT_ENCODER):
            pytest.skip(get_skip_reason(case, ComponentType.TEXT_ENCODER))
        duplicate_reason = get_component_duplicate_skip_reason(
            case, ComponentType.TEXT_ENCODER
        )
        if duplicate_reason:
            pytest.skip(duplicate_reason)
        run_text_encoder_accuracy_case(
            AccuracyEngine,
            case,
            case.server_args.num_gpus,
        )


@pytest.mark.parametrize(
    "case", VAE_CHANNELS_LAST_3D_PARITY_CASES, ids=lambda case: case.id
)
class TestVAEChannelsLast3DParity2GPU:
    """2-GPU VAE guard for channels_last_3d drift."""

    def test_vae_channels_last_3d_parity(self, case):
        AccuracyEngine.run_vae_channels_last_3d_parity(
            case,
            case.server_args.num_gpus,
        )
