import pytest

from sglang.multimodal_gen.test.server.accuracy_config import (
    ComponentType,
    get_skip_reason,
    should_skip_component,
)
from sglang.multimodal_gen.test.server.accuracy_testcase_configs import (
    ACCURACY_ONE_GPU_CASES,
    get_component_duplicate_skip_reason,
)
from sglang.multimodal_gen.test.server.accuracy_utils import (
    run_native_component_accuracy_case,
    run_text_encoder_accuracy_case,
)
from sglang.multimodal_gen.test.server.component_accuracy import AccuracyEngine

VAE_CHANNELS_LAST_3D_PARITY_CASE_IDS = {
    "wan2_1_t2v_1.3b",
}
VAE_CHANNELS_LAST_3D_PARITY_CASES = [
    case
    for case in ACCURACY_ONE_GPU_CASES
    if case.id in VAE_CHANNELS_LAST_3D_PARITY_CASE_IDS
]


@pytest.mark.parametrize("case", ACCURACY_ONE_GPU_CASES, ids=lambda case: case.id)
class TestComponentAccuracy1GPU:
    """1-GPU component accuracy suite."""

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
class TestVAEChannelsLast3DParity1GPU:
    """1-GPU VAE guard for channels_last_3d drift."""

    def test_vae_channels_last_3d_parity(self, case):
        AccuracyEngine.run_vae_channels_last_3d_parity(
            case,
            case.server_args.num_gpus,
        )
