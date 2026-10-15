import pytest

from sglang.multimodal_gen.test.single_test_file.component_accuracy.config import (
    ComponentType,
    get_skip_reason,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.engine import (
    AccuracyEngine,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.testcase_configs import (
    get_component_duplicate_skip_reason,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    run_native_component_accuracy_case,
    run_text_encoder_accuracy_case,
)


def _skip_component_if_needed(case, component):
    reason = get_skip_reason(case, component)
    if reason is not None:
        pytest.skip(reason)
    duplicate_reason = get_component_duplicate_skip_reason(case, component)
    if duplicate_reason:
        pytest.skip(duplicate_reason)


class ComponentAccuracySuite:
    def test_vae_accuracy(self, case):
        _skip_component_if_needed(case, ComponentType.VAE)
        run_native_component_accuracy_case(
            AccuracyEngine,
            case,
            ComponentType.VAE,
            "diffusers",
            case.server_args.num_gpus,
        )

    def test_transformer_accuracy(self, case):
        _skip_component_if_needed(case, ComponentType.TRANSFORMER)
        run_native_component_accuracy_case(
            AccuracyEngine,
            case,
            ComponentType.TRANSFORMER,
            "diffusers",
            case.server_args.num_gpus,
        )

    def test_encoder_accuracy(self, case):
        _skip_component_if_needed(case, ComponentType.TEXT_ENCODER)
        run_text_encoder_accuracy_case(
            AccuracyEngine,
            case,
            case.server_args.num_gpus,
        )


class VAEChannelsLast3DParitySuite:
    def test_vae_channels_last_3d_parity(self, case):
        AccuracyEngine.run_vae_channels_last_3d_parity(
            case,
            case.server_args.num_gpus,
        )
