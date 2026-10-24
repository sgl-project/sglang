import pytest

from sglang.multimodal_gen.test.single_test_file.component_accuracy.suite import (
    ComponentAccuracySuite,
    VAEChannelsLast3DParitySuite,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.testcase_configs import (
    ACCURACY_TWO_GPU_CASES,
)

VAE_CHANNELS_LAST_3D_PARITY_CASES = [
    case for case in ACCURACY_TWO_GPU_CASES if case.id == "wan2_2_i2v_a14b_2gpu"
]


@pytest.mark.parametrize("case", ACCURACY_TWO_GPU_CASES, ids=lambda case: case.id)
class TestComponentAccuracy2GPU(ComponentAccuracySuite):
    """2-GPU component accuracy suite."""


@pytest.mark.parametrize(
    "case", VAE_CHANNELS_LAST_3D_PARITY_CASES, ids=lambda case: case.id
)
class TestVAEChannelsLast3DParity2GPU(VAEChannelsLast3DParitySuite):
    """2-GPU VAE guard for channels_last_3d drift."""
