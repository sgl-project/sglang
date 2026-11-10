import pytest

from sglang.multimodal_gen.test.single_test_file.component_accuracy.suite import (
    ComponentAccuracySuite,
    VAEChannelsLast3DParitySuite,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.testcase_configs import (
    ACCURACY_ONE_GPU_CASES,
)

VAE_CHANNELS_LAST_3D_PARITY_CASES = [
    case for case in ACCURACY_ONE_GPU_CASES if case.id == "wan2_1_t2v_1.3b"
]


@pytest.mark.parametrize("case", ACCURACY_ONE_GPU_CASES, ids=lambda case: case.id)
class TestComponentAccuracy1GPU(ComponentAccuracySuite):
    """1-GPU component accuracy suite."""


@pytest.mark.parametrize(
    "case", VAE_CHANNELS_LAST_3D_PARITY_CASES, ids=lambda case: case.id
)
class TestVAEChannelsLast3DParity1GPU(VAEChannelsLast3DParitySuite):
    """1-GPU VAE guard for channels_last_3d drift."""
