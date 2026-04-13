import pytest
import torch

from sglang.srt.utils import get_available_device_memory


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="Not supported"
                )
            ],
        ),
        pytest.param(
            "npu",
            marks=[
                pytest.mark.skipif(not torch.npu.is_available(), reason="Not supported")
            ],
        ),
    ],
)
@pytest.mark.parametrize("distributed", [False, True])
@pytest.mark.parametrize("empty_cache", [False, True])
def test_get_available_device_memory(device, distributed, empty_cache):
    """
    Check if the available device memory can be queried.
    """
    assert get_available_device_memory(device, 0, distributed, empty_cache, None) > 0
