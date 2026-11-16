from unittest.mock import patch

from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.platforms import current_platform


def _init_tensor_group_with_cuda_flag(is_cuda: bool) -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeCoordinator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    with (
        patch.object(current_platform, "is_cuda", return_value=is_cuda),
        patch.object(parallel_state, "GroupCoordinator", FakeCoordinator),
    ):
        parallel_state.init_parallel_group_coordinator(
            [[0, 1]], local_rank=0, backend="mccl", parallel_mode="tensor"
        )
    return captured


def test_tensor_group_uses_native_collective_outside_cuda():
    kwargs = _init_tensor_group_with_cuda_flag(is_cuda=False)
    assert kwargs["use_srt_custom_allreduce"] is False


def test_tensor_group_keeps_custom_collective_on_cuda():
    kwargs = _init_tensor_group_with_cuda_flag(is_cuda=True)
    assert kwargs["use_srt_custom_allreduce"] is True
