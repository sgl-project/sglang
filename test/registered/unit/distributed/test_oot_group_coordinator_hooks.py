import sys
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")


def _platform(out_of_tree: bool, device: str = "cpu", pynccl: bool = False) -> Mock:
    platform = Mock(spec=SRTPlatform)
    platform.is_out_of_tree.return_value = out_of_tree
    platform.get_group_coordinator_device.side_effect = lambda local_rank: torch.device(
        device
    )
    platform.capabilities = PlatformCapabilities(supports_pynccl=pynccl)
    return platform


def _no_in_tree_device():
    return (
        patch.object(parallel_state, "is_cuda_alike", return_value=False),
        patch.object(parallel_state, "_is_npu", False),
        patch.object(parallel_state, "_is_xpu", False),
        patch.object(parallel_state, "_is_musa", False),
    )


def _with_platform(platform):
    return patch.object(
        parallel_state.platforms, "current_platform", platform, create=True
    )


def test_out_of_tree_platform_chooses_group_device():
    platform = _platform(out_of_tree=True, device="meta")
    p1, p2, p3, p4 = _no_in_tree_device()
    with p1, p2, p3, p4, _with_platform(platform):
        device = parallel_state._group_coordinator_device(3)
    assert device == torch.device("meta")
    platform.get_group_coordinator_device.assert_called_once_with(3)


def test_in_tree_unknown_device_falls_back_to_cpu():
    platform = _platform(out_of_tree=False)
    p1, p2, p3, p4 = _no_in_tree_device()
    with p1, p2, p3, p4, _with_platform(platform):
        device = parallel_state._group_coordinator_device(3)
    assert device == torch.device("cpu")
    platform.get_group_coordinator_device.assert_not_called()


def test_base_platform_defaults():
    platform = SRTPlatform()
    assert platform.get_group_coordinator_device(0) == torch.device("cpu")
    assert platform.capabilities.supports_pynccl is False


@pytest.mark.parametrize(
    "out_of_tree, platform_pynccl, explicit, expected",
    [
        (True, False, None, False),
        (True, True, None, True),
        (False, False, None, True),
        (True, False, True, True),
        (False, True, False, False),
    ],
)
def test_pynccl_default(out_of_tree, platform_pynccl, explicit, expected):
    platform = _platform(out_of_tree=out_of_tree, pynccl=platform_pynccl)
    with (
        _with_platform(platform),
        patch.object(parallel_state, "_is_npu", False),
        patch.object(parallel_state, "_is_xpu", False),
        patch.object(parallel_state, "GroupCoordinator") as coordinator,
    ):
        parallel_state.init_model_parallel_group(
            group_ranks=[[0, 1]],
            local_rank=0,
            backend="gloo",
            use_pynccl=explicit,
        )
    assert coordinator.call_args.kwargs["use_pynccl"] is expected


@pytest.mark.parametrize("out_of_tree, key", [(True, "PrivateUse1"), (False, "CUDA")])
def test_custom_op_dispatch_key(out_of_tree, key):
    from torch.library import Library

    from sglang.srt.utils import common

    lib_name = f"sglang_oot_test_{int(out_of_tree)}"
    lib = Library(lib_name, "FRAGMENT")

    def op(x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    with (
        patch.object(common, "current_platform", _platform(out_of_tree=out_of_tree)),
        patch.object(common, "is_npu", return_value=False),
        patch.object(common, "is_xpu", return_value=False),
        patch.object(common, "is_musa", return_value=False),
    ):
        common.direct_register_custom_op("op", op, mutates_args=[], target_lib=lib)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(f"{lib_name}::op", key)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
