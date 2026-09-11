import sys
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from sglang.kernels.jit.utils.arch import ArchInfo
from sglang.srt.layers.attention.linear.kernels import kda_triton
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture
def inputs():
    mode = FakeTensorMode()

    def tensor(shape, dtype=torch.bfloat16, device="cuda:0"):
        return FakeTensor(
            mode, torch.empty(shape, dtype=dtype, device="meta"), torch.device(device)
        )

    values = dict(
        A_log=tensor((16,), torch.float32),
        dt_bias=tensor((2048,), torch.float32),
        q=tensor((1, 96, 16, 128)),
        k=tensor((1, 96, 16, 128)),
        v=tensor((1, 96, 16, 128)),
        a=tensor((96, 2048)),
        b=tensor((96, 16)),
        ssm_states=tensor((19, 16, 128, 128), torch.float32),
        cache_indices=tensor((16,), torch.int32),
        query_start_loc=tensor((17,), torch.int32),
        intermediate_states_buffer=tensor((19, 6, 16, 128, 128), torch.float32),
        intermediate_state_indices=tensor((16,), torch.int32),
        cache_steps=6,
        retrieve_parent_token=None,
        lower_bound=-5,
        dense_verify=True,
    )
    return values, tensor


def selected_warps(values, major=10, hip=False):
    with (
        patch.object(
            kda_triton, "get_jit_cuda_arch", return_value=ArchInfo(major, 3, "a")
        ),
        patch.object(kda_triton, "is_hip", return_value=hip),
        patch.object(kda_triton, "fused_sigmoid_gating_delta_rule_update") as launch,
    ):
        kda_triton.TritonKDAKernel().target_verify(**values)
    return launch.call_args.kwargs.get("num_warps", 1)


def test_dense_safe_gate_verify_selects_multiwarp(inputs):
    values, _ = inputs
    assert selected_warps(values) == 4


@pytest.mark.parametrize(
    "case",
    [
        "default_layout",
        "ragged",
        "standard_gate",
        "other_safe_gate",
        "draft_count",
        "tree",
        "ring",
        "fp16_gate",
        "bf16_state",
        "missing_snapshot",
        "bf16_snapshot",
        "strided_state",
        "strided_snapshot",
        "bf16_gate_parameter",
        "int64_indices",
        "head_width",
        "head_count",
        "cpu",
        "hopper",
        "hip",
    ],
)
def test_unvalidated_verify_contract_keeps_reference_launch(inputs, case):
    values, tensor = inputs
    major, hip = 10, False
    if case == "default_layout":
        values.pop("dense_verify")
    elif case == "ragged":
        values["dense_verify"] = False
    elif case == "standard_gate":
        values["lower_bound"] = None
    elif case == "other_safe_gate":
        values["lower_bound"] = -4
    elif case == "draft_count":
        values["cache_steps"] = 8
    elif case == "tree":
        values["retrieve_parent_token"] = tensor((16, 6), torch.int32)
    elif case == "ring":
        values["cache_ring"] = True
    elif case == "fp16_gate":
        values["a"] = tensor((96, 2048), torch.float16)
    elif case == "bf16_state":
        values["ssm_states"] = tensor((19, 16, 128, 128))
    elif case == "missing_snapshot":
        values["intermediate_states_buffer"] = None
    elif case == "bf16_snapshot":
        values["intermediate_states_buffer"] = tensor((19, 6, 16, 128, 128))
    elif case == "strided_state":
        values["ssm_states"] = values["ssm_states"].transpose(-1, -2)
    elif case == "strided_snapshot":
        values["intermediate_states_buffer"] = values[
            "intermediate_states_buffer"
        ].transpose(-1, -2)
    elif case == "bf16_gate_parameter":
        values["dt_bias"] = tensor((2048,))
    elif case == "int64_indices":
        values["intermediate_state_indices"] = tensor((16,), torch.int64)
    elif case == "head_width":
        values["v"] = tensor((1, 96, 16, 64))
    elif case == "head_count":
        values["k"] = tensor((1, 96, 8, 128))
    elif case == "cpu":
        values["q"] = tensor((1, 96, 16, 128), device="cpu")
    elif case == "hopper":
        major = 9
    elif case == "hip":
        hip = True
    assert selected_warps(values, major, hip) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
