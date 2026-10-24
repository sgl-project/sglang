from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.runtime_context import get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-b-test-cpu-intel")


@pytest.mark.parametrize(
    "backend",
    [MoeRunnerBackend.FLASHINFER_TRTLLM, MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED],
)
@pytest.mark.parametrize("name", ["w13_weight", "w2_weight"])
def test_bf16_reload_restores_checkpoint_shape_without_reallocating(backend, name):
    layer = SimpleNamespace(
        num_local_experts=2,
        hidden_size=2048,
        intermediate_size_per_partition=256,
        moe_runner_config=SimpleNamespace(is_gated=True),
    )
    shape = (2, 512, 2048) if name == "w13_weight" else (2, 2048, 256)
    weight = torch.nn.Parameter(
        torch.zeros(shape, dtype=torch.bfloat16).reshape(2, -1, 64),
        requires_grad=False,
    )
    address = weight.data_ptr()
    with get_flags().moe.override(runner_backend=backend):
        method = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
        method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            layer, weight, f"model.layers.0.mlp.experts.{name}"
        )
    assert tuple(weight.shape) == shape
    assert weight.data_ptr() == address
    checkpoint = torch.ones(shape, dtype=torch.bfloat16)
    weight.data.copy_(checkpoint)
    assert torch.equal(weight, checkpoint)


def test_other_moe_backends_leave_weight_layout_unchanged():
    weight = torch.nn.Parameter(torch.zeros(2, 128, 64), requires_grad=False)
    with get_flags().moe.override(runner_backend=MoeRunnerBackend.TRITON):
        method = UnquantizedFusedMoEMethod()
        method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
            SimpleNamespace(), weight, "model.layers.0.mlp.experts.w13_weight"
        )
    assert tuple(weight.shape) == (2, 128, 64)
