from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _StaticMxfp4Config:
    @staticmethod
    def get_name():
        return "mxfp4"

    @staticmethod
    def is_static_cfg():
        return True


def test_static_mxfp4_expert_zero_uses_per_expert_loader():
    load_impl = Mock()
    layer = SimpleNamespace(
        quant_config=_StaticMxfp4Config(),
        _map_global_expert_id_to_local_expert_id=lambda expert_id: expert_id,
        _weight_loader_impl=load_impl,
    )
    param = torch.nn.Parameter(torch.empty(1))
    loaded_weight = torch.empty((4, 8))

    with patch(
        "sglang.srt.layers.moe.fused_moe_triton.layer."
        "get_global_expert_location_metadata",
        return_value=None,
    ):
        FusedMoE.weight_loader(
            layer,
            param,
            loaded_weight,
            "experts.w1.weight",
            "w1",
            expert_id=0,
        )

    load_impl.assert_called_once_with(
        param=param,
        loaded_weight=loaded_weight,
        weight_name="experts.w1.weight",
        shard_id="w1",
        expert_id=0,
    )
