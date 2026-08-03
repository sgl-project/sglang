import subprocess
import sys

import pytest
import torch
from torch import nn

from sglang.srt.layers.moe.dwdp.common import restore_storage_rank
from sglang.srt.layers.moe.dwdp.tensor_schema import (
    DwdpTensorSchema,
    existing_tensor_names,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


class _ExpertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_weight = nn.Parameter(torch.empty(2, 8, 4))
        self.w2_weight = nn.Parameter(torch.empty(2, 4, 4))
        self.w13_weight_scale = nn.Parameter(torch.empty(2, 8, 1))


def test_schema_requires_explicit_partitioned_tensors():
    layer = _ExpertLayer()
    schema = DwdpTensorSchema(
        partitioned=("w13_weight", "w2_weight", "w13_weight_scale")
    )
    schema.validate(layer)

    del layer.w2_weight
    with pytest.raises(RuntimeError, match="w2_weight"):
        schema.validate(layer)


def test_main_weights_must_be_partitioned():
    layer = _ExpertLayer()
    schema = DwdpTensorSchema(
        main_weights=("w13_weight", "w2_weight"),
        partitioned=("w13_weight",),
    )
    with pytest.raises(ValueError, match="main weights"):
        schema.validate(layer)


def test_existing_tensor_names_preserves_contract_order():
    layer = _ExpertLayer()
    assert existing_tensor_names(
        layer,
        ("w2_weight", "missing", "w13_weight_scale"),
    ) == ("w2_weight", "w13_weight_scale")


def test_mxfp4_flattened_scale_restores_expert_axis():
    method = object.__new__(Mxfp4MoEMethod)
    method.num_experts = 2
    layer = _ExpertLayer()
    layer.w13_weight_scale = nn.Parameter(torch.empty(16, 4))
    restored = method.get_dwdp_tensor(layer, "w13_weight_scale")
    assert restored.shape == (2, 8, 4)
    assert (
        restored.untyped_storage().data_ptr()
        == layer.w13_weight_scale.untyped_storage().data_ptr()
    )


def test_gathered_scale_restores_original_storage_rank():
    flattened_storage = torch.empty(16, 4)
    gathered_expert_view = torch.empty(8, 8, 4)
    restored = restore_storage_rank(flattened_storage, gathered_expert_view)
    assert restored.shape == (64, 4)

    fp8_storage = torch.empty(2, 4, 3)
    fp8_gathered = torch.empty(8, 4, 3)
    assert restore_storage_rank(fp8_storage, fp8_gathered).shape == (8, 4, 3)


def test_dwdp_package_import_does_not_load_cuda_driver():
    code = """
import sys
import sglang.srt.layers.moe.dwdp
assert "cuda.bindings.driver" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
