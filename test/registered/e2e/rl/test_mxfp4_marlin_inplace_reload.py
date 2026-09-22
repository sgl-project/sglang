import torch

from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_mxfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")


def test_mxfp4_marlin_inplace_reload_matches_initial_repack():
    """A reload must land in the original Parameters, byte-identical to a fresh repack."""
    device = torch.device("cuda")
    layer = torch.nn.Module()
    layer.orig_dtype = torch.bfloat16
    checkpoint_tensors = {
        "w13_weight": torch.randint(
            0, 256, (2, 256, 128), dtype=torch.uint8, device=device
        ),
        "w13_weight_scale": torch.randint(
            120, 136, (2, 256, 8), dtype=torch.uint8, device=device
        ),
        "w13_weight_bias": torch.randn(2, 256, dtype=torch.bfloat16, device=device),
        "w2_weight": torch.randint(
            0, 256, (2, 256, 64), dtype=torch.uint8, device=device
        ),
        "w2_weight_scale": torch.randint(
            120, 136, (2, 256, 4), dtype=torch.uint8, device=device
        ),
        "w2_weight_bias": torch.randn(2, 256, dtype=torch.bfloat16, device=device),
    }
    source_values = {
        name: tensor.clone() for name, tensor in checkpoint_tensors.items()
    }
    for name, tensor in checkpoint_tensors.items():
        param = torch.nn.Parameter(tensor, requires_grad=False)
        param.weight_loader = object()
        layer.register_parameter(name, param)

    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.use_marlin = True
    method.use_mega_moe = False
    layer._mxfp4_marlin_checkpoint_parameter_specs = {
        name: (tuple(tensor.shape), tensor.dtype)
        for name, tensor in checkpoint_tensors.items()
    }

    prepare_moe_mxfp4_layer_for_marlin(layer)

    expected = {
        name: getattr(layer, name).detach().clone()
        for name in method._MARLIN_CHECKPOINT_PARAMETER_NAMES
    }
    layer._mxfp4_marlin_runtime_parameter_specs = {
        name: (tuple(tensor.shape), tensor.dtype) for name, tensor in expected.items()
    }
    parameter_ids = {
        name: id(getattr(layer, name))
        for name in method._MARLIN_CHECKPOINT_PARAMETER_NAMES
    }
    storage_ptrs = {
        name: getattr(layer, name).untyped_storage().data_ptr()
        for name in method._MARLIN_CHECKPOINT_PARAMETER_NAMES
    }

    method.restore_weights_before_loading(layer)
    for name, source in source_values.items():
        param = getattr(layer, name)
        assert hasattr(param, "weight_loader"), name
        param.data.copy_(source)
    prepare_moe_mxfp4_layer_for_marlin(layer, reuse_parameter_storage=True)

    for name, expected_tensor in expected.items():
        actual = getattr(layer, name)
        assert id(actual) == parameter_ids[name]
        assert actual.untyped_storage().data_ptr() == storage_ptrs[name]
        assert torch.equal(actual, expected_tensor)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-x"]))
