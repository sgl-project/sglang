import gc
import unittest

import torch

import sglang as sgl
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_npu_ci(est_time=150, suite="nightly-2-npu-a3", nightly=True)


def _check_param(engine, param_name, expect_values):
    """Check if the first 5 values of the specified parameter match the expected values."""
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.002
    ), f"{actual_values=}"


class TestUpdateWeightsFromTensor(CustomTestCase):
    """Testcase: Verify weight update functionality with custom weight loader on NPU .

    [Test Category] Parameter
    [Test Target] --custom-weight-loader
    """

    def test_update_weights_from_tensor_load_format_custom(self):
        # Path to the custom weight loader function
        custom_loader_name = (
            "sglang.srt.model_executor.model_runner._model_load_weights_direct"
        )

        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            tp_size=1,
            custom_weight_loader=[custom_loader_name],
        )

        write_param_names = [
            f"model.layers.{i}.self_attn.qkv_proj.weight" for i in range(6, 16)
        ]
        read_param_names = [
            f"model.layers.{i}.self_attn.k_proj.weight" for i in range(6, 16)
        ]

        # Check original weight values before update
        _check_param(
            engine, read_param_names[0], [-0.0198, 0.0227, 0.0168, 0.0232, -0.0178]
        )

        # Create new tensor filled with constant value 1.5
        new_tensor = torch.full((3072, 2048), 1.5)

        # Update model weights using the custom loader
        engine.update_weights_from_tensor(
            [(name, new_tensor.clone()) for name in write_param_names],
            load_format=custom_loader_name,
        )

        # Verify weights are updated successfully
        for read_param_name in read_param_names[:3]:
            _check_param(engine, read_param_name, [1.5] * 5)

        # Shutdown engine and release resources
        engine.shutdown()

        del new_tensor
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
