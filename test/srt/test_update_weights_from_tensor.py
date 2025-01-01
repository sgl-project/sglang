import time
import unittest

import sglang as sgl
import torch
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestUpdateWeightsFromTensor(unittest.TestCase):
    def test_update_weights_from_tensor(self):
        engine = sgl.Engine(model_path=DEFAULT_MODEL_NAME_FOR_TEST)

        param_name = "model.layers.2.self_attn.k_proj.weight"

        _check_param(engine, param_name, [0.0078, -0.0060, -0.0134, -0.0403, -0.0103])

        new_tensor = torch.full((3072, 2048), 1.5)

        time_start = time.time()
        engine.update_weights_from_tensor(param_name, new_tensor)
        print(f'Time delta: {time.time() - time_start:.03f}')

        _check_param(engine, param_name, [1.5] * 5)

        engine.shutdown()


def _check_param(engine, param_name, expect_values):
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.001
    ), f"{actual_values=}"


if __name__ == "__main__":
    unittest.main()
