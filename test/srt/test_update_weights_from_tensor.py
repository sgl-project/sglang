import time
import unittest

import sglang as sgl
import torch
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestUpdateWeightsFromTensor(unittest.TestCase):
    def test_update_weights_from_tensor_load_format_default(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        param_names = [f"model.layers.{i}.mlp.up_proj.weight" for i in range(6, 16)]

        _check_param(engine, param_names[0], [0.0087, -0.0214, -0.0004, 0.0039, 0.0110])

        new_tensor = torch.full((16384, 2048), 1.5)

        time_start = time.time()
        engine.update_weights_from_tensor([(x, new_tensor) for x in param_names])
        print(f"Time delta: {time.time() - time_start:.03f}")

        for param_name in param_names[:3]:
            _check_param(engine, param_name, [1.5] * 5)

        engine.shutdown()

    def test_update_weights_from_tensor_load_format_direct(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        write_param_name = f"model.layers.6.self_attn.qkv_proj.weight"
        read_param_name = f"model.layers.6.self_attn.k_proj.weight"

        _check_param(engine, read_param_name, [-0.0198, 0.0227, 0.0168, 0.0232, -0.0178])

        new_tensor = torch.full((3072, 2048), 1.5)
        engine.update_weights_from_tensor([(write_param_name, new_tensor)], load_format='direct')

        _check_param(engine, read_param_name, [1.5] * 5)

        engine.shutdown()


def _check_param(engine, param_name, expect_values):
    actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
    assert torch.allclose(
        actual_values, torch.tensor(expect_values), atol=0.002
    ), f"{actual_values=}"


if __name__ == "__main__":
    unittest.main()
