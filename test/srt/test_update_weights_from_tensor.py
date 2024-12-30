import unittest

import torch

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestReleaseGPUOccupation(unittest.TestCase):
    def test_release_and_resume_occupation(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        param_name = "model.layers.2.self_attn.k_proj.weight"

        def _check_param(expect_values):
            actual_values = torch.tensor(engine.get_weights_by_name(param_name))[0, :5]
            assert torch.allclose(
                actual_values, torch.tensor(expect_values), atol=0.001
            ), f"{actual_values=}"

        _check_param([0.0571, -0.0114, 0.0444, 0.0215, -0.0149])

        new_tensor = torch.full((3072, 2048), 1.5)
        engine.update_weights_from_tensor(param_name, new_tensor)

        _check_param([1.5] * 5)

        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
