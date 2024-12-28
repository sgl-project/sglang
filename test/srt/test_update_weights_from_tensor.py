import unittest

import sglang as sgl
import torch
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestReleaseGPUOccupation(unittest.TestCase):
    def test_release_and_resume_occupation(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        param_name = "model.layers.2.self_attn.k_proj.weight"

        param_value = _get_sample_sub_tensor(engine.get_weights_by_name(param_name))
        assert param_value == ['TODO'], f'{param_value=}'

        new_tensor = torch.full((100,), 42)  # TODO
        engine.update_weights_from_tensor(param_name, new_tensor)

        param_value = _get_sample_sub_tensor(engine.get_weights_by_name(param_name))
        assert param_value == ['TODO'], f'{param_value=}'

        engine.shutdown()


def _get_sample_sub_tensor(x):
    return torch.tensor(x)[0, :5]


if __name__ == "__main__":
    unittest.main()
