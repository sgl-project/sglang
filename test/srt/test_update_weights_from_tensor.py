import unittest

import sglang as sgl
import torch
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestReleaseGPUOccupation(unittest.TestCase):
    def test_release_and_resume_occupation(self):
        engine = sgl.Engine(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        param_name = "model.layers.2.self_attn.k_proj.weight"

        assert engine.get_weights_by_name(param_name, truncate_size=8) == ['TODO']

        new_tensor = torch.full((100,), 42)  # TODO
        engine.update_weights_from_tensor(param_name, new_tensor)

        assert engine.get_weights_by_name(param_name, truncate_size=8) == ['TODO']

        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
