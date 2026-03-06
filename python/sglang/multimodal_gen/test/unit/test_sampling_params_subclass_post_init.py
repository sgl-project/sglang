import unittest

from sglang.multimodal_gen.configs.sample.diffusers_generic import (
    DiffusersGenericSamplingParams,
)
from sglang.multimodal_gen.configs.sample.flux import FluxSamplingParams


class TestSamplingParamsSubclassPostInit(unittest.TestCase):
    def test_flux_defaults_resolution_when_not_provided(self):
        params = FluxSamplingParams()

        self.assertEqual(params.height, 1024)
        self.assertEqual(params.width, 1024)
        self.assertTrue(params.height_not_provided)
        self.assertTrue(params.width_not_provided)

    def test_flux_preserves_user_resolution(self):
        params = FluxSamplingParams(height=640, width=768)

        self.assertEqual(params.height, 640)
        self.assertEqual(params.width, 768)
        self.assertFalse(params.height_not_provided)
        self.assertFalse(params.width_not_provided)

    def test_diffusers_generic_calls_base_post_init(self):
        with self.assertRaises(AssertionError):
            DiffusersGenericSamplingParams(num_frames=0)


if __name__ == "__main__":
    unittest.main()
