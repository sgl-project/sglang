import json
import unittest

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestEngineUpdateWeightsFromDisk(unittest.TestCase):
    def setUp(self):
        self.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        # Initialize the engine in offline mode.
        self.engine = sgl.Engine(model_path=self.model)

    def tearDown(self):
        self.engine.shutdown()

    def run_decode(self):
        # Use a list of prompts with the given sampling parameters.
        prompts = ["The capital of France is"]
        sampling_params = {"temperature": 0, "max_new_tokens": 32}
        outputs = self.engine.generate(prompts, sampling_params)
        print("=" * 100)
        print(f"Prompt: {prompts[0]}\nGenerated text: {outputs[0]['text']}")
        return outputs[0]["text"]

    def run_update_weights(self, model_path):
        # Update weights from disk via the engine.
        ret = self.engine.update_weights_from_disk(model_path)
        print(json.dumps(ret))
        return ret

    def test_update_weights(self):
        # Capture the original decode output.
        origin_response = self.run_decode()

        # Update weights to a new model (removing "-Instruct").
        new_model_path = self.model.replace("-Instruct", "")
        ret = self.run_update_weights(new_model_path)
        self.assertTrue(ret[0])  # Use tuple indexing

        # Verify that decode output changes after weight update.
        updated_response = self.run_decode()
        self.assertNotEqual(origin_response[:32], updated_response[:32])

        # Revert weights back to the original model.
        ret = self.run_update_weights(self.model)
        self.assertTrue(ret[0])  # Use tuple indexing

        # Verify that decode output returns to its original state.
        reverted_response = self.run_decode()
        self.assertEqual(origin_response[:32], reverted_response[:32])

    def test_update_weights_unexist_model(self):
        # Capture the original decode output.
        origin_response = self.run_decode()

        # Attempt to update weights using a non-existent model.
        new_model_path = self.model.replace("-Instruct", "wrong")
        ret = self.run_update_weights(new_model_path)
        self.assertFalse(ret[0])  # Use tuple indexing

        # Verify that decode output remains unchanged.
        updated_response = self.run_decode()
        self.assertEqual(origin_response[:32], updated_response[:32])


if __name__ == "__main__":
    unittest.main()
