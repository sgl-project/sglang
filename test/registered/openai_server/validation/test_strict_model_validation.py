import unittest
import openai
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestStrictModelValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.served_model_name = "test_model_name"

        # Start server with a specific served model name
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=("--served-model-name", cls.served_model_name),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_invalid_model_name_chat(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model="wrong_model_name",
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertIn(f"The model `wrong_model_name` does not exist", str(cm.exception))
        self.assertIn(
            f"The served model name is `{self.served_model_name}`", str(cm.exception)
        )

    def test_invalid_model_name_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        with self.assertRaises(openai.BadRequestError) as cm:
            client.completions.create(
                model="wrong_model_name",
                prompt="hi",
            )
        self.assertIn(f"The model `wrong_model_name` does not exist", str(cm.exception))

    def test_valid_model_name_chat(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        # This should not raise an error
        client.chat.completions.create(
            model=self.served_model_name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )

    def test_lora_model_name(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        # LoRA syntax base:adapter should be allowed if base matches served_model_name
        # Note: This will likely fail later in the pipeline because the adapter doesn't exist,
        # but it should pass the initial model name validation.
        # However, since we are only testing the model name validation, we expect it to
        # either succeed or fail with a LoRA-related error, NOT a "model does not exist" error.

        try:
            client.chat.completions.create(
                model=f"{self.served_model_name}:some_adapter",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
        except openai.BadRequestError as e:
            self.assertNotIn("does not exist", str(e))
            self.assertIn("LoRA is not enabled", str(e))


if __name__ == "__main__":
    unittest.main()
