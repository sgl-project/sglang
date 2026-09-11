"""End-to-end negative-branch contracts for OpenAI-compatible LoRA routing."""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="weekly", runner_config="1-gpu-large")


def setup_class(cls, *, enable_lora):
    """Start the shared server for one routing contract."""
    cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    cls.base_url = DEFAULT_URL_FOR_TEST

    other_args = [
        "--max-running-requests",
        "10",
        "--disable-radix-cache",
    ]

    if enable_lora:
        other_args.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                "8",
                "--lora-target-modules",
                "q_proj",
            ]
        )

    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")


class TestLoRAOpenAICompatible(CustomTestCase):
    """Verify that ``model:adapter`` reaches the LoRA registry."""

    @classmethod
    def setUpClass(cls):
        setup_class(cls, enable_lora=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_unknown_model_adapter_is_rejected(self):
        """An adapter suffix must not be ignored and routed to the base model."""
        with self.assertRaises(openai.APIError) as context:
            self.client.chat.completions.create(
                model=f"{self.model}:nonexistent",
                messages=[{"role": "user", "content": "Hello!"}],
                max_tokens=1,
            )

        error_message = str(context.exception)
        self.assertIn("never been loaded", error_message)
        self.assertIn("nonexistent", error_message)


class TestLoRADisabledError(CustomTestCase):
    """Verify the disabled-LoRA request contract."""

    @classmethod
    def setUpClass(cls):
        setup_class(cls, enable_lora=False)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_lora_disabled_error(self):
        """A requested adapter must fail clearly when LoRA is disabled."""
        with self.assertRaises(openai.APIError) as context:
            self.client.chat.completions.create(
                model=f"{self.model}:tool_calling",
                messages=[
                    {"role": "user", "content": "What tools do you have available?"}
                ],
                max_tokens=1,
            )

        error_message = str(context.exception)
        self.assertIn("LoRA", error_message)
        self.assertIn("not enabled", error_message)
        self.assertIn("tool_calling", error_message)
        self.assertIn("--enable-lora", error_message)


if __name__ == "__main__":
    unittest.main()
