import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

LORA_ADAPTERS = [
    "Nutanix/Meta-Llama-3.1-8B-Instruct_lora_4_alpha_16",
    "philschmid/code-llama-3-1-8b-text-to-sql-lora",
    "pbevan11/llama-3.1-8b-ocr-correction",
]


class TestUpdateLoRAAdapters(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "meta-llama/Llama-3.1-8B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.expected_adapters = set([LORA_ADAPTERS[0]])
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-radix-cache",  # TODO: remove this after https://github.com/sgl-project/sglang/pull/7216 is merged.
                "--lora-paths",
                LORA_ADAPTERS[0],
                "--cuda-graph-max-bs",
                "1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def try_generate(self):
        lora_paths = list(self.expected_adapters)
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": ["SGLang is a"] * len(lora_paths),
                "lora_paths": lora_paths,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
        )
        print("=" * 100)
        print(f"Response: {response.json()}")
        self.assertTrue(response.ok)
        self.assertEqual(len(response.json()), len(self.expected_adapters))

    def try_load_lora_adapter(self, lora_name, lora_path=None):
        if lora_path is None:
            lora_path = lora_name

        self.expected_adapters.add(lora_name)
        response = requests.post(
            self.base_url + "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
        )

        print("=" * 100)
        print(f"Response: {response.json()}")
        self.assertTrue(response.ok)
        self.assertEqual(
            set(response.json()["loaded_adapters"]), self.expected_adapters
        )

    def try_unload_lora_adapter(self, lora_name):
        self.expected_adapters.remove(lora_name)
        response = requests.post(
            self.base_url + "/unload_lora_adapter",
            json={"lora_name": lora_name},
        )

        print("=" * 100)
        print(f"Response: {response.json()}")
        self.assertTrue(response.ok)
        self.assertEqual(
            set(response.json()["loaded_adapters"]), self.expected_adapters
        )

    def test_update_lora(self):
        # loaded adapters: 0
        self.try_generate()

        # loaded adapters: 0, 1
        self.try_load_lora_adapter(LORA_ADAPTERS[1])
        self.try_generate()

        # loaded adapters: 0, 1, 2
        self.try_load_lora_adapter(LORA_ADAPTERS[2])
        self.try_generate()

        # loaded adapters: 1, 2
        self.try_unload_lora_adapter(LORA_ADAPTERS[0])
        self.try_generate()

        # loaded adapters: 2
        self.try_unload_lora_adapter(LORA_ADAPTERS[1])
        self.try_generate()

        # loaded adapters: 1, 2
        self.try_load_lora_adapter(LORA_ADAPTERS[1])
        self.try_generate()

        # loaded adapters: 0, 1, 2
        self.try_load_lora_adapter(LORA_ADAPTERS[0])
        self.try_generate()


if __name__ == "__main__":
    unittest.main()
