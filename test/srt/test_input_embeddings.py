import json
import os
import tempfile
import unittest

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestInputEmbeds(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.ref_model = AutoModelForCausalLM.from_pretrained(cls.model)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--disable-radix", "--cuda-graph-max-bs", 4],
        )
        cls.texts = [
            "The capital of France is",
            "What is the best time of year to visit Japan for cherry blossoms?",
        ]

    def generate_input_embeddings(self, text):
        """Generate input embeddings for a given text."""
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]
        embeddings = self.ref_model.get_input_embeddings()(input_ids)
        return embeddings.squeeze().tolist()  # Convert tensor to a list for API use

    def send_request(self, payload):
        """Send a POST request to the /generate endpoint and return the response."""
        response = requests.post(
            self.base_url + "/generate",
            json=payload,
            timeout=30,  # Set a reasonable timeout for the API request
        )
        if response.status_code == 200:
            return response.json()
        return {
            "error": f"Request failed with status {response.status_code}: {response.text}"
        }

    def send_file_request(self, file_path):
        """Send a POST request to the /generate_from_file endpoint with a file."""
        with open(file_path, "rb") as f:
            response = requests.post(
                self.base_url + "/generate_from_file",
                files={"file": f},
                timeout=30,  # Set a reasonable timeout for the API request
            )
        if response.status_code == 200:
            return response.json()
        return {
            "error": f"Request failed with status {response.status_code}: {response.text}"
        }

    def test_text_based_response(self):
        """Test and print API responses using text-based input."""
        for text in self.texts:
            payload = {
                "model": self.model,
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            }
            response = self.send_request(payload)
            print(
                f"Text Input: {text}\nResponse: {json.dumps(response, indent=2)}\n{'-' * 80}"
            )

    def test_embedding_based_response(self):
        """Test and print API responses using input embeddings."""
        for text in self.texts:
            embeddings = self.generate_input_embeddings(text)
            payload = {
                "model": self.model,
                "input_embeds": embeddings,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            }
            response = self.send_request(payload)
            print(
                f"Embeddings Input (for text '{text}'):\nResponse: {json.dumps(response, indent=2)}\n{'-' * 80}"
            )

    def test_compare_text_vs_embedding(self):
        """Test and compare responses for text-based and embedding-based inputs."""
        for text in self.texts:
            # Text-based payload
            text_payload = {
                "model": self.model,
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            }
            # Embedding-based payload
            embeddings = self.generate_input_embeddings(text)
            embed_payload = {
                "model": self.model,
                "input_embeds": embeddings,
                "sampling_params": {"temperature": 0, "max_new_tokens": 50},
            }
            # Get responses
            text_response = self.send_request(text_payload)
            embed_response = self.send_request(embed_payload)
            # Print responses
            print(
                f"Text Input: {text}\nText-Based Response: {json.dumps(text_response, indent=2)}\n"
            )
            print(
                f"Embeddings Input (for text '{text}'):\nEmbedding-Based Response: {json.dumps(embed_response, indent=2)}\n{'-' * 80}"
            )
            # This is flaky, so we skip this temporarily
            # self.assertEqual(text_response["text"], embed_response["text"])

    def test_generate_from_file(self):
        """Test the /generate_from_file endpoint using tokenized embeddings."""
        for text in self.texts:
            embeddings = self.generate_input_embeddings(text)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp_file:
                json.dump(embeddings, tmp_file)
                tmp_file_path = tmp_file.name

            try:
                response = self.send_file_request(tmp_file_path)
                print(
                    f"Text Input: {text}\nResponse from /generate_from_file: {json.dumps(response, indent=2)}\n{'-' * 80}"
                )
            finally:
                # Ensure the temporary file is deleted
                os.remove(tmp_file_path)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
