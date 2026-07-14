import base64
import json
import os
import unittest
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import soundfile as sf
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Define the models to be tested in a structured way.
MODELS = [SimpleNamespace(model="Qwen/Qwen3-Omni-30B-A3B-Instruct")]


# Set default mem_fraction_static to 0.8
DEFAULT_MEM_FRACTION_STATIC = 0.8


class MultiModalClient:
    """A client for interacting with a multimodal API that follows the OpenAI format."""

    def __init__(self, base_url: str, model_name: str, verbose: bool = False):
        """
        Initializes the client.

        Args:
            base_url: The base URL of the API server.
            model_name: The name of the model to use for requests.
            verbose: If True, prints detailed request and response info.
        """
        self.base_url = base_url
        self.model_name = model_name
        self.verbose = verbose

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
        timeout: int = 120,
    ) -> Optional[str]:
        """
        Sends a chat completion request with multimodal content.

        Args:
            messages: A list of message dictionaries.
            max_tokens: The maximum number of tokens to generate.
            temperature: The sampling temperature.
            timeout: The request timeout in seconds.

        Returns:
            The model's response content as a string, or None on failure.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                return content
            if self.verbose:
                print(
                    f"Request failed with status code {response.status_code}: {response.text}"
                )
            return None  # Request failed

        except (requests.Timeout, requests.RequestException):
            return None  # Request timed out or other network error
        except (requests.Timeout, requests.RequestException) as e:
            if self.verbose:
                print(f"Request failed with exception: {e}")
            return None  # Request timed out or other network error


class OmniBenchEvaluator:
    """Evaluator for the OmniBench dataset, handling data preparation, model inference, and result analysis."""

    def __init__(self, base_url: str, model_name: str, verbose: bool = False):
        """
        Initializes the evaluator.

        Args:
            base_url: The API server's base URL.
            model_name: The model name to be evaluated.
            verbose: If True, enables detailed logging during evaluation.
        """
        self.client = MultiModalClient(base_url, model_name, verbose)
        self.verbose = verbose
        self.results = []

    def encode_audio(self, audio_array: np.ndarray, sampling_rate: int) -> str:
        """Encodes a NumPy audio array into a base64 string."""
        buffered = BytesIO()
        sf.write(buffered, audio_array, sampling_rate, format="WAV")
        audio_bytes = buffered.getvalue()
        return base64.b64encode(audio_bytes).decode("utf-8")

    def encode_image(self, image: Image.Image) -> str:
        """Encodes a PIL Image object into a base64 string."""
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        return base64.b64encode(image_bytes).decode("utf-8")

    def prepare_message(self, sample: Dict[str, Any]) -> List[Dict]:
        """Constructs the multimodal message payload for the API from a dataset sample."""
        content = []
        if sample.get("audio") is not None:
            audio_data = sample["audio"]
            audio_base64 = self.encode_audio(
                audio_data["array"], audio_data["sampling_rate"]
            )
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
                }
            )

        if sample.get("image") is not None:
            image_base64 = self.encode_image(sample["image"])
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )

        question_text = self._format_question(sample)
        content.append({"type": "text", "text": question_text})

        return [{"role": "user", "content": content}]

    def _format_question(self, sample: Dict[str, Any]) -> str:
        """Formats the question and options into a single string."""
        question_text = sample["question"]
        if sample.get("options"):
            options_text = "\n".join(
                [f"{chr(65+i)}. {opt}" for i, opt in enumerate(sample["options"])]
            )
            question_text += (
                f"\n\nOptions:\n{options_text}\n\nPlease select the correct answer."
            )
        return question_text

    def extract_answer(self, response: str, options: List[str]) -> str:
        """Extracts the most likely answer from the model's free-form text response."""
        if not response:
            return ""
        response_upper = response.strip().upper()

        for i, opt in enumerate(options):
            option_letter = chr(65 + i)
            if (
                response_upper.startswith(option_letter)
                or f"({option_letter})" in response_upper
                or f"[{option_letter}]" in response_upper
            ):
                return opt

        for opt in options:
            if opt.lower() in response.lower():
                return opt

        return response

    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Checks if the predicted answer is correct against the ground truth."""
        if not predicted:
            return False
        predicted = predicted.strip()
        ground_truth = ground_truth.strip()
        return (
            predicted.lower() == ground_truth.lower()
            or ground_truth.lower() in predicted.lower()
        )

    def evaluate(self, dataset, num_samples: int = 10000) -> Dict[str, Any]:
        """Runs the evaluation loop over the dataset."""
        correct, total, errors = 0, 0, 0
        samples = dataset["train"].select(
            range(min(num_samples, len(dataset["train"])))
        )
        for idx, sample in enumerate(
            tqdm(samples, desc="Evaluating", disable=self.verbose)
        ):
            try:
                messages = self.prepare_message(sample)
                response = self.client.chat(messages, timeout=120)

                if response is None:
                    errors += 1
                    total += 1
                    self.results.append(
                        {
                            "index": idx,
                            "error": "API request failed",
                            "is_correct": False,
                        }
                    )
                    continue

                predicted_answer = self.extract_answer(
                    response, sample.get("options", [])
                )
                is_correct = self.check_answer(predicted_answer, sample["answer"])

                if is_correct:
                    correct += 1
                total += 1

                self.results.append(
                    {
                        "index": sample.get("index", idx),
                        "task_type": sample.get("task type", "unknown"),
                        "audio_type": sample.get("audio type", "unknown"),
                        "question": sample["question"],
                        "ground_truth": sample["answer"],
                        "predicted": predicted_answer,
                        "full_response": response,
                        "is_correct": is_correct,
                        "options": sample.get("options", []),
                    }
                )
            except Exception as e:
                errors += 1
                total += 1
                self.results.append(
                    {"index": idx, "error": str(e), "is_correct": False}
                )
                continue

        return self._calculate_metrics(correct, total, errors)

    def _calculate_metrics(
        self, correct: int, total: int, errors: int
    ) -> Dict[str, Any]:
        """Calculates overall and per-category accuracy metrics."""
        accuracy = correct / total if total > 0 else 0
        task_stats = {}
        audio_stats = {}

        for r in self.results:
            if "task_type" in r:
                t = r["task_type"]
                task_stats.setdefault(t, {"correct": 0, "total": 0})
                task_stats[t]["total"] += 1
                if r.get("is_correct"):
                    task_stats[t]["correct"] += 1
            if "audio_type" in r:
                a = r["audio_type"]
                audio_stats.setdefault(a, {"correct": 0, "total": 0})
                audio_stats[a]["total"] += 1
                if r.get("is_correct"):
                    audio_stats[a]["correct"] += 1

        for stats in [task_stats, audio_stats]:
            for k in stats:
                stats[k]["accuracy"] = (
                    stats[k]["correct"] / stats[k]["total"]
                    if stats[k]["total"] > 0
                    else 0
                )

        return {
            "overall_accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": errors,
            "task_type_stats": task_stats,
            "audio_type_stats": audio_stats,
        }

    def save_results(self, output_dir: str):
        """Saves the detailed evaluation results to a JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "detailed_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

    def print_metrics(self, metrics: Dict[str, Any]):
        """Prints a formatted summary of the evaluation metrics."""
        print("\n" + "=" * 70 + "\n📈 Evaluation Summary\n" + "=" * 70)
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(
            f"Correct/Total: {metrics['correct']}/{metrics['total']}, Errors: {metrics['errors']}"
        )

        if metrics["task_type_stats"]:
            print("\n📋 By Task Type:")
            for t, s in sorted(metrics["task_type_stats"].items()):
                print(
                    f"  • {t:30s}: {s['accuracy']:6.2%} ({s['correct']:3d}/{s['total']:3d})"
                )

        if metrics["audio_type_stats"]:
            print("\n🔊 By Audio Type:")
            for a, s in sorted(metrics["audio_type_stats"].items()):
                print(
                    f"  • {a:30s}: {s['accuracy']:6.2%} ({s['correct']:3d}/{s['total']:3d})"
                )
        print("=" * 70)


class TestQwen3OmniOmniBench(CustomTestCase):
    """A comprehensive test for OmniBench."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODELS[0].model
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                DEFAULT_MEM_FRACTION_STATIC,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_omnibench(self):
        """Tests the evaluation on 100 samples, checking for a reasonable accuracy."""
        dataset = load_dataset("m-a-p/OmniBench")
        evaluator = OmniBenchEvaluator(self.base_url, self.model, verbose=False)
        metrics = evaluator.evaluate(dataset, num_samples=100)
        evaluator.print_metrics(metrics)
        output_dir = "./tmp/qwen3_omni_omnibench"
        evaluator.save_results(output_dir)

        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self.assertGreater(
            metrics["overall_accuracy"], 0.2, f"Accuracy should be > 0.2"
        )
        self.assertEqual(metrics["total"], 100, f"Should evaluate 100 samples")


class TestAudioUrlMode(CustomTestCase):
    """Tests the server's ability to handle audio provided via an HTTP URL."""

    @classmethod
    def setUpClass(cls):
        cls.model = MODELS[0].model
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_audio_url_mode(self):
        """Sends a request with an audio URL and verifies a non-empty response is received."""
        client = MultiModalClient(self.base_url, self.model, verbose=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/asr_en.wav"
                        },
                    },
                    {"type": "text", "text": "How long is this audio file?"},
                ],
            }
        ]

        response = client.chat(messages)
        self.assertIsNotNone(response, "Response should not be None")
        print(f"Response from audio URL test: {response}")


if __name__ == "__main__":
    print("\n" + "=" * 70 + "\n🧪 OmniBench Evaluation Test Suite\n" + "=" * 70)
    print("\nAvailable Tests:")
    print("  - TestAudioUrlMode (tests HTTP audio URL input)")
    print("  - TestQwen3OmniOmniBench (100 samples)")
    print("\nUsage Example:")
    print("  python test_omni_models.py TestQwen3OmniOmniBench")
    print("  # To run all tests:")
    print("  python test_omni_models.py")
    print("=" * 70 + "\n")

    unittest.main(verbosity=2)
