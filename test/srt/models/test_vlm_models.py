import unittest
import os
from types import SimpleNamespace
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    add_common_sglang_args_and_parse
)
import argparse
import time

import openai

from mmmu_utils.data_utils import save_json
from mmmu_utils.eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm

from sglang.srt.utils import kill_process_tree

# CI VLM model for testing
CI_MODELS = [
    SimpleNamespace(model="google/gemma-3-27b-it", chat_template="gemma-it", mmmu_accuracy=0.39),
    SimpleNamespace(model="Qwen/Qwen2.5-VL-7B-Instruct", chat_template="qwen2-vl", mmmu_accuracy=0.45),
    SimpleNamespace(model="meta-llama/Llama-3.2-11B-Vision-Instruct", chat_template="llama_3_vision", mmmu_accuracy=0.31),
    SimpleNamespace(model="openbmb/MiniCPM-V-2_6", chat_template="minicpmv", mmmu_accuracy=0.4)
]

class TestVLMModels(CustomTestCase):

    @classmethod
    def setUpClass(cls):  # Fixed method name (was setUPClass)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.time_out = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    def eval_mmmu(self):
        """Evaluate model performance on MMMU benchmark."""
        parser = argparse.ArgumentParser()
        EvalArgs.add_cli_args(parser)
        # Fixed duplicate parsing - only use one parsing method
        args = add_common_sglang_args_and_parse(parser)
        eval_args = EvalArgs.from_cli_args(args)

        out_samples = dict()
        sampling_params = get_sampling_params(eval_args)
        samples = prepare_samples(eval_args)
        answer_dict = {}

        # Create OpenAI client for API access
        client = openai.Client(
            api_key=self.api_key,
            base_url=f"{self.base_url}/v1"
        )

        start = time.time()
        for i, sample in enumerate(tqdm(samples)):
            prompt = sample["final_input_prompt"]
            
            # Handle prompt splitting more robustly
            try:
                prefix = prompt.split("<")[0]
                suffix = prompt.split(">")[1]
            except IndexError:
                print(f"Warning: Could not properly split prompt for sample {i}. Using full prompt.")
                prefix = prompt
                suffix = ""
                
            image = sample.get("image")
            image_path = sample.get("image_path")
            
            if image is None or image_path is None:
                print(f"Warning: Sample {i} missing image data. Skipping.")
                continue
                
            try:
                response = client.chat.completions.create(
                    model="default",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prefix},
                                {"type": "image_url", "image_url": {"url": image_path}},
                                {"type": "text", "text": suffix},
                            ],
                        }
                    ],
                    temperature=0,
                    max_completion_tokens=sampling_params["max_new_tokens"],
                    max_tokens=sampling_params["max_new_tokens"],
                )
                response_text = response.choices[0].message.content
                process_result(response_text, sample, answer_dict, out_samples)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Add error handling - continue with next sample

        benchmark_time = time.time() - start
        print(f"Benchmark time: {benchmark_time:.2f} seconds")

        output_path = os.path.join(".", "val_sglang.json")
        save_json(output_path, out_samples)
        return eval_result(model_answer_path=output_path, answer_dict=answer_dict)

    def test_ci_models(self):
        """Test CI models against MMMU benchmark."""
        for cli_model in CI_MODELS:
            print(f"\nTesting model: {cli_model.model}")
            
            process = None
            mmmu_accuracy = 0  # Initialize to handle potential exceptions
            
            try:
                # Launch server for testing
                process = popen_launch_server(
                    cli_model.model,
                    base_url=self.base_url,
                    timeout=self.time_out,
                    api_key=self.api_key,
                    other_args=[
                        "--chat-template",
                        cli_model.chat_template,
                        "--trust-remote-code",
                    ],
                )

                # Run evaluation
                mmmu_accuracy = self.eval_mmmu()
                print(f"Model {cli_model.model} achieved accuracy: {mmmu_accuracy:.4f}")
                
                # Assert performance meets expected threshold
                self.assertGreaterEqual(
                    mmmu_accuracy, 
                    cli_model.mmmu_accuracy,
                    f"Model {cli_model.model} accuracy ({mmmu_accuracy:.4f}) below expected threshold ({cli_model.mmmu_accuracy:.4f})"
                )
                
            except Exception as e:
                print(f"Error testing {cli_model.model}: {e}")
                self.fail(f"Test failed for {cli_model.model}: {e}")
                
            finally:
                # Ensure process cleanup happens regardless of success/failure
                if process is not None and process.poll() is None:
                    print(f"Cleaning up process {process.pid}")
                    try:
                        kill_process_tree(process.pid)
                    except Exception as e:
                        print(f"Error killing process: {e}")

if __name__ == "__main__":
    unittest.main()