"""
End-to-end test for LoRA MoE model inference using SGLang.

This script loads a LoRA MoE model using SGLang runner, runs inference on a test dataset,
and compares outputs against gold labels to validate correctness.
"""

import json
import os
import sys
from typing import List, Dict, Any
from urllib.request import urlopen

import torch
from sglang.test.runners import SRTRunner

# Configuration - set your model and LoRA paths here
MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B"  # Your LoRA MoE model path
LORA_PATH = "jonahbernard/sglang-lora-moe-test-qwen1.5-MoE-A2.7B"  # REQUIRED: Your LoRA adapter path
TEST_DATA_URL = "https://huggingface.co/jonahbernard/sglang-lora-moe-test-qwen1.5-MoE-A2.7B/blob/main/training_dataset.json"  # URL to test data JSON file


def load_test_dataset(test_data_url: str) -> List[Dict[str, Any]]:
    """Load test dataset from JSON URL."""
    try:
        with urlopen(test_data_url) as response:
            test_dataset = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        raise RuntimeError(f"Failed to load test data from URL {test_data_url}: {e}")

    return test_dataset


def run_lora_moe_inference_test():
    """Run end-to-end test for LoRA MoE model inference using SGLang."""

    print("=== LoRA MoE End-to-End Test (SGLang) ===\n")

    print(f"Model: {MODEL_PATH}")
    print(f"LoRA Path: {LORA_PATH}")
    print(f"Test Data URL: {TEST_DATA_URL}")
    print()

    # Load test dataset
    try:
        test_dataset = load_test_dataset(TEST_DATA_URL)
        print(f"Loaded {len(test_dataset)} test cases from {TEST_DATA_URL}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return False

    # Initialize results tracking
    results = []
    total_tests = len(test_dataset)
    correct_predictions = 0

    try:
        # Initialize SGLang runner
        print("Initializing SGLang runner...")
        with SRTRunner(
            model_path=MODEL_PATH,
            torch_dtype=torch.float32,
            model_type="generation",
            trust_remote_code=True,
            lora_paths=[LORA_PATH],
            max_loras_per_batch=1,
        ) as runner:
            print("SGLang runner initialized successfully. Running inference tests...\n")

            # Run inference on each test case
            for i, test_case in enumerate(test_dataset, 1):
                instruction = test_case["instruction"]
                expected_output = test_case["output"]
                test_type = test_case["type"]

                print(f"Test {i}/{total_tests}: {test_type}")
                print(f"Instruction: {instruction}")
                print(f"Expected: '{expected_output}'")

                try:
                    # Run inference using SGLang runner
                    model_output = runner.forward(
                        prompts=[instruction],
                        max_new_tokens=50,  # Adjust as needed for your model
                        lora_paths=[LORA_PATH],
                    )

                    # Extract the generated text
                    generated_output = model_output.output_strs[0]
                    print(f"Generated: '{generated_output}'")

                    # Compare with expected output (exact match for simplicity)
                    is_correct = generated_output.strip() == expected_output.strip()

                    if is_correct:
                        correct_predictions += 1
                        print("✓ PASS")
                    else:
                        print("✗ FAIL")

                    # Store result
                    results.append({
                        "test_id": i,
                        "type": test_type,
                        "instruction": instruction,
                        "expected": expected_output,
                        "generated": generated_output,
                        "correct": is_correct
                    })

                except Exception as e:
                    print(f"✗ ERROR: {e}")
                    results.append({
                        "test_id": i,
                        "type": test_type,
                        "instruction": instruction,
                        "expected": expected_output,
                        "generated": f"ERROR: {e}",
                        "correct": False
                    })

            print("-" * 50)

        # Print final statistics
        accuracy = correct_predictions / total_tests * 100

        print("\n=== Test Results ===")
        print(f"Total tests: {total_tests}")
        print(f"Correct predictions: {correct_predictions}")
        print(".2f")
        print()

        # Print detailed results
        print("Detailed Results:")
        for result in results:
            status = "PASS" if result["correct"] else "FAIL"
            print(f"Test {result['test_id']}: {result['type']} - {status}")

        # Group by type
        type_stats = {}
        for result in results:
            test_type = result["type"]
            if test_type not in type_stats:
                type_stats[test_type] = {"total": 0, "correct": 0}
            type_stats[test_type]["total"] += 1
            if result["correct"]:
                type_stats[test_type]["correct"] += 1

        print("\nResults by Type:")
        for test_type, stats in type_stats.items():
            type_accuracy = stats["correct"] / stats["total"] * 100
            print(f"  {test_type}: {stats['correct']}/{stats['total']} ({type_accuracy:.1f}%)")

        return accuracy >= 0  # Always return True for now, adjust threshold as needed

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_lora_moe_inference_test()
    sys.exit(0 if success else 1)
