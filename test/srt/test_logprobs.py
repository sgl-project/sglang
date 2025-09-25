import io
import os
import pickle
import random
import time
import unittest

import numpy as np
import requests
import torch

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    write_github_step_summary,
)

# Dense model configuration
DENSE_MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
if torch.version.hip is not None:
    print("Running on AMD ROCm GPU")
    DENSE_INPUT_PKL_URL = "https://huggingface.co/datasets/yushengsu/logprobs/resolve/main/sglang_baseline_2000_amd.pkl"
    DENSE_TOLERANCE_MAX_DIFF = 1.4
    DENSE_TOLERANCE_MEAN_DIFF = 0.1
elif torch.version.cuda is not None:
    print("Running on NVIDIA CUDA GPU")
    DENSE_INPUT_PKL_URL = "https://huggingface.co/datasets/font-info/logprobs/resolve/main/sglang_baseline_2000.pkl"
    DENSE_TOLERANCE_MAX_DIFF = 1.5
    DENSE_TOLERANCE_MEAN_DIFF = 0.1
else:
    print("No GPU backend (CPU only)")

# Common configuration
TOP_K = 20
MAX_RETRIES = 3
RETRY_DELAY = 2
NUM_SAMPLES = 1000
LOGPROB_SAMPLE_RATIO = 0.5
TEMPERATURE = 1.0


class TestLogprobsDense(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class - initialize the engine once for all tests."""
        print(f"Launching SGLang Engine with {DENSE_MODEL_NAME}...")
        cls.engine = sgl.Engine(
            model_path=DENSE_MODEL_NAME,
            random_seed=42,
            skip_tokenizer_init=True,
            mem_fraction_static=0.80,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - shutdown the engine."""
        cls.engine.shutdown()
        torch.cuda.empty_cache()

    def load_test_data(self):
        """Load test data from Hugging Face dataset with retry mechanism."""
        print(f"Loading data from {DENSE_INPUT_PKL_URL}...")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(DENSE_INPUT_PKL_URL, timeout=30)
                response.raise_for_status()

                with io.BytesIO(response.content) as f:
                    records = pickle.load(f)

                if not records:
                    raise ValueError("Empty dataset")

                print(f"Successfully loaded {len(records)} records")
                return records

            except Exception as e:
                print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise Exception(
                        f"Failed to load data after {MAX_RETRIES} attempts: {e}"
                    )
                time.sleep(RETRY_DELAY)

    def compare_meta(self, baseline_meta, sglang_meta):
        """Compare metadata between two outputs and return max and mean differences."""
        diffs = []
        for key in ["input_top_logprobs", "output_top_logprobs"]:
            baseline_logprobs, sglang_logprobs = baseline_meta[key], sglang_meta[key]
            self.assertEqual(
                len(baseline_logprobs),
                len(sglang_logprobs),
                f"Length of {key} is not equal, sglang did not return the correct number of log probs(should be top 20)",
            )
            for baseline_entry, sglang_entry in zip(baseline_logprobs, sglang_logprobs):
                if not baseline_entry or not sglang_entry:
                    continue
                baseline_token_map = {tid: lp for lp, tid, _ in baseline_entry}
                sglang_token_map = {tid: lp for lp, tid, _ in sglang_entry}
                common_tokens = baseline_token_map.keys() & sglang_token_map.keys()
                self.assertGreaterEqual(
                    len(common_tokens),
                    TOP_K / 2,
                    f"there are only {len(common_tokens)} common topk tokens that matches",
                )
                for token_id in common_tokens:
                    diffs.append(
                        abs(baseline_token_map[token_id] - sglang_token_map[token_id])
                    )
        return max(diffs), float(np.mean(diffs))

    def test_logprobs_comparison(self):
        """Test the logprobs comparison functionality with different parameter combinations."""
        # Load test data with retry mechanism
        records = self.load_test_data()

        with self.subTest(
            config={
                "num_samples": NUM_SAMPLES,
                "logprob_sample_ratio": LOGPROB_SAMPLE_RATIO,
                "temperature": TEMPERATURE,
            }
        ):

            # Sample records for this config
            test_records = random.sample(records, k=min(NUM_SAMPLES, len(records)))
            random.shuffle(test_records)

            # Calculate how many samples should return logprobs
            logprob_count = int(len(test_records) * LOGPROB_SAMPLE_RATIO)
            print(
                f"Testing with {len(test_records)} samples, temperature={TEMPERATURE}"
            )
            print(
                f"Will return logprobs for {logprob_count} samples (ratio: {LOGPROB_SAMPLE_RATIO})"
            )

            all_max, all_mean = [], []
            logprob_returned_count = 0

            # Process all records at once
            input_ids = [rec["ids"] for rec in test_records]
            logprob_start_lens = [rec["start_pos"] for rec in test_records]

            # Determine which samples should return logprobs (randomly selected)
            logprob_indices = set(
                random.sample(range(len(test_records)), logprob_count)
            )
            return_logprob_array = [
                sample_idx in logprob_indices for sample_idx in range(len(test_records))
            ]

            # Sampling param per request
            sampling_params = [
                {
                    "temperature": TEMPERATURE,
                    "top_p": 1.0,
                    "top_k": TOP_K,
                    "max_new_tokens": 1,
                }
                for _ in test_records
            ]

            outputs = self.engine.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_logprob=return_logprob_array,
                logprob_start_len=logprob_start_lens,
                top_logprobs_num=TOP_K,
            )

            for sample_idx, (rec, output) in enumerate(zip(test_records, outputs)):
                # Only compare logprobs for samples that should have them
                if sample_idx in logprob_indices:
                    # Safe access to meta_info and input_top_logprobs
                    meta_info = output.get("meta_info")
                    input_top_logprobs = (
                        meta_info.get("input_top_logprobs") if meta_info else None
                    )

                    self.assertIsNotNone(
                        input_top_logprobs,
                        f"return_logprob enabled on this sample, but input_top_logprobs is None (length: {len(input_top_logprobs) if input_top_logprobs is not None else 'N/A'})",
                    )
                    baseline_meta = rec["meta"]
                    sglang_meta = meta_info

                    max_diff, mean_diff = self.compare_meta(baseline_meta, sglang_meta)
                    all_max.append(max_diff)
                    all_mean.append(mean_diff)
                    logprob_returned_count += 1
                else:
                    # Verify that logprobs were not returned for this sample
                    meta_info = output.get("meta_info")
                    input_top_logprobs = (
                        meta_info.get("input_top_logprobs") if meta_info else None
                    )
                    output_token_ids_logprobs = (
                        meta_info.get("output_token_ids_logprobs")
                        if meta_info
                        else None
                    )

                    self.assertFalse(
                        input_top_logprobs,
                        f"return_logprob is disabled on this sample, Sample {sample_idx} should not have logprobs, content: {output_token_ids_logprobs}",
                    )

            max_of_max = max(all_max) if all_max else 0.0
            mean_of_mean = np.mean(all_mean) if all_mean else 0.0

            print(f"max Δ={max_of_max:.6g}")
            print(f"mean Δ={mean_of_mean:.6g}")
            print(
                f"logprobs returned for {logprob_returned_count} samples (expected: {logprob_count})"
            )

            # Verify correct number of logprobs returned
            self.assertEqual(
                logprob_returned_count,
                logprob_count,
                f"Expected {logprob_count} samples with logprobs, got {logprob_returned_count}",
            )

            # Write results to GitHub summary
            summary_content = f"""
- **Configuration**: {{"num_samples": {NUM_SAMPLES}, "logprob_sample_ratio": {LOGPROB_SAMPLE_RATIO}, "temperature": {TEMPERATURE}}}
- **Max of max Δ**: {max_of_max:.6g}
- **Mean of mean Δ**: {mean_of_mean:.6g}
- **Status**: {'✅ Passed' if max_of_max <= DENSE_TOLERANCE_MAX_DIFF and mean_of_mean <= DENSE_TOLERANCE_MEAN_DIFF else '❌ Failed'}
"""
            write_github_step_summary(summary_content)

            # Basic validation
            self.assertIsInstance(all_max, list)
            self.assertIsInstance(all_mean, list)
            self.assertGreater(
                len(all_max),
                0,
                f"No test samples processed for config {{'num_samples': {NUM_SAMPLES}, 'logprob_sample_ratio': {LOGPROB_SAMPLE_RATIO}, 'temperature': {TEMPERATURE}}}",
            )

            # Tolerance checks with clear error messages
            failed_samples = []
            for sample_idx, (max_diff, mean_diff) in enumerate(zip(all_max, all_mean)):
                if max_diff > DENSE_TOLERANCE_MAX_DIFF:
                    failed_samples.append(
                        f"Sample {sample_idx}: max_diff={max_diff:.6g} > {DENSE_TOLERANCE_MAX_DIFF}"
                    )
                if mean_diff > DENSE_TOLERANCE_MEAN_DIFF:
                    failed_samples.append(
                        f"Sample {sample_idx}: mean_diff={mean_diff:.6g} > {DENSE_TOLERANCE_MEAN_DIFF}"
                    )

            if failed_samples:
                self.fail(
                    f"Config {{'num_samples': {NUM_SAMPLES}, 'logprob_sample_ratio': {LOGPROB_SAMPLE_RATIO}, 'temperature': {TEMPERATURE}}} - Tolerance exceeded in {len(failed_samples)} samples:\n"
                    + "\n".join(failed_samples[:5])
                )


if __name__ == "__main__":
    unittest.main()
