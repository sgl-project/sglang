"""
Logprobs Accuracy Test for SGLang

======================
With deterministic/batch invariant kernels, we can ensure that SGLang produces exactly the same
logprobs results for identical inputs. However, logprobs are highly sensitive to GPU hardware,
kernels, torch versions, and other factors, so we cannot maintain a unified logprobs baseline
across different machines.

This test is designed to be run locally by contributors to verify logprobs accuracy
before making changes to related code.
When submitting changes that affect logprobs computation, please:
1. Generate baseline
2. Run test
3. Submit results

We really appreciate your effort and contribution to SGLang!

======================
What does this test do?
This test fetches 1000 samples from the ShareGPT dataset, generates logprobs for each sample,
and saves them as a baseline. Then, by running the test mode, it validates the accuracy of
logprobs by comparing them against the baseline.

This test ensures that:
- the boundary of log probs requests are correct, eg, the index for tokens that required log probs are strictly followed
- logprobs remain invariant between test runs, and also before and after your code changes;

======================
Usage

Step 1: Generate Baseline (Before Code Changes)
```bash
python test/srt/test_logprobs.py gen
```

Step 2: Test Against Baseline (After Code Changes)
```bash
python test/srt/test_logprobs.py test
```
This tests your changes against the locally generated baseline from Step 1.
The test passes if the maximum and mean differences are within the tolerance thresholds.
======================
"""

import argparse
import json
import os
import pickle
import random
import unittest

import numpy as np
import requests
import torch
from transformers import AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

# Configuration
DENSE_MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/"
    "ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)

# Hardware-specific configuration
if torch.version.cuda is not None:
    print("Running on NVIDIA CUDA GPU")
    DENSE_TOLERANCE_MAX_DIFF = 1e-5
    DENSE_TOLERANCE_MEAN_DIFF = 1e-5
else:
    print("No GPU backend (CPU only)")
    raise ValueError("No GPU backend (CPU only)")

# Common configuration
TOP_K = 20
NUM_SAMPLES = 1000
LOGPROB_SAMPLE_RATIO = 0.5
TEMPERATURE = 1.0
MAX_LEN = 20000

# Default output files
DEFAULT_BASELINE_PKL = "sglang_baseline_local.pkl"
DEFAULT_META_JSON = "baseline_meta_preview.json"

# Default engine configuration
DEFAULT_ENGINE_CONFIG = {
    "model_path": DENSE_MODEL_NAME,
    "random_seed": 42,
    "skip_tokenizer_init": True,
    "mem_fraction_static": 0.8,
    "enable_deterministic_inference": True,
    "attention_backend": "flashinfer",
}


def generate_baseline(
    baseline_file=DEFAULT_BASELINE_PKL,
    meta_file=DEFAULT_META_JSON,
    num_samples=NUM_SAMPLES,
):
    """Generate a local baseline for logprobs testing.

    Args:
        baseline_file: Path to save the baseline pickle file
        meta_file: Path to save the metadata preview JSON file
        num_samples: Number of samples to generate
    """
    print(f"SGLang version: {sgl.__version__}")
    print("Downloading ShareGPT dataset...")

    # Download ShareGPT dataset
    try:
        response = requests.get(SHAREGPT_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"Dataset size: {len(data)}")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download ShareGPT dataset: {e}") from e

    # Filter and prepare texts
    texts = []
    for s in data:
        if "conversations" in s and len(s["conversations"]) > 0:
            try:
                text = s["conversations"][0]["value"]
                if isinstance(text, str) and len(text) <= MAX_LEN and len(text) >= 5500:
                    texts.append(text)
                    if len(texts) >= num_samples * 40:  # Get more samples for filtering
                        break
            except (KeyError, IndexError, TypeError) as e:
                print(f"Warning: Skipping invalid conversation data: {e}")
                continue

    if not texts:
        raise ValueError("No valid texts found in the dataset")

    print(f"Loading tokenizer for {DENSE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_NAME, use_fast=True)

    rng = np.random.default_rng(42)

    print(f"Launching SGLang Engine with {DENSE_MODEL_NAME}...")
    engine = sgl.Engine(
        model_path=DENSE_MODEL_NAME,
        attention_backend="flashinfer",
        enable_deterministic_inference=True,
        random_seed=42,
        skip_tokenizer_init=True,
        mem_fraction_static=0.8,
        max_running_requests=1,
    )

    records = []
    prompt_lengths = []

    try:
        for i, text in enumerate(texts):
            if len(records) >= num_samples:
                break

            try:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) < 5:
                    continue

                start_pos = int(rng.integers(0, max(1, len(ids) - 3)))

                outputs = engine.generate(
                    input_ids=[ids],
                    sampling_params={
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "top_k": TOP_K,
                        "max_new_tokens": 1,
                    },
                    return_logprob=True,
                    logprob_start_len=start_pos,
                    top_logprobs_num=TOP_K,
                )
                meta = outputs[0]["meta_info"]

                records.append(
                    dict(id=i, text=text, ids=ids, start_pos=start_pos, meta=meta)
                )
                prompt_lengths.append(len(ids))

                if (i + 1) % 50 == 0:
                    print(f"Processed {len(records)}/{num_samples} samples")

            except Exception as e:
                print(f"Warning: Failed to process sample {i}: {e}")
                continue

        if not records:
            raise RuntimeError(
                "Failed to generate any baseline records. Please check the warnings above for errors."
            )

        # Save baseline files
        with open(baseline_file, "wb") as f:
            pickle.dump(records, f)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(records[:2], f, ensure_ascii=False, indent=2)

        print(f"✅ Saved {len(records)} samples to {baseline_file}")
        print(f"✅ Meta preview saved to {meta_file}")

        if prompt_lengths:
            avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
            print(f"📊 Average prompt length: {avg_prompt_length:.2f} tokens")

    finally:
        engine.shutdown()
        torch.cuda.empty_cache()


class TestLogprobsDense(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class - initialize the engine once for all tests."""
        print(f"Launching SGLang Engine with {DENSE_MODEL_NAME}...")
        cls.engine = sgl.Engine(**DEFAULT_ENGINE_CONFIG)

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - shutdown the engine."""
        cls.engine.shutdown()
        torch.cuda.empty_cache()

    @classmethod
    def restart_engine_with_config(cls, **kwargs):
        """Create engine with custom configuration"""
        # Safely shutdown existing engine
        cls.engine.shutdown()
        torch.cuda.empty_cache()

        # Set chunk size
        chunk_size = kwargs.pop("chunk_size", None)
        if chunk_size is not None:
            print(f"Setting chunk size to {chunk_size}")
            os.environ["SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK"] = "True"
            os.environ["SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"] = str(chunk_size)
        else:
            os.environ["SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK"] = "False"

        # Create engine with merged configuration
        engine_config = {**DEFAULT_ENGINE_CONFIG, **kwargs}
        cls.engine = sgl.Engine(**engine_config)

    def load_test_data(self, baseline_file=None):
        """Load test data from local baseline file. In test mode, only local baseline is supported."""
        if not baseline_file:
            raise ValueError("baseline_file is required in test mode")

        if not os.path.exists(baseline_file):
            raise FileNotFoundError(
                f"Baseline file not found: {baseline_file}. Please run 'gen' mode first to generate the baseline."
            )

        print(f"Loading local baseline from {baseline_file}...")
        try:
            with open(baseline_file, "rb") as f:
                records = pickle.load(f)
            print(f"Successfully loaded {len(records)} records from local baseline")
            return records
        except (IOError, pickle.PickleError) as e:
            raise Exception(f"Failed to load local baseline: {e}") from e

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
                    TOP_K,
                    f"there are only {len(common_tokens)} common topk tokens that matches",
                )
                for token_id in common_tokens:
                    diffs.append(
                        abs(baseline_token_map[token_id] - sglang_token_map[token_id])
                    )
        if not diffs:
            return 0.0, 0.0
        return max(diffs), float(np.mean(diffs))

    def test_logprobs_comparison(self, baseline_file=None):
        """Test the logprobs comparison functionality with different parameter combinations."""
        # Load test data with retry mechanism
        records = self.load_test_data(baseline_file)

        # Fast configs for CI
        test_configs = [
            {"num_samples": NUM_SAMPLES},
            {"num_samples": 42, "chunk_size": 1, "max_running_requests": 16},
            {"num_samples": 42, "chunk_size": 2, "max_running_requests": 16},
            {"num_samples": 42, "chunk_size": 3, "max_running_requests": 16},
            {"num_samples": NUM_SAMPLES, "chunk_size": 16, "max_running_requests": 128},
            {"num_samples": NUM_SAMPLES, "chunk_size": 128, "max_running_requests": 16},
            {"num_samples": NUM_SAMPLES, "chunk_size": 128, "max_running_requests": 8},
            {"num_samples": NUM_SAMPLES, "chunk_size": 128, "max_running_requests": 32},
            {
                "num_samples": NUM_SAMPLES,
                "chunk_size": 128,
                "max_running_requests": 128,
            },
            {"num_samples": NUM_SAMPLES, "chunk_size": 256, "max_running_requests": 8},
            {"num_samples": NUM_SAMPLES, "chunk_size": 256, "max_running_requests": 32},
            {
                "num_samples": NUM_SAMPLES,
                "chunk_size": 256,
                "max_running_requests": 128,
            },
        ]

        # Run tests
        for config in test_configs:
            with self.subTest(config=config):
                print(f"Testing with config: {config}")

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
                    sample_idx in logprob_indices
                    for sample_idx in range(len(test_records))
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

                # Some configs must restart the engine to take effect
                chunk_size = config.get("chunk_size", None)
                max_running_requests = config.get("max_running_requests", None)
                if chunk_size is not None or max_running_requests is not None:
                    self.restart_engine_with_config(
                        chunk_size=chunk_size,
                        max_running_requests=max_running_requests,
                    )

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

                        max_diff, mean_diff = self.compare_meta(
                            baseline_meta, sglang_meta
                        )
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
                for sample_idx, (max_diff, mean_diff) in enumerate(
                    zip(all_max, all_mean)
                ):
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


def main():
    """Main function to handle command line arguments and run either generation or testing."""
    parser = argparse.ArgumentParser(
        description="SGLang Logprobs Test and Baseline Generation"
    )
    parser.add_argument(
        "mode",
        choices=["gen", "test"],
        help="Mode to run: 'gen' to generate baseline, 'test' to run tests",
    )

    args = parser.parse_args()

    if args.mode == "gen":
        print("🚀 Generating baseline...")
        generate_baseline()
        print(f"\n✅ Baseline generation complete!")
        print(f"📁 Baseline saved to: {DEFAULT_BASELINE_PKL}")
        print(f"📁 Metadata preview saved to: {DEFAULT_META_JSON}")
        print(f"\n💡 Next steps:")
        print(f"   1. Make your code changes")
        print(f"   2. Run: python {__file__} test")

    elif args.mode == "test":
        print("🧪 Running logprobs test...")
        if not os.path.exists(DEFAULT_BASELINE_PKL):
            print(f"❌ Baseline file not found: {DEFAULT_BASELINE_PKL}")
            print(f"💡 Generate baseline first by running:")
            print(f"   python {__file__} gen")
            print(f"   This will download ShareGPT data and generate a local baseline.")
            return 1

        # Set environment variable for testing
        os.environ["RETURN_ORIGINAL_LOGPROB"] = "True"

        # Create test instance and run
        test_instance = TestLogprobsDense()
        test_instance.setUpClass()
        try:
            test_instance.test_logprobs_comparison(baseline_file=DEFAULT_BASELINE_PKL)
            print("\n✅ Test completed successfully!")
        finally:
            test_instance.tearDownClass()

    return 0


if __name__ == "__main__":
    exit(main())
