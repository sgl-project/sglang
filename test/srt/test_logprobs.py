import os, pickle, numpy as np
import torch
import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST, write_github_step_summary
import random
import unittest
import requests
import io
import time

# MOE model configuration
MOE_MODEL_NAME = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST
MOE_INPUT_PKL_URL = "https://huggingface.co/datasets/font-info/logprobs/resolve/main/sglang_baseline_moe.pkl"
MOE_TOLERANCE_MAX_DIFF = 10
MOE_TOLERANCE_MEAN_DIFF = 0.1
MOE_TOLERANCE_MEAN_DIFF_SAMPLE = 0.3

# Dense model configuration
DENSE_MODEL_NAME = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
DENSE_INPUT_PKL_URL = "https://huggingface.co/datasets/font-info/logprobs/resolve/main/sglang_baseline_2000.pkl"
DENSE_TOLERANCE_MAX_DIFF = 1.0
DENSE_TOLERANCE_MEAN_DIFF = 0.05
DENSE_TOLERANCE_MEAN_DIFF_SAMPLE = 0.1


# Common configuration
TOP_K = 20
MAX_RETRIES = 3
RETRY_DELAY = 2

# Test configurations
TEST_CONFIGS = [
    {"batch_size": 50, "num_samples": 200, "temperature": 0.5},
    {"batch_size": 100, "num_samples": 300, "temperature": 10.0},
    {"batch_size": 20, "num_samples": 500, "temperature": 2.0},
    {"batch_size": 20, "num_samples": 500, "temperature": 1.0},
]

os.environ["RETURN_ORIGINAL_LOGPROB"] = "True"


@unittest.skip("Skipping MOE test case for now")
class TestLogprobsMOE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class - initialize the engine once for all tests."""
        print(f"Launching SGLang Engine with {MOE_MODEL_NAME}...")
        cls.engine = sgl.Engine(
            model_path=MOE_MODEL_NAME,
            random_seed=42,
            skip_tokenizer_init=True,
            mem_fraction_static=0.6,
            max_running_requests=1,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests - shutdown the engine."""
        cls.engine.shutdown()
        torch.cuda.empty_cache()

    def load_test_data(self):
        """Load test data from Hugging Face dataset with retry mechanism."""
        print(f"Loading data from {MOE_INPUT_PKL_URL}...")
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(MOE_INPUT_PKL_URL, timeout=30)
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
                    raise Exception(f"Failed to load data after {MAX_RETRIES} attempts: {e}")
                time.sleep(RETRY_DELAY)

    def compare_meta(self, metaA, metaB):
        """Compare metadata between two outputs and return max and mean differences."""
        diffs = []
        for key in ["input_top_logprobs", "output_top_logprobs"]:
            arrA, arrB = metaA[key], metaB[key]
            for e1, e2 in zip(arrA, arrB):
                if not e1 or not e2:
                    continue
                dmapA = {tid: lp for lp, tid, _ in e1}
                dmapB = {tid: lp for lp, tid, _ in e2}
                common = dmapA.keys() & dmapB.keys()
                for tid in common:
                    diffs.append(abs(dmapA[tid] - dmapB[tid]))
        if not diffs:
            return 0.0, 0.0
        return max(diffs), float(np.mean(diffs))

    def test_logprobs_comparison(self):
        """Test the logprobs comparison functionality with different parameter combinations."""
        # Load test data with retry mechanism
        records = self.load_test_data()
        
        for config_idx, config in enumerate(TEST_CONFIGS):
            with self.subTest(config=config):
                
                # Sample records for this config
                test_records = random.sample(records, k=min(config["num_samples"], len(records)))
                random.shuffle(test_records)
                print(f"Testing with {len(test_records)} samples, batch_size={config['batch_size']}, temperature={config['temperature']}")

                all_max, all_mean = [], []
                
                for i in range(0, len(test_records), config["batch_size"]):
                    batch = test_records[i:i+config["batch_size"]]
                    input_ids = [rec["ids"] for rec in batch]
                    logprob_start_lens = [rec["start_pos"] for rec in batch]

                    # Sampling param per request
                    sampling_params = [ 
                        {
                            "temperature": config["temperature"],
                            "top_p": 1.0,
                            "top_k": TOP_K,
                            "max_new_tokens": 1
                        } for _ in batch
                    ]

                    outputs = self.engine.generate(
                        input_ids=input_ids,
                        sampling_params=sampling_params,
                        return_logprob=True,
                        logprob_start_len=logprob_start_lens,
                        top_logprobs_num=TOP_K,
                    )

                    for rec, output in zip(batch, outputs):
                        metaA = rec["meta"]
                        metaB = output["meta_info"]

                        max_diff, mean_diff = self.compare_meta(metaA, metaB)
                        all_max.append(max_diff)
                        all_mean.append(mean_diff)

                max_of_max = max(all_max)
                mean_of_mean = np.mean(all_mean)
                
                print(f"Config {config_idx + 1} - max of max Δ={max_of_max:.6g}")
                print(f"Config {config_idx + 1} - mean of mean Δ={mean_of_mean:.6g}")

                # Write results to GitHub summary
                summary_content = f"""
## MOE Logprobs Test - Config {config_idx + 1}
- **Configuration**: {config}
- **Max of max Δ**: {max_of_max:.6g}
- **Mean of mean Δ**: {mean_of_mean:.6g}
- **Status**: {'✅ Passed' if max_of_max <= MOE_TOLERANCE_MAX_DIFF and mean_of_mean <= MOE_TOLERANCE_MEAN_DIFF_SAMPLE else '❌ Failed'}

"""
                write_github_step_summary(summary_content)

                # Basic validation
                self.assertIsInstance(all_max, list)
                self.assertIsInstance(all_mean, list)
                self.assertGreater(len(all_max), 0, f"No test samples processed for config {config}")
                
                # Tolerance checks with clear error messages
                failed_samples = []
                for i, (max_diff, mean_diff) in enumerate(zip(all_max, all_mean)):
                    if max_diff > MOE_TOLERANCE_MAX_DIFF:
                        failed_samples.append(f"Sample {i}: max_diff={max_diff:.6g} > {MOE_TOLERANCE_MAX_DIFF}")
                    if mean_diff > MOE_TOLERANCE_MEAN_DIFF_SAMPLE:
                        failed_samples.append(f"Sample {i}: mean_diff={mean_diff:.6g} > {MOE_TOLERANCE_MEAN_DIFF_SAMPLE}")
                
                if failed_samples:
                    self.fail(f"Config {config} - Tolerance exceeded in {len(failed_samples)} samples:\n" + "\n".join(failed_samples[:5]))
                
                print(f"✅ Config {config_idx + 1} - All {len(all_max)} samples passed tolerance checks")


class TestLogprobsDense(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test class - initialize the engine once for all tests."""
        print(f"Launching SGLang Engine with {DENSE_MODEL_NAME}...")
        cls.engine = sgl.Engine(
            model_path=DENSE_MODEL_NAME,
            random_seed=42,
            skip_tokenizer_init=True,
            mem_fraction_static=0.6,
            max_running_requests=1,
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
                    raise Exception(f"Failed to load data after {MAX_RETRIES} attempts: {e}")
                time.sleep(RETRY_DELAY)

    def compare_meta(self, metaA, metaB):
        """Compare metadata between two outputs and return max and mean differences."""
        diffs = []
        for key in ["input_top_logprobs", "output_top_logprobs"]:
            self.assertEqual(len(metaA[key]), len(metaB[key]), f"Length of {key} is not equal, sglang did not return the log probs from the correct starting index")
            arrA, arrB = metaA[key], metaB[key]
            self.assertEqual(len(arrA), len(arrB), f"Length of {key} is not equal, sglang did not return the correct number of log probs(should be top 20)")
            for e1, e2 in zip(arrA, arrB):
                if not e1 or not e2:
                    continue
                dmapA = {tid: lp for lp, tid, _ in e1}
                dmapB = {tid: lp for lp, tid, _ in e2}
                common = dmapA.keys() & dmapB.keys()
                for tid in common:
                    diffs.append(abs(dmapA[tid] - dmapB[tid]))
        if not diffs:
            return 0.0, 0.0
        return max(diffs), float(np.mean(diffs))

    def test_logprobs_comparison(self):
        """Test the logprobs comparison functionality with different parameter combinations."""
        # Load test data with retry mechanism
        records = self.load_test_data()
        
        for config_idx, config in enumerate(TEST_CONFIGS):
            with self.subTest(config=config):
                
                # Sample records for this config
                test_records = random.sample(records, k=min(config["num_samples"], len(records)))
                random.shuffle(test_records)
                print(f"Testing with {len(test_records)} samples, batch_size={config['batch_size']}, temperature={config['temperature']}")

                all_max, all_mean = [], []
                
                for i in range(0, len(test_records), config["batch_size"]):
                    batch = test_records[i:i+config["batch_size"]]
                    input_ids = [rec["ids"] for rec in batch]
                    logprob_start_lens = [rec["start_pos"] for rec in batch]

                    # Sampling param per request
                    sampling_params = [ 
                        {
                            "temperature": config["temperature"],
                            "top_p": 1.0,
                            "top_k": TOP_K,
                            "max_new_tokens": 1
                        } for _ in batch
                    ]

                    outputs = self.engine.generate(
                        input_ids=input_ids,
                        sampling_params=sampling_params,
                        return_logprob=True,
                        logprob_start_len=logprob_start_lens,
                        top_logprobs_num=TOP_K,
                    )

                    for rec, output in zip(batch, outputs):
                        metaA = rec["meta"]
                        metaB = output["meta_info"]

                        max_diff, mean_diff = self.compare_meta(metaA, metaB)
                        all_max.append(max_diff)
                        all_mean.append(mean_diff)

                max_of_max = max(all_max)
                mean_of_mean = np.mean(all_mean)
                
                print(f"Config {config_idx + 1} - max of max Δ={max_of_max:.6g}")
                print(f"Config {config_idx + 1} - mean of mean Δ={mean_of_mean:.6g}")

                # Write results to GitHub summary
                summary_content = f"""
## Dense Logprobs Test - Config {config_idx + 1}
- **Configuration**: {config}
- **Max of max Δ**: {max_of_max:.6g}
- **Mean of mean Δ**: {mean_of_mean:.6g}
- **Status**: {'✅ Passed' if max_of_max <= DENSE_TOLERANCE_MAX_DIFF and mean_of_mean <= DENSE_TOLERANCE_MEAN_DIFF_SAMPLE else '❌ Failed'}
"""
                write_github_step_summary(summary_content)

                # Basic validation
                self.assertIsInstance(all_max, list)
                self.assertIsInstance(all_mean, list)
                self.assertGreater(len(all_max), 0, f"No test samples processed for config {config}")
                
                # Tolerance checks with clear error messages
                failed_samples = []
                for i, (max_diff, mean_diff) in enumerate(zip(all_max, all_mean)):
                    if max_diff > DENSE_TOLERANCE_MAX_DIFF:
                        failed_samples.append(f"Sample {i}: max_diff={max_diff:.6g} > {DENSE_TOLERANCE_MAX_DIFF}")
                    if mean_diff > DENSE_TOLERANCE_MEAN_DIFF_SAMPLE:
                        failed_samples.append(f"Sample {i}: mean_diff={mean_diff:.6g} > {DENSE_TOLERANCE_MEAN_DIFF_SAMPLE}")
                
                if failed_samples:
                    self.fail(f"Config {config} - Tolerance exceeded in {len(failed_samples)} samples:\n" + "\n".join(failed_samples[:5]))
                
                print(f"✅ Config {config_idx + 1} - All {len(all_max)} samples passed tolerance checks")


if __name__ == "__main__":
    unittest.main()
