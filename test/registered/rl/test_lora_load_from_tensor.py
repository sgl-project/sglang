import json
import os
import unittest
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import sglang as sgl
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=102, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_REPO = "charent/self_cognition_Alice"
TEST_PROMPT = "Hello, my name is"
EXPECTED_OUTPUT = (
    " Alice, and I am a software engineer. I am excited to share my journey"
)
MAX_NEW_TOKENS = 16


def load_lora_via_stream(engine, lora_name, tensors, config_dict):
    """Wire-load an adapter over the stream path: register (control plane) +
    prefixed tensors through a sync_base=False weight-update session."""
    result = engine.register_lora_adapter(lora_name=lora_name, config_dict=config_dict)
    if not result.success:
        return SimpleNamespace(success=False, error_message=result.error_message)
    engine.begin_weight_update(selector="all", sync_base=False)
    engine.update_weights_from_tensor(
        named_tensors=[
            (f"{lora_name}:{key}", tensor) for key, tensor in tensors.items()
        ]
    )
    success, message = engine.end_weight_update()
    return SimpleNamespace(success=success, error_message=message)


def loaded_adapter_names(engine):
    return list(engine.tokenizer_manager.lora_registry._registry.keys())


class TestLoRALoadFromTensor(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=MODEL_PATH,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            mem_fraction_static=0.6,
            log_level="error",
        )

        lora_adapter = snapshot_download(
            repo_id=LORA_REPO,
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"],
        )
        # Load tensors and config from downloaded adapter
        cls.lora_tensors = load_file(
            os.path.join(lora_adapter, "adapter_model.safetensors")
        )
        with open(os.path.join(lora_adapter, "adapter_config.json"), "r") as f:
            cls.lora_config_dict = json.load(f)

    def test_lora_stream_register_rejected_over_cap(self):
        """Streamed adapters have no path to reload from, so the cap rejects new
        registrations instead of evicting; same-name upserts stay allowed."""
        MAX_LOADED_LORAS = 8
        test_engine = sgl.Engine(
            model_path=MODEL_PATH,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            mem_fraction_static=0.6,
            log_level="error",
            max_loaded_loras=MAX_LOADED_LORAS,
        )

        TEST_LORA_COUNT = 10
        for i in range(TEST_LORA_COUNT):
            result = load_lora_via_stream(
                test_engine,
                f"self_cognition_Alice_{i}",
                self.lora_tensors,
                self.lora_config_dict,
            )
            if i < MAX_LOADED_LORAS:
                self.assertTrue(
                    result.success,
                    f"Failed to load LoRA adapter {i}: {result.error_message}",
                )
            else:
                self.assertFalse(
                    result.success,
                    f"Adapter {i} should have been rejected over the cap",
                )
                self.assertIn("max-loaded-loras", result.error_message)

        loaded = loaded_adapter_names(test_engine)
        self.assertEqual(
            loaded,
            [f"self_cognition_Alice_{i}" for i in range(MAX_LOADED_LORAS)],
            f"Loaded adapters do not match the first {MAX_LOADED_LORAS} registrations: {loaded}",
        )

        # Same-name upsert must still pass at the cap.
        result = load_lora_via_stream(
            test_engine,
            "self_cognition_Alice_0",
            self.lora_tensors,
            self.lora_config_dict,
        )
        self.assertTrue(result.success, f"Upsert at cap failed: {result.error_message}")

    def test_register_rejects_name_holding_the_separator(self):
        """A name holding ':' could never be matched by its own streamed tensors,
        since the first ':' is what splits adapter from tensor key."""
        result = self.engine.register_lora_adapter(
            lora_name="team:alice",
            config_dict=self.lora_config_dict,
        )
        self.assertFalse(result.success, "':' in an adapter name must be rejected")
        self.assertIn("must not contain ':'", result.error_message)
        self.assertNotIn("team:alice", loaded_adapter_names(self.engine))

    def test_lora_e2e_load_from_tensor_params(self):
        print("[Test]Testing LoRA load from tensor params...")

        result = load_lora_via_stream(
            self.engine,
            "self_cognition_Alice",
            self.lora_tensors,
            self.lora_config_dict,
        )
        self.assertTrue(
            result.success,
            f"Failed to load LoRA from tensors: {result.error_message}",
        )

        output_without_lora = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
        )

        output_lora = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
            lora_path=["self_cognition_Alice"],
        )

        print(f"[Without LoRA] {output_without_lora[0]}")
        print(f"[With LoRA]  {output_lora[0]}")
        self.assertNotEqual(
            output_without_lora[0]["text"][: len(EXPECTED_OUTPUT)],
            EXPECTED_OUTPUT,
            "Output before applying LoRA should not match expected result",
        )

        self.assertEqual(
            output_lora[0]["text"][: len(EXPECTED_OUTPUT)],
            EXPECTED_OUTPUT,
            "Output after applying LoRA does not match expected result",
        )

    def test_lora_load_unload_load_from_tensor_params(self):
        print("[Test]Testing LoRA load, unload, load from tensor params...")

        # Load LoRA adapter from tensors
        result = load_lora_via_stream(
            self.engine,
            "self_cognition_Alice_multiple",
            self.lora_tensors,
            self.lora_config_dict,
        )
        self.assertTrue(
            result.success,
            f"Failed to load LoRA from tensors: {result.error_message}",
        )

        # Unload LoRA adapter
        result = self.engine.unload_lora_adapter("self_cognition_Alice_multiple")
        self.assertTrue(
            result.success, f"Failed to unload LoRA: {result.error_message}"
        )
        with self.assertRaises(ValueError) as context:
            output_lora = self.engine.generate(
                prompt=[TEST_PROMPT],
                sampling_params={
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.0,
                },
                lora_path=["self_cognition_Alice_multiple"],
            )
        # Load LoRA adapter again
        result_again = load_lora_via_stream(
            self.engine,
            "self_cognition_Alice_multiple",
            self.lora_tensors,
            self.lora_config_dict,
        )
        self.assertTrue(
            result_again.success,
            f"Failed to load LoRA from tensors: {result_again.error_message}",
        )
        output_lora_loaded_again = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
            lora_path=["self_cognition_Alice_multiple"],
        )

        print(f"[With LoRA Loaded again]  {output_lora_loaded_again[0]}")
        self.assertEqual(
            output_lora_loaded_again[0]["text"][: len(EXPECTED_OUTPUT)],
            EXPECTED_OUTPUT,
            "Output after applying LoRA does not match expected result",
        )

    def test_lora_logp_diff_with_huggingface(self):
        """
        Test comparing SGLang and HuggingFace LoRA logprobs when loading LoRA from tensors.
        This verifies that loading LoRA adapters from tensors produces consistent logprobs
        with HuggingFace.
        """

        from sglang.test.runners import HFRunner, SRTRunner
        from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER

        print("[Test]Testing LoRA logprob difference with HuggingFace...")

        lora_name = "self_cognition_Alice_logprob_test"
        prompts = [TEST_PROMPT]

        # Step 1: Run SGLang with LoRA loaded from tensors
        print("[Test]Running SGLang with LoRA from tensors...")
        with SRTRunner(
            MODEL_PATH,
            torch_dtype=torch.float16,
            model_type="generation",
            tp_size=1,
            max_loras_per_batch=1,
            lora_backend="triton",
            disable_cuda_graph=False,
            disable_radix_cache=True,
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
            mem_fraction_static=0.6,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ) as srt_runner:
            result = load_lora_via_stream(
                srt_runner.engine, lora_name, self.lora_tensors, self.lora_config_dict
            )
            self.assertTrue(
                result.success,
                f"Failed to load LoRA from tensors: {result.error_message}",
            )

            # Run inference with loaded LoRA
            srt_outputs = srt_runner.forward(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[lora_name],
            )

        # Step 2: Run HuggingFace with LoRA
        print("[Test]Running HuggingFace with LoRA...")
        torch.cuda.empty_cache()

        with HFRunner(
            MODEL_PATH,
            torch_dtype=torch.float16,
            model_type="generation",
            patch_model_do_sample_false=True,
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[LORA_REPO],
            )

        # Step 3: Compare results
        sglang_text = srt_outputs.output_strs[0]
        hf_text = hf_outputs.output_strs[0]

        print(f"[Text Output]")
        print(f"  SGLang:      {sglang_text}")
        print(f"  HuggingFace: {hf_text}")

        # Compare prefill (input) logprobs
        sglang_prefill = torch.tensor(srt_outputs.top_input_logprobs[0])
        hf_prefill = torch.tensor(hf_outputs.top_input_logprobs[0])

        prefill_diff = torch.abs(sglang_prefill - hf_prefill)
        prefill_max_diff = torch.max(prefill_diff).item()
        prefill_mean_diff = torch.mean(prefill_diff).item()

        print(f"\n[Prefill Logprob Comparison]")
        print(f"  Shape:           {list(sglang_prefill.shape)}")
        print(f"  Max difference:  {prefill_max_diff:.6e}")
        print(f"  Mean difference: {prefill_mean_diff:.6e}")

        # Compare decode (output) logprobs
        sglang_decode = torch.tensor(srt_outputs.top_output_logprobs[0])
        hf_decode = torch.tensor(hf_outputs.top_output_logprobs[0])

        decode_diff = torch.abs(sglang_decode - hf_decode)
        decode_max_diff = torch.max(decode_diff).item()
        decode_mean_diff = torch.mean(decode_diff).item()

        print(f"\n[Decode Logprob Comparison]")
        print(f"  Shape:           {list(sglang_decode.shape)}")
        print(f"  Max difference:  {decode_max_diff:.6e}")
        print(f"  Mean difference: {decode_mean_diff:.6e}")

        # Assert logprobs are close (threshold 1e-1)
        LOGPROB_THRESHOLD = 1e-1
        self.assertLess(
            prefill_max_diff,
            LOGPROB_THRESHOLD,
            f"Prefill logprob max difference too large: {prefill_max_diff:.6e} > {LOGPROB_THRESHOLD:.0e}",
        )
        self.assertLess(
            decode_max_diff,
            LOGPROB_THRESHOLD,
            f"Decode logprob max difference too large: {decode_max_diff:.6e} > {LOGPROB_THRESHOLD:.0e}",
        )

        # Verify text outputs match expected
        self.assertEqual(
            sglang_text[: len(EXPECTED_OUTPUT)],
            EXPECTED_OUTPUT,
            "SGLang output does not match expected result",
        )

        print("\n[Test]LoRA logprob comparison test passed!")

    def test_lora_e2e_load_from_flattened_bucket(self):
        """Test loading LoRA via FlattenedTensorBucket format (RL weight sync path)."""
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        lora_name = "self_cognition_Alice_flattened"
        named_tensors = [
            (f"{lora_name}:{key}", tensor) for key, tensor in self.lora_tensors.items()
        ]
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        bucket_dict = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized = MultiprocessingSerializer.serialize(bucket_dict, output_str=True)

        result = self.engine.register_lora_adapter(
            lora_name=lora_name, config_dict=self.lora_config_dict
        )
        self.assertTrue(result.success, f"Failed: {result.error_message}")
        self.engine.begin_weight_update(selector="all", sync_base=False)
        self.engine.update_weights_from_tensor(
            named_tensors=[serialized], load_format="flattened_bucket"
        )
        success, message = self.engine.end_weight_update()
        self.assertTrue(success, f"Failed: {message}")

        output = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
            lora_path=["self_cognition_Alice_flattened"],
        )
        self.assertEqual(
            output[0]["text"][: len(EXPECTED_OUTPUT)],
            EXPECTED_OUTPUT,
            "Output after applying LoRA via flattened bucket does not match expected",
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()


if __name__ == "__main__":
    unittest.main()
