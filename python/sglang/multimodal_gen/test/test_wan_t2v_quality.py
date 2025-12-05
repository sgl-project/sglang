import glob
import json
import logging
import os
import re
import shutil
import subprocess
import time  # Added for file handle delay
import unittest
import uuid

import torch

from sglang.test.test_utils import (  # DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, # Not used here; is_in_ci, # Not used here
    CustomTestCase,
)

# --- Library Imports ---
# NOTE: Ensure qwen_vl_utils.py is available in the Python path
try:
    from qwen_vl_utils import process_vision_info
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        Qwen2VLForConditionalGeneration,
    )
except ImportError:
    # Dummy implementation for environments without transformers/qwen
    class Dummy:
        pass

    (
        Qwen2VLForConditionalGeneration,
        AutoProcessor,
        AutoModelForCausalLM,
        AutoTokenizer,
    ) = [Dummy] * 4

    def process_vision_info(*args, **kwargs):
        pass


class TestWanT2VQuality(CustomTestCase):
    """Test Wan T2V model quality with automated evaluation."""

    # --- Configuration and Setup ---

    # Static configuration shared across all tests
    WAN_MODEL = os.getenv(
        "WAN_MODEL_PATH", "/data01/models/Wan-AI/Wan2.1-T2V-14B-Diffusers"
    )
    BASE_OUT_DIR = os.getenv("CI_OUTPUT_DIR", "/tmp/sglang_ci_t2v_outputs")
    # MODEL_ROOT = os.getenv("MODEL_ROOT", "~/.cache/huggingface/hub")
    MODEL_ROOT = os.getenv("MODEL_ROOT", "/data03/models")

    CAPTION_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
    JUDGE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    PASS_THRESHOLD = 70
    NUM_GPUS = os.getenv("NUM_GPUS", "1")
    FILE_DELAY_SEC = 3  # Delay to ensure file handle release

    JUDGE_SYSTEM_PROMPT = """
You are a video evaluation engine.
Compare the PROMPT to the VIDEO DESCRIPTION.
Evaluate: Object correctness, Action correctness, Consistency.

Scoring (0-100):
90-100: Perfect match.
70-89: Good match, minor errors.
40-69: Weak match, key elements missing.
0-39: Fail, completely wrong.

Output format (EXACTLY 7 lines):
Overall Score: <int>
Object: <int>
Attribute: <int>
Action: <int>
Spatial: <int>
Temporal: <int>
Reason: <short sentence>
"""
    # Class-level caches for lazy loading
    _caption_model = None
    _caption_processor = None
    _judge_model = None
    _judge_tokenizer = None

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources (logging, model validation)."""
        super().setUpClass()
        cls.logger = logging.getLogger(__name__)

        if not os.path.exists(cls.WAN_MODEL):
            cls.logger.warning(
                f"T2V Model path missing: {cls.WAN_MODEL}. Skipping test."
            )
            # Unittest standard way to skip if setup fails
            raise unittest.SkipTest("T2V model path not found.")

        # Ensure base output directory exists
        os.makedirs(cls.BASE_OUT_DIR, exist_ok=True)

    def setUp(self):
        """Setup for each individual test."""
        super().setUp()
        self.run_id = str(uuid.uuid4())[:8]
        self.job_out_dir = os.path.join(self.BASE_OUT_DIR, f"ci_run_{self.run_id}")
        os.makedirs(self.job_out_dir, exist_ok=True)
        self.logger.info(f"Test Setup: Job Workspace is {self.job_out_dir}")
        self.video_path = None  # To be set during the test

    def tearDown(self):
        """Cleanup after each individual test."""
        super().tearDown()
        if os.path.exists(self.job_out_dir):
            self.logger.info(f"Test Teardown: Cleaning up workspace {self.job_out_dir}")
            shutil.rmtree(self.job_out_dir)

    # --- Helper Methods ---

    def resolve(self, model_name: str) -> str:
        """Resolves model name to a path, prioritizing MODEL_ROOT."""
        path = os.path.join(self.MODEL_ROOT, model_name)
        return path if os.path.exists(path) else model_name

    def run_command(self, command, check=True, cwd=None, env=None):
        """Runs a command with full environment passing."""
        cmd_str = " ".join(command)
        self.logger.info(f"Executing: {cmd_str}")
        full_output = ""

        # Prepare the environment (CRITICAL for quality/library path issues)
        effective_env = os.environ.copy()
        if env:
            effective_env.update(env)

        try:
            with subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=effective_env,  # Pass the resolved environment
            ) as proc:
                for line in proc.stdout:
                    print(line, end="")
                    full_output += line

                return_code = proc.wait()

                if check and return_code != 0:
                    raise RuntimeError(
                        f"Command failed with return code {return_code}: {cmd_str}"
                    )

            return full_output

        except FileNotFoundError:
            raise RuntimeError(
                f"Command not found: {command[0]}. Ensure it is in PATH."
            )
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {e}")

    # --- LLM/VLM Judger Methods ---

    def _get_caption_model(self):
        """Lazy load the Qwen-VL captioning model."""
        if self._caption_model is None:
            self.logger.info("Loading Caption Model...")
            path = self.resolve(self.CAPTION_MODEL_NAME)
            self._caption_model = Qwen2VLForConditionalGeneration.from_pretrained(
                path, torch_dtype="auto", device_map="auto"
            )
            self._caption_processor = AutoProcessor.from_pretrained(path)
        return self._caption_model, self._caption_processor

    def _get_judge_model(self):
        """Lazy load the Qwen-Instruct judge model."""
        if self._judge_model is None:
            self.logger.info("Loading Judge Model...")
            path = self.resolve(self.JUDGE_MODEL_NAME)
            self._judge_tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True
            )
            self._judge_model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype="auto", device_map="auto"
            )
        return self._judge_model, self._judge_tokenizer

    def _caption_video(self, video_path: str) -> str:
        """Generates a text description of the video using Qwen-VL."""
        model, processor = self._get_caption_model()

        # ... (Video captioning logic remains the same) ...
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{video_path}", "fps": 1.0},
                    {"type": "text", "text": "Describe this video in detail."},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=256)
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def _parse_judge_output(self, raw_text):
        """Parses the 7-line judge output into a dictionary."""
        # ... (Parsing logic remains the same) ...
        data = {"raw_output": raw_text}
        try:
            matches = re.findall(r"(.*?):\s*(.*)", raw_text)
            for key, value in matches:
                k = key.strip()
                v = value.strip()
                data[k] = int(v) if v.isdigit() else v
        except Exception:
            pass
        return data

    def _judge_t2v(self, prompt: str, caption: str):
        """Uses the Qwen-Instruct model to score the generated caption against the prompt."""
        model, tokenizer = self._get_judge_model()

        # ... (Judging logic remains the same) ...
        user_msg = f"PROMPT:\n{prompt}\n\nVIDEO DESCRIPTION:\n{caption}\n"
        messages = [
            {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=256, temperature=0.01
            )

        generated_ids = [
            out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return self._parse_judge_output(response)

    # --- Main Test Execution ---

    def test_t2v_quality(self):
        """Main test function to orchestrate video generation and evaluation."""
        full_prompt = "A cute cat eating a large piece of pepperoni pizza."  # Better, harder prompt

        # 1. Run Generation
        original_cwd = os.getcwd()
        os.chdir(self.job_out_dir)

        # 1.1 Define environment (CRITICAL for quality/library paths)
        sglang_env = os.environ.copy()
        # NOTE: Add crucial library paths/cache dirs here, e.g.,
        # sglang_env["LD_LIBRARY_PATH"] = "/path/to/fast/cuda/libs"

        sglang_command = [
            "sglang",
            "generate",
            "--model-path",
            self.WAN_MODEL,
            "--num-gpus",
            self.NUM_GPUS,
            "--text-encoder-cpu-offload",
            "false",
            "--dit-cpu-offload",
            "false",
            "--vae-cpu-offload",
            "false",
            "--num-frames",
            "8",
            "--width",
            "256",
            "--height",
            "256",
            "--num-inference-steps",
            "20",
            "--prompt",
            full_prompt,
            "--save-output",
            "--override-protected-fields",
        ]

        self.run_command(sglang_command, cwd=self.job_out_dir, env=sglang_env)
        os.chdir(original_cwd)

        # 2. Locate result
        search_pattern = os.path.join(self.job_out_dir, "**", "*.mp4")
        list_of_files = glob.glob(search_pattern, recursive=True)

        self.assertTrue(
            list_of_files, f"T2V generation failed: No .mp4 found in {self.job_out_dir}"
        )

        # CRITICAL FIX: Add delay for file handles to release
        self.logger.info(
            f"Waiting {self.FILE_DELAY_SEC} seconds for file handles to release..."
        )
        time.sleep(self.FILE_DELAY_SEC)

        self.video_path = max(list_of_files, key=os.path.getmtime)
        self.logger.info(f"Video found: {self.video_path}")
        self.assertTrue(
            os.path.exists(self.video_path), "Generated video not found after delay."
        )

        # 3. Run Evaluation (Internal Call)
        # We must clear cache since we are reusing the GPU context for the VLM/LLM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("Starting Captioning and Judging.")

        try:
            caption = self._caption_video(self.video_path)
            results = self._judge_t2v(full_prompt, caption)
        except Exception as e:
            # If VLM/LLM fails to load/run, treat it as a test failure.
            self.fail(f"VLM/LLM judging process failed: {e}")
            return

        score = results.get("Overall Score", results.get("OverallScore", 0))

        if not isinstance(score, int):
            try:
                score = int(score)
            except:
                score = 0

        self.logger.info(f"Final Score: {score}. Reason: {results.get('Reason')}")

        # 4. Final Assertion
        self.assertGreaterEqual(
            score,
            self.PASS_THRESHOLD,
            f"T2V quality test failed! Score ({score}) is below threshold ({self.PASS_THRESHOLD}). Details: {json.dumps(results, indent=2)}",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[TEST] %(message)s")
    unittest.main()
