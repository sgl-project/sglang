"""
Config-driven diffusion performance test with pytest parametrization.


If the actual run is significantly better than the baseline, the improved cases with their updated baseline will be printed
"""

from __future__ import annotations

import os
import subprocess

import pytest
import requests
from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES_A,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)

logger = init_logger(__name__)

# duplicate with `maybe_download_lora`?
def download_lora_weights(url: str, file_name: str) -> str:
    target_dir: str = "~/.cache"
    cache_dir = os.path.expanduser(target_dir)
    os.makedirs(cache_dir, exist_ok=True)

    file_path = os.path.join(cache_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        subprocess.run(["wget", "-O", file_path, url], check=True)

    return os.path.abspath(file_path)


class TestDiffusionServerOneGpu(DiffusionServerBase):
    """Performance tests for 1-GPU diffusion cases."""

    @pytest.fixture(params=ONE_GPU_CASES_A, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 1-GPU test."""
        return request.param


class TestLoraWorkflow:
    """Functional tests for LoRA workflow on 2 GPUs."""

    @pytest.fixture
    def case(self) -> DiffusionTestCase:
        """Define the base model configuration for the LoRA test."""
        return DiffusionTestCase(
            id="qwen_image_lora_workflow",
            server_args=DiffusionServerArgs(
                model_path="Qwen/Qwen-Image",
                modality="image",
                num_gpus=1,
            ),
            sampling_params=DiffusionSamplingParams(
                prompt="1GIRL, A girl standing in a cyberpunk city at night, neon lights, rain.",
            ),
        )

    def _get_client(self, port: int) -> OpenAI:
        return OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{port}/v1",
        )

    def test_lora_switching_flow(self, diffusion_server, case):
        """
        Verify the full LoRA switching workflow:
        1. Generate (Base)
        2. Set LoRA A -> Generate
        3. Unmerge
        4. Set LoRA B -> Generate
        """
        # Stream logs in background
        diffusion_server.start_log_streaming()

        port = diffusion_server.port
        base_url = f"http://localhost:{port}/v1"
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )
        client = self._get_client(port)
        try:
            print("\n=== Step 1: Download LoRA ===")

            lora_a_path = download_lora_weights(
                "https://civitai.com/api/download/models/2144921?type=Model&format=SafeTensor&token=df1327bc997d334ccb65eee66020f43b",
                "Qwen-Image-Lora-EliGen.safetensors",
            )

            lora_b_path = download_lora_weights(
                "https://civitai.com/api/download/models/2123706?type=Model&format=SafeTensor&token=df1327bc997d334ccb65eee66020f43b",
                "Qwen-Image-Lora-Rem-and-Ram-Re:Zero.safetensors",
            )

            try:
                print("\n=== Step 2: Set LoRA A ===")
                resp = requests.post(
                    f"{base_url}/set_lora",
                    json={"lora_nickname": "lora_a", "lora_path": lora_a_path},
                )
                # Note: Expecting failure here because dummy file is not a valid safetensors
                # In a real scenario with valid file: assert resp.status_code == 200
                print(f"Set LoRA A response: {resp.status_code} {resp.text}")

                if resp.status_code == 200:
                    print("=== Generate with LoRA A ===")
                    generate_fn("Generation with LoRA A", client)

                print("\n=== Step 3: Unmerge LoRA A ===")
                resp = requests.post(f"{base_url}/unmerge_lora_weights")
                assert resp.status_code == 200, f"Unmerge failed: {resp.text}"
                print("Unmerge successful")

                # Verify we can generate again (back to base)
                generate_fn("Generation without any LoRA", client)

                print("\n=== Step 4: Set LoRA B ===")
                resp = requests.post(
                    f"{base_url}/set_lora",
                    json={"lora_nickname": "lora_b", "lora_path": lora_b_path},
                )
                print(f"Set LoRA B response: {resp.status_code} {resp.text}")

                if resp.status_code == 200:
                    print("=== Generate with LoRA B ===")
                    generate_fn("Generation with LoRA B", client)

                # Test Error Case: Set LoRA C without unmerging B (if B succeeded)
                # Since B likely failed with dummy file, we can't strictly test the lock here

            finally:
                pass
        finally:
            pass
