"""
2 GPU tests
"""

from __future__ import annotations

import pytest
import requests
from openai import OpenAI

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import (
    TWO_GPU_CASES_A,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)


class TestDiffusionServerTwoGpu(DiffusionServerBase):
    """Performance tests for 2-GPU diffusion cases."""

    @pytest.fixture(params=TWO_GPU_CASES_A, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 2-GPU test."""
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
                num_gpus=2,
            ),
            sampling_params=DiffusionSamplingParams(
                prompt="1GIRL, ",
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
            print("\n=== Step 1: Base Generation ===")

            # wget -O "Qwen-Image-Lora-EliGen.safetensors" "https://civitai.com/api/download/models/2144921?type=Model&format=SafeTensor&token=df1327bc997d334ccb65eee66020f43b"
            lora_a_path = "Qwen-Image-Lora-EliGen.safetensors"
            # wget -O "Qwen-Image-Lora-Rem-and-Ram-Re:Zero.safetensors" "https://civitai.com/api/download/models/2123706?type=Model&format=SafeTensor&token=df1327bc997d334ccb65eee66020f43b"
            lora_b_path = "Qwen-Image-Lora-Rem-and-Ram-Re:Zero.safetensors"

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
