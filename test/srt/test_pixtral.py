"""
python3 -m unittest test_pixtral

"""

import asyncio
import base64
import json
import math
import os
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

from python.sglang.srt.configs.model_config import ModelConfig
from python.sglang.srt.model_executor.model_runner import ModelRunner
from python.sglang.srt.server_args import PortArgs, ServerArgs


class RawPixtralTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Define the models to test
        self.models = {
            "large": "mistralai/Pixtral-Large-Instruct-2411",
            "base": "mistralai/Pixtral-12B-2409",
        }

        # Initialize objects for each model
        self.tokenizers = {}
        self.models_obj = {}
        self.processors = {}
        self.devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for key, model_name in self.models.items():
            # Load tokenizer
            self.tokenizers[key] = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            # Load model
            self.models_obj[key] = (
                LlavaForConditionalGeneration.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
                )
                .eval()
                .to(self.devices)
            )

            # Load processor
            self.processors[key] = AutoProcessor.from_pretrained(model_name)

    def test_vision_encoder(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true",
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        for key, model_name in self.models.items():
            print(f"\n=== Testing {model_name} ===")

            # Get the tokenizer, model, and processor for the current model
            tokenizer = self.tokenizers[key]
            model = self.models_obj[key]
            processor = self.processors[key]

            # Apply chat template to generate text input
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Download the image
            response = requests.get(messages[0]["content"][0]["image"])
            main_image = Image.open(BytesIO(response.content))

            # Process inputs using the processor
            inputs = processor(
                text=[text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )

            # Hugging Face model output
            with torch.no_grad():
                hf_output = model.visual(
                    inputs["pixel_values"].to(self.devices),
                    grid_thw=inputs["image_grid_thw"].to(self.devices),
                )

            # SGLang model setup
            model_config = ModelConfig(model_name, model_override_args="{}")
            server_args = ServerArgs(model_path=model_name)
            model_runner = ModelRunner(
                model_config=model_config,
                mem_fraction_static=0.8,
                gpu_id=0,
                tp_rank=0,
                tp_size=1,
                nccl_port=12435,
                server_args=server_args,
            )

            # SGLang model output
            with torch.no_grad():
                sglang_output = model_runner.model.visual(
                    inputs["pixel_values"].to(self.devices),
                    grid_thw=inputs["image_grid_thw"].to(self.devices),
                )

            # Comparison metrics
            hf = hf_output.float()
            sg = sglang_output.float()

            # Basic shape and dtype comparison
            print("\n=== Basic Properties ===")
            print(f"Shapes match: {hf.shape == sg.shape}")
            print(f"HF shape: {hf.shape}, SGLang shape: {sg.shape}")
            print(f"HF dtype: {hf.dtype}, SGLang dtype: {sg.dtype}")

            # Statistical metrics
            print("\n=== Statistical Metrics ===")
            print(
                f"Mean absolute difference: {torch.mean(torch.abs(hf - sg)).item():.6f}"
            )
            print(
                f"Max absolute difference: {torch.max(torch.abs(hf - sg)).item():.6f}"
            )
            print(f"Mean squared error: {torch.mean((hf - sg) ** 2).item():.6f}")
            print(
                f"Root mean squared error: {torch.sqrt(torch.mean((hf - sg) ** 2)).item():.6f}"
            )

            # Cosine similarity
            cos_sim = F.cosine_similarity(hf, sg)
            print(f"Mean cosine similarity: {torch.mean(cos_sim).item():.6f}")
            print(f"Min cosine similarity: {torch.min(cos_sim).item():.6f}")

            # Largest absolute differences
            print("\n=== Largest Absolute Differences ===")
            diffs = torch.abs(hf - sg)
            flat_diffs = diffs.flatten()
            top_k = 10
            top_values, top_flat_indices = torch.topk(flat_diffs, top_k)
            top_indices = np.unravel_index(top_flat_indices.cpu().numpy(), diffs.shape)

            print(f"\nTop {top_k} largest absolute differences:")
            print(
                "Index".ljust(30)
                + "Difference".ljust(15)
                + "HF Value".ljust(15)
                + "SGLang Value"
            )
            print("-" * 75)

            for i in range(top_k):
                idx = tuple(dim[i] for dim in top_indices)
                diff_val = top_values[i].item()
                hf_val = hf[idx].item()
                sg_val = sg[idx].item()
                print(f"{str(idx):<30}{diff_val:<15.6f}{hf_val:<15.6f}{sg_val:.6f}")

            # Assert outputs are close
            np.testing.assert_allclose(hf.cpu().numpy(), sg.cpu().numpy(), rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
